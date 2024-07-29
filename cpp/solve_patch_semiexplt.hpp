#pragma once

#include "eigen3/Eigen/Dense"

#include "KernelData.hpp"
#include "Patch.hpp"
#include "PatchCstm.hpp"
#include "PatchData.hpp"
#include "ProblemDataFluxCstm.hpp"
#include "minimise_flux.hpp"
#include "solve_patch_weaksym.hpp"
#include "utils.hpp"

#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/graph/AdjacencyList.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <ios>
#include <iostream>
#include <iterator>
#include <span>
#include <tuple>
#include <vector>

using namespace dolfinx;

namespace dolfinx_eqlb
{

namespace stdex = std::experimental;

template <typename T>
void calculate_jump(
    mdspan_t<T, 2> GtHat_Ea, const std::size_t nq, const std::size_t iq_Ta,
    const std::int8_t Ea_reversed, std::span<const std::int32_t> dofs_G_Ea,
    std::span<const T> G_Tap1Ea, smdspan_t<const double, 2> shp_Tap1Ea,
    smdspan_t<const double, 1> hat_Tap1Ea, std::span<const T> G_TaEa,
    smdspan_t<const double, 2> shp_TaEa, smdspan_t<const double, 1> hat_TaEa)
{
  // Number of DOFs per facet
  const int ndofs_projflux_fct = dofs_G_Ea.size() / 2;

  // Id of quadrature point on reversed facet
  const std::size_t iq_Tap1 = (Ea_reversed) ? nq - iq_Ta - 1 : iq_Ta;
  // const std::size_t iq_Tap1 = (Ea_reversed) ? iq_Ta : iq_Ta;

  // Initialise jump with zero
  for (std::size_t i = 0; i < GtHat_Ea.extent(1); ++i)
  {
    GtHat_Ea(0, i) = 0.0;
    GtHat_Ea(1, i) = 0.0;
  }

  // Interpolate projected flux
  for (std::size_t i = 0; i < ndofs_projflux_fct; ++i)
  {
    // Local and global IDs of first DOF on facet
    int id_Ta = dofs_G_Ea[i + ndofs_projflux_fct];
    int id_Tap1 = dofs_G_Ea[i];
    int offs_Ta = 2 * id_Ta;
    int offs_Tap1 = 2 * id_Tap1;

    // Cell Ta
    GtHat_Ea(0, 0) += G_TaEa[offs_Ta] * shp_TaEa(iq_Ta, id_Ta);
    GtHat_Ea(0, 1) += G_TaEa[offs_Ta + 1] * shp_TaEa(iq_Ta, id_Ta);

    // Cell Tap1
    GtHat_Ea(1, 0) += G_Tap1Ea[offs_Tap1] * shp_Tap1Ea(iq_Tap1, id_Tap1);
    GtHat_Ea(1, 1) += G_Tap1Ea[offs_Tap1 + 1] * shp_Tap1Ea(iq_Tap1, id_Tap1);
  }

  // Multiplication with hat-function
  GtHat_Ea(0, 0) *= hat_TaEa(iq_Ta);
  GtHat_Ea(0, 1) *= hat_TaEa(iq_Ta);

  GtHat_Ea(1, 0) *= hat_Tap1Ea(iq_Tap1);
  GtHat_Ea(1, 1) *= hat_Tap1Ea(iq_Tap1);
}

/// Calculate equilibrated fluxes on patch
///
/// Calculates sig in pice-wise H(div) that fulfills jump and divergence
/// condition on patch (see [1, Appendix A, Algorithm 2]). The explicit setp is
/// followed by a unconstrained minimisation on a patch-wise H(div=0) space.
///
/// [1] Bertrand, F.; Carstensen, C.; Gräßle, B. & Tran, N. T.:
///     Stabilization-free HHO a posteriori error control, 2022
///
/// @tparam T              The scalar type
/// @tparam id_flux_order  Parameter for flux order (1->RT1, 2->RT2, 3->general)
/// @param geometry        The geometry
/// @param fct_perms       The permutation data for the facets of each cell
/// @param patch           The patch
/// @param patch_data      The temporary storage for the patch
/// @param problem_data    The problem data (Functions of flux, flux_dg, RHS_dg)
/// @param kernel_data     The kernel data (Quadrature data, tabulated basis)
/// @param minkernel       The kernel for unconstrained minimisation
/// @param minkernel_rhs   The kernel (RHS) for unconstrained minimisation
template <typename T, int id_flux_order>
void equilibrate_flux_semiexplt(const mesh::Geometry& geometry,
                                std::span<const std::uint8_t> fct_perms,
                                PatchFluxCstm<T, id_flux_order>& patch,
                                PatchDataCstm<T, id_flux_order>& patch_data,
                                ProblemDataFluxCstm<T>& problem_data,
                                KernelDataEqlb<T>& kernel_data,
                                kernel_fn<T, true>& minkernel,
                                kernel_fn<T, false>& minkernel_rhs)
{
  /* Extract data */
  // Spacial dimension
  const int dim = 2;

  // The geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();
  std::span<const double> x = geometry.x();

  // The patch
  std::span<const std::int32_t> cells = patch.cells();
  std::span<const std::int32_t> fcts = patch.fcts();
  const int ncells = patch.ncells();
  const int nfcts = patch.nfcts();

  // The cell
  const int nnodes_cell = kernel_data.nnodes_cell();
  const int nfcts_cell = kernel_data.nfacets_cell();

  // DOF-counts function spaces
  const int degree_flux_rt = patch.degree_raviart_thomas();

  const int ndofs_flux = patch.ndofs_flux();
  const int ndofs_flux_fct = patch.ndofs_flux_fct();
  const int ndofs_flux_cell_div = patch.ndofs_flux_cell_div();
  const int ndofs_flux_cell_add = patch.ndofs_flux_cell_add();

  const int ndofs_projflux = patch.ndofs_fluxdg_cell();
  const int ndofs_projflux_fct = patch.ndofs_fluxdg_fct();

  const int ndofs_rhs = patch.ndofs_rhs_cell();

  /* Initialise Mappings */
  // Representation/Storage isoparametric mapping
  std::array<double, 9> Jb, Kb;
  std::array<double, 18> detJ_scratch;
  mdspan_t<double, 2> J(Jb.data(), 2, 2), K(Kb.data(), 2, 2);

  // Storage cell geometry
  std::array<double, 12> coordinate_dofs_e;
  mdspan_t<const double, 2> coords(coordinate_dofs_e.data(), nnodes_cell, 3);

  // Storage pre-factors (due to orientation of the normal)
  mdspan_t<T, 2> prefactor_dof = patch_data.prefactors_facet_per_cell();

  // Storage for markers of reversed edges
  mdspan_t<std::uint8_t, 2> reversed_fct
      = patch_data.reversed_facets_per_cell();

  /* Initialise Step 1 */
  // The interpolation matrix
  mdspan_t<const double, 4> M = kernel_data.interpl_matrix_facte();
  mdspan_t<double, 4> M_mapped = patch_data.mapped_interpolation_matrix();

  // Coefficient arrays for RHS/ projected flux
  std::span<T> coefficients_f = patch_data.coefficients_rhs();
  std::span<T> coefficients_G_Ta = patch_data.coefficients_projflux_Ta();
  std::span<T> coefficients_G_Tap1 = patch_data.coefficients_projflux_Tap1();

  // Flux-jumps over facets
  std::array<T, 2> dGtHat_Ei, dGtHat_Ei_mapped;

  std::array<T, 4> dGtHat_Ea;
  std::array<T, 2> jGtHat;

  mdspan_t<T, 2> GtHat_Ea(dGtHat_Ea.data(), 2, 2);
  mdspan_t<T, 3> GtHat_Eam1 = patch_data.jumpG_Eam1();

  // Storage for cell-wise solution
  T c_ta_ea = 0, c_ta_eam1 = 0, c_tam1_eam1 = 0, c_t1_e0 = 0;
  std::span<T> c_ta_div = patch_data.c_ta_div(),
               cj_ta_ea = patch_data.cj_ta_ea(),
               cj_interm = patch_data.cj_intermediate();

  /* Initialise Step 2 */
  // Number of DOFs on patch-wise H(div=0) space
  const int ndofs_hdivz = patch.ndofs_flux_hdiz_zero();
  const int ndofs_hdivz_per_cell = 2 * ndofs_flux_fct + ndofs_flux_cell_add;

  /* Pre-evaluate repeatedly used cell data */
  // Jacobi transformation, facet orientation and interpolation matrix
  // TODO - Check if patch_has_reversed_facets is required
  bool patch_has_reversed_facets = false;

  for (std::size_t index = 0; index < ncells; ++index)
  {
    // Index using patch nomenclature
    int a = index + 1;

    // Get current cell
    std::int32_t c = cells[a];

    // Copy points of current cell
    std::span<const std::int32_t> x_dofs = x_dofmap.links(c);
    copy_cell_data<double, 3>(x, x_dofs, coordinate_dofs_e, 3);

    /* Piola mapping */
    // Calculate Jacobi, inverse, and determinant
    double detJ = kernel_data.compute_jacobian(J, K, detJ_scratch, coords);
    patch_data.store_piola_mapping(index, detJ, J, K);

    /* DOF transformation */
    // Local facet ids
    std::int8_t fctloc_ta_eam1, fctloc_ta_ea;
    std::tie(fctloc_ta_eam1, fctloc_ta_ea) = patch.fctid_local(a);

    // Normal orientation
    bool noutward_ta_eam1, noutward_ta_ea;
    std::tie(noutward_ta_eam1, noutward_ta_ea)
        = kernel_data.fct_normal_is_outward(fctloc_ta_eam1, fctloc_ta_ea);

    // Check for reversed facets
    if (patch.type(0) != PatchType::internal && ((a == 1) || (a == ncells)))
    {
      if (a == 1)
      {
        // Check if facet 1 is reversed
        std::int32_t c_ap1 = cells[a + 1];
        std::int8_t fctloc_tap1_ea = patch.fctid_local(a, a + 1);

        std::uint8_t perm_ta_ea = fct_perms[c * nfcts_cell + fctloc_ta_ea];
        std::uint8_t perm_tap1_ea
            = fct_perms[c_ap1 * nfcts_cell + fctloc_tap1_ea];

        if (perm_ta_ea != perm_tap1_ea)
        {
          patch_has_reversed_facets = true;
          reversed_fct(index, 1) = true;
        }
      }
      else if (a = ncells)
      {
        // Check if facet a-1 is reversed
        std::int32_t c_am1 = cells[a - 1];
        std::int8_t fctloc_tam1_eam1 = patch.fctid_local(a - 1, a - 1);

        std::uint8_t perm_tam1_eam1
            = fct_perms[c_am1 * nfcts_cell + fctloc_tam1_eam1];
        std::uint8_t perm_ta_eam1 = fct_perms[c * nfcts_cell + fctloc_ta_eam1];

        reversed_fct(index, 0)
            = (perm_tam1_eam1 != perm_ta_eam1) ? true : false;

        if (perm_tam1_eam1 != perm_ta_eam1)
        {
          patch_has_reversed_facets = true;
          reversed_fct(index, 0) = true;
        }
      }
    }
    else
    {
      // The neighboring cells
      std::int32_t c_am1 = cells[a - 1];
      std::int32_t c_ap1 = cells[a + 1];

      // Local facet ids of ea resp. eam1 on neighboring cells
      std::int8_t fctloc_tam1_eam1 = patch.fctid_local(a - 1, a - 1);
      std::int8_t fctloc_tap1_ea = patch.fctid_local(a, a + 1);

      // The facet permutation data
      std::uint8_t perm_tam1_eam1
          = fct_perms[c_am1 * nfcts_cell + fctloc_tam1_eam1];
      std::uint8_t perm_ta_eam1 = fct_perms[c * nfcts_cell + fctloc_ta_eam1];
      std::uint8_t perm_ta_ea = fct_perms[c * nfcts_cell + fctloc_ta_ea];
      std::uint8_t perm_tap1_ea
          = fct_perms[c_ap1 * nfcts_cell + fctloc_tap1_ea];

      // Set markers if facets are reversed
      if (perm_tam1_eam1 != perm_ta_eam1)
      {
        patch_has_reversed_facets = true;
        reversed_fct(index, 0) = true;
      }

      if (perm_ta_ea != perm_tap1_ea)
      {
        patch_has_reversed_facets = true;
        reversed_fct(index, 1) = true;
      }
    }

    std::cout << "Cell " << c << " has facets " << fcts[a - 1] << " ("
              << unsigned(reversed_fct(index, 0)) << ")" << " and " << fcts[a]
              << " (" << unsigned(reversed_fct(index, 1)) << ")" << std::endl;

    // Sign of the jacobian
    const double sgn_detJ = detJ / std::fabs(detJ);

    // Set prefactors
    prefactor_dof(index, 0) = (noutward_ta_eam1) ? sgn_detJ : -sgn_detJ;
    prefactor_dof(index, 1) = (noutward_ta_ea) ? sgn_detJ : -sgn_detJ;

    std::cout << "Cell has normal-orientations " << noutward_ta_eam1 << " and "
              << noutward_ta_ea << "with sign_detJ " << sgn_detJ
              << "--> pfkt: " << prefactor_dof(index, 0) << ", "
              << prefactor_dof(index, 1) << std::endl;

    /* Apply push-back on interpolation matrix M */
    for (std::size_t i = 0; i < M_mapped.extent(1); ++i)
    {
      for (std::size_t j = 0; j < M_mapped.extent(3); ++j)
      {
        // Select facet
        // (Zero-order moment on facet Eam1 and Ea, higher-order moments on Ea)
        const std::int8_t fctid = (i == 0) ? fctloc_ta_eam1 : fctloc_ta_ea;
        const std::int8_t ii = (i < 2) ? 0 : i - 1;

        // Map interpolation cell Tam1
        M_mapped(index, i, 0, j)
            = detJ
              * (M(fctid, ii, 0, j) * K(0, 0) + M(fctid, ii, 1, j) * K(1, 0));
        M_mapped(index, i, 1, j)
            = detJ
              * (M(fctid, ii, 0, j) * K(0, 1) + M(fctid, ii, 1, j) * K(1, 1));
      }
    }
  }

  // DOFmap for minimisation
  patch.set_assembly_informations(kernel_data.fct_normal_is_outward(),
                                  patch_data.jacobi_determinant());

  mdspan_t<const std::int32_t, 3> dofmap_flux
      = patch.assembly_info_minimisation();
  std::span<const std::int32_t> offs_dofmap = patch.offset_dofmap();

  const int offs_ffEa = offs_dofmap[1];
  const int offs_fcadd = offs_dofmap[2];
  const int offs_fcdiv = offs_dofmap[4];

  // Binomial coefficients for DOF tranformation on reversed facet
  mdspan_t<const T, 2> pascals_triangle
      = patch_data.transformation_factors_facet_integrals();

  /* Evaluate DOFs of sigma_tilde (for each flux separately) */
  for (std::size_t i_rhs = 0; i_rhs < problem_data.nrhs(); ++i_rhs)
  {
    /* Extract data */
    // Patch type
    PatchType type_patch = patch.type(i_rhs);

    // Check if reversion is requierd
    bool reversion_required = patch.reversion_required(i_rhs);

    // Equilibarted flux
    mdspan_t<T, 2> coefficients_flux = patch_data.coefficients_flux(i_rhs);

    // Projected primal flux
    const graph::AdjacencyList<std::int32_t>& fluxdg_dofmap
        = problem_data.fspace_flux_dg()->dofmap()->list();
    std::span<const T> x_flux_proj
        = problem_data.projected_flux(i_rhs).x()->array();

    // Projected RHS
    const graph::AdjacencyList<std::int32_t>& rhs_dofmap
        = problem_data.fspace_flux_dg()->dofmap()->list();
    std::span<const T> x_rhs_proj
        = problem_data.projected_rhs(i_rhs).x()->array();

    // Boundary values
    std::span<const T> boundary_values = problem_data.boundary_values(i_rhs);

    /* Step 1: Calculate sigma_tilde */
    // Initialisations
    copy_cell_data<T, 2>(x_flux_proj, fluxdg_dofmap.links(cells[1]),
                         coefficients_G_Tap1, 2);

    // Reinitialise history storage
    if (i_rhs > 0)
    {
      c_tam1_eam1 = 0.0;
      c_t1_e0 = 0.0;
      patch_data.reinitialise_jumpG_Eam1();
    }

    // Loop over all cells
    for (std::size_t a = 1; a < ncells + 1; ++a)
    {
      // Set id for accessing storage
      int id_a = a - 1;

      // Global cell id
      std::int32_t c_a = cells[a];

      // Cell-local id of patch-central node
      std::int8_t node_i_Ta = patch.inode_local(a);
      std::int8_t node_i_Tap1 = patch.inode_local(a + 1);

      // Check if facet is on boundary
      bool fct_on_boundary
          = ((patch.is_on_boundary()) && (a == 1 || a == ncells)) ? true
                                                                  : false;

      // Check if bcs have to be applied
      bool fct_has_bc = false;

      if (fct_on_boundary == true)
      {
        if (type_patch == PatchType::bound_essnt_dual)
        {
          fct_has_bc = true;
        }
        else if (type_patch == PatchType::bound_mixed)
        {
          if (a == 1)
          {
            fct_has_bc = patch.requires_flux_bcs(i_rhs, 0);
          }
          else if (a == ncells)
          {
            fct_has_bc = patch.requires_flux_bcs(i_rhs, ncells);
          }
          else
          {
            fct_has_bc = false;
          }
        }
        else
        {
          fct_has_bc = false;
        }
      }

      /* Extract required data */
      // Local facet IDs
      std::int8_t fl_TaEam1, fl_TaEa;
      std::tie(fl_TaEam1, fl_TaEa) = patch.fctid_local(a);
      std::int8_t fl_Tap1Ea = patch.fctid_local(a, a + 1);

      // Isoparametric mapping
      const double detJ = patch_data.jacobi_determinant(id_a);
      const double sign_detJ = (detJ > 0.0) ? 1.0 : -1.0;

      // Coefficient arrays
      std::swap(coefficients_G_Ta, coefficients_G_Tap1);

      copy_cell_data<T, 2>(x_flux_proj, fluxdg_dofmap.links(cells[a + 1]),
                           coefficients_G_Tap1, 2);
      copy_cell_data<T, 1>(x_rhs_proj, rhs_dofmap.links(c_a), coefficients_f,
                           1);

      /* Tabulate shape functions */
      // Shape functions RHS
      smdspan_t<const double, 2> shp_TaEa
          = kernel_data.shapefunctions_fct_rhs(fl_TaEa);
      smdspan_t<const double, 2> shp_Tap1Ea
          = kernel_data.shapefunctions_fct_rhs(fl_Tap1Ea);

      // Shape-functions hat-function
      smdspan_t<const double, 1> hat_TaEam1
          = kernel_data.shapefunctions_fct_hat(fl_TaEam1, node_i_Ta);
      smdspan_t<const double, 1> hat_TaEa
          = kernel_data.shapefunctions_fct_hat(fl_TaEa, node_i_Ta);
      smdspan_t<const double, 1> hat_Tap1Ea
          = kernel_data.shapefunctions_fct_hat(fl_Tap1Ea, node_i_Tap1);

      /* Prepare data for inclusion of flux BCs */
      // DOFs on facet E0
      std::span<const std::int32_t> pflux_ldofs_E0;

      // Tabulated shape functions on facet E0
      smdspan_t<const double, 2> shp_TaEam1;

      if ((a == 1) && (fct_has_bc || type_patch == PatchType::bound_mixed))
      {
        // DOFs (cell-local) projected flux on facet E0
        pflux_ldofs_E0 = patch.dofs_projflux_fct(0);

        // Tabulate shape functions RHS on facet 0
        shp_TaEam1 = kernel_data.shapefunctions_fct_rhs(fl_TaEam1);
      }

      /* DOFs from facet integrals */
      // Initialisation
      c_ta_eam1 = -c_tam1_eam1;
      std::fill(cj_ta_ea.begin(), cj_ta_ea.end(), 0.0);

      // Consider flux BCs
      if (fct_has_bc)
      {
        // Get (global) boundary facet
        std::int32_t bfct_global, offs_bdofs;
        if (a == 1)
        {
          offs_bdofs = 0;
          bfct_global = patch.fct(0);
        }
        else
        {
          offs_bdofs = ndofs_flux_fct;
          bfct_global = patch.fct(a);
        }

        // Calculate patch bcs
        mdspan_t<const double, 2> J = patch_data.jacobian(0);
        mdspan_t<const double, 2> K = patch_data.inverse_jacobian(0);

        problem_data.calculate_patch_bc(i_rhs, bfct_global, node_i_Ta, J, detJ,
                                        K);

        // Contribution to c_ta_eam1
        if (a == 1)
        {
          c_ta_eam1 += prefactor_dof(id_a, 0)
                       * boundary_values[dofmap_flux(1, a, offs_bdofs)];
        }

        // Contribution to cj_ta_ea
        if constexpr (id_flux_order > 1)
        {
          if constexpr (id_flux_order == 2)
          {
            coefficients_flux(id_a, dofmap_flux(0, a, offs_bdofs + 1))
                += boundary_values[dofmap_flux(1, a, offs_bdofs + 1)];
          }
          else
          {
            for (std::size_t j = 1; j < ndofs_flux_fct; ++j)
            {
              coefficients_flux(id_a, dofmap_flux(0, a, offs_bdofs + j))
                  += boundary_values[dofmap_flux(1, a, offs_bdofs + j)];
            }
          }
        }

        // Handle mixed patch with E_0 on dirichlet boundary
        if (reversion_required)
        {
          c_t1_e0 -= prefactor_dof(id_a, 1)
                     * boundary_values[dofmap_flux(1, a, offs_bdofs)];
        }
      }

      // Interpolate DOFs
      int nq_fct = kernel_data.nipoints_facet();

      T surfint_c_ta_eam1 = 0.0, test = 0.0, test2 = 0.0;

      for (std::size_t n = 0; n < nq_fct; ++n)
      {
        // Interpolate jump (facet Ea) at quadrature point
        std::size_t nq_Tap1 = (reversed_fct(id_a, 1)) ? nq_fct - 1 - n : n;

        if (fct_on_boundary)
        {
          if (a == 1)
          {
            // Handle BCs on facet 0
            if (fct_has_bc || type_patch == PatchType::bound_mixed)
            {
              // Evaluate jump
              for (std::size_t i = 0; i < ndofs_projflux_fct; ++i)
              {
                // Local and global IDs of first DOF on facet
                int id_Ta = pflux_ldofs_E0[i + ndofs_projflux_fct];
                int offs_Ta = 2 * id_Ta;

                // Evaluate jump
                GtHat_Eam1(n, 0, 0)
                    -= coefficients_G_Ta[offs_Ta] * shp_TaEam1(n, id_Ta);
                GtHat_Eam1(n, 0, 1)
                    -= coefficients_G_Ta[offs_Ta + 1] * shp_TaEam1(n, id_Ta);
              }

              GtHat_Eam1(n, 0, 0) *= hat_TaEam1(n);
              GtHat_Eam1(n, 0, 1) *= hat_TaEam1(n);

              // Evaluate higher order DOFs on boundary facet
              if constexpr (id_flux_order > 1)
              {
                // Extract mapping data
                mdspan_t<const double, 2> J = patch_data.jacobian(id_a);
                mdspan_t<const double, 2> K = patch_data.inverse_jacobian(id_a);

                // Pull back flux to reference cell
                mdspan_t<T, 2> GtHat_E0(dGtHat_Ei.data(), 1, dim);
                mdspan_t<T, 2> GtHat_mapped_E0(dGtHat_Ei_mapped.data(), 1, dim);

                if ((type_patch == PatchType::bound_mixed)
                    && (fct_has_bc == false))
                {
                  GtHat_E0(0, 0) = -GtHat_Eam1(n, 0, 0);
                  GtHat_E0(0, 1) = -GtHat_Eam1(n, 0, 1);
                }
                else
                {
                  GtHat_E0(0, 0) = GtHat_Eam1(n, 0, 0);
                  GtHat_E0(0, 1) = GtHat_Eam1(n, 0, 1);
                }

                kernel_data.pull_back_flux(GtHat_mapped_E0, GtHat_E0, J, detJ,
                                           K);

                // Evaluate higher-order DOFs on facet E0
                if constexpr (id_flux_order == 2)
                {
                  coefficients_flux(id_a, dofmap_flux(0, a, 1))
                      += M(fl_TaEam1, 1, 0, n) * GtHat_mapped_E0(0, 0)
                         + M(fl_TaEam1, 1, 1, n) * GtHat_mapped_E0(0, 1);
                }
                else
                {
                  // Evaluate facet DOFs
                  for (std::size_t j = 1; j < ndofs_flux_fct; ++j)
                  {
                    coefficients_flux(id_a, dofmap_flux(0, a, j))
                        += M(fl_TaEam1, j, 0, n) * GtHat_mapped_E0(0, 0)
                           + M(fl_TaEam1, j, 1, n) * GtHat_mapped_E0(0, 1);
                  }
                }
              }

              // Handle mixed patch with E_0 on dirichlet boundary
              if (fct_has_bc == false)
              {
                // Unset jump on facet E0
                GtHat_Eam1(n, 0, 0) = 0.0;
                GtHat_Eam1(n, 0, 1) = 0.0;
              }
            }

            // Evaluate jump on facet E1
            calculate_jump<T>(GtHat_Ea, nq_fct, n, reversed_fct(id_a, 1),
                              patch.dofs_projflux_fct(a), coefficients_G_Tap1,
                              shp_Tap1Ea, hat_Tap1Ea, coefficients_G_Ta,
                              shp_TaEa, hat_TaEa);
          }
          else
          {
            // DOFs (cell-local) projected flux on facet Ea
            std::span<const std::int32_t> dofs_G_Ea
                = patch.dofs_projflux_fct(a);

            // Evaluate jump on facet En
            double pfctr = (fct_has_bc) ? -1.0 : 1.0;
            std::fill(dGtHat_Ea.begin(), dGtHat_Ea.end(), 0.0);

            for (std::size_t i = 0; i < ndofs_projflux_fct; ++i)
            {
              // Local and global IDs of first DOF on facet
              int id_Ta = dofs_G_Ea[i + ndofs_projflux_fct];
              int offs_Ta = 2 * id_Ta;

              // Jump
              double sshp_TaEa = pfctr * shp_TaEa(n, id_Ta);
              GtHat_Ea(1, 0) += coefficients_G_Ta[offs_Ta] * sshp_TaEa;
              GtHat_Ea(1, 1) += coefficients_G_Ta[offs_Ta + 1] * sshp_TaEa;
            }

            GtHat_Ea(1, 0) *= hat_TaEa(n);
            GtHat_Ea(1, 1) *= hat_TaEa(n);

            // Add boundary contribution to c_t1_e0
            // (required for mixed patch with facte E0 on dirichlet boundary)
            if (reversion_required)
            {
              // Extract mapping data
              mdspan_t<const double, 2> J = patch_data.jacobian(id_a);
              mdspan_t<const double, 2> K = patch_data.inverse_jacobian(id_a);

              // Pull back flux to reference cell
              mdspan_t<T, 2> GtHat_En(dGtHat_Ei.data(), 1, dim);
              mdspan_t<T, 2> GtHat_mapped_En(dGtHat_Ei_mapped.data(), 1, dim);

              GtHat_En(0, 0) = GtHat_Ea(1, 0);
              GtHat_En(0, 1) = GtHat_Ea(1, 1);

              kernel_data.pull_back_flux(GtHat_mapped_En, GtHat_En, J, detJ, K);

              // Evaluate boundary contribution
              T aux = M(fl_TaEa, 0, 0, n) * GtHat_mapped_En(0, 0)
                      + M(fl_TaEa, 0, 1, n) * GtHat_mapped_En(0, 1);
              c_t1_e0 -= prefactor_dof(id_a, 1) * aux;
            }
          }
        }
        else
        {
          calculate_jump<T>(GtHat_Ea, nq_fct, n, reversed_fct(id_a, 1),
                            patch.dofs_projflux_fct(a), coefficients_G_Tap1,
                            shp_Tap1Ea, hat_Tap1Ea, coefficients_G_Ta, shp_TaEa,
                            hat_TaEa);
        }

        // Zero order facet moments: Jump constribution
        jGtHat[0] = GtHat_Eam1(n, 1, 0) - GtHat_Eam1(n, 0, 0);
        jGtHat[1] = GtHat_Eam1(n, 1, 1) - GtHat_Eam1(n, 0, 1);

        std::cout << "Jump Eam1: " << jGtHat[0] << ", " << jGtHat[1]
                  << std::endl;

        surfint_c_ta_eam1 -= M_mapped(id_a, 0, 0, n) * jGtHat[0]
                             + M_mapped(id_a, 0, 1, n) * jGtHat[1];

        // Higher order facet moments: Jump constribution
        if constexpr (id_flux_order > 1)
        {
          if constexpr (id_flux_order == 2)
          {
            if (reversed_fct(id_a, 1))
            {
              // The jump
              jGtHat[0] = GtHat_Ea(1, 0) - GtHat_Ea(0, 0);
              jGtHat[1] = GtHat_Ea(1, 1) - GtHat_Ea(0, 1);

              // The second order facet moment
              if (a != ncells)
              {
                cj_ta_ea[0] += M_mapped(id_a, 2, 0, n) * jGtHat[0]
                               + M_mapped(id_a, 2, 1, n) * jGtHat[1]
                               - M_mapped(id_a, 1, 0, n) * jGtHat[0]
                               - M_mapped(id_a, 1, 1, n) * jGtHat[1];
              }
              else
              {
                cj_ta_ea[0] += M_mapped(id_a, 2, 0, n) * jGtHat[0]
                               + M_mapped(id_a, 2, 1, n) * jGtHat[1];
              }
            }
            else
            {
              // The jump
              jGtHat[0] = GtHat_Ea(1, 0) - GtHat_Ea(0, 0);
              jGtHat[1] = GtHat_Ea(1, 1) - GtHat_Ea(0, 1);

              // The second order facet moment
              cj_ta_ea[0] += M_mapped(id_a, 2, 0, n) * jGtHat[0]
                             + M_mapped(id_a, 2, 1, n) * jGtHat[1];
            }
          }
          else
          {
            if (reversed_fct(id_a, 1))
            {
              // The jump
              jGtHat[0] = GtHat_Ea(1, 0) - GtHat_Ea(0, 0);
              jGtHat[1] = GtHat_Ea(1, 1) - GtHat_Ea(0, 1);

              // The higher order facet moment
              if (a != ncells)
              {
                cj_ta_ea[0] += M_mapped(id_a, 2, 0, n) * jGtHat[0]
                               + M_mapped(id_a, 2, 1, n) * jGtHat[1]
                               - M_mapped(id_a, 1, 0, n) * jGtHat[0]
                               - M_mapped(id_a, 1, 1, n) * jGtHat[1];
                cj_ta_ea[1] += M_mapped(id_a, 3, 0, n) * jGtHat[0]
                               + M_mapped(id_a, 3, 1, n) * jGtHat[1]
                               - M_mapped(id_a, 1, 0, n) * jGtHat[0]
                               - M_mapped(id_a, 1, 1, n) * jGtHat[1];
              }
              else
              {
                // The jump
                jGtHat[0] = GtHat_Ea(1, 0) - GtHat_Ea(0, 0);
                jGtHat[1] = GtHat_Ea(1, 1) - GtHat_Ea(0, 1);

                cj_ta_ea[0] += M_mapped(id_a, 2, 0, n) * jGtHat[0]
                               + M_mapped(id_a, 2, 1, n) * jGtHat[1];
                cj_ta_ea[1] += M_mapped(id_a, 3, 0, n) * jGtHat[0]
                               + M_mapped(id_a, 3, 1, n) * jGtHat[1];
              }
            }
            else
            {
              // The jump
              jGtHat[0] = GtHat_Ea(1, 0) - GtHat_Ea(0, 0);
              jGtHat[1] = GtHat_Ea(1, 1) - GtHat_Ea(0, 1);

              // The higher order facet moments
              for (std::size_t j = 2; j < M_mapped.extent(1); ++j)
              {
                cj_ta_ea[j - 2] += M_mapped(id_a, j, 0, n) * jGtHat[0]
                                   + M_mapped(id_a, j, 1, n) * jGtHat[1];
              }
            }
          }
        }

        // Store jump-data
        GtHat_Eam1(n, 0, 0) = GtHat_Ea(0, 0);
        GtHat_Eam1(n, 0, 1) = GtHat_Ea(0, 1);
        GtHat_Eam1(n, 1, 0) = GtHat_Ea(1, 0);
        GtHat_Eam1(n, 1, 1) = GtHat_Ea(1, 1);
      }

      // Reverse calculated jumps on facet Ea
      if (reversed_fct(id_a, 1))
      {
        for (std::size_t i = 0; i < std::floor(nq_fct / 2); ++i)
        {
          // Index of reversed entry
          std::size_t ri = nq_fct - 1 - i;

          T interm = GtHat_Eam1(i, 0, 0);
          GtHat_Eam1(i, 0, 0) = GtHat_Eam1(ri, 0, 0);
          GtHat_Eam1(ri, 0, 0) = interm;
        }
      }

      c_ta_eam1 += prefactor_dof(id_a, 0) * surfint_c_ta_eam1;
      c_t1_e0 -= prefactor_dof(id_a, 0) * surfint_c_ta_eam1;

      // std::cout << "cell: " << c_a << std::endl;
      // std::cout << "-surfint_c_tap1_ea: " << cj_ta_ea[0] << std::endl;

      /* DOFs from cell integrals */
      if constexpr (id_flux_order == 1)
      {
        // Set DOF on facet Ea
        T vol_int = coefficients_f[0] * (std::fabs(detJ) / 6);
        c_ta_ea = vol_int - c_ta_eam1;

        c_t1_e0 += vol_int;
      }
      else
      {
        // Isoparametric mapping
        mdspan_t<const double, 2> K = patch_data.inverse_jacobian(id_a);

        // Quadrature points and weights
        mdspan_t<const double, 2> qpoints = kernel_data.quadrature_points(0);
        std::span<const double> weights = kernel_data.quadrature_weights(0);
        const int nqpoints = weights.size();

        // Shape-functions RHS
        smdspan_t<const double, 3> shp_rhs
            = kernel_data.shapefunctions_cell_rhs(K);

        // Shape-functions hat-function
        smdspan_t<const double, 2> shp_hat
            = kernel_data.shapefunctions_cell_hat();

        // Quadrature loop
        c_ta_ea = -c_ta_eam1;
        std::fill(c_ta_div.begin(), c_ta_div.end(), 0.0);

        for (std::size_t n = 0; n < nqpoints; ++n)
        {
          // Interpolation
          double f = 0.0;
          double div_g = 0.0;
          for (std::size_t i = 0; i < ndofs_rhs; ++i)
          {
            // RHS
            f += coefficients_f[i] * shp_rhs(0, n, i);

            // Divergence of projected flux
            const int offs = 2 * i;
            div_g += coefficients_G_Ta[offs] * shp_rhs(1, n, i)
                     + coefficients_G_Ta[offs + 1] * shp_rhs(2, n, i);
          }

          // Auxiliary data
          const double aux
              = (f - div_g) * shp_hat(n, node_i_Ta) * weights[n] * detJ;
          const double vol_int = aux * sign_detJ;

          // Evaluate facet DOF
          c_ta_ea += vol_int;

          // Contribution to c_t1_e0
          c_t1_e0 += vol_int;

          // Evaluate cell DOFs
          if constexpr (id_flux_order == 2)
          {
            c_ta_div[0] += aux * qpoints(n, 1);
            c_ta_div[1] += aux * qpoints(n, 0);
          }
          else
          {
            int count = 0;
            const int degree = degree_flux_rt + 1;

            for (std::size_t l = 0; l < degree; ++l)
            {
              for (std::size_t m = 0; m < degree - l; m++)
              {
                if ((l + m) > 0)
                {
                  // Calculate DOF
                  c_ta_div[count] += aux * std::pow(qpoints(n, 0), l)
                                     * std::pow(qpoints(n, 1), m);

                  // Increment counter
                  count += 1;
                }
              }
            }
          }
        }
      }

      /* Correct higher order facet moments */
      if constexpr (id_flux_order > 1)
      {
        if (reversed_fct(id_a, 1) && (a != ncells))
        {
          cj_ta_ea[0] += prefactor_dof(id_a + 1, 0) * c_ta_ea;

          if constexpr (id_flux_order > 2)
          {
            for (std::size_t i = 2; i < ndofs_flux_fct; ++i)
            {
              cj_ta_ea[i - 1] += prefactor_dof(id_a + 1, 0) * c_ta_ea;
            }
          }
        }
      }

      // std::cout << "surfint_c_ta_ea, cj_ta_ea, test, test2: "
      //           << surfint_c_ta_eam1 << ", " << cj_ta_ea[0] << ", " << test
      //           << ", " << test2 << std::endl;

      /* Store DOFs into patch-wise solution vector */
      // Set zero order DOFs
      coefficients_flux(id_a, dofmap_flux(0, a, 0))
          += prefactor_dof(id_a, 0) * c_ta_eam1;
      coefficients_flux(id_a, dofmap_flux(0, a, offs_ffEa))
          += prefactor_dof(id_a, 1) * c_ta_ea;

      if constexpr (id_flux_order > 1)
      {
        if constexpr (id_flux_order == 2)
        {
          // Set higher-order DOFs on facets
          coefficients_flux(id_a, dofmap_flux(0, a, offs_ffEa + 1))
              += cj_ta_ea[0];

          // Set DOFs on cell
          coefficients_flux(id_a, dofmap_flux(0, a, offs_fcdiv)) += c_ta_div[0];
          coefficients_flux(id_a, dofmap_flux(0, a, offs_fcdiv + 1))
              += c_ta_div[1];
        }
        else
        {
          // Set higher-order DOFs on facets
          for (std::size_t i = 1; i < ndofs_flux_fct; ++i)
          {
            // DOFs on facet Ea
            coefficients_flux(id_a, dofmap_flux(0, a, offs_ffEa + i))
                += cj_ta_ea[i - 1];
          }

          // Set divergence DOFs on cell
          for (std::size_t i = 0; i < ndofs_flux_cell_div; ++i)
          {
            coefficients_flux(id_a, dofmap_flux(0, a, offs_fcdiv + i))
                += c_ta_div[i];
          }
        }
      }

      // Update c_tam1_eam1
      c_tam1_eam1 = c_ta_ea;
    }

    /* Correct zero-order DOFs on reversed patch */
    if (reversion_required)
    {
      for (std::size_t a = 1; a < ncells + 1; ++a)
      {
        // Set id for accessing storage
        std::size_t id_a = a - 1;

        // Set zero-order DOFs on facets
        coefficients_flux(id_a, dofmap_flux(0, a, 0))
            += prefactor_dof(id_a, 0) * c_t1_e0;
        coefficients_flux(id_a, dofmap_flux(0, a, offs_ffEa))
            -= prefactor_dof(id_a, 1) * c_t1_e0;
      }
    }

    /* Step 2: Minimse sigma_delta */
    // Set boundary markers
    if (type_patch == PatchType::bound_essnt_dual
        || type_patch == PatchType::bound_mixed)
    {
      set_boundary_markers(patch_data.boundary_markers(false), {type_patch},
                           {reversion_required}, ncells, ndofs_hdivz,
                           ndofs_flux_fct);
    }

    // Check if assembly of entire system is required
    bool assemble_entire_system = false;

    if (i_rhs == 0)
    {
      assemble_entire_system = true;
    }
    else
    {
      if (patch.is_on_boundary())
      {
        if (patch.type(i_rhs) != patch.type(i_rhs - 1)
            || patch.type(i_rhs) == PatchType::bound_mixed)
        {
          assemble_entire_system = true;
        }
      }
    }

    // Assemble equation system
    if (assemble_entire_system)
    {
      // Assemble system
      assemble_fluxminimiser<T, id_flux_order, true>(
          minkernel, patch_data, dofmap_flux, i_rhs,
          patch.requires_flux_bcs(i_rhs));

      // Factorisation of system matrix
      patch_data.factorise_matrix_A();
    }
    else
    {
      // Assemble linear form
      assemble_fluxminimiser<T, id_flux_order, false>(
          minkernel_rhs, patch_data, dofmap_flux, i_rhs,
          patch.requires_flux_bcs(i_rhs));
    }

    // Solve system
    patch_data.solve_unconstrained_minimisation();

    /* Move patch-wise solution into global storage */
    // The patch-local solution
    Eigen::Matrix<T, Eigen::Dynamic, 1>& u_sigma = patch_data.vector_u_sigma();

    // Global solution vector and DOFmap
    std::span<T> x_flux_dhdiv = problem_data.flux(i_rhs).x()->mutable_array();
    const graph::AdjacencyList<std::int32_t>& flux_dofmap
        = problem_data.fspace_flux_hdiv()->dofmap()->list();

    // Move cell contributions
    for (std::int32_t a = 1; a < ncells + 1; ++a)
    {
      // Set id for accessing storage
      std::size_t id_a = a - 1;

      // Global DOFs
      std::span<const std::int32_t> gdofs = flux_dofmap.links(cells[a]);

      // Map solution from H(div=0) to H(div) space
      // for (std::size_t i = 0; i < ndofs_hdivz_per_cell; ++i)
      // {
      //   // Apply correction
      //   coefficients_flux(id_a, dofmap_flux(0, a, i))
      //       += dofmap_flux(3, a, i) * u_sigma(dofmap_flux(2, a, i));
      // }

      // Loop over DOFs an cell
      for (std::size_t i = 0; i < ndofs_flux; ++i)
      {
        // Set zero-order DOFs on facets
        x_flux_dhdiv[gdofs[i]] += coefficients_flux(id_a, i);
      }
    }
  }
}

/// Calculate equilibrated fluxes on patch with weak symmetry constraint
///
/// Calculates sig in pice-wise H(div) that fulfills jump and divergence
/// condition on patch (see [1, Appendix A, Algorithm 2]). The explicit setp is
/// followed by a unconstrained minimisation on a patch-wise H(div=0) space. The
/// first gdim fluxes form the stress tensor, wich has to fulfill a weak
/// symmetry condition. Is is enforced based on a constrained minimisation
/// problem, follwoing [2].
///
/// [1] Bertrand, F., Carstensen, C., Gräßle, B. & Tran, N. T.:
///     Stabilization-free HHO a posteriori error control, 2022
/// [2] Bertrand, F., Kober, B., Moldenhauer M. & Starke, G.: Weakly
///     symmetric stress equilibration and a posteriori error estimation for
///     linear elasticity, 2021
///
/// @tparam T              The scalar type
/// @tparam id_flux_order  Parameter for flux order (1->RT1, 2->RT2, 3->general)
/// @param geometry        The geometry
/// @param fct_perms       The permutation data for the facets of each cell
/// @param patch           The patch
/// @param patch_data      The temporary storage for the patch
/// @param problem_data    The problem data (Functions of flux, flux_dg, RHS_dg)
/// @param kernel_data     The kernel data (Quadrature data, tabulated basis)
/// @param minkernel       The kernel for unconstrained minimisation
/// @param minkernel_rhs   The kernel (RHS) for unconstrained minimisation
/// @param kernel_weaksym  The kernel for imposition of the weak symmetry
///                        constraind
template <typename T, int id_flux_order>
void equilibrate_flux_semiexplt(const mesh::Geometry& geometry,
                                std::span<const std::uint8_t> fct_perms,
                                PatchFluxCstm<T, id_flux_order>& patch,
                                PatchDataCstm<T, id_flux_order>& patch_data,
                                ProblemDataFluxCstm<T>& problem_data,
                                KernelDataEqlb<T>& kernel_data,
                                kernel_fn<T, true>& minkernel,
                                kernel_fn<T, false>& minkernel_rhs,
                                kernel_fn_schursolver<T>& kernel_weaksym)
{
  /* Step 1: Unconstrained flux equilibration */
  equilibrate_flux_semiexplt<T, id_flux_order>(
      geometry, fct_perms, patch, patch_data, problem_data, kernel_data,
      minkernel, minkernel_rhs);

  /* Step 2: Enforce weak symmetry constraint */
  impose_weak_symmetry<T, id_flux_order, false>(
      geometry, patch, patch_data, problem_data, kernel_data, kernel_weaksym);
}

} // namespace dolfinx_eqlb