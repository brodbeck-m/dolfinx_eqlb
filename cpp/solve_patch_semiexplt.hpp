#pragma once

#include "eigen3/Eigen/Dense"

#include "KernelData.hpp"
#include "Patch.hpp"
#include "PatchFluxCstm.hpp"
#include "ProblemDataFluxCstm.hpp"
#include "assemble_patch_semiexplt.hpp"
#include "minimise_flux.hpp"
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

/// Evaluate jump (projected flux) on internal surface
/// @tparam T The data type of the Flux
/// @param[in] ipoint_n            Id of the interpolation point
/// @param[in] coefficients_G_Tap1 The flux DOFs (cell Tap1)
/// @param[in] shp_Tap1Ea          The shape functions (cell Tap1, facet Ea)
/// @param[in] coefficients_G_Ta   The flux DOFs (cell Ta)
/// @param[in] shp_TaEa            The shape functions (cell Ta, facet Ea)
/// @param[in] dofs_Ea             The DOFs on facet Ea
/// @param[in,out] jG_Ea           The evaluated flux jump
template <typename T>
void calculate_jump(std::size_t ipoint_n,
                    std::span<const T> coefficients_G_Tap1,
                    mdspan_t<const double, 2> shp_Tap1Ea,
                    std::span<const T> coefficients_G_Ta,
                    mdspan_t<const double, 2> shp_TaEa,
                    std::span<const std::int32_t> dofs_Ea, std::span<T> jG_Ea)
{
  // Number of DOFs per facet
  const int ndofs_projflux_fct = dofs_Ea.size() / 2;

  // Initialise jump with zero
  std::fill(jG_Ea.begin(), jG_Ea.end(), 0.0);

  // Evaluate jump at
  for (std::size_t i = 0; i < ndofs_projflux_fct; ++i)
  {
    // Local and global IDs of first DOF on facet
    int id_Ta = dofs_Ea[i + ndofs_projflux_fct];
    int id_Tap1 = dofs_Ea[i];
    int offs_Ta = 2 * id_Ta;
    int offs_Tap1 = 2 * id_Tap1;

    // Evaluate jump
    // jump = (flux_proj_Tap1 - flux_proj_Ta)
    jG_Ea[0] += coefficients_G_Tap1[offs_Tap1] * shp_Tap1Ea(ipoint_n, id_Tap1)
                - coefficients_G_Ta[offs_Ta] * shp_TaEa(ipoint_n, id_Ta);
    jG_Ea[1]
        += coefficients_G_Tap1[offs_Tap1 + 1] * shp_Tap1Ea(ipoint_n, id_Tap1)
           - coefficients_G_Ta[offs_Ta + 1] * shp_TaEa(ipoint_n, id_Ta);
  }
}

/// Step 1: Calculate fluxes with jump/divergence condition on patch
///
/// Calculates sig in pice-wise H(div) that fulfills jump and divergence
/// condition on patch (see [1, Appendix A, Algorithm 2])
///
/// [1] Bertrand, F.; Carstensen, C.; Gräßle, B. & Tran, N. T.:
///     Stabilization-free HHO a posteriori error control, 2022
///
/// @tparam T             The scalar type
/// @tparam id_flux_order Parameter for flux order (1->RT1, 2->RT2, 3->general)
/// @param geometry     The geometry
/// @param patch        The patch
/// @param problem_data The problem data (Functions of flux, flux_dg, RHS_dg)
/// @param kernel_data  The kernel data (Quadrature data, tabulated basis
/// functions)
template <typename T, int id_flux_order = 3>
void equilibrate_flux_semiexplt(const mesh::Geometry& geometry,
                                PatchFluxCstm<T, id_flux_order>& patch,
                                ProblemDataFluxCstm<T>& problem_data,
                                KernelDataEqlb<T>& kernel_data)
{
  /* Geometry data */
  const int dim = 2;
  const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();
  std::span<const double> x = geometry.x();

  /* Extract Data */
  // Cells on patch
  std::span<const std::int32_t> cells = patch.cells();
  const int ncells = patch.ncells();
  const int nfcts = patch.nfcts();

  // Geometry of cells
  int nnodes_cell = kernel_data.nnodes_cell();

  // Number of DOFs
  const int degree_flux_rt = patch.degree_raviart_thomas();
  const int ndofs_flux = patch.ndofs_flux();
  const int ndofs_flux_fct = patch.ndofs_flux_fct();
  const int ndofs_flux_cell_div = patch.ndofs_flux_cell_div();
  const int ndofs_projflux = patch.ndofs_fluxdg_cell();
  const int ndofs_projflux_fct = patch.ndofs_fluxdg_fct();
  const int ndofs_rhs = patch.ndofs_rhs_cell();

  // Interpolation matrix (reference cell)
  mdspan_t<const double, 4> M = kernel_data.interpl_matrix_facte();

  /* Initialise solution process */
  // Jacobian J, inverse K and determinant detJ
  std::array<double, 9> Jb;
  mdspan2_t J(Jb.data(), 2, 2);
  std::array<double, 9> Kb;
  mdspan2_t K(Kb.data(), 2, 2);
  std::array<double, 18> detJ_scratch;

  // Interpolation data
  // (Structure M_mapped: cells x dofs x dim x points)
  // (DOF zero on facet Eam1, higher order DOFs on Ea)
  const int nipoints_facet = kernel_data.nipoints_facet();

  std::vector<double> data_M_mapped(
      ncells * nipoints_facet * ndofs_flux_fct * 2, 0);
  mdspan_t<double, 4> M_mapped(data_M_mapped.data(), ncells, ndofs_flux_fct, 2,
                               nipoints_facet);

  // Coefficient arrays for RHS/ projected flux
  std::vector<T> coefficients_f(ndofs_rhs, 0),
      coefficients_G_Tap1(ndofs_projflux, 0),
      coefficients_G_Ta(ndofs_projflux, 0),
      data_coefficients_flux(ncells * ndofs_flux, 0);

  mdspan_t<T, 2> coefficients_flux(data_coefficients_flux.data(), ncells,
                                   ndofs_flux);

  // Storage of inter-facet jumps
  std::array<T, 2> jG_Ea, data_jG_E0, data_jG_mapped_E0;

  const int size_jump = dim * nipoints_facet;
  std::vector<T> data_jG_Eam1(size_jump, 0);
  mdspan_t<T, 2> jG_Eam1(data_jG_Eam1.data(), nipoints_facet, dim);

  // History array for c_tam1_eam1
  T c_ta_ea = 0, c_ta_eam1 = 0, c_tam1_eam1 = 0, c_t1_e0 = 0;
  std::vector<T> c_ta_div, cj_ta_ea;

  /* Initialise storage */
  // Jacobian J, inverse K and determinant detJ
  bool store_J = false, store_K = false;
  std::vector<double> storage_detJ(ncells, 0);
  std::vector<double> storage_J, storage_K;

  // Initialise storage only required for higher order cases
  if constexpr (id_flux_order > 1)
  {
    // Inverse of the Jacobian
    store_K = true;
    storage_K.resize(ncells * 4);

    // Storage of DOFs
    c_ta_div.resize(ndofs_flux_cell_div, 0);
    cj_ta_ea.resize(ndofs_flux_fct - 1, 0);
  }

  if (patch.is_on_boundary())
  {
    // Jacobian
    store_J = true;
    storage_J.resize(ncells * 4, 0);

    if (store_K == false)
    {
      // Inverse of the Jacobian
      store_K = true;
      storage_K.resize(ncells * 4, 0);
    }
  }

  /* Storage cell geometries/ normal orientation */
  // Initialisations
  const int cstride_geom = 3 * nnodes_cell;
  std::vector<double> coordinate_dofs(ncells * cstride_geom, 0);

  std::int8_t fctloc_ea, fctloc_eam1;
  bool noutward_ea, noutward_eam1;
  std::vector<double> dprefactor_dof(ncells * 2, 1.0);
  mdspan2_t prefactor_dof(dprefactor_dof.data(), ncells, 2);

  for (std::size_t index = 0; index < ncells; ++index)
  {
    // Index using patch nomenclature
    int a = index + 1;

    // Get current cell
    std::int32_t c = cells[index];

    /* Copy cell geometry */
    std::span<double> coordinate_dofs_e(
        coordinate_dofs.data() + index * cstride_geom, cstride_geom);
    std::span<const std::int32_t> x_dofs = x_dofmap.links(c);
    copy_cell_data<double, 3>(x, x_dofs, coordinate_dofs_e, 3);

    /* Piola mapping */
    // Reshape geometry infos
    cmdspan2_t coords(coordinate_dofs_e.data(), nnodes_cell, 3);

    // Calculate Jacobi, inverse, and determinant
    storage_detJ[index]
        = kernel_data.compute_jacobian(J, K, detJ_scratch, coords);

    const double detJ = storage_detJ[index];

    // Storage of (inverse) Jacobian
    if (store_J)
    {
      store_mapping_data(index, storage_J, J);
    }

    if (store_K)
    {
      store_mapping_data(index, storage_K, K);
    }

    /* DOF transformation */
    std::tie(fctloc_eam1, fctloc_ea) = patch.fctid_local(a);
    std::tie(noutward_eam1, noutward_ea)
        = kernel_data.fct_normal_is_outward(fctloc_eam1, fctloc_ea);

    // Sign of the jacobian
    const double sgn_detJ = detJ / std::fabs(detJ);

    // Set prefactors
    prefactor_dof(index, 0) = (noutward_eam1) ? sgn_detJ : -sgn_detJ;
    prefactor_dof(index, 1) = (noutward_ea) ? sgn_detJ : -sgn_detJ;

    /* Apply push-back on interpolation matrix M */
    for (std::size_t i = 0; i < ndofs_flux_fct; ++i)
    {
      for (std::size_t j = 0; j < nipoints_facet; ++j)
      {
        // Select facet
        // (Zero-order moment on facet Eama, higer order moments on Ea)
        const std::int8_t fctid = (i == 0) ? fctloc_eam1 : fctloc_ea;

        // Map interpolation cell Tam1
        M_mapped(index, i, 0, j)
            = detJ
              * (M(fctid, i, 0, j) * K(0, 0) + M(fctid, i, 1, j) * K(1, 0));
        M_mapped(index, i, 1, j)
            = detJ
              * (M(fctid, i, 0, j) * K(0, 1) + M(fctid, i, 1, j) * K(1, 1));
      }
    }
  }

  /* Evaluate DOFs of sigma_tilde (for each flux separately) */
  for (std::size_t i_rhs = 0; i_rhs < problem_data.nrhs(); ++i_rhs)
  {
    /* Extract data */
    // Patch type
    PatchType type_patch = patch.type(i_rhs);

    // Check if reversion is requierd
    bool reversion_required = patch.reversion_required(i_rhs);

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

    // Boundary DOFs
    std::span<const T> boundary_values = problem_data.boundary_values(i_rhs);

    /* Step 1: Calculate sigma_tilde */
    // Initialisations
    copy_cell_data<T, 2>(x_flux_proj, fluxdg_dofmap.links(cells[0]),
                         coefficients_G_Tap1, 2);

    // Reinitialise history storage
    if (i_rhs > 0)
    {
      c_tam1_eam1 = 0.0;
      c_t1_e0 = 0.0;
      std::fill(data_jG_Eam1.begin(), data_jG_Eam1.end(), 0.0);
      std::fill(data_coefficients_flux.begin(), data_coefficients_flux.end(),
                0.0);
    }

    // Loop over all cells
    for (std::size_t a = 1; a < ncells + 1; ++a)
    {
      // Set id for accessing storage
      int id_a = a - 1;

      // Global cell id
      std::int32_t c_a = cells[id_a];

      // Cell-local id of patch-central node
      std::int8_t node_i_Ta = patch.inode_local(a);

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
      const double detJ = storage_detJ[id_a];
      const double sign_detJ = (detJ > 0.0) ? 1.0 : -1.0;

      // DOFs (cell-local) projected flux on facet Ea
      std::span<const std::int32_t> dofs_Ea = patch.dofs_projflux_fct(a);

      // Coefficient arrays
      std::swap(coefficients_G_Ta, coefficients_G_Tap1);

      copy_cell_data<T, 2>(x_flux_proj, fluxdg_dofmap.links(patch.cell(a + 1)),
                           coefficients_G_Tap1, 2);
      copy_cell_data<T, 1>(x_rhs_proj, rhs_dofmap.links(c_a), coefficients_f,
                           1);

      /* Tabulate shape functions */
      // Shape functions RHS
      s_cmdspan2_t shp_TaEa = kernel_data.shapefunctions_fct_rhs(fl_TaEa);
      s_cmdspan2_t shp_Tap1Ea = kernel_data.shapefunctions_fct_rhs(fl_Tap1Ea);

      // Shape-functions hat-function
      s_cmdspan2_t hat_TaEam1 = kernel_data.shapefunctions_fct_hat(fl_TaEam1);
      s_cmdspan2_t hat_TaEa = kernel_data.shapefunctions_fct_hat(fl_TaEa);

      /* Prepare data for inclusion of flux BCs */
      // DOFs on facet E0
      std::span<const std::int32_t> pflux_ldofs_E0, ldofs_E0;

      // Tabulated shape functions on facet E0
      s_cmdspan2_t shp_TaEam1;

      if ((a == 1) && (fct_has_bc || type_patch == PatchType::bound_mixed))
      {
        // DOFs (cell-local) projected flux on facet E0
        pflux_ldofs_E0 = patch.dofs_projflux_fct(0);

        // DOFs (cell-local) equilibrated flux on facet E0
        ldofs_E0 = patch.dofs_flux_fct_local(1, 0);

        // Tabulate shape functions RHS on facet 0
        shp_TaEam1 = kernel_data.shapefunctions_fct_rhs(fl_TaEam1);
      }

      /* DOFs from interpolation */
      // Initialisation
      c_ta_eam1 = -c_tam1_eam1;
      std::fill(cj_ta_ea.begin(), cj_ta_ea.end(), 0.0);

      // Consider flux BCs
      if (fct_has_bc)
      {
        // Get (global) boundary facet/DOFs
        std::span<const std::int32_t> bdofs_local, bdofs_global;
        std::int32_t bfct_global;
        if (a == 1)
        {
          bdofs_local = patch.dofs_flux_fct_local(1, 0);
          bdofs_global = patch.dofs_flux_fct_global(1, 0);

          bfct_global = patch.fct(0);
        }
        else
        {
          bdofs_local = patch.dofs_flux_fct_local(a, a);
          bdofs_global = patch.dofs_flux_fct_global(a, a);

          bfct_global = patch.fct(a);
        }

        // Calculate patch bcs
        std::int8_t node_i_Ta = patch.inode_local(a);
        mdspan_t<const double, 2> J = extract_mapping_data(0, storage_J);
        mdspan_t<const double, 2> K = extract_mapping_data(0, storage_K);

        problem_data.calculate_patch_bc(i_rhs, bfct_global, node_i_Ta, J, detJ,
                                        K);

        // Contribution to c_ta_eam1
        if (a == 1)
        {
          c_ta_eam1
              += prefactor_dof(id_a, 0) * boundary_values[bdofs_global[0]];
        }

        // Contribution to cj_ta_ea
        if constexpr (id_flux_order > 1)
        {
          if constexpr (id_flux_order == 2)
          {
            // x_flux_dhdiv[bdofs_global[1]] +=
            // boundary_values[bdofs_global[1]];
            coefficients_flux(id_a, bdofs_local[1])
                += boundary_values[bdofs_global[1]];
          }
          else
          {
            for (std::size_t j = 1; j < ndofs_flux_fct; ++j)
            {
              // x_flux_dhdiv[bdofs_global[j]] +=
              // boundary_values[bdofs_global[j]];
              coefficients_flux(id_a, bdofs_local[j])
                  += boundary_values[bdofs_global[j]];
            }
          }
        }

        // Handle mixed patch with E_0 on dirichlet boundary
        if (reversion_required)
        {
          c_t1_e0 -= prefactor_dof(id_a, 1) * boundary_values[bdofs_global[0]];
        }
      }

      // Interpolate DOFs
      for (std::size_t n = 0; n < nipoints_facet; ++n)
      {
        // Global index of Tap1
        std::int32_t c_ap1 = (a < ncells) ? cells[id_a + 1] : cells[0];

        // Interpolate jump at quadrature point
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
                jG_Eam1(n, 0)
                    += coefficients_G_Ta[offs_Ta] * shp_TaEam1(n, id_Ta);
                jG_Eam1(n, 1)
                    += coefficients_G_Ta[offs_Ta + 1] * shp_TaEam1(n, id_Ta);
              }

              // Evaluate higher order DOFs on boundary facet
              if constexpr (id_flux_order > 1)
              {
                // Extract mapping data
                mdspan_t<const double, 2> J
                    = extract_mapping_data(id_a, storage_J);
                mdspan_t<const double, 2> K
                    = extract_mapping_data(id_a, storage_K);

                // Pull back flux to reference cell
                mdspan_t<T, 2> jG_E0(data_jG_E0.data(), 1, dim);
                mdspan_t<T, 2> jG_mapped_E0(data_jG_mapped_E0.data(), 1, dim);

                if ((type_patch == PatchType::bound_mixed)
                    && (fct_has_bc == false))
                {
                  jG_E0(0, 0) = jG_Eam1(n, 0) * hat_TaEam1(n, node_i_Ta);
                  jG_E0(0, 1) = jG_Eam1(n, 1) * hat_TaEam1(n, node_i_Ta);
                }
                else
                {
                  jG_E0(0, 0) = -jG_Eam1(n, 0) * hat_TaEam1(n, node_i_Ta);
                  jG_E0(0, 1) = -jG_Eam1(n, 1) * hat_TaEam1(n, node_i_Ta);
                }

                kernel_data.pull_back_flux(jG_mapped_E0, jG_E0, J, detJ, K);

                // Evaluate higher-order DOFs on facet E0
                if constexpr (id_flux_order == 2)
                {
                  // x_flux_dhdiv[dofs_global_E0[1]]
                  //     += M(fl_TaEam1, 1, 0, n) * jG_mapped_E0(0, 0)
                  //        + M(fl_TaEam1, 1, 1, n) * jG_mapped_E0(0, 1);
                  coefficients_flux(id_a, ldofs_E0[1])
                      += M(fl_TaEam1, 1, 0, n) * jG_mapped_E0(0, 0)
                         + M(fl_TaEam1, 1, 1, n) * jG_mapped_E0(0, 1);
                }
                else
                {
                  // Evaluate facet DOFs
                  for (std::size_t j = 1; j < ndofs_flux_fct; ++j)
                  {
                    // x_flux_dhdiv[dofs_global_E0[j]]
                    //     += M(fl_TaEam1, j, 0, n) * jG_mapped_E0(0, 0)
                    //        + M(fl_TaEam1, j, 1, n) * jG_mapped_E0(0, 1);
                    coefficients_flux(id_a, ldofs_E0[j])
                        += M(fl_TaEam1, j, 0, n) * jG_mapped_E0(0, 0)
                           + M(fl_TaEam1, j, 1, n) * jG_mapped_E0(0, 1);
                  }
                }
              }

              // Handle mixed patch with E_0 on dirichlet boundary
              if (fct_has_bc == false)
              {
                // Unset jump on facet E0
                jG_Eam1(n, 0) = 0.0;
                jG_Eam1(n, 1) = 0.0;
              }
            }

            // Evaluate jump on facet E1
            calculate_jump<T>(n, coefficients_G_Tap1, shp_Tap1Ea,
                              coefficients_G_Ta, shp_TaEa, dofs_Ea, jG_Ea);
          }
          else
          {
            // Evaluate jump on facet En
            double pfctr = (fct_has_bc) ? -1.0 : 1.0;
            std::fill(jG_Ea.begin(), jG_Ea.end(), 0.0);

            for (std::size_t i = 0; i < ndofs_projflux_fct; ++i)
            {
              // Local and global IDs of first DOF on facet
              int id_Ta = dofs_Ea[i + ndofs_projflux_fct];
              int offs_Ta = 2 * id_Ta;

              // Jump
              double sshp_TaEa = pfctr * shp_TaEa(n, id_Ta);
              jG_Ea[0] += coefficients_G_Ta[offs_Ta] * sshp_TaEa;
              jG_Ea[1] += coefficients_G_Ta[offs_Ta + 1] * sshp_TaEa;
            }

            // Add boundary contribution to c_t1_e0
            // (required for mixed patch with facte E0 on dirichlet boundary)
            if (reversion_required)
            {
              // Extract mapping data
              mdspan_t<const double, 2> J
                  = extract_mapping_data(id_a, storage_J);
              mdspan_t<const double, 2> K
                  = extract_mapping_data(id_a, storage_K);

              // Pull back flux to reference cell
              mdspan_t<T, 2> jG_En(jG_Ea.data(), 1, dim);
              mdspan_t<T, 2> jG_mapped_En(data_jG_mapped_E0.data(), 1, dim);

              kernel_data.pull_back_flux(jG_mapped_En, jG_En, J, detJ, K);

              // Evaluate boundary contribution
              T aux = M(fl_TaEa, 0, 0, n) * jG_mapped_En(0, 0)
                      + M(fl_TaEa, 0, 1, n) * jG_mapped_En(0, 1);
              c_t1_e0 -= prefactor_dof(id_a, 1) * hat_TaEa(n, node_i_Ta) * aux;
            }
          }
        }
        else
        {
          calculate_jump<T>(n, coefficients_G_Tap1, shp_Tap1Ea,
                            coefficients_G_Ta, shp_TaEa, dofs_Ea, jG_Ea);
        }

        // Evaluate facet DOFs
        const T aux = M_mapped(id_a, 0, 0, n) * jG_Eam1(n, 0)
                      + M_mapped(id_a, 0, 1, n) * jG_Eam1(n, 1);
        const T fct_int
            = prefactor_dof(id_a, 0) * hat_TaEam1(n, node_i_Ta) * aux;
        c_ta_eam1 -= fct_int;

        c_t1_e0 += fct_int;

        if constexpr (id_flux_order > 1)
        {
          if constexpr (id_flux_order == 2)
          {
            T MjG = M_mapped(id_a, 1, 0, n) * jG_Ea[0]
                    + M_mapped(id_a, 1, 1, n) * jG_Ea[1];
            cj_ta_ea[0] += hat_TaEa(n, node_i_Ta) * MjG;
          }
          else
          {
            // Multiply jump and hat-function
            std::array<T, 2> jGha_Ea = {jG_Ea[0] * hat_TaEa(n, node_i_Ta),
                                        jG_Ea[1] * hat_TaEa(n, node_i_Ta)};

            // Evaluate facet DOFs
            for (std::size_t j = 1; j < ndofs_flux_fct; ++j)
            {
              cj_ta_ea[j - 1] += M_mapped(id_a, j, 0, n) * jGha_Ea[0]
                                 + M_mapped(id_a, j, 1, n) * jGha_Ea[1];
            }
          }
        }

        // Store jump
        jG_Eam1(n, 0) = jG_Ea[0];
        jG_Eam1(n, 1) = jG_Ea[1];
      }

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
        cmdspan2_t K = extract_mapping_data(id_a, storage_K);

        // Quadrature points and weights
        cmdspan2_t qpoints = kernel_data.quadrature_points(0);
        std::span<const double> weights = kernel_data.quadrature_weights(0);
        const int nqpoints = weights.size();

        // Shape-functions RHS
        s_cmdspan3_t shp_rhs = kernel_data.shapefunctions_cell_rhs(K);

        // Shape-functions hat-function
        s_cmdspan2_t shp_hat = kernel_data.shapefunctions_cell_hat();

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

      /* Store DOFs into patch-wise solution evctor */
      // Global DOF ids
      std::span<const std::int32_t> ldofs_fct = patch.dofs_flux_fct_local(a);

      // Set zero order DOFs
      // x_flux_dhdiv[gdofs_fct[0]] += prefactor_dof(id_a, 0) * c_ta_eam1;
      // x_flux_dhdiv[gdofs_fct[ndofs_flux_fct]]
      //     += prefactor_dof(id_a, 1) * c_ta_ea;
      coefficients_flux(id_a, ldofs_fct[0])
          += prefactor_dof(id_a, 0) * c_ta_eam1;
      coefficients_flux(id_a, ldofs_fct[ndofs_flux_fct])
          += prefactor_dof(id_a, 1) * c_ta_ea;

      if constexpr (id_flux_order > 1)
      {
        // Global DOF ids
        std::span<const std::int32_t> ldofs_cell
            = patch.dofs_flux_cell_local(a);

        if constexpr (id_flux_order == 2)
        {
          // Set higher-order DOFs on facets
          // x_flux_dhdiv[gdofs_fct[3]] += cj_ta_ea[0];
          coefficients_flux(id_a, ldofs_fct[3]) += cj_ta_ea[0];

          // Set DOFs on cell
          // x_flux_dhdiv[gdofs_cell[0]] += c_ta_div[0];
          // x_flux_dhdiv[gdofs_cell[1]] += c_ta_div[1];
          coefficients_flux(id_a, ldofs_cell[0]) += c_ta_div[0];
          coefficients_flux(id_a, ldofs_cell[1]) += c_ta_div[1];
        }
        else
        {
          // Set higher-order DOFs on facets
          for (std::size_t i = 1; i < ndofs_flux_fct; ++i)
          {
            const int offs = ldofs_fct[ndofs_flux_fct + i];

            // DOFs on facet Ea
            // x_flux_dhdiv[offs] += cj_ta_ea[i - 1];
            coefficients_flux(id_a, offs) += cj_ta_ea[i - 1];
          }

          // Set divergence DOFs on cell
          for (std::size_t i = 0; i < ndofs_flux_cell_div; ++i)
          {
            // x_flux_dhdiv[gdofs_cell[i]] += c_ta_div[i];
            coefficients_flux(id_a, ldofs_cell[i]) += c_ta_div[i];
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

        // Global DOF ids
        std::span<const std::int32_t> ldofs_fct = patch.dofs_flux_fct_local(a);

        // Set zero-order DOFs on facets
        // x_flux_dhdiv[gdofs_fct[0]] += prefactor_dof(id_a, 0) * c_t1_e0;
        // x_flux_dhdiv[gdofs_fct[ndofs_flux_fct]]
        //     -= prefactor_dof(id_a, 1) * c_t1_e0;
        coefficients_flux(id_a, ldofs_fct[0])
            += prefactor_dof(id_a, 0) * c_t1_e0;
        coefficients_flux(id_a, ldofs_fct[ndofs_flux_fct])
            -= prefactor_dof(id_a, 1) * c_t1_e0;
      }
    }

    /* Step 2: Minimse sigma_delta */

    /* Move patch-wise solution into global storage */
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
      std::span<const std::int32_t> gdofs = flux_dofmap.links(cells[id_a]);

      // Loop over DOFs an cell
      for (std::size_t i = 0; i < ndofs_flux; ++i)
      {
        // Set zero-order DOFs on facets
        x_flux_dhdiv[gdofs[i]] += coefficients_flux(id_a, i);
      }
    }
  }
}

// Step 2: Minimise flux on patch-wise ansatz space
///
/// Minimises the in step 1 calculated flux in an patch-wise,
/// divergence-free H(div) space. Explicite ansatz for such a space see [1,
/// Lemma 12]. The DOFs are therefore locally numbered as follows:
///
/// dofs_patch = [d0, d^l_E0, ..., d^l_En, d^r_T1, ..., d^r_Tn]
///
/// [1] Bertrand, F.; Carstensen, C.; Gräßle, B. & Tran, N. T.:
///     Stabilization-free HHO a posteriori error control, 2022
///
/// @tparam T             The scalar type
/// @tparam id_flux_order The flux order (1->RT1, 2->RT2, 3->general)
/// @param geometry     The geometry
/// @param patch        The patch
/// @param problem_data The problem data
/// @param kernel_data  The kernel data
template <typename T, int id_flux_order = 3>
void minimise_flux(const mesh::Geometry& geometry,
                   PatchFluxCstm<T, id_flux_order>& patch,
                   ProblemDataFluxCstm<T>& problem_data,
                   KernelDataEqlb<T>& kernel_data)
{
  assert(id_flux_order < 0);

  /* Geometry data */
  const int dim = 2;
  const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();
  std::span<const double> x = geometry.x();

  /* Extract patch data */
  // Data flux element
  const int degree_rt = patch.degree_raviart_thomas();
  const int ndofs_flux = patch.ndofs_flux();
  const int ndofs_flux_fct = patch.ndofs_flux_fct();
  const int ndofs_flux_cell_add = patch.ndofs_flux_cell_add();

  const graph::AdjacencyList<std::int32_t>& flux_dofmap
      = problem_data.fspace_flux_hdiv()->dofmap()->list();

  // Cells on patch
  std::span<const std::int32_t> cells = patch.cells();
  const int ncells = patch.ncells();

  // Facets on patch
  const int nfcts = patch.nfcts();

  // DOFs per cell on patch
  const int ndofs_cell_local = 2 * ndofs_flux_fct + ndofs_flux_cell_add;
  const int ndofs_cell_patch = ndofs_cell_local - 1;

  /* Initialize Patch-LGS */
  // Patch arrays
  const int size_psystem
      = 1 + degree_rt * nfcts + 0.5 * degree_rt * (degree_rt - 1) * ncells;

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A_patch;
  Eigen::Matrix<T, Eigen::Dynamic, 1> L_patch, u_patch;

  A_patch.resize(size_psystem, size_psystem);
  L_patch.resize(size_psystem);
  u_patch.resize(size_psystem);

  // Local solver
  Eigen::LLT<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> solver;

  /* Initialize solution process */
  // Number of nodes on reference cell
  int nnodes_cell = kernel_data.nnodes_cell();

  // Storage DOFmap
  // dim: (dof_local, dof_patch, dof_global, pfctr) x cell x dofs_per_cell
  // Soring per cell: [d^0_Eam1, d^0_Ea, d^l_Eam1, d^l_Ea, d^r_Ta]
  std::vector<std::int32_t> ddofmap_patch(4 * ncells * ndofs_cell_local, 0);
  mdspan_t<std::int32_t, 3> dofmap_patch(ddofmap_patch.data(), 4,
                                         (std::size_t)ncells,
                                         (std::size_t)ndofs_cell_local);

  // Marker boundary DOFs
  std::vector<std::int8_t> boundary_markers(size_psystem, false);

  /* Storage cell geometries and DOFmap*/
  // Initialisations
  const int cstride_geom = 3 * nnodes_cell;
  std::vector<double> coordinate_dofs(ncells * cstride_geom, 0);

  std::int8_t fctloc_ea, fctloc_eam1;
  bool noutward_ea, noutward_eam1;

  std::vector<T> coefficients(ncells * ndofs_flux, 0);

  // Evaluate quantities on all cells
  for (std::size_t index = 0; index < ncells; ++index)
  {
    // Get current cell
    std::int32_t c = cells[index];

    /* Copy cell coordinates */
    std::span<double> coordinate_dofs_e(
        coordinate_dofs.data() + index * cstride_geom, cstride_geom);
    std::span<const std::int32_t> x_dofs = x_dofmap.links(c);
    copy_cell_data<double, 3>(x, x_dofs, coordinate_dofs_e, 3);
  }

  /* Perform minimisation */
  for (std::size_t i_rhs = 0; i_rhs < problem_data.nrhs(); ++i_rhs)
  {
    /* Extract data */
    // Patch type
    PatchType type_patch = patch.type(i_rhs);

    // Check if reversion is requierd
    bool reversion_required = patch.reversion_required(i_rhs);

    // Solution vector (flux, picewise-H(div))
    std::span<T> x_flux_dhdiv = problem_data.flux(i_rhs).x()->mutable_array();

    // Stoarge result minimisation
    std::span<T> x_minimisation = problem_data.x_minimisation(i_rhs);

    // Set coefficients (copy solution data into flattend structure)
    copy_cell_data<T, 1>(cells, flux_dofmap, x_minimisation, coefficients,
                         ndofs_flux, 1);

    /* Set bounday markers */
    // (Required, as DOFs of H(div=0) space has non-global DOF ordering)
    if (patch.requires_flux_bcs(i_rhs))
    {
      // Unset all previous markers
      std::fill(boundary_markers.begin(), boundary_markers.end(), false);

      // Set boundary markers
      boundary_markers[0] = true;

      if constexpr (id_flux_order > 1)
      {
        for (std::size_t i = 1; i < ndofs_flux_fct; ++i)
        {
          if (type_patch == PatchType::bound_essnt_dual)
          {
            // Mark DOFs on facet E0 and En
            boundary_markers[i] = true;
            boundary_markers[ncells * (ndofs_flux_fct - 1) + i] = true;
          }
          else
          {
            if (reversion_required)
            {
              // Mark DOFs in facet En
              // (Mixed patch with reversed order)
              boundary_markers[ncells * (ndofs_flux_fct - 1) + i] = true;
            }
            else
            {
              // Mark DOFs in facet E0
              // (Mixed patch with original order)
              boundary_markers[i] = true;
            }
          }
        }
      }
    }

    /* Perform minimisation */
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

    // Assemble system
    if (assemble_entire_system)
    {
      // Initialize tangents
      A_patch.setZero();
      L_patch.setZero();

      // Assemble tangents
      assemble_minimisation<T, id_flux_order, true>(
          A_patch, L_patch, patch, kernel_data, dofmap_patch, boundary_markers,
          coefficients, coordinate_dofs, type_patch);

      // Factorization of system matrix
      if constexpr (id_flux_order > 1)
      {
        solver.compute(A_patch);
      }
    }
    else
    {
      // Initialise linear form
      L_patch.setZero();

      // Assemble linear form
      assemble_minimisation<T, id_flux_order, false>(
          A_patch, L_patch, patch, kernel_data, dofmap_patch, boundary_markers,
          coefficients, coordinate_dofs, type_patch);
    }

    // Solve system
    if constexpr (id_flux_order == 1)
    {
      u_patch(0) = L_patch(0) / A_patch(0, 0);
    }
    else
    {
      u_patch = solver.solve(L_patch);
    }

    /* Apply correction onto flux */
    // Set prefactors due to construction of H(div) subspace
    std::vector<T> crr_fct(ndofs_cell_local, 1.0);
    crr_fct[1] = -1.0;

    if constexpr (id_flux_order > 1)
    {
      if constexpr (id_flux_order == 2)
      {
        crr_fct[3] = -1.0;
      }
      else
      {
        for (std::size_t i = ndofs_flux_fct + 1; i < 2 * ndofs_flux_fct; ++i)
        {
          crr_fct[i] = -1.0;
        }
      }
    }

    // Move patch-wise solution to global solution vector
    for (std::size_t a = 1; a < ncells + 1; ++a)
    {
      int id_a = a - 1;

      for (std::size_t i = 0; i < ndofs_cell_local; ++i)
      {
        // Overall correction factor (facet orientation and ansatz space)
        T crr = crr_fct[i] * dofmap_patch(3, id_a, i);

        // Apply correction
        x_flux_dhdiv[dofmap_patch(2, id_a, i)]
            += crr * u_patch(dofmap_patch(1, id_a, i));
      }
    }
  }
}
} // namespace dolfinx_eqlb