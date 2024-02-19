#pragma once

#include "eigen3/Eigen/Dense"

#include "KernelData.hpp"
#include "Patch.hpp"
#include "PatchFluxCstm.hpp"
#include "ProblemDataFluxCstm.hpp"
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
void equilibrate_flux_semiexplt(
    const mesh::Geometry& geometry, PatchFluxCstm<T, id_flux_order>& patch,
    PatchFluxCstmNew<T, id_flux_order, false>& patch_test,
    ProblemDataFluxCstm<T>& problem_data, KernelDataEqlb<T>& kernel_data)
{
  /* Extract data */
  // Spacial dimension
  const int dim = 2;

  // The geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();
  std::span<const double> x = geometry.x();

  // The patch
  std::span<const std::int32_t> cells = patch_test.cells();
  const int ncells = patch_test.ncells();
  const int nfcts = patch_test.nfcts();

  // Nodes constructing one element
  const int nnodes_cell = kernel_data.nnodes_cell();

  // DOF-counts function spaces
  const int degree_flux_rt = patch_test.degree_raviart_thomas();
  const int ndofs_flux = patch_test.ndofs_flux();
  const int ndofs_flux_fct = patch_test.ndofs_flux_fct();
  const int ndofs_flux_cell_div = patch_test.ndofs_flux_cell_div();
  const int ndofs_flux_cell_add = patch_test.ndofs_flux_cell_add();
  const int ndofs_projflux = patch_test.ndofs_fluxdg_cell();
  const int ndofs_projflux_fct = patch_test.ndofs_fluxdg_fct();
  const int ndofs_rhs = patch_test.ndofs_rhs_cell();

  /* Initialise Mappings */
  // Representation/Storage isoparametric mapping
  std::array<double, 9> Jb, Kb;
  mdspan_t<double, 2> J(Jb.data(), 2, 2), K(Kb.data(), 2, 2);
  std::array<double, 18> detJ_scratch;

  bool store_K = false;
  std::vector<double> storage_detJ(ncells, 0), storage_J(ncells * 4, 0),
      storage_K;

  if constexpr (id_flux_order > 1)
  {
    store_K = true;
    storage_K.resize(ncells * 4);
  }

  if (patch_test.is_on_boundary())
  {
    if (store_K == false)
    {
      // Inverse of the Jacobian
      store_K = true;
      storage_K.resize(ncells * 4, 0);
    }
  }

  // Storage cell geometry
  const int cstride_geom = 3 * nnodes_cell;
  std::vector<double> coordinate_dofs_e(cstride_geom, 0);

  // Storage pre-factors (due to orientation of the normal)
  std::int8_t fctloc_ea, fctloc_eam1;
  bool noutward_ea, noutward_eam1;
  std::vector<double> dprefactor_dof(ncells * 2, 1.0);
  mdspan_t<double, 2> prefactor_dof(dprefactor_dof.data(), ncells, 2);

  /* The patch-wise solution */
  std::vector<T> dcoefficients_flux(ncells * ndofs_flux, 0);
  mdspan_t<T, 2> coefficients_flux(dcoefficients_flux.data(), ncells,
                                   ndofs_flux);

  /* Initialise Step 1 */
  // The interpolation matrix (reference cell)
  mdspan_t<const double, 4> M = kernel_data.interpl_matrix_facte();

  const int nipoints_facet = kernel_data.nipoints_facet();

  // The mapped interpolation matrix
  // (Structure M_mapped: cells x dofs x dim x points)
  // (DOF zero on facet Eam1, higher order DOFs on Ea)
  std::vector<double> dM_mapped(ncells * nipoints_facet * ndofs_flux_fct * 2,
                                0);
  mdspan_t<double, 4> M_mapped(dM_mapped.data(), ncells, ndofs_flux_fct, 2,
                               nipoints_facet);

  // Coefficient arrays for RHS/ projected flux
  std::vector<T> coefficients_f(ndofs_rhs, 0),
      coefficients_G_Tap1(ndofs_projflux, 0),
      coefficients_G_Ta(ndofs_projflux, 0);

  // Flux-jumps over facets
  std::array<T, 2> jG_Ea, djG_E0, djG_mapped_E0;

  std::vector<T> djG_Eam1(dim * nipoints_facet, 0);
  mdspan_t<T, 2> jG_Eam1(djG_Eam1.data(), nipoints_facet, dim);

  // Storage for cell-wise solution
  T c_ta_ea = 0, c_ta_eam1 = 0, c_tam1_eam1 = 0, c_t1_e0 = 0;
  std::vector<T> c_ta_div, cj_ta_ea;

  if constexpr (id_flux_order > 1)
  {
    c_ta_div.resize(ndofs_flux_cell_div, 0);
    cj_ta_ea.resize(ndofs_flux_fct - 1, 0);
  }

  /* Initialise Step 2 */
  // Number of DOFs on patch-wise H(div=0) space
  const int ndofs_hdivz
      = 1 + degree_flux_rt * nfcts
        + 0.5 * degree_flux_rt * (degree_flux_rt - 1) * ncells;
  const int ndofs_hdivz_per_cell = 2 * ndofs_flux_fct + ndofs_flux_cell_add;

  // The equation system for the minimisation step
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A_patch;
  Eigen::Matrix<T, Eigen::Dynamic, 1> L_patch, u_patch;

  A_patch.resize(ndofs_hdivz, ndofs_hdivz);
  L_patch.resize(ndofs_hdivz);
  u_patch.resize(ndofs_hdivz);

  // Local solver (Cholesky decomposition)
  Eigen::LLT<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> solver;

  // Boundary markers
  std::vector<std::int8_t> boundary_markers
      = initialise_boundary_markers(ndofs_hdivz);

  /* Pre-evaluate repeatedly used cell data */
  // Jacobi transformation and interpolation matrix
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
    // Reshape geometry infos
    mdspan_t<const double, 2> coords(coordinate_dofs_e.data(), nnodes_cell, 3);

    // Calculate Jacobi, inverse, and determinant
    storage_detJ[index]
        = kernel_data.compute_jacobian(J, K, detJ_scratch, coords);

    const double detJ = storage_detJ[index];

    // Storage of (inverse) Jacobian
    store_mapping_data(index, storage_J, J);

    if (store_K)
    {
      store_mapping_data(index, storage_K, K);
    }

    /* DOF transformation */
    std::tie(fctloc_eam1, fctloc_ea) = patch_test.fctid_local(a);
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

  // DOFmap for minimisation
  std::vector<std::int32_t> ddofmap_minms = set_flux_dofmap<T, id_flux_order>(
      patch, kernel_data.fct_normal_is_outward(), storage_detJ);
  mdspan_t<const std::int32_t, 3> dofmap_minms(
      ddofmap_minms.data(), 5, (std::size_t)ncells,
      (std::size_t)(2 * ndofs_flux_fct + ndofs_flux_cell_add));

  patch_test.set_assembly_informations(kernel_data.fct_normal_is_outward(),
                                       storage_detJ);
  mdspan_t<const std::int32_t, 3> dofmap_flux
      = patch_test.assembly_info_minimisation();

  const int offs_ffEa = ndofs_flux_fct;
  const int offs_fcadd = 2 * ndofs_flux_fct;
  const int offs_fcdiv = offs_fcadd + ndofs_flux_cell_add;

  /* Evaluate DOFs of sigma_tilde (for each flux separately) */
  for (std::size_t i_rhs = 0; i_rhs < problem_data.nrhs(); ++i_rhs)
  {
    /* Extract data */
    // Patch type
    PatchType type_patch = patch_test.type(i_rhs);

    // Check if reversion is requierd
    bool reversion_required = patch_test.reversion_required(i_rhs);

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
      std::fill(djG_Eam1.begin(), djG_Eam1.end(), 0.0);
      std::fill(dcoefficients_flux.begin(), dcoefficients_flux.end(), 0.0);
    }

    // Loop over all cells
    for (std::size_t a = 1; a < ncells + 1; ++a)
    {
      // Set id for accessing storage
      int id_a = a - 1;

      // Global cell id
      std::int32_t c_a = cells[a];

      // Cell-local id of patch-central node
      std::int8_t node_i_Ta = patch_test.inode_local(a);

      // Check if facet is on boundary
      bool fct_on_boundary
          = ((patch_test.is_on_boundary()) && (a == 1 || a == ncells)) ? true
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
            fct_has_bc = patch_test.requires_flux_bcs(i_rhs, 0);
          }
          else if (a == ncells)
          {
            fct_has_bc = patch_test.requires_flux_bcs(i_rhs, ncells);
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
      std::tie(fl_TaEam1, fl_TaEa) = patch_test.fctid_local(a);
      std::int8_t fl_Tap1Ea = patch_test.fctid_local(a, a + 1);

      // Isoparametric mapping
      const double detJ = storage_detJ[id_a];
      const double sign_detJ = (detJ > 0.0) ? 1.0 : -1.0;

      // DOFs (cell-local) projected flux on facet Ea
      std::span<const std::int32_t> dofs_Ea = patch_test.dofs_projflux_fct(a);

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
      smdspan_t<const double, 2> hat_TaEam1
          = kernel_data.shapefunctions_fct_hat(fl_TaEam1);
      smdspan_t<const double, 2> hat_TaEa
          = kernel_data.shapefunctions_fct_hat(fl_TaEa);

      /* Prepare data for inclusion of flux BCs */
      // DOFs on facet E0
      std::span<const std::int32_t> pflux_ldofs_E0;

      // Tabulated shape functions on facet E0
      smdspan_t<const double, 2> shp_TaEam1;

      if ((a == 1) && (fct_has_bc || type_patch == PatchType::bound_mixed))
      {
        // DOFs (cell-local) projected flux on facet E0
        pflux_ldofs_E0 = patch_test.dofs_projflux_fct(0);

        // DOFs (cell-local) equilibrated flux on facet E0
        // ldofs_E0 = patch.dofs_flux_fct_local(1, 0);

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
        // std::span<const std::int32_t> bdofs_local, bdofs_global;
        std::int32_t bfct_global;
        int offs_bdofs;
        if (a == 1)
        {
          // bdofs_local = patch.dofs_flux_fct_local(1, 0);
          // bdofs_global = patch.dofs_flux_fct_global(1, 0);
          offs_bdofs = 0;

          bfct_global = patch_test.fct(0);
        }
        else
        {
          // bdofs_local = patch.dofs_flux_fct_local(a, a);
          // bdofs_global = patch.dofs_flux_fct_global(a, a);
          offs_bdofs = ndofs_flux_fct;

          bfct_global = patch_test.fct(a);
        }

        // Calculate patch bcs
        // std::int8_t node_i_Ta = patch.inode_local(a);
        mdspan_t<const double, 2> J = extract_mapping_data(0, storage_J);
        mdspan_t<const double, 2> K = extract_mapping_data(0, storage_K);

        problem_data.calculate_patch_bc(i_rhs, bfct_global, node_i_Ta, J, detJ,
                                        K);

        // Contribution to c_ta_eam1
        if (a == 1)
        {
          // c_ta_eam1
          //     += prefactor_dof(id_a, 0) * boundary_values[bdofs_global[0]];
          c_ta_eam1 += prefactor_dof(id_a, 0)
                       * boundary_values[dofmap_flux(1, a, offs_bdofs)];
        }

        // Contribution to cj_ta_ea
        if constexpr (id_flux_order > 1)
        {
          if constexpr (id_flux_order == 2)
          {
            // coefficients_flux(id_a, bdofs_local[1])
            //     += boundary_values[bdofs_global[1]];
            coefficients_flux(id_a, dofmap_flux(0, a, offs_bdofs + 1))
                += boundary_values[dofmap_flux(1, a, offs_bdofs + 1)];
          }
          else
          {
            for (std::size_t j = 1; j < ndofs_flux_fct; ++j)
            {
              // coefficients_flux(id_a, bdofs_local[j])
              //     += boundary_values[bdofs_global[j]];
              coefficients_flux(id_a, dofmap_flux(0, a, offs_bdofs + j))
                  += boundary_values[dofmap_flux(1, a, offs_bdofs + j)];
            }
          }
        }

        // Handle mixed patch with E_0 on dirichlet boundary
        if (reversion_required)
        {
          // c_t1_e0 -= prefactor_dof(id_a, 1) *
          // boundary_values[bdofs_global[0]];
          c_t1_e0 -= prefactor_dof(id_a, 1)
                     * boundary_values[dofmap_flux(1, a, offs_bdofs)];
        }
      }

      // Interpolate DOFs
      for (std::size_t n = 0; n < nipoints_facet; ++n)
      {
        // Global index of Tap1
        std::int32_t c_ap1 = cells[a + 1];

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
                mdspan_t<T, 2> jG_E0(djG_E0.data(), 1, dim);
                mdspan_t<T, 2> jG_mapped_E0(djG_mapped_E0.data(), 1, dim);

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
                  // coefficients_flux(id_a, ldofs_E0[1])
                  // += M(fl_TaEam1, 1, 0, n) * jG_mapped_E0(0, 0)
                  //        + M(fl_TaEam1, 1, 1, n) * jG_mapped_E0(0, 1);
                  coefficients_flux(id_a, dofmap_flux(0, a, 1))
                      += M(fl_TaEam1, 1, 0, n) * jG_mapped_E0(0, 0)
                         + M(fl_TaEam1, 1, 1, n) * jG_mapped_E0(0, 1);
                }
                else
                {
                  // Evaluate facet DOFs
                  for (std::size_t j = 1; j < ndofs_flux_fct; ++j)
                  {
                    // coefficients_flux(id_a, ldofs_E0[j])
                    // += M(fl_TaEam1, j, 0, n) * jG_mapped_E0(0, 0)
                    //        + M(fl_TaEam1, j, 1, n) * jG_mapped_E0(0, 1);
                    coefficients_flux(id_a, dofmap_flux(0, a, j))
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
              mdspan_t<T, 2> jG_mapped_En(djG_mapped_E0.data(), 1, dim);

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
        mdspan_t<const double, 2> K = extract_mapping_data(id_a, storage_K);

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

      /* Store DOFs into patch-wise solution evctor */
      // Global DOF ids
      // std::span<const std::int32_t> ldofs_fct = patch.dofs_flux_fct_local(a);

      // Set zero order DOFs
      // coefficients_flux(id_a, ldofs_fct[0])
      //     += prefactor_dof(id_a, 0) * c_ta_eam1;
      // coefficients_flux(id_a, ldofs_fct[ndofs_flux_fct])
      //     += prefactor_dof(id_a, 1) * c_ta_ea;
      coefficients_flux(id_a, dofmap_flux(0, a, 0))
          += prefactor_dof(id_a, 0) * c_ta_eam1;
      coefficients_flux(id_a, dofmap_flux(0, a, offs_ffEa))
          += prefactor_dof(id_a, 1) * c_ta_ea;

      if constexpr (id_flux_order > 1)
      {
        // Global DOF ids
        // std::span<const std::int32_t> ldofs_cell
        //     = patch.dofs_flux_cell_local(a);

        if constexpr (id_flux_order == 2)
        {
          // Set higher-order DOFs on facets
          // coefficients_flux(id_a, ldofs_fct[3]) += cj_ta_ea[0];
          coefficients_flux(id_a, dofmap_flux(0, a, 3)) += cj_ta_ea[0];

          // Set DOFs on cell
          // coefficients_flux(id_a, ldofs_cell[0]) += c_ta_div[0];
          // coefficients_flux(id_a, ldofs_cell[1]) += c_ta_div[1];
          coefficients_flux(id_a, dofmap_flux(0, a, 4)) += c_ta_div[0];
          coefficients_flux(id_a, dofmap_flux(0, a, 5)) += c_ta_div[1];
        }
        else
        {
          // Set higher-order DOFs on facets
          for (std::size_t i = 1; i < ndofs_flux_fct; ++i)
          {
            // const int offs = ldofs_fct[ndofs_flux_fct + i];

            // DOFs on facet Ea
            // coefficients_flux(id_a, offs) += cj_ta_ea[i - 1];
            coefficients_flux(id_a, dofmap_flux(0, a, offs_ffEa + i))
                += cj_ta_ea[i - 1];
          }

          // Set divergence DOFs on cell
          for (std::size_t i = 0; i < ndofs_flux_cell_div; ++i)
          {
            // coefficients_flux(id_a, ldofs_cell[i]) += c_ta_div[i];
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

        // Global DOF ids
        // std::span<const std::int32_t> ldofs_fct =
        // patch.dofs_flux_fct_local(a);

        // Set zero-order DOFs on facets
        // coefficients_flux(id_a, ldofs_fct[0])
        //     += prefactor_dof(id_a, 0) * c_t1_e0;
        // coefficients_flux(id_a, ldofs_fct[ndofs_flux_fct])
        //     -= prefactor_dof(id_a, 1) * c_t1_e0;
        coefficients_flux(id_a, dofmap_flux(0, a, 0))
            += prefactor_dof(id_a, 0) * c_t1_e0;
        coefficients_flux(id_a, dofmap_flux(0, a, offs_ffEa))
            -= prefactor_dof(id_a, 1) * c_t1_e0;
      }
    }

    /* Step 2: Minimse sigma_delta */
    // Set boundary markers
    set_boundary_markers(boundary_markers, type_patch, ncells, ndofs_flux_fct,
                         reversion_required);

    // Check if assembly of entire system is required
    bool assemble_entire_system = false;

    if (i_rhs == 0)
    {
      assemble_entire_system = true;
    }
    else
    {
      if (patch_test.is_on_boundary())
      {
        if (patch_test.type(i_rhs) != patch_test.type(i_rhs - 1)
            || patch_test.type(i_rhs) == PatchType::bound_mixed)
        {
          assemble_entire_system = true;
        }
      }
    }

    // Assemble equation system
    if (assemble_entire_system)
    {
      // Initialize tangents
      A_patch.setZero();
      L_patch.setZero();

      // Assemble system
      assemble_fluxminimiser<T, id_flux_order, true>(
          A_patch, L_patch, patch, kernel_data, dofmap_minms, boundary_markers,
          dcoefficients_flux, storage_detJ, storage_J, storage_K,
          patch_test.requires_flux_bcs(i_rhs));

      // Factorise of system matrix
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
      assemble_fluxminimiser<T, id_flux_order, true>(
          A_patch, L_patch, patch, kernel_data, dofmap_minms, boundary_markers,
          dcoefficients_flux, storage_detJ, storage_J, storage_K,
          patch_test.requires_flux_bcs(i_rhs));
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
      std::span<const std::int32_t> gdofs = flux_dofmap.links(cells[a]);

      // Map solution from H(div=0) to H(div=0) space
      for (std::size_t i = 0; i < ndofs_hdivz_per_cell; ++i)
      {
        // Apply correction
        coefficients_flux(id_a, dofmap_flux(0, a, i))
            += dofmap_flux(3, a, i) * u_patch(dofmap_flux(2, a, i));
      }

      // Loop over DOFs an cell
      for (std::size_t i = 0; i < ndofs_flux; ++i)
      {
        // Set zero-order DOFs on facets
        x_flux_dhdiv[gdofs[i]] += coefficients_flux(id_a, i);
      }
    }
  }
}
} // namespace dolfinx_eqlb