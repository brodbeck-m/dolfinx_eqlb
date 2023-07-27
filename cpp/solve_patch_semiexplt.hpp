#pragma once

#include "KernelData.hpp"
#include "PatchFluxCstm.hpp"
#include "ProblemDataFluxCstm.hpp"
#include "assemble_patch_semiexplt.hpp"
#include "eigen3/Eigen/Dense"
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

namespace dolfinx_adaptivity::equilibration
{

namespace stdex = std::experimental;

/// Copy cell data from global storage into flattened array (per cell)
/// @param data_global The global data storage
/// @param dofs_cell   The DOFs on current cell
/// @param data_cell   The flattened storage of current cell
/// @param bs_data     Block size of data
template <typename T, int _bs_data = 1>
void copy_cell_data(std::span<const T> data_global,
                    std::span<const std::int32_t> dofs_cell,
                    std::span<T> data_cell, const int bs_data)
{
  for (std::size_t j = 0; j < dofs_cell.size(); ++j)
  {
    if constexpr (_bs_data == 1)
    {
      std::copy_n(std::next(data_global.begin(), dofs_cell[j]), 1,
                  std::next(data_cell.begin(), j));
    }
    else if constexpr (_bs_data == 2)
    {
      std::copy_n(std::next(data_global.begin(), 2 * dofs_cell[j]), 2,
                  std::next(data_cell.begin(), 2 * j));
    }
    else if constexpr (_bs_data == 3)
    {
      std::copy_n(std::next(data_global.begin(), 3 * dofs_cell[j]), 3,
                  std::next(data_cell.begin(), 3 * j));
    }
    else
    {
      std::copy_n(std::next(data_global.begin(), bs_data * dofs_cell[j]),
                  bs_data, std::next(data_cell.begin(), bs_data * j));
    }
  }
}

/// Copy cell data from global storage into flattened array (per patch)
/// @param cells        List of cells on patch
/// @param dofmap_data  DOFmap of data
/// @param data_global  The global data storage
/// @param data_cell    The flattened storage of current patch
/// @param cstride_data Number of data-points per cell
/// @param bs_data      Block size of data
template <typename T, int _bs_data = 4>
void copy_cell_data(std::span<const std::int32_t> cells,
                    const graph::AdjacencyList<std::int32_t>& dofmap_data,
                    std::span<const T> data_global, std::vector<T>& data_cell,
                    const int cstride_data, const int bs_data)
{
  for (std::size_t index = 0; index < cells.size(); ++index)
  {
    // Extract cell
    std::int32_t c = cells[index];

    // DOFs on current cell
    std::span<const std::int32_t> data_dofs = dofmap_data.links(c);

    // Copy DOFs into flattend storage
    std::span<T> data_dofs_e(data_cell.data() + index * cstride_data,
                             cstride_data);
    if constexpr (_bs_data == 1)
    {
      copy_cell_data<T, 1>(data_global, data_dofs, data_dofs_e, bs_data);
    }
    else if constexpr (_bs_data == 2)
    {
      copy_cell_data<T, 2>(data_global, data_dofs, data_dofs_e, bs_data);
    }
    else if constexpr (_bs_data == 3)
    {
      copy_cell_data<T, 3>(data_global, data_dofs, data_dofs_e, bs_data);
    }
    else
    {
      copy_cell_data<T>(data_global, data_dofs, data_dofs_e, bs_data);
    }
  }
}

/// Calculate prefactor of vector-values DOFs on facet
/// General: Explicite formulas assume that RT-Funtions are calculated based on
///          outward pointing normals. This is not the case in FenicsX.
///          Therefore transformation +1 resp. -1 is necessary.
/// Determination: Orientation of facet-normal on reference cell is stored
///                within basix. Correction required, as during contra-varinat
///                Piola mapping basis func- tions stay normal to element edges
///                but can change their orientation with respect to the cell
///                (inward or outwad pointing).This change is indetified
///                by the sign of the determinant of the Jacobian of the
///                mapping.
/// @param a             The patch-local index of a cell (a>1!)
/// @param noutward_eam1 True if normal of facet Eam1 points outward
/// @param noutward_ea   True if normal of facet Ea points outward
/// @param detj          Determinant of the Jacobian of the mapping
/// @param prefactor_dof The prefactors of N_(Ta,Ea) and N_(Ta,Eam1)
void set_dof_prefactors(int a, bool noutward_eam1, bool noutward_ea,
                        double detj, std::span<double> prefactor_dof)
{
  // Sign of the jacobian
  double sgn_detj = detj / std::fabs(detj);

  // Set prefactors
  int index = 2 * a - 2;
  prefactor_dof[index] = (noutward_eam1) ? sgn_detj : -sgn_detj;
  prefactor_dof[index + 1] = (noutward_ea) ? sgn_detj : -sgn_detj;
}

void store_mapping_data(const int cell_id, std::span<double> storage,
                        dolfinx_adaptivity::mdspan2_t matrix)
{
  // Set offset
  const int offset = 4 * cell_id;

  storage[offset] = matrix(0, 0);
  storage[offset + 1] = matrix(0, 1);
  storage[offset + 2] = matrix(1, 0);
  storage[offset + 3] = matrix(1, 1);
}

dolfinx_adaptivity::cmdspan2_t extract_mapping_data(const int cell_id,
                                                    std::span<double> storage)
{
  // Set offset
  const int offset = 4 * cell_id;

  return dolfinx_adaptivity::cmdspan2_t(storage.data() + offset, 2, 2);
}

template <typename T, int id_flux_order = 3>
void calc_fluxtilde_explt(const mesh::Geometry& geometry,
                          PatchFluxCstm<T, id_flux_order>& patch,
                          ProblemDataFluxCstm<T>& problem_data,
                          KernelData<T>& kernel_data)
{
  assert(id_flux_order < 0);

  /* Geometry data */
  const int dim = 2;
  const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();
  std::span<const double> x = geometry.x();

  /* Extract Data */
  // Cells on patch
  std::span<const std::int32_t> cells = patch.cells();
  int ncells = patch.ncells();

  // Geometry of cells
  int nnodes_cell = kernel_data.nnodes_cell();

  // Number of DOFs
  const int degree_flux_rt = patch.degree_raviart_thomas();
  const int ndofs_flux_fct = patch.ndofs_flux_fct();
  const int ndofs_flux_cell_div = patch.ndofs_flux_cell_div();
  const int ndofs_projflux = patch.ndofs_fluxdg_cell();
  const int ndofs_projflux_fct = patch.ndofs_fluxdg_fct();
  const int ndofs_rhs = patch.ndofs_rhs_cell();

  /* Initialise solution process */
  // Jacobian J, inverse K and determinant detJ
  std::array<double, 9> Jb;
  dolfinx_adaptivity::mdspan2_t J(Jb.data(), 2, 2);
  std::array<double, 9> Kb;
  dolfinx_adaptivity::mdspan2_t K(Kb.data(), 2, 2);
  std::array<double, 18> detJ_scratch;

  // Coefficient arrays for RHS/ projected flux
  std::vector<T> coefficients_f(ndofs_rhs, 0),
      coefficients_G_Tap1(ndofs_projflux, 0),
      coefficients_G_Ta(ndofs_projflux, 0);

  // Jump within projected flux
  std::array<T, 2> diff_proj_flux;
  std::vector<T> jump_proj_flux_Eam1(1, 0);

  const int nipoints_facet = kernel_data.nipoints_facet();
  const int size_jump = dim * nipoints_facet;
  std::vector<T> data_jG_Eam1(size_jump, 0);
  std::vector<T> data_jGhat(2 * size_jump, 0),
      data_jGhat_mapped(2 * size_jump, 0);

  dolfinx_adaptivity::mdspan_t<T, 2> jG_Eam1(data_jG_Eam1.data(),
                                             nipoints_facet, dim);
  dolfinx_adaptivity::mdspan_t<T, 2> jGhat(data_jGhat.data(), nipoints_facet,
                                           dim);
  dolfinx_adaptivity::mdspan_t<T, 2> jGhat_mapped(data_jGhat_mapped.data(),
                                                  nipoints_facet, dim);

  // DOFs flux (cell-wise H(div))
  T c_ta_ea, c_ta_eam1, c_tam1_eam1;
  std::vector<T> c_ta_div;
  std::vector<T> cj_ta_ea;

  /* Initialise storage */
  // Jacobian J, inverse K and determinant detJ
  std::vector<double> storage_detJ(ncells, 0);
  std::vector<double> storage_detJf(ncells + 1, 0);
  std::vector<double> storage_K, storage_J;

  // Physical normal
  std::vector<double> storage_normal_phys((ncells + 1) * dim, 0);

  // Initialise storage only required for higher order cases
  if constexpr (id_flux_order > 1)
  {
    // Inverse of the Jocobian
    storage_J.resize(ncells * 4);
    storage_K.resize(ncells * 4);

    // Storage of DOFs
    c_ta_div.resize(ndofs_flux_cell_div);
    cj_ta_ea.resize(ndofs_flux_fct - 1);
  }

  /* Storage cell geometries/ normal orientation */
  // Initialisations
  const int cstride_geom = 3 * nnodes_cell;
  std::vector<double> coordinate_dofs(ncells * cstride_geom, 0);

  std::int8_t fctloc_ea, fctloc_eam1;
  bool noutward_ea, noutward_eam1;
  std::vector<double> dprefactor_dof(ncells * 2, 1.0);
  dolfinx_adaptivity::mdspan2_t prefactor_dof(dprefactor_dof.data(), ncells, 2);

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
    dolfinx_adaptivity::cmdspan2_t coords(coordinate_dofs_e.data(), nnodes_cell,
                                          3);

    // Calculate Jacobi, inverse, and determinant
    storage_detJ[index]
        = kernel_data.compute_jacobian(J, K, detJ_scratch, coords);

    // Storage of Jacobian and its inverse
    if (id_flux_order > 1)
    {
      store_mapping_data(index, storage_K, K);
      store_mapping_data(index, storage_J, J);
    }

    // Calculate determinant of facet jacobian
    if (a == 1 && patch.type(0) > 0)
    {
      // Calculate detJf on facet E0
      std::span<const std::int32_t> nodes_fct = patch.nodes_on_fct(0);
      double dx = x[3 * nodes_fct[0]] - x[3 * nodes_fct[1]];
      double dy = x[3 * nodes_fct[0] + 1] - x[3 * nodes_fct[1] + 1];
      storage_detJf[0] = std::sqrt(dx * dx + dy * dy);

      // Calculate detJf on facet E1
      nodes_fct = patch.nodes_on_fct(1);
      dx = x[3 * nodes_fct[0]] - x[3 * nodes_fct[1]];
      dy = x[3 * nodes_fct[0] + 1] - x[3 * nodes_fct[1] + 1];
      storage_detJf[1] = std::sqrt(dx * dx + dy * dy);
    }
    else
    {
      // Calculate detJf on facet Ea
      std::span<const std::int32_t> nodes_fct = patch.nodes_on_fct(a);
      double dx = x[3 * nodes_fct[0]] - x[3 * nodes_fct[1]];
      double dy = x[3 * nodes_fct[0] + 1] - x[3 * nodes_fct[1] + 1];
      storage_detJf[a] = std::sqrt(dx * dx + dy * dy);
    }

    /* DOF transformation */
    std::tie(fctloc_eam1, fctloc_ea) = patch.fctid_local(a);
    std::tie(noutward_eam1, noutward_ea)
        = kernel_data.fct_normal_is_outward(fctloc_eam1, fctloc_ea);

    set_dof_prefactors(a, noutward_eam1, noutward_ea, storage_detJ[index],
                       dprefactor_dof);

    /* Calculation of physical normals */
    if (a == 1 && patch.type(0) > 0)
    {
      // Get local facet id of E_0 on T_0
      std::int8_t fctid_loc = patch.fctid_local(0, 1);

      // Get storage of normal
      std::span<double> normal_e(storage_normal_phys.data(), dim);

      // Transform normal into physical space
      kernel_data.physical_fct_normal(normal_e, K, fctid_loc);
    }

    // Get local facet id of E_a on T_a
    std::int8_t fctid_loc = patch.fctid_local(a, a);

    // Get storage of normal
    std::span<double> normal_e(storage_normal_phys.data() + a * dim, dim);

    // Transform normal into physical space
    kernel_data.physical_fct_normal(normal_e, K, fctid_loc);

    // Check if last cell is reached (only internal patch!)
    if (a == ncells && patch.type(0) == 0)
    {
      // Get data for last facet from facet 0
      storage_normal_phys[0] = storage_normal_phys[2 * ncells];
      storage_normal_phys[1] = storage_normal_phys[2 * ncells + 1];

      storage_detJf[0] = storage_detJf[2 * ncells];
    }
  }

  /* Evaluate DOFs of sigma_tilde (for each flux separately) */
  for (std::size_t i_rhs = 0; i_rhs < problem_data.nlhs(); ++i_rhs)
  {
    /* Extract data */
    // Patch type
    int type_patch = patch.type(i_rhs);

    // Solution vector (flux, picewise-H(div))
    std::span<T> x_flux_dhdiv = problem_data.flux(i_rhs).x()->mutable_array();

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

    // Initialise coefficients_G
    copy_cell_data<T, 2>(x_flux_proj, fluxdg_dofmap.links(cells[0]),
                         coefficients_G_Tap1, 2);

    /* Calculate sigma_tilde */
    c_tam1_eam1 = 0.0;

    for (std::size_t a = 1; a < ncells + 1; ++a)
    {
      // Set id for accessing storage
      int id_a = a - 1;

      // Global cell id
      std::int32_t c_a = cells[id_a];

      // Cell-local id of patch-central node
      std::int8_t node_i_Ta = patch.inode_local(a);

      /* Extract required data */
      // Isoparametric mapping
      const double detJ = std::fabs(storage_detJ[id_a]);
      const double detJ_Eam1 = storage_detJf[a - 1];
      const double detJ_Ea = storage_detJf[a];

      // Coefficient arrays
      std::swap(coefficients_G_Ta, coefficients_G_Tap1);

      copy_cell_data<T, 2>(x_flux_proj, fluxdg_dofmap.links(patch.cell(a + 1)),
                           coefficients_G_Tap1, 2);
      copy_cell_data<T, 1>(x_rhs_proj, rhs_dofmap.links(c_a), coefficients_f,
                           1);

      /* DOFs from facet integrals */
      if constexpr (id_flux_order == 1)
      {
        // Extract facet normal
        std::span<double> normal_Ea(storage_normal_phys.data() + a * 2, 2);

        // Extract gradients (+/- side) on facet E_am1
        std::span<const std::int32_t> dofs_projflux_fct
            = patch.dofs_projflux_fct(a - 1);

        // Evaluate jump (facet Ea)
        if (type_patch > 0 && a == ncells)
        {
          diff_proj_flux[0] = coefficients_G_Ta[0];
          diff_proj_flux[1] = coefficients_G_Ta[1];
        }
        else
        {
          diff_proj_flux[0] = coefficients_G_Tap1[0] - coefficients_G_Ta[0];
          diff_proj_flux[1] = coefficients_G_Tap1[1] - coefficients_G_Ta[1];
        }

        double jump_proj_flux_Ea = diff_proj_flux[0] * normal_Ea[0]
                                   + diff_proj_flux[1] * normal_Ea[1];

        // Set DOF
        // (Positive jump as it is calculated with n_(Ta,Ea)=-n_(Tap1,Ea))
        c_ta_eam1 = jump_proj_flux_Eam1[0] * 0.5 * detJ_Eam1 - c_tam1_eam1;

        // Store calculated jump
        jump_proj_flux_Eam1[0] = jump_proj_flux_Ea;
      }
      else
      {
        // Extract mapping data
        dolfinx_adaptivity::cmdspan2_t J
            = extract_mapping_data(id_a, storage_J);
        dolfinx_adaptivity::cmdspan2_t K
            = extract_mapping_data(id_a, storage_K);

        // Local facet ids
        std::int8_t fl_TaEam1, fl_TaEa;
        std::tie(fl_TaEam1, fl_TaEa) = patch.fctid_local(a);
        std::int8_t fl_Tap1Ea = patch.fctid_local(a, a + 1);

        // DOFs (cell local) projected flux on facet Ea
        std::span<const std::int32_t> dofs_Ea = patch.dofs_projflux_fct(a);

        // Interpolation data
        dolfinx_adaptivity::mdspan_t<const double, 4> M
            = kernel_data.interpl_matrix_facte();

        // Shape functions RHS
        dolfinx_adaptivity::s_cmdspan2_t shp_TaEa
            = kernel_data.shapefunctions_fct_rhs(fl_TaEa);
        dolfinx_adaptivity::s_cmdspan2_t shp_Tap1Ea
            = kernel_data.shapefunctions_fct_rhs(fl_Tap1Ea);

        // Shape-functions hat-function
        dolfinx_adaptivity::s_cmdspan2_t hat_TaEam1
            = kernel_data.shapefunctions_fct_hat(fl_TaEam1);
        dolfinx_adaptivity::s_cmdspan2_t hat_TaEa
            = kernel_data.shapefunctions_fct_hat(fl_TaEa);

        // Quadrature loop
        c_ta_eam1 = -c_tam1_eam1;
        std::fill(cj_ta_ea.begin(), cj_ta_ea.end(), 0.0);

        for (std::size_t n = 0; n < nipoints_facet; ++n)
        {
          // Global index of Tap1
          std::int32_t c_ap1 = (a < ncells) ? cells[id_a + 1] : cells[0];

          // Interpolate jump at quadrature point
          std::fill(diff_proj_flux.begin(), diff_proj_flux.end(), 0.0);

          if (type_patch > 0 && a == ncells)
          {
            for (std::size_t i = 0; i < ndofs_projflux_fct; ++i)
            {
              // Local and global IDs of first DOF on facet
              int s_Tap1 = dofs_Ea[i];
              int s_Ta = dofs_Ea[i + ndofs_projflux_fct];

              // Evaluate jump
              // jump = flux_proj_Ta on boundary!
              diff_proj_flux[0] = coefficients_G_Ta[s_Ta] * shp_TaEa(n, s_Ta);
              diff_proj_flux[1]
                  = coefficients_G_Ta[s_Ta + 1] * shp_TaEa(n, s_Ta);
            }
          }
          else
          {
            // Interpolate jump
            for (std::size_t i = 0; i < ndofs_projflux_fct; ++i)
            {
              // Local and global IDs of first DOF on facet
              int id_Tap1 = dofs_Ea[i];
              int id_Ta = dofs_Ea[i + ndofs_projflux_fct];
              int offs_Tap1 = 2 * id_Tap1;
              int offs_Ta = 2 * id_Ta;

              // Evaluate jump
              // jump = (flux_proj_Tap1 - flux_proj_Ta)
              diff_proj_flux[0]
                  += coefficients_G_Tap1[offs_Tap1] * shp_Tap1Ea(n, id_Tap1)
                     - coefficients_G_Ta[offs_Ta] * shp_TaEa(n, id_Ta);
              diff_proj_flux[1]
                  += coefficients_G_Tap1[offs_Tap1 + 1] * shp_Tap1Ea(n, id_Tap1)
                     - coefficients_G_Ta[offs_Ta + 1] * shp_TaEa(n, id_Ta);
            }
          }

          // Multiply jump with hat-function
          jGhat(0, 0) = -jG_Eam1(n, 0) * hat_TaEam1(n, node_i_Ta);
          jGhat(0, 1) = -jG_Eam1(n, 1) * hat_TaEam1(n, node_i_Ta);
          jGhat(1, 0) = -diff_proj_flux[0] * hat_TaEa(n, node_i_Ta);
          jGhat(1, 1) = -diff_proj_flux[1] * hat_TaEa(n, node_i_Ta);

          // Pull back jump
          kernel_data.pull_back_flux(jGhat_mapped, jGhat, J, storage_detJ[id_a],
                                     K);

          // std::cout << "jGhat_0, jGhat_1, jGhat_mapped_0, jGhat_mapped_1: "
          //           << jGhat(0, 0) << ", " << jGhat(0, 1) << ", "
          //           << jGhat_mapped(0, 0) << ", " << jGhat_mapped(0, 1)
          //           << std::endl;

          // std::cout << "K_00, K_01, K_10, K_11, detJ: " << K(0, 0) << ", "
          //           << K(0, 1) << ", " << K(1, 0) << ", " << K(1, 1) << ", "
          //           << storage_detJ[id_a] << std::endl;

          // Evaluate facet DOFs
          // (Positive jump, as calculated with n_(Ta,Ea)=-n_(Tap1,Ea))
          T aux = M(fl_TaEam1, 0, 0, n) * jGhat_mapped(0, 0)
                  + M(fl_TaEam1, 0, 1, n) * jGhat_mapped(0, 1);
          c_ta_eam1 += prefactor_dof(id_a, 0) * aux;

          // std::cout << "a, M_0, M_1, jGhat_0, jGhat_1, fid_Eam1, prefact: "
          // << a
          //           << ", " << M(fl_TaEam1, 0, 0, n) << ", "
          //           << M(fl_TaEam1, 0, 1, n) << ", " << jGhat_mapped(0, 0)
          //           << ", " << jGhat_mapped(0, 1) << ", " <<
          //           unsigned(fl_TaEam1)
          //           << ", " << prefactor_dof(id_a, 0) << std::endl;

          if constexpr (id_flux_order == 2)
          {
            cj_ta_ea[0] += M(fl_TaEa, 1, 0, n) * jGhat_mapped(1, 0)
                           + M(fl_TaEa, 1, 1, n) * jGhat_mapped(1, 1);
          }
          else
          {
            for (std::size_t j = 1; j < ndofs_flux_fct; ++j)
            {
              cj_ta_ea[j - 1] += M(fl_TaEa, j, 0, n) * jGhat_mapped(1, 0)
                                 + M(fl_TaEa, j, 1, n) * jGhat_mapped(1, 1);
            }
          }

          // Store jump
          jG_Eam1(n, 0) = diff_proj_flux[0];
          jG_Eam1(n, 1) = diff_proj_flux[1];
        }
      }

      // Treatment of facet 0
      if (a == 1)
      {
        if (type_patch == 0 | type_patch == 2)
        {
          // Set c_ta_eam1 to zero
          c_ta_eam1 = 0.0;
        }
        else if (type_patch == 1)
        {
          throw std::runtime_error("Equilibration: Neumann BCs not supported!");
        }
        else
        {
          throw std::runtime_error("Equilibration: Neumann BCs not supported!");
        }
      }

      /* DOFs from cell integrals */
      if constexpr (id_flux_order == 1)
      {
        // Set DOF on facet Ea
        c_ta_ea = coefficients_f[0] * (detJ / 6) - c_ta_eam1;
      }
      else
      {
        // Isoparametric mapping
        dolfinx_adaptivity::cmdspan2_t K
            = extract_mapping_data(id_a, storage_K);

        // Quadrature points and weights
        const int nqpoints = kernel_data.nqpoints_cell();
        dolfinx_adaptivity::cmdspan2_t qpoints
            = kernel_data.quadrature_points_cell();
        std::span<const double> weights = kernel_data.quadrature_weights_cell();

        // Shape-functions RHS
        dolfinx_adaptivity::s_cmdspan3_t shp_rhs
            = kernel_data.shapefunctions_cell_rhs(K);

        // Shape-functions hat-function
        dolfinx_adaptivity::s_cmdspan2_t shp_hat
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
              = (f + div_g) * shp_hat(n, node_i_Ta) * weights[n] * detJ;

          // Evaluate facet DOF
          c_ta_ea += aux;

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

      // std::cout << "Results cell (local, global): " << a << ", " << c_a
      //           << std::endl;
      // std::cout << "c_ta_eam1=" << c_ta_eam1 << std::endl;
      // std::cout << "c_ta_ea=" << c_ta_ea << std::endl;
      // std::cout << "cj_ta_ea=" << cj_ta_ea[0] << std::endl;

      /* Store DOFs to global solution vector */
      if constexpr (id_flux_order == 1)
      {
        // Extract global DOF ids
        std::span<const std::int32_t> gdofs_flux
            = patch.dofs_flux_fct_global(a);

        // Set DOF values
        x_flux_dhdiv[gdofs_flux[0]] += prefactor_dof(id_a, 0) * c_ta_eam1;
        x_flux_dhdiv[gdofs_flux[1]] += prefactor_dof(id_a, 1) * c_ta_ea;
      }
      else
      {
        // Global DOF ids
        std::span<const std::int32_t> gdofs_fct = patch.dofs_flux_fct_global(a);
        std::span<const std::int32_t> gdofs_cell
            = patch.dofs_flux_cell_global(a);

        if constexpr (id_flux_order == 2)
        {
          // Set DOF values facet Eam1
          x_flux_dhdiv[gdofs_fct[0]] += prefactor_dof(id_a, 0) * c_ta_eam1;

          // Set DOF values facet Ea
          x_flux_dhdiv[gdofs_fct[2]] += prefactor_dof(id_a, 1) * c_ta_ea;
          x_flux_dhdiv[gdofs_fct[3]] += cj_ta_ea[0];

          // Set DOFs on cell
          x_flux_dhdiv[gdofs_cell[0]] += c_ta_div[0];
          x_flux_dhdiv[gdofs_cell[1]] += c_ta_div[1];
        }
        else
        {
          // Set zero-order DOFs on facets
          x_flux_dhdiv[gdofs_fct[0]] += prefactor_dof(id_a, 0) * c_ta_eam1;
          x_flux_dhdiv[gdofs_fct[ndofs_flux_fct]]
              += prefactor_dof(id_a, 1) * c_ta_ea;

          // Set higher-order DOFs on facets
          for (std::size_t i = 0; i < ndofs_flux_fct - 1; ++i)
          {
            const int offs = gdofs_fct[ndofs_flux_fct + i];

            // DOFs on facet Ea
            x_flux_dhdiv[offs] += cj_ta_ea[i];
          }

          // Set divergence DOFs on cell
          for (std::size_t i = 0; i < ndofs_flux_cell_div; ++i)
          {
            x_flux_dhdiv[gdofs_cell[i]] += c_ta_div[i];
          }
        }
      }

      // Update c_tam1_eam1
      c_tam1_eam1 = c_ta_ea;
    }
  }
}

template <typename T, int id_flux_order = 3>
void minimise_flux(const mesh::Geometry& geometry,
                   PatchFluxCstm<T, id_flux_order>& patch,
                   ProblemDataFluxCstm<T>& problem_data,
                   KernelData<T>& kernel_data)
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

  const graph::AdjacencyList<std::int32_t>& flux_dofmap
      = problem_data.fspace_flux_hdiv()->dofmap()->list();

  // Cells on patch
  std::span<const std::int32_t> cells = patch.cells();
  const int ncells = patch.ncells();

  // Facets on patch
  const int nfcts = patch.nfcts();

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
  Eigen::PartialPivLU<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
      solver;

  /* Initialize solution process */
  // Number of nodes on reference cell
  int nnodes_cell = kernel_data.nnodes_cell();

  /* Storage cell geometries and DOF prefactors*/
  // Initialisations
  const int cstride_geom = 3 * nnodes_cell;
  std::vector<double> coordinate_dofs(ncells * cstride_geom, 0);

  std::int8_t fctloc_ea, fctloc_eam1;
  bool noutward_ea, noutward_eam1;

  std::vector<double> dprefactor_dof(ncells * 2, 1.0);
  dolfinx_adaptivity::mdspan2_t prefactor_dof(dprefactor_dof.data(), ncells, 2);

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

    /* DOF transformation */
    std::tie(fctloc_eam1, fctloc_ea) = patch.fctid_local(index + 1);
    std::tie(noutward_eam1, noutward_ea)
        = kernel_data.fct_normal_is_outward(fctloc_eam1, fctloc_ea);

    set_dof_prefactors(index + 1, noutward_eam1, noutward_ea, 1,
                       dprefactor_dof);
  }

  /* Perform minimisation */
  for (std::size_t i_rhs = 0; i_rhs < problem_data.nlhs(); ++i_rhs)
  {
    /* Extract data */
    // Patch type
    const int type_patch = patch.type(i_rhs);

    // Solution vector (flux, picewise-H(div))
    std::span<T> x_flux_dhdiv = problem_data.flux(i_rhs).x()->mutable_array();

    // Set coefficients (copy solution data into flattend structure)
    copy_cell_data<T, 1>(cells, flux_dofmap, x_flux_dhdiv, coefficients,
                         ndofs_flux, 1);

    /* Perform minimisation */
    if (i_rhs == 0)
    {
      // Initialize tangents
      A_patch.setZero();
      L_patch.setZero();

      // Assemble tangents
      assemble_minimisation<T, id_flux_order, true>(
          A_patch, L_patch, cells, patch, kernel_data, prefactor_dof,
          coefficients, ndofs_flux, coordinate_dofs, type_patch);

      // LU-factorization of system matrix
      if constexpr (id_flux_order > 1)
      {
        solver.compute(A_patch);
      }
    }
    else
    {
      if (patch.equal_patch_types())
      {
        // Assemble only vector
        throw std::runtime_error("Not Implemented!");
      }
      else
      {
        // Recreate patch and reassemble entire system
        throw std::runtime_error("Not Implemented!");
      }
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

    // Correct flux
    for (std::size_t a = 1; a < ncells + 1; ++a)
    {
      int id_a = a - 1;

      // Set d_0
      std::span<const std::int32_t> gdofs_flux = patch.dofs_flux_fct_global(a);

      x_flux_dhdiv[gdofs_flux[0]] += prefactor_dof(id_a, 0) * u_patch(0);
      x_flux_dhdiv[gdofs_flux[1]] -= prefactor_dof(id_a, 1) * u_patch(0);

      // Set d_E
      if constexpr (id_flux_order > 1)
      {
        throw std::runtime_error("Not Implemented!");
      }

      // Set d_T
      if constexpr (id_flux_order > 2)
      {
        throw std::runtime_error("Not Implemented!");
      }
    }
  }
}
} // namespace dolfinx_adaptivity::equilibration