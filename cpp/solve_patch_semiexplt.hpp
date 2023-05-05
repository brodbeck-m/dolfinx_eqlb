#pragma once

#include "KernelData.hpp"
#include "PatchFluxCstm.hpp"
#include "ProblemDataFluxCstm.hpp"
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
#include <iostream>
#include <iterator>
#include <span>
#include <tuple>
#include <vector>

using namespace dolfinx;

namespace dolfinx_adaptivity::equilibration
{
template <typename T>
void copy_cell_data(std::span<const T> data_global,
                    std::span<const std::int32_t> data_dofs,
                    std::vector<T>& data_cell)
{
  for (std::size_t j = 0; j < data_dofs.size(); ++j)
  {
    std::copy_n(std::next(data_global.begin(), 3 * data_dofs[j]), 3,
                std::next(data_cell.begin(), 3 * j));
  }
}

template <typename T>
void copy_cell_data(std::span<const std::int32_t> cells,
                    const graph::AdjacencyList<std::int32_t>& dofmap_data,
                    std::span<const T> data_global, std::vector<T>& data_cell,
                    const int cstride_data)
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
    copy_cell_data<T>(data_global, data_dofs, data_dofs_e);
  }
}

template <typename T, int id_flux_order = -1>
void calc_fluxtilde_explt(const mesh::Geometry& geometry,
                          PatchFluxCstm<T, id_flux_order>& patch,
                          ProblemDataFluxCstm<T>& problem_data,
                          KernelData& kernel_data)
{
  assert(flux_order < 0);

  /* Geometry data */
  const int dim = 2;
  const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();
  std::span<const double> x = geometry.x();

  /* Extract patch data */
  // Elements on patch
  std::span<const std::int32_t> cells = patch.cells();
  int ncells = patch.ncells();

  /* Initialize solution process */
  // Number of nodes on reference cell
  int nnodes_cell = kernel_data.nnodes_cell();

  // Jacobian J, inverse K and determinant detJ
  std::array<double, 9> Jb;
  dolfinx_adaptivity::mdspan2_t J(Jb.data(), 2, 2);
  std::array<double, 9> Kb;
  dolfinx_adaptivity::mdspan2_t K(Kb.data(), 2, 2);
  std::array<double, 18> detJ_scratch;
  std::vector<double> storage_detJ(ncells, 0);

  // +/- cells
  std::int32_t cell_plus, cell_minus, cell_plus_eam1, cell_minus_eam1;

  // Physical normal
  std::vector<double> storage_normal_phys((ncells + 1) * dim, 0);

  // Jump within projected flux
  std::array<double, 2> jump_proj_flux;

  // DOFs flux (cell-wise H(div))
  T c_ta_ea, c_ta_eam1, c_tam1_eam1;

  // Storage cell geometries/ normal orientation
  const int cstride_geom = 3 * nnodes_cell;
  std::vector<double> coordinate_dofs(ncells * cstride_geom, 0);

  std::int8_t fctloc_ea, fctloc_eam1;
  bool noutward_ea, noutward_eam1;
  std::vector<double> dprefactor_dof(ncells * 2, 1.0);
  dolfinx_adaptivity::mdspan2_t prefactor_dof(dprefactor_dof.data(), ncells, 2);

  bool eval_normal_am1 = false;
  for (std::size_t index = 0; index < ncells; ++index)
  {
    // Get current cell
    std::int32_t c = cells[index];

    /* Copy cell geometry */
    std::span<double> coordinate_dofs_e(
        coordinate_dofs.data() + index * cstride_geom, cstride_geom);

    auto x_dofs = x_dofmap.links(c);
    for (std::size_t j = 0; j < x_dofs.size(); ++j)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs[j]), 3,
                  std::next(coordinate_dofs_e.begin(), 3 * j));
    }

    /* Piola mapping */
    // Reshape geometry infos
    dolfinx_adaptivity::cmdspan2_t coords(coordinate_dofs_e.data(), nnodes_cell,
                                          3);
    // Calculate Jacobi, inverse, and determinant
    storage_detJ[index]
        = kernel_data.compute_jacobian(J, K, detJ_scratch, coords);

    /* DOF transformation */
    // Get local fact ids on cell_i
    std::tie(fctloc_ea, fctloc_eam1) = patch.fctid_local(index + 1);

    // Set prefactor of facte DOFs (H(div) flux)
    std::tie(noutward_eam1, noutward_ea)
        = kernel_data.fct_normal_is_outward(fctloc_ea, fctloc_eam1);

    dprefactor_dof[2 * index] = (noutward_ea) ? 1.0 : -1.0;
    dprefactor_dof[2 * index + 1] = (noutward_eam1) ? 1.0 : -1.0;

    /* Calculation of physical normals */
    int a = index + 1;

    // Check if last cell is reached (only internal patch!)
    if ((patch.type(0) > 0) && (a == 1))
    {
      // Get local facet id of T_0 on E_0
      std::int8_t fctid_loc_plus = patch.fctid_local(0, 1);

      // Get storage of normal
      std::span<double> normal_e(storage_normal_phys.data(), dim);

      // Transform normal into physical space
      kernel_data.physical_fct_normal(normal_e, K, fctid_loc_plus);
    }

    // Transform normal n_am1 within cell T_a
    if (eval_normal_am1)
    {
      int am1 = a - 1;

      // Unset idetifire
      eval_normal_am1 = false;

      // Get local facet id of E_am1 on T_a
      std::int8_t fctid_loc_plus = patch.fctid_local(am1, a);

      // Get storage of normal
      std::span<double> normal_e(storage_normal_phys.data() + am1 * dim, dim);

      // Transform normal into physical space
      kernel_data.physical_fct_normal(normal_e, K, fctid_loc_plus);
    }

    // Get +/- cell on facet E_a
    std::tie(cell_plus, cell_minus) = patch.cellpm(a);

    if (c == patch.cell(cell_plus))
    {
      // Get local facet id on T_a
      std::int8_t fctid_loc_plus = patch.fctid_local(a, cell_plus);

      // Get storage of normal
      std::span<double> normal_e(storage_normal_phys.data() + a * dim, dim);

      // Transform normal into physical space
      kernel_data.physical_fct_normal(normal_e, K, fctid_loc_plus);
    }
    else
    {
      // Set idetifier to transform n_a within cell T_ap1
      eval_normal_am1 = true;
    }

    // Check if last cell is reached (only internal patch!)
    if ((patch.type(0) == 0) && (a == ncells))
    {
      if (eval_normal_am1)
      {
        // Recalculate Jacobi on E_1
        std::span<double> coordinate_dofs_0(coordinate_dofs.data(),
                                            cstride_geom);
        dolfinx_adaptivity::cmdspan2_t coords(coordinate_dofs_0.data(),
                                              nnodes_cell, 3);
        double detJ = kernel_data.compute_jacobian(J, K, detJ_scratch, coords);

        // Get local facet id of E_0 on T_1
        std::int8_t fctid_loc_plus = patch.fctid_local(0, 1);

        // Get storage of normal
        std::span<double> normal_e(storage_normal_phys.data() + ncells * dim,
                                   dim);

        // Transform normal into physical space
        kernel_data.physical_fct_normal(normal_e, K, fctid_loc_plus);
      }

      // Store normal on E_0
      storage_normal_phys[0] = storage_normal_phys[2 * ncells];
      storage_normal_phys[1] = storage_normal_phys[2 * ncells + 1];
    }
  }

  for (std::size_t i_rhs = 0; i_rhs < problem_data.nlhs(); ++i_rhs)
  {
    /* Extract data */
    // Patch type
    int type_patch = patch.type(i_rhs);

    // Solution vector (flux, picewise-H(div))
    std::span<T> x_flux_dhdiv = problem_data.flux(i_rhs).x()->mutable_array();

    // Projected primal flux
    std::span<T> x_flux_proj
        = problem_data.projected_flux(i_rhs).x()->mutable_array();

    // Projected RHS
    std::span<T> x_rhs_proj
        = problem_data.projected_rhs(i_rhs).x()->mutable_array();

    /* Calculate sigma_tilde */
    int loop_end;
    if (type_patch == 0 | type_patch == 2)
    {
      loop_end = ncells + 1;

      // Isoparametric mappring for cell
      const double detJ = storage_detJ[0];

      // Extract RHS value on current cell
      T f_i = x_rhs_proj[cells[0]];

      // Set DOFs for cell 1
      c_ta_ea = prefactor_dof(0, 1) * f_i * detJ / 6;

      // Store coefficients and set history values
      std::span<const std::int32_t> gdofs_flux = patch.dofs_flux_fct_global(1);

      x_flux_dhdiv[gdofs_flux[0]] = 0;
      x_flux_dhdiv[gdofs_flux[1]] = c_ta_ea;

      c_tam1_eam1 = c_ta_ea;
    }
    else if (type_patch == 1)
    {
      loop_end = ncells;
      throw std::runtime_error("Equilibration: Neumann BCs not supported!");
    }
    else
    {
      loop_end = ncells + 1;
      throw std::runtime_error("Equilibration: Neumann BCs not supported!");
    }

    for (std::size_t a = 2; a < loop_end; ++a)
    {
      // Set id for acessing storage
      int id_a = a - 1;

      // Global cell id
      std::int32_t c = cells[id_a];

      // Isoparametric mappring cell
      const double detJ = storage_detJ[id_a];

      // Extract physical normal
      std::span<double> normal_phys(storage_normal_phys.data() + a * dim, dim);

      // Extract RHS value
      T f_i = x_rhs_proj[c];

      // Extract gadients (+/- side) ond facet E_am1
      std::span<const std::int32_t> dofs_projflux_fct
          = patch.dofs_projflux_fct(a - 1);

      jump_proj_flux[0] = x_flux_proj[dofs_projflux_fct[0]]
                          - x_flux_proj[dofs_projflux_fct[2]];
      jump_proj_flux[1] = x_flux_proj[dofs_projflux_fct[1]]
                          - x_flux_proj[dofs_projflux_fct[3]];

      double jump_i = jump_proj_flux[0] * normal_phys[0]
                      + jump_proj_flux[1] * normal_phys[1];

      // Set DOFs for cell
      // FIXME - Consider integral of hat function over facet
      c_ta_eam1 = prefactor_dof(id_a, 0) * (jump_i - c_tam1_eam1);
      c_ta_ea = prefactor_dof(id_a, 1) * (f_i * detJ / 6 - c_ta_eam1);

      // Store coefficients and set history values
      std::span<const std::int32_t> gdofs_flux = patch.dofs_flux_fct_global(a);

      x_flux_dhdiv[gdofs_flux[0]] = c_ta_eam1;
      x_flux_dhdiv[gdofs_flux[1]] = c_ta_ea;

      c_tam1_eam1 = c_ta_ea;
    }
  }
}

template <typename T, int id_flux_order = -1>
void minimise_flux(const mesh::Geometry& geometry,
                   PatchFluxCstm<T, id_flux_order>& patch,
                   ProblemDataFluxCstm<T>& problem_data,
                   KernelData& kernel_data)
{
  assert(flux_order < 0);

  /* Geometry data */
  const int dim = 2;
  const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();
  std::span<const double> x = geometry.x();

  /* Extract patch data */
  // Data flux element
  const int degree_rt = patch.degree_raviart_thomas();
  const int ndofs_flux = patch.ndofs_flux();
  const graph::AdjacencyList<std::int32_t>& flux_dofmap
      = problem_data.fspace_flux_hdiv()->dofmap()->list();

  // Cells on patch// Get cells
  std::span<const std::int32_t> cells = patch.cells();
  const int ncells = patch.ncells();

  // Factes on patch
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

  // Jacobian J and determinant detJ
  std::array<double, 9> Jb;
  dolfinx_adaptivity::mdspan2_t J(Jb.data(), 2, 2);
  std::array<double, 18> detJ_scratch;
  std::vector<double> storage_detJ(ncells, 0);

  // Storage cell geometries/ normal orientation and coefficients
  const int cstride_geom = 3 * nnodes_cell;
  std::vector<double> coordinate_dofs_e(3 * nnodes_cell, 0);

  std::int8_t fctloc_ea, fctloc_eam1;
  bool noutward_ea, noutward_eam1;
  std::vector<double> dprefactor_dof(ncells * 2, 1.0);
  dolfinx_adaptivity::mdspan2_t prefactor_dof(dprefactor_dof.data(), ncells, 2);

  std::vector<T> coefficients(ncells * ndofs_flux, 0);

  for (std::size_t index = 0; index < ncells; ++index)
  {
    // Get current cell
    std::int32_t c = cells[index];

    /* Copy cell coordinates */
    std::span<const std::int32_t> x_dofs = x_dofmap.links(c);
    copy_cell_data<double>(x, x_dofs, coordinate_dofs_e);

    /* Piola mapping */
    // Reshape geometry infos
    dolfinx_adaptivity::cmdspan2_t coords(coordinate_dofs_e.data(), nnodes_cell,
                                          3);
    // Calculate Jacobi, inverse, and determinant
    storage_detJ[index] = kernel_data.compute_jacobian(J, detJ_scratch, coords);

    /* DOF transformation */
    // Get local fact ids on cell_i
    std::tie(fctloc_ea, fctloc_eam1) = patch.fctid_local(index + 1);

    // Set prefactor of facte DOFs (H(div) flux)
    std::tie(noutward_eam1, noutward_ea)
        = kernel_data.fct_normal_is_outward(fctloc_ea, fctloc_eam1);

    dprefactor_dof[2 * index] = (noutward_ea) ? 1.0 : -1.0;
    dprefactor_dof[2 * index + 1] = (noutward_eam1) ? 1.0 : -1.0;
  }

  for (std::size_t i_rhs = 0; i_rhs < problem_data.nlhs(); ++i_rhs)
  {
    /* Extract data */
    // Patch type
    const int type_patch = patch.type(i_rhs);

    // Solution vector (flux, picewise-H(div))
    std::span<T> x_flux_dhdiv = problem_data.flux(i_rhs).x()->mutable_array();

    /* Perform minimisation */
    if (i_rhs == 0)
    {
      // Prepare coefficients
      copy_cell_data<T>(cells, flux_dofmap, x_flux_dhdiv, coefficients,
                        ndofs_flux);

      // Initialize tangents
      A_patch.setZero();
      L_patch.setZero();

      // Assemble tangents
      impl::assemble_tangents(A_patch, L_patch, cells, patch, kernel_data,
                              coefficients, storage_detJ, type_patch);

      // LU-factorization of system matrix
      solver.compute(A_patch);
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
  }
}

} // namespace dolfinx_adaptivity::equilibration