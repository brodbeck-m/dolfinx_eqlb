#pragma once

#include "PatchFluxEV.hpp"
#include "ProblemData.hpp"
#include "StorageStiffness.hpp"
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Sparse"
#include <algorithm>
#include <cmath>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <functional>
#include <iostream>
#include <iterator>
#include <span>
#include <vector>

namespace dolfinx_adaptivity::equilibration
{
/// Set hat-function for cell_i on patch
///
/// Modify coefficients of cell_i such that the function-value of the hat
/// function has value at the central node of the patch.
/// @param coeffs_l      Span of the coefficients of all elements
/// @param info_coeffs_l Infos about the corfficients
///                      [cstride, begin_hat, begin_flux]
/// @param cell_i        Identifier (processor level) for current cell
/// @param node_i        Cell-local id of the central node of the patch
/// @param value         Value onto whcih the function has to be sed

template <typename T>
void set_hat_function(std::span<T> coeffs_l,
                      const std::vector<int>& info_coeffs_l,
                      const int32_t cell_i, const std::int8_t node_i, T value)
{
  coeffs_l[info_coeffs_l[0] * cell_i + info_coeffs_l[1] + node_i] = value;
}

template <typename T>
void assemble_tangents(
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& A_patch,
    Eigen::Matrix<T, Eigen::Dynamic, 1>& L_patch,
    std::span<const std::int32_t> cells,
    std::vector<dolfinx::fem::impl::scalar_value_type_t<T>>& coordinate_dofs,
    const int cstride_geom, PatchFluxEV& patch,
    const std::function<void(const std::span<T>&,
                             const std::span<const std::uint32_t>&,
                             std::int32_t, int)>& dof_transform,
    const std::function<void(const std::span<T>&,
                             const std::span<const std::uint32_t>&,
                             std::int32_t, int)>& dof_transform_to_transpose,
    std::span<const std::uint32_t> cell_info, fem::FEkernel<T> auto kernel_a,
    fem::FEkernel<T> auto kernel_lpen, fem::FEkernel<T> auto kernel_l,
    std::span<const T> constants_l, std::span<T> coefficients_l,
    const int cstride_l, std::span<const std::int8_t> bmarkers,
    std::span<const T> bvalues, StorageStiffness<T>& storage_stiffness);

template <typename T>
void assemble_vector(
    Eigen::Matrix<T, Eigen::Dynamic, 1>& L_patch,
    std::span<const std::int32_t> cells,
    std::vector<dolfinx::fem::impl::scalar_value_type_t<T>>& coordinate_dofs,
    const int cstride_geom, PatchFluxEV& patch,
    const std::function<void(const std::span<T>&,
                             const std::span<const std::uint32_t>&,
                             std::int32_t, int)>& dof_transform,
    std::span<const std::uint32_t> cell_info, fem::FEkernel<T> auto kernel_l,
    std::span<const T> constants_l, std::span<T> coefficients_l,
    const int cstride_l, std::span<const std::int8_t> bmarkers,
    std::span<const T> bvalues, StorageStiffness<T>& storage_stiffness)
{
  throw std::runtime_error("assembly_vector: Not implemented!")
}

template <typename T>
void apply_lifting(std::span<T> Ae, std::span<T> Le,
                   std::span<const std::int8_t> bmarkers,
                   std::span<const T> bvalues,
                   std::span<const int32_t> dofs_elmt,
                   std::span<const int32_t> dofs_global, const int type_patch,
                   const int ndof_elmt_nz, const int ndof_elmt);

/// Assembly and solution of patch problems
///
/// Assembly of the patch-wise equation systems, following [1]. Element
/// stiffness-matrizes - as these stay constant but appear in multiple
//  patches - are stored within a adjacency-list. The LHS is assembled
//  patch-wise and not stored!
///
/// [1] Ern, A. & Vohralík, M.: Polynomial-Degree-Robust A Posteriori
///     Estimates in a Unified Setting for Conforming, Nonconforming,
///     Discontinuous Galerkin, and Mixed Discretizations, 2015
///
/// @param geometry                   msh->geomtry of the problem
/// @param patch                      The patch
/// @param dofmap_global              dofmap.list() of global FEspace
/// @param dof_transform              DOF-transformation function
/// @param dof_transform_to_transpose DOF-transformation function
/// @param cell_info                  Information for DOF transformation
/// @param kernel_a                   Kernel bilinar form
/// @param kernel_lpen                Kernel for penalisation terms
/// @param probelm_data               Linear forms and problem dependent
///                                   input data
/// @param storage_stiffness          Storage element tangents
/// @param x_flux                     DOFs projected flux function

template <typename T>
void equilibrate_flux_constrmin(
    const mesh::Geometry& geometry, PatchFluxEV& patch,
    const graph::AdjacencyList<std::int32_t>& dofmap_global,
    const std::function<void(const std::span<T>&,
                             const std::span<const std::uint32_t>&,
                             std::int32_t, int)>& dof_transform,
    const std::function<void(const std::span<T>&,
                             const std::span<const std::uint32_t>&,
                             std::int32_t, int)>& dof_transform_to_transpose,
    std::span<const std::uint32_t> cell_info, fem::FEkernel<T> auto kernel_a,
    fem::FEkernel<T> auto kernel_lpen, ProblemData<T>& problem_data,
    StorageStiffness<T>& storage_stiffness, std::span<T> x_flux_dg)
{
  /* Initialize Patch-LGS*/
  const int ndof_ppatch = patch.ndofs_patch() + 1;

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A_patch;
  Eigen::Matrix<T, Eigen::Dynamic, 1> L_patch, u_patch;

  A_patch.resize(ndof_ppatch, ndof_ppatch);
  L_patch.resize(ndof_ppatch);
  u_patch.resize(ndof_ppatch);

  /* Initialize hat-function and cell-geometries */
  // Required patch-data
  const int ncells = patch.ncells();
  std::span<const std::int8_t> inode_local = patch.inodes_local();

  // Get geometry data
  const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();
  std::span<const dolfinx::fem::impl::scalar_value_type_t<T>> x = geometry.x();

  // Initialize geometry storage
  const int cstride_geom = 3 * geometry.cmap().dim();
  std::vector<dolfinx::fem::impl::scalar_value_type_t<T>> coordinate_dofs(
      ncells * cstride_geom, 0);

  for (std::size_t index = 0; index < ncells; ++index)
  {
    // Get current cell
    std::int32_t c = cells[index];

    // Set hat function
    set_hat_function(coeffs_l, info_coeffs_l, c, inode_local[index], 1.0);

    // Copy cell geometry
    std::span<dolfinx::fem::impl::scalar_value_type_t<T>> coordinate_dofs_e(
        coordinate_dofs.data() + index * cstride_geom, cstride_geom);

    auto x_dofs = x_dofmap.links(c);
    for (std::size_t j = 0; j < x_dofs.size(); ++j)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs[j]), 3,
                  std::next(coordinate_dofs_e.begin(), 3 * j));
    }
  }

  /* Solve equilibartion */
  for (std::size_t i_lhs = 0; i_lhs < problem_data.nlhs(); ++i_lhs)
  {
    /* Extract data for current LHS */
    // Integration kernel
    const auto kernel_l = problem_data.kernel(i_lhs);

    // Constants and coefficients
    std::span<const T> consts_l = problem_data.constants(i_lhs);
    std::span<T> coeffs_l = problem_data.coefficients(i_lhs);

    // Infos about coefficients
    std::vector<int> info_coeffs_l(3, 0);
    info_coeffs_l[0] = problem_data.cstride(i_lhs);
    info_coeffs_l[1] = problem_data.begin_hat(i_lhs);

    // Boundary data
    std::span<const std::int8_t> bmarkers
        = problem_data.boundary_markers(i_lhs);
    std::span<const T> bvalues = problem_data.boundary_values(i_lhs);

    /* Extract patch informations */
    // Type patch
    const int type_patch = patch.type(i_lhs);

    // Assemble system
    if (i_lhs == 0)
    {
      // Get cells
      std::span<const std::int32_t> cells = patch.cells();

      // Assemble tangents
      assemble_tangents(A_patch, L_patch, cells, coordinate_dofs, cstride_geom,
                        patch, dof_transform, dof_transform_to_transpose,
                        cell_info, kernel_a, kernel_lpen, kernel_l, constants_l,
                        coefficients_l, cstride_l, bmarkers, bvalues,
                        storage_stiffness);
    }
    else
    {
      if (patch.equal_patch_types)
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

  /* Unset hat-functions */
  for (std::size_t index = 0; index < ncells; ++index)
  {
    // Get current cell
    std::int32_t c = cells[index];

    // Set hat function
    set_hat_function(coeffs_l, info_coeffs_l, c, inode_local[index], 0.0);
  }

  // // Debug
  // std::fill(x_flux_dg.begin(), x_flux_dg.end(), 0);
  // std::cout << "Type-Patch: " << type_patch << std::endl;
  // std::cout << "nDOFs patch: " << ndof_patch << std::endl;
  // int offset = 0;
  // for (std::size_t k = 0; k < ndof_patch; ++k)
  // {
  //   // Calculate offset
  //   offset = k * ndof_patch;

  //   for (std::size_t l = 0; l < ndof_patch; ++l)
  //   {
  //     // Assemble stiffness matrix
  //     x_flux_dg[offset + l] = A_patch(k, l);
  //   }
  // }
}
} // namespace dolfinx_adaptivity::equilibration