#pragma once

#include "PatchFluxEV.hpp"
#include "StorageStiffness.hpp"
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Sparse"
#include <algorithm>
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
/// @param kernel_a                   Kernel bilinar form
/// @param kernel_lpen                Kernel for penalisation terms
/// @param kernel_l                   Kernel linear form
/// @param consts_l                   Constants linar form
/// @param coeffs_l                   Coefficients linar form
/// @param info_coeffs_l              Information about storage of coeffs
///                                   (linear form)
/// @param cell_info                  Information for DOF transformation
/// @param storage_stiffness          Storage element tangents
/// @param x_flux                     DOFs flux function (Hdiv)

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
    fem::FEkernel<T> auto kernel_a, fem::FEkernel<T> auto kernel_lpen,
    fem::FEkernel<T> auto kernel_l, std::span<const T> consts_l,
    std::span<T> coeffs_l, const std::vector<int>& info_coeffs_l,
    std::span<const std::uint32_t> cell_info, std::span<T> x_flux,
    std::span<T> x_flux_dg)
{
  // Initilaize storage cell geoemtry
  const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();
  std::span<const double> x = geometry.x();

  std::vector<double> coordinate_dofs(3 * geometry.cmap().dim());

  /* Extract patch informations */
  // Type patch
  const int type_patch = patch.type();

  // Cells on patch
  std::span<const std::int32_t> cells = patch.cells();

  // List local node-ids on central node
  std::span<const std::int8_t> inode_local = patch.inodes_local();

  // DOF counters
  const int ndim0 = patch.ndofs_elmt();
  const int ndof_elmt_nz = patch.ndofs_elmt_nz();
  const int ndof_patch = patch.ndofs_patch();
  const int ndof_ppatch = ndof_patch + 1;

  // Initialize storage of tangent arrays (Penalty for pure Neumann
  // problems)
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A_patch;
  Eigen::Matrix<T, Eigen::Dynamic, 1> L_patch, flux_patch;

  std::vector<T> Le(ndim0);
  std::vector<T> Ae(ndim0 * ndim0);
  std::fill(Ae.begin(), Ae.end(), 0);
  std::span<T> _Le(Le);

  if (type_patch < 2)
  {
    A_patch.resize(ndof_ppatch, ndof_ppatch);
    L_patch.resize(ndof_ppatch);
    flux_patch.resize(ndof_ppatch);
  }
  else
  {
    A_patch.resize(ndof_patch, ndof_patch);
    L_patch.resize(ndof_patch);
    flux_patch.resize(ndof_patch);
  }

  for (std::size_t index = 0; index < cells.size(); ++index)
  {
    // Get current cell
    std::int32_t c = cells[index];

    // Get current stiffness-matrix in storage
    // std::span<T> Ae = storage_stiffness.stiffness_elmt(c);
    // std::span<T> Pe = storage_stiffness.penalty_elmt(c);

    // Set hat-function appropriately
    set_hat_function(coeffs_l, info_coeffs_l, c, inode_local[index], 1.0);

    /* Evaluate tangent arrays if not already done */
    // Extract cell geometry
    auto x_dofs = x_dofmap.links(c);
    for (std::size_t j = 0; j < x_dofs.size(); ++j)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs[j]), 3,
                  std::next(coordinate_dofs.begin(), 3 * j));
    }

    // Evaluate bilinar form
    // if (storage_stiffness.evaluation_status(c) == 0)
    // {
    //   std::cout << unsigned(storage_stiffness.evaluation_status(c))
    //             << std::endl;
    //   // // Initialize bilinear form
    //   // std::fill(Ae.begin(), Ae.end(), 0);
    //   // std::fill(Pe.begin(), Pe.end(), 0);

    //   // // Evaluate bilinar form
    //   // kernel_a(Ae.data(), nullptr, nullptr, coordinate_dofs.data(),
    //   nullptr,
    //   //          nullptr);

    //   // // Evaluate penalty terms
    //   // kernel_lpen(Pe.data(), nullptr, nullptr, coordinate_dofs.data(),
    //   // nullptr,
    //   //             nullptr);

    //   // // DOF transformation
    //   // dof_transform(Ae, cell_info, c, ndim0);
    //   // dof_transform_to_transpose(Ae, cell_info, c, ndim0);

    //   // Set identifire for evaluated data
    //   storage_stiffness.mark_cell_evaluated(c);
    // }

    // Evaluate linear form
    std::fill(Le.begin(), Le.end(), 0);

    kernel_l(Le.data(), coeffs_l.data() + c * info_coeffs_l[0], consts_l.data(),
             coordinate_dofs.data(), nullptr, nullptr);

    dof_transform(_Le, cell_info, c, 1);

    /* Assemble into patch system */
    // Element-local and patch-local DOFmap
    std::span<const int32_t> dofs_elmt = patch.dofs_elmt(index);
    std::span<const int32_t> dofs_patch = patch.dofs_patch(index);

    for (std::size_t k = 0; k < ndof_elmt_nz; ++k)
    {
      // Calculate offset
      int offset = dofs_elmt[k] * ndim0;

      // Assemble load vector
      L_patch(dofs_patch[k]) += Le[dofs_elmt[k]];

      for (std::size_t l = 0; l < ndof_elmt_nz; ++l)
      {
        // Assemble stiffness matrix
        A_patch(dofs_patch[k], dofs_patch[l]) += Ae[offset + dofs_elmt[l]];
      }
    }

    // Add penalyt terms
    if (type_patch < 2)
    {
      // Required counters
      const int ndofs_cons = patch.ndofs_cons();
      const int offset = patch.ndofs_flux_nz();

      std::vector<T> Pe(ndofs_cons);
      std::fill(Pe.begin(), Pe.end(), 0);

      // Loop over DOFs
      for (std::size_t k = 0; k < ndofs_cons; ++k)
      {
        // Add K_ql
        A_patch(dofs_patch[offset + k], ndof_ppatch) += Pe[k];

        // Add K_lq
        A_patch(ndof_ppatch, dofs_patch[offset + k]) += Pe[k];
      }
    }

    // Unset hat function
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