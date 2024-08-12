// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Patch.hpp"
#include "PatchFluxEV.hpp"
#include "ProblemDataFluxEV.hpp"
#include "StorageStiffness.hpp"
#include "assemble_patch_constrmin.hpp"
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

using namespace dolfinx;

namespace dolfinx_eqlb
{
/// Assembly and solution of patch problems
///
/// Assembly of the patch-wise equation systems, following [1]. Element
/// stiffness-matrices - as these stay constant but appear in multiple
//  patches - are stored within a adjacency-list. The LHS is assembled
//  patch-wise and not stored!
///
/// [1] Ern, A. & Vohralík, M.: Polynomial-Degree-Robust A Posteriori
///     Estimates in a Unified Setting for Conforming, Nonconforming,
///     Discontinuous Galerkin, and Mixed Discretizations, 2015
///
/// @param geometry                   msh->geometry of the problem
/// @param patch                      The patch
/// @param dofmap_global              dofmap.list() of global FEspace
/// @param dof_transform              DOF-transformation function
/// @param dof_transform_to_transpose DOF-transformation function
/// @param cell_info                  Information for DOF transformation
/// @param kernel_a                   Kernel bilinear form
/// @param kernel_lpen                Kernel for penalization terms
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
    fem::FEkernel<T> auto kernel_lpen, ProblemDataFluxEV<T>& problem_data,
    StorageStiffness<T>& storage_stiffness)
{
  /* Extract required data */
  // Cells on patch
  const int ncells = patch.ncells();
  std::span<const std::int32_t> cells = patch.cells();

  // DOFs per patch
  const int ndof_patch = patch.ndofs_patch();

  // Patch-central node
  std::span<const std::int8_t> inode_local = patch.inodes_local();

  /* Initialize Patch-LGS */
  // Patch arrays
  const int ndof_ppatch = ndof_patch + 1;

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A_patch;
  Eigen::Matrix<T, Eigen::Dynamic, 1> L_patch, u_patch;

  A_patch.resize(ndof_ppatch, ndof_ppatch);
  L_patch.resize(ndof_ppatch);
  u_patch.resize(ndof_ppatch);

  // Local solver
  Eigen::PartialPivLU<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> solver;

  /* Initialize hat-function and cell-geometries */
  // Get geometry data
  const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();
  std::span<const fem::impl::scalar_value_type_t<T>> x = geometry.x();

  // Initialize geometry storage/ hat-function
  const int cstride_geom = 3 * geometry.cmap().dim();
  std::vector<fem::impl::scalar_value_type_t<T>> coordinate_dofs(
      ncells * cstride_geom, 0);

  for (std::size_t index = 0; index < ncells; ++index)
  {
    // Get current cell
    std::int32_t c = cells[index];

    // Set hat function
    problem_data.set_hat_function(c, inode_local[index], 1.0);

    // Copy cell geometry
    std::span<fem::impl::scalar_value_type_t<T>> coordinate_dofs_e(
        coordinate_dofs.data() + index * cstride_geom, cstride_geom);

    auto x_dofs = x_dofmap.links(c);
    for (std::size_t j = 0; j < x_dofs.size(); ++j)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs[j]), 3,
                  std::next(coordinate_dofs_e.begin(), 3 * j));
    }
  }

  /* Calculate boundary conditions for current patch */
  if (patch.is_on_boundary())
  {
    // Get facet list
    const int nfcts = patch.nfcts();
    std::span<const std::int32_t> fcts = patch.fcts();

    // Calculate boundary conditions
    std::vector<std::int32_t> boundary_fcts{fcts[0], fcts[nfcts - 1]};
    std::vector<std::int8_t> inode_boundary_fcts{inode_local[0],
                                                 inode_local[ncells - 1]};

    problem_data.calculate_patch_bc(boundary_fcts, inode_boundary_fcts);
  }

  /* Solve equilibration */
  for (std::size_t i_lhs = 0; i_lhs < problem_data.nlhs(); ++i_lhs)
  {
    /* Extract data for current LHS */
    // Integration kernel
    const auto& kernel_l = problem_data.kernel(i_lhs);

    // Constants and coefficients
    std::span<const T> constants_l = problem_data.constants(i_lhs);
    std::span<T> coefficients_l = problem_data.coefficients(i_lhs);

    // Infos about coefficients
    int cstride_l = problem_data.cstride(i_lhs);

    // Boundary data
    std::span<const std::int8_t> bmarkers
        = problem_data.boundary_markers(i_lhs);
    std::span<const T> bvalues = problem_data.boundary_values(i_lhs);

    /* Solve equilibration on current patch */
    // Check if assembly of entire system is required
    bool assemble_entire_system = false;

    if (i_lhs == 0)
    {
      assemble_entire_system = true;
    }
    else
    {
      if (patch.is_on_boundary())
      {
        PatchType patch_i = patch.type(i_lhs);

        if (patch_i != patch.type(i_lhs - 1)
            || patch_i == PatchType::bound_mixed)
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
      assemble_tangents<T, true>(
          A_patch, L_patch, cells, coordinate_dofs, cstride_geom, patch,
          dof_transform, dof_transform_to_transpose, cell_info, kernel_a,
          kernel_lpen, kernel_l, constants_l, coefficients_l, cstride_l,
          bmarkers, bvalues, storage_stiffness, i_lhs);

      // LU-factorization of system matrix
      solver.compute(A_patch);
    }
    else
    {
      // Initialize RHS
      L_patch.setZero();

      // Assemble tangents
      assemble_tangents<T, false>(
          A_patch, L_patch, cells, coordinate_dofs, cstride_geom, patch,
          dof_transform, dof_transform_to_transpose, cell_info, kernel_a,
          kernel_lpen, kernel_l, constants_l, coefficients_l, cstride_l,
          bmarkers, bvalues, storage_stiffness, i_lhs);
    }

    // Solve equation system
    u_patch = solver.solve(L_patch);

    /* Add patch contribution to H(div) flux */
    // Extract solution vector
    std::span<T> x_flux_hdiv = problem_data.flux(i_lhs).x()->mutable_array();

    // Extract DOFs
    std::span<const int32_t> dofs_flux_patch = patch.dofs_fluxhdiv_patch();
    std::span<const int32_t> dofs_flux_global = patch.dofs_fluxhdiv_global();

    // Add local solution
    for (std::size_t k = 0; k < patch.ndofs_flux_patch_nz(); ++k)
    {
      x_flux_hdiv[dofs_flux_global[k]] += u_patch[dofs_flux_patch[k]];
    }
  }

  /* Unset hat-functions */
  for (std::size_t index = 0; index < ncells; ++index)
  {
    // Get current cell
    std::int32_t c = cells[index];

    // Set hat function
    problem_data.set_hat_function(c, inode_local[index], 0.0);
  }
}
} // namespace dolfinx_eqlb