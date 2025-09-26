// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Patch.hpp"
#include "StorageStiffness.hpp"
#include "eigen3/Eigen/Dense"

#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx_eqlb/base/Patch.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <iterator>
#include <span>
#include <vector>

using namespace dolfinx;

namespace base = dolfinx_eqlb::base;

namespace dolfinx_eqlb::ev
{
/// Apply lifting of non-homogenous boundary conditions
///
/// A lifting routine for non-homogeneous boundary conditions on patches. As
/// patch problems are always linear and therefore solvable within one
/// newton-setp a correction of the boundary values by the pervious solution is
/// unnecessary.
///
/// @param Ae           The element stiffness matrix
/// @param Le           The element load vector
/// @param bmarkers     The boundary markers
/// @param bvalues      The boundary values
/// @param dofs_elmt    The element-local DOFmap
/// @param dofs_patch   The patch-local DOFmap
/// @param dofs_global  The global DOFmap
/// @param type_patch   The patch type
/// @param ndof_elmt_nz The number of non-zero DOFs on element
/// @param ndof_elmt    The number of DOFs on element
template <dolfinx::scalar T>
void apply_lifting(std::span<T> Ae, std::vector<T>& Le,
                   std::span<const std::int8_t> bmarkers,
                   std::span<const T> bvalues,
                   std::span<const int32_t> dofs_elmt,
                   std::span<const int32_t> dofs_patch,
                   std::span<const int32_t> dofs_global,
                   const base::PatchType type_patch, const int ndof_elmt_nz,
                   const int ndof_elmt)
{
  if (type_patch == base::PatchType::bound_essnt_dual
      || type_patch == base::PatchType::bound_mixed)
  {
    for (std::size_t k = 0; k < ndof_elmt_nz; ++k)
    {
      std::int32_t dof_global_k = dofs_global[k];

      if (bmarkers[dof_global_k] == 0)
      {
        // Calculate offset
        int offset = dofs_elmt[k] * ndof_elmt;

        for (std::size_t l = 0; l < ndof_elmt_nz; ++l)
        {
          std::int32_t dof_global_l = dofs_global[l];

          if (bmarkers[dof_global_l] != 0)
          {
            Le[dofs_elmt[k]]
                -= Ae[offset + dofs_elmt[l]] * bvalues[dof_global_l];
          }
        }
      }
    }
  }
}

/// Patch-wise assembly of equation-system
///
/// Assembles the patch-wise equation system for the constrained minimization,
/// determining the equilibrated flux for some general RHS. Depending of the
/// template parameter either the entire system or only the load vector is
/// assembled. The elemental stiffness matrices are computed only once. If this
/// is already done, values are extracted from storage and no recalculations is
/// required.
///
/// @tparam T              The scalar type
/// @tparam U              The geometry type
/// @tparam asmbl_systmtrx True if entire system is assembled
/// @param A_patch                    The patch stiffness matrix
/// @param L_patch                    The patch load vector
/// @param cells                      The cells of the patch
/// @param coordinate_dofs            The coordinate DOFs of each cell
/// @param cstride_geom               The stride of the coordinate DOFs
/// @param patch                      The patch
/// @param P0                         The DOF transformation function
/// @param P0T The DOF transformation function
/// @param cell_info                  The cell transformation information
/// @param kernel_a                   The kernel for the bilinear form
/// @param kernel_lpen                The kernel for the penalization terms
/// @param kernel_l                   The kernel for the linear form
/// @param constants_l                The constants of the linear form
/// @param coefficients_l             The coefficients of the linear form
/// @param cstride_l                  The stride of the coefficients
/// @param bmarkers                   The boundary markers
/// @param bvalues                    The boundary values
/// @param storage_stiffness          The storage for the stiffness matrices
/// @param index_rhs                  The index of teh currently assembled RHS
template <dolfinx::scalar T, std::floating_point U, bool asmbl_systmtrx = true>
void assemble_tangents(
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& A_patch,
    Eigen::Matrix<T, Eigen::Dynamic, 1>& L_patch,
    std::span<const std::int32_t> cells, std::vector<U>& coordinate_dofs,
    const int cstride_geom, Patch<U>& patch, fem::DofTransformKernel<T> auto P0,
    fem::DofTransformKernel<T> auto P0T,
    std::span<const std::uint32_t> cell_info, fem::FEkernel<T> auto kernel_a,
    fem::FEkernel<T> auto kernel_lpen, fem::FEkernel<T> auto kernel_l,
    std::span<const T> constants_l, std::span<T> coefficients_l,
    const int cstride_l, std::span<const std::int8_t> bmarkers,
    std::span<const T> bvalues, StorageStiffness<T>& storage_stiffness,
    int index_rhs)
{
  // Counters
  const int ndim0 = patch.ndofs_elmt();
  const int ndof_elmt_nz = patch.ndofs_elmt_nz();
  const int ndof_patch = patch.ndofs_patch();
  const int ndof_ppatch = ndof_patch + 1;

  const int ncells = cells.size();

  // Initialize storage Load-vector
  std::vector<T> Le(ndim0, 0);
  std::span<T> _Le(Le);

  for (std::size_t index = 0; index < ncells; ++index)
  {
    // Get current cell
    std::int32_t c = cells[index];

    // Get coordinates of current element
    std::span<U> coordinate_dofs_e(
        coordinate_dofs.data() + index * cstride_geom, cstride_geom);

    /* Evaluate stiffness matrix */
    std::span<T> Ae, Pe;
    if constexpr (asmbl_systmtrx)
    {
      // Get current stiffness-matrix in storage
      Ae = storage_stiffness.stiffness_elmt(c);
      Pe = storage_stiffness.penalty_elmt(c);

      // Evaluate bilinear form
      if (storage_stiffness.evaluation_status(c) == 0)
      {
        // Initialize bilinear form
        std::fill(Ae.begin(), Ae.end(), 0);
        std::fill(Pe.begin(), Pe.end(), 0);

        // Evaluate bilinar form
        kernel_a(Ae.data(), nullptr, nullptr, coordinate_dofs_e.data(), nullptr,
                 nullptr);

        // Evaluate penalty terms
        kernel_lpen(Pe.data(), nullptr, nullptr, coordinate_dofs_e.data(),
                    nullptr, nullptr);

        // DOF transformation
        P0(Ae, cell_info, c, ndim0);
        P0T(Ae, cell_info, c, ndim0);

        // Set identifire for evaluated data
        storage_stiffness.mark_cell_evaluated(c);
      }
    }

    // Evaluate linear form
    std::fill(Le.begin(), Le.end(), 0);
    kernel_l(Le.data(), coefficients_l.data() + c * cstride_l,
             constants_l.data(), coordinate_dofs_e.data(), nullptr, nullptr);

    P0(_Le, cell_info, c, 1);

    /* Assemble into patch system */
    // Get patch type
    const base::PatchType type_patch = patch.type(index_rhs);

    // Element-local and patch-local DOFmap
    std::span<const int32_t> dofs_elmt = patch.dofs_elmt(index);
    std::span<const int32_t> dofs_patch = patch.dofs_patch(index);
    std::span<const int32_t> dofs_global = patch.dofs_global(index);

    if (patch.requires_flux_bcs(index_rhs))
    {
      // Extract stiffness matrix
      if constexpr (!asmbl_systmtrx)
      {
        Ae = storage_stiffness.stiffness_elmt(c);
      }

      // Apply lifting
      apply_lifting(Ae, Le, bmarkers, bvalues, dofs_elmt, dofs_patch,
                    dofs_global, type_patch, ndof_elmt_nz, ndim0);

      // Assemble tangents
      for (std::size_t k = 0; k < ndof_elmt_nz; ++k)
      {
        // Check for boundary condition
        if (bmarkers[dofs_global[k]] != 0)
        {
          // Set main-diagonal of stiffness matrix
          if constexpr (asmbl_systmtrx)
          {
            A_patch(dofs_patch[k], dofs_patch[k]) = 1;
          }

          // Set boundary value
          L_patch(dofs_patch[k]) = bvalues[dofs_global[k]];
        }
        else
        {
          // Calculate offset
          int offset = dofs_elmt[k] * ndim0;

          // Assemble load vector
          L_patch(dofs_patch[k]) += Le[dofs_elmt[k]];

          if constexpr (asmbl_systmtrx)
          {
            for (std::size_t l = 0; l < ndof_elmt_nz; ++l)
            {
              // Assemble stiffness matrix
              if (bmarkers[dofs_global[l]] == 0)
              {
                A_patch(dofs_patch[k], dofs_patch[l])
                    += Ae[offset + dofs_elmt[l]];
              }
            }
          }
        }
      }
    }
    else
    {
      for (std::size_t k = 0; k < ndof_elmt_nz; ++k)
      {
        // Calculate offset
        int offset = dofs_elmt[k] * ndim0;

        // Assemble load vector
        L_patch(dofs_patch[k]) += Le[dofs_elmt[k]];

        // Assemble stiffness matrix
        if constexpr (asmbl_systmtrx)
        {
          for (std::size_t l = 0; l < ndof_elmt_nz; ++l)
          {
            A_patch(dofs_patch[k], dofs_patch[l]) += Ae[offset + dofs_elmt[l]];
          }
        }
      }
    }

    // Add penalty terms
    if constexpr (asmbl_systmtrx)
    {
      if (type_patch == base::PatchType::internal
          || type_patch == base::PatchType::bound_essnt_dual)
      {
        // Required counters
        const int ndofs_cons = patch.ndofs_cons();
        const int offset = patch.ndofs_flux_nz();

        // Loop over DOFs
        for (std::size_t k = 0; k < ndofs_cons; ++k)
        {
          // Add K_ql
          A_patch(dofs_patch[offset + k], ndof_ppatch - 1) += Pe[k];

          // Add K_lq
          A_patch(ndof_ppatch - 1, dofs_patch[offset + k]) += Pe[k];
        }
      }
      else
      {
        // Set penalty to zero
        A_patch(ndof_ppatch - 1, ndof_ppatch - 1) = 1.0;
      }
    }
  }
}

} // namespace dolfinx_eqlb::ev