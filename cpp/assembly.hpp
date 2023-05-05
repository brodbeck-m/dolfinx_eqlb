#pragma once

#include "PatchFluxEV.hpp"
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

using namespace dolfinx;

namespace dolfinx_adaptivity::equilibration::impl
{
template <typename T>
void apply_lifting(std::span<T> Ae, std::vector<T>& Le,
                   std::span<const std::int8_t> bmarkers,
                   std::span<const T> bvalues,
                   std::span<const int32_t> dofs_elmt,
                   std::span<const int32_t> dofs_patch,
                   std::span<const int32_t> dofs_global, const int type_patch,
                   const int ndof_elmt_nz, const int ndof_elmt)
{
  if (type_patch == 1 || type_patch == 3)
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

template <typename T>
void assemble_tangents(
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& A_patch,
    Eigen::Matrix<T, Eigen::Dynamic, 1>& L_patch,
    std::span<const std::int32_t> cells,
    std::vector<fem::impl::scalar_value_type_t<T>>& coordinate_dofs,
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
    std::span<const T> bvalues, StorageStiffness<T>& storage_stiffness,
    int index_lhs)
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

    // Get current stiffness-matrix in storage
    std::span<T> Ae = storage_stiffness.stiffness_elmt(c);
    std::span<T> Pe = storage_stiffness.penalty_elmt(c);

    // Get coordinates of current element
    std::span<fem::impl::scalar_value_type_t<T>> coordinate_dofs_e(
        coordinate_dofs.data() + index * cstride_geom, cstride_geom);

    /* Evaluate tangent arrays if not already done */
    // Evaluate bilinar form
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
      dof_transform(Ae, cell_info, c, ndim0);
      dof_transform_to_transpose(Ae, cell_info, c, ndim0);

      // Set identifire for evaluated data
      storage_stiffness.mark_cell_evaluated(c);
    }

    // Evaluate linear form
    std::fill(Le.begin(), Le.end(), 0);
    kernel_l(Le.data(), coefficients_l.data() + c * cstride_l,
             constants_l.data(), coordinate_dofs_e.data(), nullptr, nullptr);

    dof_transform(_Le, cell_info, c, 1);

    /* Assemble into patch system */
    // Get patch type
    const int type_patch = patch.type(index_lhs);

    // Element-local and patch-local DOFmap
    std::span<const int32_t> dofs_elmt = patch.dofs_elmt(index);
    std::span<const int32_t> dofs_patch = patch.dofs_patch(index);
    std::span<const int32_t> dofs_global = patch.dofs_global(index);

    if (type_patch == 1 || type_patch == 3)
    {
      // Apply lifting
      apply_lifting(Ae, Le, bmarkers, bvalues, dofs_elmt, dofs_patch,
                    dofs_global, type_patch, ndof_elmt_nz, ndim0);

      // Assemble tangents
      for (std::size_t k = 0; k < ndof_elmt_nz; ++k)
      {
        // Check for boundary condition
        if (bmarkers[dofs_global[k]] != 0)
        {
          // Set main-digonal of stiffness matrix
          A_patch(dofs_patch[k], dofs_patch[k]) = 1;

          // Set boundary value
          L_patch(dofs_patch[k]) = bvalues[dofs_global[k]];
        }
        else
        {
          // Calculate offset
          int offset = dofs_elmt[k] * ndim0;

          // Assemble load vector
          L_patch(dofs_patch[k]) += Le[dofs_elmt[k]];

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
    else
    {
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
    }

    // Add penalyt terms
    if (type_patch < 2)
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

template <typename T, int id_flux_order = -1>
void assemble_tangents(
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& A_patch,
    Eigen::Matrix<T, Eigen::Dynamic, 1>& L_patch,
    std::span<const std::int32_t> cells, PatchFluxCstm<T, id_flux_order>& patch,
    KernelData& kernel_data, std::span<T> coefficients,
    std::vector<double> storage_detJ, const int type_patch)
{
  throw std::runtime_error("assembly_tangents: Not implemented!");
}

template <typename T>
void assemble_vector(
    Eigen::Matrix<T, Eigen::Dynamic, 1>& L_patch,
    std::span<const std::int32_t> cells,
    std::vector<fem::impl::scalar_value_type_t<T>>& coordinate_dofs,
    const int cstride_geom, PatchFluxEV& patch,
    const std::function<void(const std::span<T>&,
                             const std::span<const std::uint32_t>&,
                             std::int32_t, int)>& dof_transform,
    std::span<const std::uint32_t> cell_info, fem::FEkernel<T> auto kernel_l,
    std::span<const T> constants_l, std::span<T> coefficients_l,
    const int cstride_l, std::span<const std::int8_t> bmarkers,
    std::span<const T> bvalues, StorageStiffness<T>& storage_stiffness,
    int index_lhs)
{
  throw std::runtime_error("assembly_vector: Not implemented!");
}

template <typename T, int id_flux_order = -1>
void assemble_vector()
{
  throw std::runtime_error("assembly_vector: Not implemented!");
}

} // namespace dolfinx_adaptivity::equilibration::impl