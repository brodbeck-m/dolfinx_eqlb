#pragma once

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
// template <typename T>
//  void equilibrate_flux_constrmin(const int ndof_patch,
//                                  std::vector<std::int32_t> cells,
//                                  std::span<T> x_flux)
template <typename T>
void equilibrate_flux_constrmin(
    const mesh::Geometry& geometry, const int type_patch, const int ndof_patch,
    std::vector<std::int32_t>& cells,
    const graph::AdjacencyList<std::int32_t>& dofmap_global,
    const graph::AdjacencyList<std::int32_t>& dofmap_elmt,
    const graph::AdjacencyList<std::int32_t>& dofmap_patch,
    const std::function<void(const std::span<T>&,
                             const std::span<const std::uint32_t>&,
                             std::int32_t, int)>& dof_transform,
    const std::function<void(const std::span<T>&,
                             const std::span<const std::uint32_t>&,
                             std::int32_t, int)>& dof_transform_to_transpose,
    fem::FEkernel<T> auto kernel_a, fem::FEkernel<T> auto kernel_l,
    std::span<const T> coeffs_a, std::span<const T> coeffs_l, int cstride_a,
    int cstride_l, std::span<const T> constants_a,
    std::span<const T> constants_l, std::span<const std::uint32_t> cell_info,
    std::vector<std::int8_t>& cell_is_evaluated,
    graph::AdjacencyList<T>& storage_stiffness_cells, std::span<T> x_flux,
    std::span<T> x_flux_dg)
{
  // Initilaize storage cell geoemtry
  const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();
  std::span<const double> x = geometry.x();

  std::vector<double> coordinate_dofs(3 * geometry.cmap().dim());

  // Get number of DOFs on element
  const int ndim0 = dofmap_global.links(0).size();

  // Initialize storage of tangent arrays (Penalty for pure Neumann problems)
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A_patch;

  if (type_patch == 0 || type_patch == 1)
  {
    const int ndof_ppatch = ndof_patch + 1;
    A_patch.resize(ndof_ppatch, ndof_ppatch);
  }
  else
  {
    A_patch.resize(ndof_patch, ndof_patch);
  }

  for (std::size_t index = 0; index < cells.size(); ++index)
  {
    // Get current cell
    std::int32_t c = cells[index];

    // Get current stiffness-matrix in storage
    std::span<T> Ae = storage_stiffness_cells.links(c);

    // Evaluate tangent arrays if not already done
    if (cell_is_evaluated[c] == 0)
    {
      // Exctract cell geometry
      auto x_dofs = x_dofmap.links(c);
      for (std::size_t j = 0; j < x_dofs.size(); ++j)
      {
        std::copy_n(std::next(x.begin(), 3 * x_dofs[j]), 3,
                    std::next(coordinate_dofs.begin(), 3 * j));
      }

      // Initialize tangent arrays
      std::fill(Ae.begin(), Ae.end(), 0);

      // Evaluate tangent arrays
      kernel_a(Ae.data(), coeffs_a.data() + cell * cstride_a,
               constants_a.data(), coordinate_dofs.data(), nullptr, nullptr);

      // DOF transformation
      dof_transform(Ae, cell_info, cell, ndim0);
      dof_transform_to_transpose(Ae, cell_info, cell, ndim0);

      // Set identifire for evaluated data
      cell_is_evaluated[c] = 1;
    }

    /* Assemble into patch system */
    // Element-local and patch-local DOFmap
    std::span<const int32_t> dofs_elmt = dofmap_elmt.links(index);
    std::span<const int32_t> dofs_patch = dofmap_patch.links(index);

    // Number of non-zero DOFs on element
    int num_nzdof_elmt = dofs_elmt.size();

    // Get number of non-zero DOFs on element
    int offset = 0;

    for (std::size_t k = 0; k < num_nzdof_elmt; ++k)
    {
      // Calculate offset
      offset = dofs_elmt[k] * ndim0;

      for (std::size_t l = 0; l < num_nzdof_elmt; ++l)
      {
        // Assemble stiffness matrix
        A_patch(dofs_patch[k], dofs_patch[l]) += Ae[offset + dofs_elmt[l]];
      }
    }
  }
}
} // namespace dolfinx_adaptivity::equilibration