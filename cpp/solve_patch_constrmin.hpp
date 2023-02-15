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
template <typename T>
void equilibrate_flux_constrmin(
    const std::size_t dim_geom, const int type_patch, const int ndof_patch,
    std::vector<std::int32_t> cells,
    const graph::AdjacencyList<std::int32_t>& dofmap, int bs,
    const std::function<void(const std::span<T>&,
                             const std::span<const std::uint32_t>&,
                             std::int32_t, int)>& dof_transform,
    const std::function<void(const std::span<T>&,
                             const std::span<const std::uint32_t>&,
                             std::int32_t, int)>& dof_transform_to_transpose,
    fem::FEkernel<T> auto kernel_a, fem::FEkernel<T> auto kernel_l,
    std::span<const T> coeffs_a, std::span<const T> coeffs_l, int cstride_a,
    int cstride_l, std::span<const T> constants_a,
    std::span<const T> constants_l, std::span<const std::uint32_t> cell_info)
{
  // // Initialize equation system (patch)

  // // Initialize storage of element constributions

  // // Loop over patch-elements
  // for (auto cell : cells)
  // {
  //   // Get cell coordinates

  //   // Set element contributions to zero

  //   // Evaluate RHS

  //   // DOF transformations

  //   // Natural boundary conditions on system matrix
  //   if (type_patch > 0)
  //   {
  //   }

  //   // Project primal-flux into DG space

  //   // Evaluate LHS

  //   // Apply boundary conditions
  //   if (type_patch > 0)
  //   {
  //   }

  //   // Assemble into patch-system
  // }

  // // Solve equation system

  // // Add flux-constribution to global storage
}
} // namespace dolfinx_adaptivity::equilibration