// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/graph/AdjacencyList.h>

#include "ProblemData.hpp"
// #include "mdspan.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <dolfinx/common/types.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/graph/AdjacencyList.h>

#include <dolfinx_eqlb/base/ProblemData.hpp>
#include <dolfinx_eqlb/base/equilibration.hpp>
#include <dolfinx_eqlb/base/mdspan.hpp>

#include <algorithm>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <span>
#include <utility>
#include <vector>

using namespace dolfinx;

namespace dolfinx_eqlb::ev
{

/// Get the id of the patch with the maximum number of cells
///
/// @param[in] nodes_on_proc The number of nodes on the processor
/// @param[in] node_to_cell  The connectivity between nodes and cells
/// @return The (local) node id in the center of the patch with the maximum
/// number of cells
std::int32_t max_patch_size(std::int32_t nodes_on_proc,
                            std::shared_ptr<const graph::AdjacencyList<std::int32_t>> node_to_cell)
{
  std::int32_t patch_id = 0, ncells_max = 0;

  // Loop over all patches
  for (std::int32_t i = 0; i < nodes_on_proc; ++i)
  {
    // Get number of cells on patch
    std::int32_t n_cells = node_to_cell->links(i).size();

    if (n_cells > ncells_max)
    {
      ncells_max = n_cells;
      patch_id = i;
    }
  }

  return patch_id;
}

/// Get the number of DOFs per entity of a given function space
///
/// @param[in] fspace The function space
/// @return The number of DOFs per entity
template <std::floating_point U>
std::vector<int> ndofs_per_entity(std::shared_ptr<const fem::FunctionSpace<U>> fspace)
{
  // Data storage
  std::vector<int> dofs_per_entity(fspace->mesh()->geometry().dim() + 1, 0);

  //   DOF ids per entity
  const std::vector<std::vector<std::vector<int>>>& entity_dofs = fspace->element()->entity_dofs();

  // Get number of DOFs per entity
  for (std::size_t i = 0; i < entity_dofs.size(); ++i)
  {
    dofs_per_entity[i] = entity_dofs[i][0].size();
  }

  return std::move(dofs_per_entity);
}

/// Get a compacting remapping of the local DOF map
/// The returned remap[0] will hold the index corresponding to the smallest value in dofs_map, remap[1] the next
/// smallest, and so on.
///
/// @param[in] dofs_map DOF map of the function space
// @return A compacting remapping of the local DOF map

std::vector<int> compact_dof_map(const std::vector<int>& dofs_map)
{
  if (dofs_map.empty())
    return {};

  // Get the number of DOFs
  std::int32_t ndofs = *std::max_element(dofs_map.begin(), dofs_map.end()) + 1;

  // Create a vector to store the remapping
  std::vector<int> remap(ndofs);
  std::iota(remap.begin(), remap.end(), 0);

  // Insertion sort the remap vector using values from dofs_map as criteria
  for (size_t index = 1; index < remap.size(); ++index)
  {
    size_t index2 = index;
    while (index2 > 0 && dofs_map[remap[index2 - 1]] > dofs_map[remap[index2]])
    {
      std::swap(remap[index2 - 1], remap[index2]);
      --index2;
    }
  }

  return remap;
}
} // namespace dolfinx_eqlb::ev
