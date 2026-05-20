// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/graph/AdjacencyList.h>

#include <dolfinx_eqlb/base/ProblemData.hpp>
#include <dolfinx_eqlb/base/equilibration.hpp>
#include <dolfinx_eqlb/base/mdspan.hpp>

#include <memory>

using namespace dolfinx;

namespace dolfinx_eqlb::ev
{

/// Get the maximum number of cells per patch
///
/// @param[in] nodes_on_proc The number of nodes on the processor
/// @param[in] node_to_cell  The connectivity between nodes and cells
/// @return The maximum number of cells per patch
int max_patch_size(
    std::int32_t nodes_on_proc,
    std::shared_ptr<const graph::AdjacencyList<std::int32_t>> node_to_cell)
{
  int ncells_max = 0;

  // Loop over all patches
  for (std::int32_t i = 0; i < nodes_on_proc; ++i)
  {
    // Get number of cells on patch
    int n_cells = node_to_cell->links(i).size();

    if (n_cells > ncells_max)
    {
      ncells_max = n_cells;
    }
  }

  return ncells_max;
}

} // namespace dolfinx_eqlb::ev