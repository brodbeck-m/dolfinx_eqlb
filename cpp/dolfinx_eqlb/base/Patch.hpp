// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx_eqlb/base/equilibration.hpp>
#include <dolfinx_eqlb/base/mdspan.hpp>

#include <algorithm>
#include <memory>
#include <span>
#include <vector>

using namespace dolfinx;

namespace dolfinx_eqlb::base
{

template <std::floating_point U>
class Patch
{
public:
  /// Create storage of patch data
  ///
  /// Storage is designed for the maximum patch size occurring within
  /// the current mesh.
  ///
  /// @param nnodes_proc Number of nodes on current processor
  /// @param mesh        The current mesh
  /// @param bfct_type   List with type of all boundary facets
  Patch(int nnodes_proc, std::shared_ptr<const mesh::Mesh<U>> mesh,
        base::mdspan_t<const std::int8_t, 2> bfct_type)
      : _mesh(mesh), _bfct_type(bfct_type), _dim(mesh->geometry().dim()),
        _dim_fct(mesh->geometry().dim() - 1),
        _type(bfct_type.extent(0), base::PatchType::internal),
        _nrhs(bfct_type.extent(0))
  {
    // Initialize connectivities
    _node_to_cell = _mesh->topology()->connectivity(0, _dim);
    _node_to_fct = _mesh->topology()->connectivity(0, _dim_fct);
    _fct_to_node = _mesh->topology()->connectivity(_dim_fct, 0);
    _fct_to_cell = _mesh->topology()->connectivity(_dim_fct, _dim);
    _cell_to_fct = _mesh->topology()->connectivity(_dim, _dim_fct);
    _cell_to_node = _mesh->topology()->connectivity(_dim, 0);

    // Determine maximum patch size
    set_max_patch_size(nnodes_proc);

    // Reserve storage for patch geometry
    _cells.resize(_ncells_max);
    _fcts.resize(_ncells_max + 1);
    _inodes_local.resize(_ncells_max);

    // Initialize number of facets per cell
    _fct_per_cell = _cell_to_fct->links(0).size();
  }

  /// Determine maximum patch size
  /// @param nnodes_proc Number of nodes on current processor
  void set_max_patch_size(int nnodes_proc)
  {
    // Initialization
    _ncells_max = 0;

    // Loop over all nodes
    for (std::size_t i_node = 0; i_node < nnodes_proc; ++i_node)
    {
      // Get number of cells on patch
      int n_cells = _node_to_cell->links(i_node).size();

      if (n_cells > _ncells_max)
      {
        _ncells_max = n_cells;
      }
    }
  }

  /* Setter functions */

  /* Getter functions */
  /// Number of considered RHS
  int nhrs() const { return _nrhs; }

  /// Return central node of patch
  /// @return Central node
  int node_i() { return _nodei; }

  /// Return patch type
  /// @param index Index of equilibrated flux
  /// @return Type of the patch
  base::PatchType type(int index) { return _type[index]; }

  /// Return patch types
  /// @return List of patch-types
  std::span<const base::PatchType> type() const
  {
    return std::span<const base::PatchType>(_type.data(), _nrhs);
  }

  /// Return type-relation
  /// @return Type relation of different LHS
  std::int8_t equal_patch_types() { return _equal_patches; }

  /// Check if patch is internal
  /// @return True if patch is internal
  bool is_internal() { return _type[0] == base::PatchType::internal; }

  /// Check if patch is on boundary
  /// @return True if patch is internal
  bool is_on_boundary() { return _type[0] != base::PatchType::internal; }

  /// Check if patch requires flux BCs
  /// @return True if patch requires flux BCs
  bool requires_flux_bcs(int index)
  {
    if (_type[index] == base::PatchType::bound_essnt_dual
        || _type[index] == base::PatchType::bound_mixed)
    {
      return true;
    }
    else
    {
      return false;
    }
  }

  /// Checks if flux BCs have to be applied on fact
  /// @param index The id of the RHS
  /// @param fct   The (patch-local) facet id
  /// @return True if BCs on the flux field are required
  bool requires_flux_bcs(int index, int fct_id)
  {
    if (_bfct_type(index, _fcts[fct_id]) == base::FacetType::essnt_dual)
    {
      return true;
    }
    else
    {
      return false;
    }
  }

  /// Return number of facets per cell
  /// @return Number of facets per cell
  int fcts_per_cell() { return _fct_per_cell; }

  // Return the maximal number of cells per patch
  int ncells_max() const { return _ncells_max; }

  /// Return number of cells on patch
  /// @return Number of cells on patch
  int ncells() { return _ncells; }

  /// Return number of facets on patch
  /// @return Number of facets on patch
  int nfcts() { return _nfcts; }

  /// Return processor-local cell ids on current patch
  /// @return cells
  std::span<const std::int32_t> cells() const
  {
    return std::span<const std::int32_t>(_cells.data(), _ncells);
  }

  /// Return processor-local cell id
  /// @param cell_i Patch-local cell id
  /// @return cell
  std::int32_t cell(int cell_i) { return _cells[cell_i]; }

  /// Return processor-local facet ids on current patch
  /// @return fcts
  std::span<const std::int32_t> fcts() const
  {
    return std::span<const std::int32_t>(_fcts.data(), _nfcts);
  }

  /// Return processor-local facet id
  /// @param fct_i Patch-local facet id
  /// @return facet
  std::int32_t fct(int fct_i) { return _fcts[fct_i]; }

  /// Return cell-local node ids of patch-central node
  /// @return inode_local
  std::span<const std::int8_t> inodes_local() const
  {
    return std::span<const std::int8_t>(_inodes_local.data(), _ncells);
  }

  /// Return cell-local node id of patch-central node
  /// @param cell_i Patch-local cell id
  /// @return inode_local
  std::int8_t inode_local(int cell_i) { return _inodes_local[cell_i]; }

protected:
  /// Determine local facet-id on cell
  /// @param fct_i Processor-local facet id
  /// @param cell_i Processor-local cell id
  /// @return Cell local facet id
  std::int8_t get_fctid_local(std::int32_t fct_i, std::int32_t cell_i)
  {
    // Get facets on cell
    std::span<const std::int32_t> fct_cell_i = _cell_to_fct->links(cell_i);

    return get_fctid_local(fct_i, fct_cell_i);
  }

  /// Determine local facet-id on cell
  /// @param fct_i      Processor-local facet id
  /// @param fct_cell_i Facets on cell_i
  /// @return Cell local facet id
  std::int8_t get_fctid_local(std::int32_t fct_i,
                              std::span<const std::int32_t> fct_cell_i)
  {
    // Initialize local id
    std::int8_t fct_loc = 0;

    // Get id (cell_local) of fct_i
    while (fct_cell_i[fct_loc] != fct_i && fct_loc < _fct_per_cell)
    {
      fct_loc += 1;
    }

    // std::cout << "Local facet id: " << static_cast<int>(fct_loc) <<
    // std::endl;

    // Check for face not on cell
    assert(fct_loc < _fct_per_cell);

    return fct_loc;
  }

  /// Determine local id of node on cell
  /// @param cell_i Processor-local cell id
  /// @param node_i Processor-local node id
  /// @return Cell local node id
  std::int8_t node_local(std::int32_t cell_i, std::int32_t node_i)
  {
    // Initialize cell-local node-id
    std::int8_t id_node_loc_ci = 0;

    // Get nodes on cell_i
    std::span<const std::int32_t> node_cell_i = _cell_to_node->links(cell_i);

    while (node_cell_i[id_node_loc_ci] != node_i)
    {
      id_node_loc_ci += 1;
    }

    return id_node_loc_ci;
  }

  /// Determine local id of patch-central node on cell
  /// @param cell_i Processor-local cell id
  /// @return Cell local node id
  std::int8_t nodei_local(std::int32_t cell_i)
  {
    return node_local(cell_i, _nodei);
  }

  // Maximum size of patch
  int _ncells_max;

  /* Geometry */
  // The mesh
  std::shared_ptr<const mesh::Mesh<U>> _mesh;

  // The connectivities
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> _node_to_cell,
      _node_to_fct, _fct_to_node, _fct_to_cell, _cell_to_fct, _cell_to_node;

  // Dimensions
  const int _dim, _dim_fct;

  // Counter element type
  int _fct_per_cell;

  // Types boundary facets
  base::mdspan_t<const std::int8_t, 2> _bfct_type;

  /* Patch */
  // Central node of patch
  int _nodei;

  // Number of considered RHS
  int _nrhs;

  // Type of patch
  std::vector<base::PatchType> _type;

  // Id if all patches are equal;
  std::int8_t _equal_patches;

  // Number of elements on patch
  int _ncells, _nfcts;

  // Factes/Cells on patch
  std::vector<std::int32_t> _cells, _fcts;
  std::vector<std::int8_t> _inodes_local;
};

} // namespace dolfinx_eqlb::base