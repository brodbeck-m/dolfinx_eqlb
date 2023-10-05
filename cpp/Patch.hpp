#pragma once

#include "utils.hpp"

#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>

#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <span>
#include <vector>

using namespace dolfinx;

namespace dolfinx_eqlb
{

enum PatchFacetType : std::int8_t
{
  internal = 0,
  essnt_primal = 1,
  essnt_dual = 2
};

enum class PatchType : std::int8_t
{
  internal = 0,
  bound_essnt_dual = 1,
  bound_essnt_primal = 2,
  bound_mixed = 3
};

class Patch
{
public:
  /// Create storage of patch data
  ///
  /// Storage is designed for the maximum patch size occurring within
  /// the current mesh.
  ///
  /// @param nnodes_proc Numbe rof nodes on current processor
  /// @param mesh        The current mesh
  /// @param bfct_type   List with type of all boundary facets
  Patch(int nnodes_proc, std::shared_ptr<const mesh::Mesh> mesh,
        mdspan_t<const std::int8_t, 2> bfct_type);

  /// Construction of a sub-DOFmap on each patch
  ///
  /// Determines type of patch and creates sorted DOFmap. Sorting of
  /// facets/elements/DOFs follows [1,2]. The sub-DOFmap is created for
  /// sub-problem 0. If patch- type differs between different patches, use
  /// recreate_subdofmap for sub-problem i>0.
  ///
  /// [1] Moldenhauer, M.: Stress reconstructionand a-posteriori error
  ///     estimationfor elasticity (PhdThesis)
  /// [2] Bertrand, F.; Carstensen, C.; Gräßle, B. & Tran, N. T.:
  ///     Stabilization-free HHO a posteriori error control, 2022
  ///
  /// @param node_i Processor-local id of current node
  void create_subdofmap(int node_i)
  {
    throw std::runtime_error("Patch-DOFmap not implemented!");
  }

  /// Recreate sub-DOFmap on each patch
  ///
  /// Change of RHS can change boundary conditions. This may require a
  /// reversed patch-orientation.
  ///
  /// @param[in] index
  /// @param[out] was_recreated true if patch-DOFmap was recreated
  bool recreate_subdofmap(int index);

  /// Implement reversion of patch orientation
  void reverse_patch_orientation()
  {
    throw std::runtime_error("Reversion of patch-DOFmap not implemented!");
  }

  /// Determine maximum patch size
  /// @param nnodes_proc Number of nodes on current processor
  void set_max_patch_size(int nnodes_proc);

  /* Setter functions */
  void set_fcts_sorted(std::span<const std::int32_t> list_fcts)
  {
    assert(_nfcts);

    // Copy data into modifiable structure
    std::copy(list_fcts.begin(), list_fcts.end(), _fcts_sorted_data.begin());

    // Set span onto relevant part of vector and sort
    _fcts_sorted = std::span<std::int32_t>(_fcts_sorted_data.data(), _nfcts);
    std::sort(_fcts_sorted.begin(), _fcts_sorted.end());
  }

  /* Getter functions */
  /// Return central node of patch
  /// @return Central node
  int node_i() { return _nodei; }

  /// Return patch type
  /// @param index Index of equilibrated flux
  /// @return Type of the patch
  PatchType type(int index) { return _type[index]; }

  /// Return type-relation
  /// @return Type relation of different LHS
  std::int8_t equal_patch_types() { return _equal_patches; }

  /// Check if patch is internal
  /// @return True if patch is internal
  bool is_internal() { return _type[0] == PatchType::internal; }

  /// Check if patch is on boundary
  /// @return True if patch is internal
  bool is_on_boundary() { return _type[0] != PatchType::internal; }

  /// Check if patch requires flux BCs
  /// @return True if patch requires flux BCs
  bool requires_flux_bcs(int index)
  {
    if (_type[index] == PatchType::bound_essnt_dual
        || _type[index] == PatchType::bound_mixed)
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

  /// Return number of cells on patch
  /// @return Number of cells on patch
  int ncells() { return _ncells; }

  /// Return number of facets on patch
  /// @return Number of facets on patch
  int nfcts() { return _nfcts; }

  /// Return processor-local cell ids on current patch
  /// @return cells
  std::span<std::int32_t> cells()
  {
    return std::span<std::int32_t>(_cells.data(), _ncells);
  }

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
  std::span<std::int32_t> fcts()
  {
    return std::span<std::int32_t>(_fcts.data(), _nfcts);
  }

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

  /// Return cell-local node id of patch-central node
  /// @return inode_local
  std::span<std::int8_t> inodes_local()
  {
    return std::span<std::int8_t>(_inodes_local.data(), _ncells);
  }

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
  /// Initializes patch
  ///
  /// Sets patch type and creates sorted list of patch-facets.
  ///
  /// Patch types:
  ///    0 -> internal
  ///    1 -> bound_esnt_flux
  ///    2 -> bound_esnt_prime
  ///    3 -> bound_mixed
  ///
  /// @param node_i Processor local id of patch-central node
  /// @return       First facte on patch,
  /// @return       Length of loop over factes
  std::pair<std::int32_t, std::int32_t> initialize_patch(int node_i);

  /// Determin cell, cell-local facet id and next facet
  ///
  /// Patch are sorted after [1,2]
  /// [1] Moldenhauer, M.: Stress reconstructionand a-posteriori error
  ///     estimationfor elasticity (PhdThesis)
  /// [2] Bertrand, F.; Carstensen, C.; Gräßle, B. & Tran, N. T.:
  ///     Stabilization-free HHO a posteriori error control, 2022
  ///
  /// @param id_l     Id of used LHS
  /// @param c_fct    Counter within loop over all facets of patch
  /// @param fct_i    Processor-local id of facet
  /// @param cell_in  Processor-local id of last cell on patch
  /// @return         Cell-local id of fct_i in cell_i,
  /// @return         Cell-local id of fct_i in cell_(i-1),
  /// @return         Next facet on patch
  std::tuple<std::int8_t, std::int8_t, std::int8_t, std::int32_t>
  fcti_to_celli(int id_l, int c_fct, std::int32_t fct_i, std::int32_t cell_i);

  /// Determine local facet-id on cell
  /// @param fct_i Processor-local facet id
  /// @param cell_i Processor-local cell id
  /// @return Cell local facet id
  std::int8_t get_fctid_local(std::int32_t fct_i, std::int32_t cell_i);

  /// Determine local facet-id on cell
  /// @param fct_i      Processor-local facet id
  /// @param fct_cell_i Facets on cell_i
  /// @return Cell local facet id
  std::int8_t get_fctid_local(std::int32_t fct_i,
                              std::span<const std::int32_t> fct_cell_i);

  /// Determine local id of patch-central node on cell
  /// @param cell_i Processor-local cell id
  /// @return Cell local node id
  std::int8_t nodei_local(std::int32_t cell_i);

  /// Determine the next facet on a patch formed by triangular elements
  ///
  /// Triangle has three facets. Determine the two facets not already used.
  /// Check if smaller or larger facet is outside the range spanned by the
  /// facets of the patch. If not, conduct full search in the list of patch
  /// facets.
  ///
  /// @param cell_i     Processor local id of current cell
  /// @param fct_cell_i List of factes (processor loacl in) of current cell
  /// @param id_fct_loc Loacl facet id on current cell
  /// @return           Next facet
  std::int32_t next_facet_triangle(std::int32_t cell_i,
                                   std::span<const std::int32_t> fct_cell_i,
                                   std::int8_t id_fct_loc);

  // Maximum size of patch
  int _ncells_max;

  /* Geometry */
  // The mesh
  std::shared_ptr<const mesh::Mesh> _mesh;

  // The connectivities
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> _node_to_cell,
      _node_to_fct, _fct_to_node, _fct_to_cell, _cell_to_fct, _cell_to_node;

  // Dimensions
  const int _dim, _dim_fct;

  // Counter element type
  int _fct_per_cell;

  // Types boundary facets
  mdspan_t<const std::int8_t, 2> _bfct_type;

  /* Patch */
  // Central node of patch
  int _nodei;

  // Type of patch
  std::vector<PatchType> _type;
  int _npatches;

  // Id if all patches are equal;
  std::int8_t _equal_patches;

  // Number of elements on patch
  int _ncells, _nfcts;

  // Factes/Cells on patch
  std::vector<std::int32_t> _cells, _fcts, _fcts_sorted_data;
  std::span<std::int32_t> _fcts_sorted;
  std::vector<std::int8_t> _inodes_local;
};
} // namespace dolfinx_eqlb