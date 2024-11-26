// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <basix/finite-element.h>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx_eqlb/base/Patch.hpp>
#include <dolfinx_eqlb/base/mdspan.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <span>
#include <tuple>
#include <vector>

using namespace dolfinx;

namespace base = dolfinx_eqlb::base;

namespace dolfinx_eqlb::ev
{

class OrientedPatch
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
  OrientedPatch(int nnodes_proc, std::shared_ptr<const mesh::Mesh> mesh,
                base::mdspan_t<const std::int8_t, 2> bfct_type);

  /// Construction of a sub-DOFmap on each patch
  ///
  /// Determines type of patch and creates sorted DOFmap. Sorting of
  /// facets/elements/DOFs follows [1]. The sub-DOFmap is created for
  /// sub-problem 0. If patch- type differs between different patches, use
  /// recreate_subdofmap for sub-problem i>0.
  ///
  /// [1] Bertrand, F. et al.: https://doi.org/10.1007/s00211-023-01366-8, 2023
  ///
  /// @param node_i Processor-local id of current node
  void create_subdofmap(int node_i)
  {
    throw std::runtime_error("Patch-DOFmap not implemented!");
  }

  /// Check if reversion of patch is required
  ///
  /// Sorting convention of patch: fct_0 is located on the neumann boundary.
  /// Changing of the RHS can violate this convention. This routine checks
  /// wether this is the case.
  ///
  /// @param[in] index     Index of sub-problem
  /// @param[out] required true if reversion is required
  bool reversion_required(int index);

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
    if (_bfct_type(index, _fcts[fct_id]) == base::PatchFacetType::essnt_dual)
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

  /// Determin cell, cell-local facet id and next facet (sorted after [1])
  ///
  /// [1] Bertrand, F. et al.: https://doi.org/10.1007/s00211-023-01366-8, 2023
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

  /// Determine local id of node on cell
  /// @param cell_i Processor-local cell id
  /// @param node_i Processor-local node id
  /// @return Cell local node id
  std::int8_t node_local(std::int32_t cell_i, std::int32_t node_i);

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
  std::vector<std::int32_t> _cells, _fcts, _fcts_sorted_data;
  std::span<std::int32_t> _fcts_sorted;
  std::vector<std::int8_t> _inodes_local;
};

class Patch : public OrientedPatch
{
public:
  /// Initialization
  ///
  /// Storage is designed for the maximum patch size occurring within
  /// the current mesh.
  ///
  /// @param nnodes_proc Number of nodes on current processor
  /// @param mesh        The current mesh
  /// @param bfct_type   List with type of all boundary facets
  Patch(int nnodes_proc, std::shared_ptr<const mesh::Mesh> mesh,
        base::mdspan_t<const std::int8_t, 2> bfct_type,
        const std::shared_ptr<const fem::FunctionSpace> function_space,
        const std::shared_ptr<const fem::FunctionSpace> function_space_fluxhdiv,
        const basix::FiniteElement& basix_element_flux);

  /* Overload functions from base-class */
  void create_subdofmap(int node_i);

  /* Setter functions */

  /* Getter functions */
  /// @return Number of mesh nodes on patch
  int npnts() const { return _nfcts + 1; }

  /// @return Number of DOFs on element
  int ndofs_elmt() { return _ndof_elmt; }

  /// @return Number of non-zero DOFs on element
  int ndofs_elmt_nz() { return _ndof_elmt_nz; }

  /// @return Number of flux-DOFs on element
  int ndofs_flux() { return _ndof_flux; }

  /// @return Number of non-zero flux-DOFs on element
  int ndofs_flux_nz() { return _ndof_flux_nz; }

  /// @return Number of non-zero flux-DOFs on patch
  int ndofs_flux_patch_nz() { return _ndof_fluxhdiv; }

  /// @return Number of flux-DOFs per facet
  int ndofs_flux_fct() { return _ndof_flux_fct; }

  /// @return Number of flux-DOFs per cell
  int ndofs_flux_cell() { return _ndof_flux_cell; }

  /// @return Number of constrained-DOFs on element
  int ndofs_cons() { return _ndof_cons; }

  /// @return Number of non-zero DOFs on patch
  int ndofs_patch() { return _ndof_patch_nz; }

  /// Extract global DOFs of cell
  /// @param cell_i Patch-local cell-id
  /// @return Global DOFs of cell:_i (zero DOFs excluded)
  std::span<std::int32_t> dofs_global(int cell_i)
  {
    return std::span<std::int32_t>(
        _dofsnz_global.data() + _offset_dofmap[cell_i],
        _offset_dofmap[cell_i + 1] - _offset_dofmap[cell_i]);
  }

  /// Extract global DOFs of cell (const. version)
  /// @param cell_i Patch-local cell-id
  /// @return Global DOFs of cell:_i (zero DOFs excluded)
  std::span<const std::int32_t> dofs_global(int cell_i) const
  {
    return std::span<const std::int32_t>(
        _dofsnz_global.data() + _offset_dofmap[cell_i],
        _offset_dofmap[cell_i + 1] - _offset_dofmap[cell_i]);
  }

  /// Extract DOFs (collapsed H(div) flux) on patch
  /// @return Global flux-DOFs
  std::span<std::int32_t> dofs_fluxhdiv_global()
  {
    return std::span<std::int32_t>(_list_dofsnz_global_fluxhdiv.data(),
                                   _ndof_fluxhdiv);
  }

  /// Extract DOFs (collapsed H(div) flux) on patch (const. version)
  /// @return Global flux-DOFs
  std::span<const std::int32_t> dofs_fluxhdiv_global() const
  {
    return std::span<const std::int32_t>(_list_dofsnz_global_fluxhdiv.data(),
                                         _ndof_fluxhdiv);
  }

  /// Extract patch-local DOFs (collapsed H(div) flux)
  /// @param cell_i Patch-local cell-id
  /// @return Patch-local flux-DOFs
  std::span<std::int32_t> dofs_fluxhdiv_patch()
  {
    return std::span<std::int32_t>(_list_dofsnz_patch_fluxhdiv.data(),
                                   _ndof_fluxhdiv);
  }

  /// Extract patch-local DOFs (collapsed H(div) flux) (const. version)
  /// @param cell_i Patch-local cell-id
  /// @return Patch-local flux-DOFs
  std::span<const std::int32_t> dofs_fluxhdiv_patch() const
  {
    return std::span<const std::int32_t>(_list_dofsnz_patch_fluxhdiv.data(),
                                         _ndof_fluxhdiv);
  }

  /// Extract patch-local DOFs of cell
  /// @param cell_i Patch-local cell-id
  /// @return Patch-local DOFs of cell:_i (zero DOFs excluded)
  std::span<std::int32_t> dofs_patch(int cell_i)
  {
    return std::span<std::int32_t>(
        _dofsnz_patch.data() + _offset_dofmap[cell_i],
        _offset_dofmap[cell_i + 1] - _offset_dofmap[cell_i]);
  }

  /// Extract patch-local DOFs of cell (const. version)
  /// @param cell_i Patch-local cell-id
  /// @return Patch-local DOFs of cell:_i (zero DOFs excluded)
  std::span<const std::int32_t> dofs_patch(int cell_i) const
  {
    return std::span<const std::int32_t>(
        _dofsnz_patch.data() + _offset_dofmap[cell_i],
        _offset_dofmap[cell_i + 1] - _offset_dofmap[cell_i]);
  }

  /// Extract cell-local DOFs of cell
  /// @param cell_i Patch-local cell-id
  /// @return Cell-local DOFs of cell:_i (zero DOFs excluded)
  std::span<std::int32_t> dofs_elmt(int cell_i)
  {
    return std::span<std::int32_t>(_dofsnz_elmt.data() + _offset_dofmap[cell_i],
                                   _offset_dofmap[cell_i + 1]
                                       - _offset_dofmap[cell_i]);
  }

  /// Extract cell-local DOFs of cell (const. version)
  /// @param cell_i Patch-local cell-id
  /// @return Cell-local DOFs of cell:_i (zero DOFs excluded)
  std::span<const std::int32_t> dofs_elmt(int cell_i) const
  {
    return std::span<const std::int32_t>(
        _dofsnz_elmt.data() + _offset_dofmap[cell_i],
        _offset_dofmap[cell_i + 1] - _offset_dofmap[cell_i]);
  }

protected:
  /*Function (sub-) space*/
  // The function space
  const std::shared_ptr<const fem::FunctionSpace> _function_space;
  const std::shared_ptr<const fem::FunctionSpace> _function_space_fluxhdiv;

  // Connectivity between entities and cell local DOFs
  const std::vector<std::vector<std::vector<int>>>& _entity_dofs_flux;

  // Storage sub-dofmap
  // Layout: fDOFs -> flux DOFs, cDOFs -> constrained DOFs
  //         [(fDOF_E0, fDOF_T1, cDOF_T1), ..., (fDOF_Eam1, fDOF_Ta, cDOF_Ta)]
  std::vector<std::int32_t> _dofsnz_elmt, _dofsnz_patch, _dofsnz_global,
      _offset_dofmap;

  // Storage DOFs H(div) flux (per patch)
  // Layout: [(fDOF_E0, fDOF_T1), ..., (fDOF_Eam1, fDOF_Ta), (fDOFs_Ea)]
  std::vector<std::int32_t> _list_dofsnz_patch_fluxhdiv,
      _list_dofsnz_global_fluxhdiv;

  // Number of DOFs on element (element definition)
  int _ndof_elmt;

  // Number of DOFs on sub-elements (element definition)
  int _ndof_flux_fct, _ndof_flux_cell, _ndof_cons_cell, _ndof_flux, _ndof_cons;

  // Number on non-zero DOFs on element (on patch)
  int _ndof_elmt_nz, _ndof_flux_nz, _ndof_patch_nz, _ndof_fluxhdiv;
};

} // namespace dolfinx_eqlb::ev