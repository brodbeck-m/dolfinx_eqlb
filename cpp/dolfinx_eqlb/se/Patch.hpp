// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <basix/finite-element.h>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx_eqlb/base/Patch.hpp>
#include <dolfinx_eqlb/base/mdspan.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <span>
#include <tuple>
#include <vector>

using namespace dolfinx;

namespace base = dolfinx_eqlb::base;

namespace dolfinx_eqlb::se
{

class OrientedPatch
{
public:
  /// Create storage of patch data
  ///
  /// Storage is designed for the maximum patch size occurring within
  /// the current mesh.
  ///
  /// @param mesh         The current mesh
  /// @param bfct_type    List with type of all boundary facets
  /// @param pnts_on_bndr Markers for all mesh nodes on stress boundary
  /// @param ncells_min   Minimum number of cells patches (below: Error)
  /// @param ncells_crit  Critical number of cells on a boundary patch
  ///                     (modification required)
  OrientedPatch(std::shared_ptr<const mesh::Mesh> mesh,
                base::mdspan_t<const std::int8_t, 2> bfct_type,
                std::span<const std::int8_t> pnts_on_essntbndr,
                const int ncells_min, const int ncells_crit);

  /// Group patches such that minimisation is possible
  ///
  /// Routine works only on boundary patches! It returns a list of adjacent
  /// patches around node_i. Thereby the patch is connected with one internal
  /// and adjacent boundary patches (type PatchType::bound_essnt_dual) which
  /// have ncells_crit cells.
  ///
  /// @param node_i           Processor-local id of patch-central node
  /// @param pnt_on_bndr      Markers for all mesh nodes on essential boundary
  ///                         (stresses)
  /// @param initial_length   Initial length of the output vector
  /// @param ncells_crit      Critical number of cells on patches
  /// @return                 The central nodes of critical, adjacent patches
  std::vector<std::int32_t>
  group_boundary_patches(const std::int32_t node_i,
                         std::span<const std::int8_t> pnt_on_bndr,
                         const int ncells_crit) const;

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
  void create_subdofmap(const int node_i)
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
  bool reversion_required(int index) const;

  /// Estimate patchs Korn constant
  ///
  /// For 2D star-shaped domains: Formula [1].
  ///
  /// [1] Kim, K.-W.: https://doi.org/10.1137/110823031, 2011
  ///
  /// @param[out] Upper bound of the patchs Korn constant
  double estimate_squared_korn_constant() const;

  /* Setter functions */

  /* Getter functions */
  /// Number of considered RHS
  int nrhs() const { return _nrhs; }

  /// Return central node of patch
  /// @return Central node
  int node_i() const { return _nodei; }

  /// Return patch type
  /// @param index Index of equilibrated flux
  /// @return Type of the patch
  base::PatchType type(int index) const { return _type[index]; }

  /// Return patch types
  /// @return List of patch-types
  std::span<const base::PatchType> type() const
  {
    return std::span<const base::PatchType>(_type.data(), _nrhs);
  }

  /// Return type-relation
  /// @return Type relation of different LHS
  std::int8_t equal_patch_types() const { return _equal_patches; }

  /// Check if patch is internal
  /// @return True if patch is internal
  bool is_internal() const { return _type[0] == base::PatchType::internal; }

  /// Check if patch is on boundary
  /// @return True if patch is internal
  bool is_on_boundary() const { return _type[0] != base::PatchType::internal; }

  /// Check if patch requires flux BCs
  /// @return True if patch requires flux BCs
  bool requires_flux_bcs() const
  {
    bool requires_bc = false;

    for (auto t : _type)
    {
      if (t == base::PatchType::bound_essnt_dual
          || t == base::PatchType::bound_mixed)
      {
        requires_bc = true;
        break;
      }
    }

    return requires_bc;
  }

  /// Check if patch requires flux BCs
  /// @return True if patch requires flux BCs
  bool requires_flux_bcs(int index) const
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
  bool requires_flux_bcs(int index, int fct_id) const
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

  /// Return spacial dimension
  /// @return The spacial dimension
  int dim() const { return _dim; }

  /// Return number of facets per cell
  /// @return Number of facets per cell
  int fcts_per_cell() const { return _fct_per_cell; }

  // Return the maximal number of grouped patches
  int groupsize_max() const { return _groupsize_max; }

  // Return the maximal number of cells per patch
  int ncells_max() const { return _ncells_max; }

  /// Return number of cells on patch
  /// @return Number of cells on patch
  int ncells() const { return _ncells; }

  /// Return number of cells on arbitrary patch
  /// @return Number of cells on patch
  int ncells(std::int32_t node_i) const
  {
    return _node_to_cell->links(node_i).size();
  }

  /// Return number of facets on patch
  /// @return Number of facets on patch
  int nfcts() const { return _nfcts; }

  /// Return processor-local cell ids on current patch
  /// @return cells
  std::span<const std::int32_t> cells() const
  {
    if (is_internal())
    {
      return std::span<const std::int32_t>(_cells.data(), _ncells + 2);
    }
    else
    {
      return std::span<const std::int32_t>(_cells.data(), _ncells + 1);
    }
  }

  /// Return processor-local cell id
  /// @param cell_i Patch-local cell id
  /// @return cell
  std::int32_t cell(int cell_i) const { return _cells[cell_i]; }

  /// Return processor-local facet ids on current patch
  /// @return fcts
  std::span<const std::int32_t> fcts() const
  {
    if (is_internal())
    {
      return std::span<const std::int32_t>(_fcts.data(), _nfcts + 1);
    }
    else
    {
      return std::span<const std::int32_t>(_fcts.data(), _nfcts);
    }
  }

  /// Return processor-local facet id
  /// @param fct_i Patch-local facet id
  /// @return facet
  std::int32_t fct(int fct_i) const { return _fcts[fct_i]; }

  /// Return cell-local node ids of patch-central node
  /// @return inode_local
  std::span<const std::int8_t> inodes_local() const
  {
    return std::span<const std::int8_t>(_inodes_local.data(), _ncells + 2);
  }

  /// Return cell-local node id of patch-central node
  /// @param cell_i Patch-local cell id
  /// @return inode_local
  std::int8_t inode_local(int cell_i) const { return _inodes_local[cell_i]; }

  /// Test
  std::span<const std::int8_t> fctid_local()
  {
    return std::span<const std::int8_t>(_fcts_local.data(), 2 * _nfcts + 2);
  }

protected:
  /// Determine maximum patch size
  /// @param nnodes_proc  Number of nodes on current processor
  /// @param pnts_on_bndr Markers for all mesh nodes on boundary
  /// @param ncells_min   Minimum number of cells on a patch
  /// @param ncells_crit  Critical number of cells on adjacent boundary patches
  void set_max_patch_size(const int nnodes_proc,
                          std::span<const std::int8_t> pnts_on_bndr,
                          const int ncells_min, const int ncells_crit);

  /// Initializes patch
  ///
  /// Sets patch type and creates sorted list of patch-facets.
  ///
  /// @param node_i Processor local id of patch-central node
  void initialize_patch(const int node_i);

  /// Determine local facet-id on cell
  /// @param fct_i Processor-local facet id
  /// @param cell_i Processor-local cell id
  /// @return Cell local facet id
  std::int8_t get_fctid_local(std::int32_t fct_i, std::int32_t cell_i) const;

  /// Determine local facet-id on cell
  /// @param fct_i      Processor-local facet id
  /// @param fct_cell_i Facets on cell_i
  /// @return Cell local facet id
  std::int8_t get_fctid_local(std::int32_t fct_i,
                              std::span<const std::int32_t> fct_cell_i) const;

  /// Determine local id of node on cell
  /// @param cell_i Processor-local cell id
  /// @param node_i Processor-local node id
  /// @return Cell local node id
  std::int8_t node_local(std::int32_t cell_i, std::int32_t node_i) const;

  /// Determine the next facet on a patch formed by triangular elements
  ///
  /// Triangle has three facets. Determine the two facets not already used.
  /// Check if smaller or larger facet is outside the range spanned by the
  /// facets of the patch. If not, conduct full search in the list of patch
  /// facets.
  ///
  /// @param cell_i     Processor local id of current cell
  /// @param fct_cell_i List of factes (processor local) of current cell
  /// @param id_fct_loc Loacl facet id on current cell
  /// @return           Next facet
  std::int32_t next_facet(std::int32_t cell_i,
                          std::span<const std::int32_t> fct_cell_i,
                          std::int8_t id_fct_loc) const;

  /// Returns an adjacent intern. patch of boundary patch
  /// @param node_i Processor-local id of patch-central node
  /// @return The patch-central node of the internal patch
  std::int32_t adjacent_internal_patch(const std::int32_t node_i) const;

  // Maximum size of patch
  int _ncells_max;

  // Maximum number of grouped patches
  int _groupsize_max;

  /* Geometry */
  // The mesh
  std::shared_ptr<const mesh::Mesh> _mesh;

  // The space dimension
  const int _dim, _dim_fct;

  // Counter element type
  int _fct_per_cell;

  // The connectivity's
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> _node_to_cell,
      _node_to_fct, _fct_to_node, _fct_to_cell, _cell_to_fct, _cell_to_node;

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

  // Facets/Cells on patch
  std::vector<std::int32_t> _cells, _fcts, _fcts_sorted;
  std::vector<std::int8_t> _inodes_local;

  // [lfct_(E0,T-1), lfct_(E0,T1), ..., lfct_(Ea,Ta), lfct_(Ea,Tap1)]
  std::vector<std::int8_t> _fcts_local;

  /* DOFmap */
  std::vector<std::int32_t> _ddofmap, _offset_dofmap;
  std::array<std::size_t, 3> _dofmap_shape;
};

template <typename T, int id_flux_order>
class Patch : public OrientedPatch
{
public:
  /// Initialization
  ///
  /// Storage is designed for the maximum patch size occurring within
  /// the current mesh.
  ///
  /// @param mesh                    The current mesh
  /// @param bfct_type               List with type of all boundary facets
  /// @param pnt_on_essntbndr        List with points on essential stress
  ///                                boundary
  ///                                (only for stress eqlb. required --> empty
  ///                                else)
  /// @param function_space_fluxhdiv Function space of H(div) flux
  /// @param symconstr_required      Flag for constrained minimisation
  /// @param ncells_min              Minimum number of cells patches
  ///                                (below: Error)
  /// @param ncells_crit             Critical number of cells on a boundary
  ///                                patch (modification required)
  Patch(std::shared_ptr<const mesh::Mesh> mesh,
        base::mdspan_t<const std::int8_t, 2> bfct_type,
        std::span<const std::int8_t> pnt_on_essntbndr,
        const std::shared_ptr<const fem::FunctionSpace> function_space_fluxhdiv,
        const std::shared_ptr<const fem::FunctionSpace> function_space_fluxdg,
        const basix::FiniteElement& basix_element_fluxdg,
        bool symconstr_required = false, int ncells_min = 1,
        int ncells_crit = 1)
      : OrientedPatch(mesh, bfct_type, pnt_on_essntbndr, ncells_min,
                      ncells_crit),
        _symconstr_required(symconstr_required),
        _degree_elmt_fluxdg(basix_element_fluxdg.degree()),
        _degree_elmt_fluxhdiv(
            function_space_fluxhdiv->element()->basix_element().degree() - 1),
        _function_space_fluxdg(function_space_fluxdg),
        _function_space_fluxhdiv(function_space_fluxhdiv),
        _entity_dofs_fluxcg(basix_element_fluxdg.entity_closure_dofs())
  {
    assert(id_flux_order < 0);

    // Set DOF counters of flux element
    _ndof_fluxdg = _function_space_fluxdg->element()->space_dimension();
    _ndof_flux = _function_space_fluxhdiv->element()->space_dimension();

    if (_degree_elmt_fluxdg == 0)
    {
      _ndof_fluxdg_fct = 1;
    }
    else
    {
      _ndof_fluxdg_fct = _entity_dofs_fluxcg[this->_dim_fct][0].size();
    }
    _ndof_flux_fct = _degree_elmt_fluxhdiv + 1;

    _ndof_flux_cell = _ndof_flux - _fct_per_cell * _ndof_flux_fct;
    _ndof_flux_add_cell
        = 0.5 * _degree_elmt_fluxhdiv * (_degree_elmt_fluxhdiv - 1);
    _ndof_flux_div_cell = _ndof_flux_cell - _ndof_flux_add_cell;

    _ndof_flux_nz = _ndof_flux - (_fct_per_cell - 2) * _ndof_flux_fct;

    // Number of DOFs (projected RHS)
    _ndof_rhsdg = _ndof_fluxdg / this->_dim;

    /* Reserve storage */
    // DOFmap
    const std::size_t ndofs_per_cell
        = (_symconstr_required) ? _ndof_flux_nz + _fct_per_cell : _ndof_flux_nz;
    _dofmap_shape = {4, (std::size_t)(_ncells_max + 2), ndofs_per_cell};
    _ddofmap.resize(_dofmap_shape[0] * _dofmap_shape[1] * _dofmap_shape[2], 0);

    _offset_dofmap.resize(5, 0);
    _offset_dofmap[1] = _ndof_flux_fct;
    _offset_dofmap[2] = _offset_dofmap[1] + _ndof_flux_fct;
    _offset_dofmap[3] = _offset_dofmap[2] + _ndof_flux_add_cell;
    _offset_dofmap[4] = (_symconstr_required)
                            ? _offset_dofmap[3] + _fct_per_cell
                            : _offset_dofmap[3];

    // DOFs of projected flux on facets
    _list_fctdofs_fluxdg.resize(2 * (this->_ncells_max + 1) * _ndof_fluxdg_fct,
                                0);
  }

  void flux_dofmap_cell(const int a, base::mdspan_t<std::int32_t, 3> dofmap)
  {
    // Cell ID
    const std::int32_t cell = _cells[a];

    // Facet IDs
    const std::int32_t fg_Eam1 = _fcts[a - 1], fg_Ea = _fcts[a];

    std::int8_t fl_Eam1, fl_Ea;
    std::tie(fl_Eam1, fl_Ea) = fctid_local(a);

    // Set DOFmap
    const std::int32_t gdof = cell * _ndof_flux;
    std::int32_t ldof, pdof, pdof_Eam1, pdof_Ea, offs;
    const bool internal_patch = is_internal();

    if constexpr (id_flux_order == 1)
    {
      /* Facet DOFs */
      // Cell-local
      dofmap(0, a, 0) = fl_Eam1;
      dofmap(0, a, 1) = fl_Ea;

      // Global
      const std::int32_t gdof = cell * _ndof_flux;

      dofmap(1, a, 0) = gdof + fl_Eam1;
      dofmap(1, a, 1) = gdof + fl_Ea;

      // Patch-local
      dofmap(2, a, 0) = 0;
      dofmap(2, a, 1) = 0;
    }
    else
    {
      /* Facet DOFs */
      offs = _offset_dofmap[1];

      if (internal_patch && (a == _ncells))
      {
        pdof_Eam1 = (a - 1) * (_ndof_flux_fct - 1);
        pdof_Ea = 0;
      }
      else
      {
        pdof_Eam1 = (a - 1) * (_ndof_flux_fct - 1);
        pdof_Ea = pdof_Eam1 + _ndof_flux_fct - 1;
      }

      for (std::size_t ii = 0; ii < _ndof_flux_fct; ++ii)
      {
        // Cell-local
        std::int32_t ldof_Eam1 = fl_Eam1 * _ndof_flux_fct + ii;
        std::int32_t ldof_Ea = fl_Ea * _ndof_flux_fct + ii;

        dofmap(0, a, ii) = ldof_Eam1;
        dofmap(0, a, offs) = ldof_Ea;

        // Global
        dofmap(1, a, ii) = gdof + ldof_Eam1;
        dofmap(1, a, offs) = gdof + ldof_Ea;

        // Patch-local
        if (ii == 0)
        {
          dofmap(2, a, ii) = 0;
          dofmap(2, a, offs) = 0;
        }
        else
        {
          dofmap(2, a, ii) = pdof_Eam1 + ii;
          dofmap(2, a, offs) = pdof_Ea + ii;
        }

        // Update offset
        offs += 1;
      }

      /* Cell DOFs: Additional */
      if constexpr (id_flux_order > 2)
      {
        offs = _offset_dofmap[2];
        ldof = _fct_per_cell * _ndof_flux_fct + _ndof_flux_div_cell;
        pdof
            = _nfcts * (_ndof_flux_fct - 1) + 1 + (a - 1) * _ndof_flux_add_cell;

        for (std::size_t ii = 0; ii < _ndof_flux_add_cell; ++ii)
        {
          // Cell-local
          dofmap(0, a, offs) = ldof;

          // Global
          dofmap(1, a, offs) = gdof + ldof;

          // Patch-local
          dofmap(2, a, offs) = pdof + ii;

          // Prefactor for construction of H(div=0) space
          dofmap(3, a, offs) = 1;

          // Update offset/ local DOF
          offs += 1;
          ldof += 1;
        }
      }

      /* Cell DOFs: Divergence */
      offs = _offset_dofmap[4];
      ldof = _fct_per_cell * _ndof_flux_fct;

      for (std::size_t ii = 0; ii < _ndof_flux_div_cell; ++ii)
      {
        // Cell-local
        dofmap(0, a, offs) = ldof;

        // Global
        dofmap(1, a, offs) = gdof + ldof;

        // Prefactor for construction of H(div=0) space
        dofmap(3, a, offs) = 0;

        // Update offset/ local DOF
        offs += 1;
        ldof += 1;
      }
    }

    /* DOFs constraint */
    if (_symconstr_required)
    {
      if (is_internal())
      {
        fctdofs_contraint_space(a, dofmap);
      }
      else
      {
        if (a == 1)
        {
          fctdofs_contraint_space(0, a, dofmap);
          fctdofs_contraint_space(1, dofmap);
        }
        if (a == _ncells)
        {
          fctdofs_contraint_space(a, a, dofmap);
        }
        else
        {
          fctdofs_contraint_space(a, dofmap);
        }
      }
    }
  }

  void fctdofs_contraint_space(const int a,
                               base::mdspan_t<std::int32_t, 3> dofmap)
  {
    // Nodes on facet ii
    std::span<const std::int32_t> nodes_on_fct = _fct_to_node->links(_fcts[a]);

    // Information of cells adjacent to facet a
    const int pcell_1 = a;
    const int pcell_2 = (a == _ncells) ? 1 : a + 1;

    const std::int32_t cell_1 = _cells[pcell_1];
    const std::int32_t cell_2 = _cells[pcell_2];

    // Determine the facet DOFs
    const int offs = _offset_dofmap[3];
    for (std::int32_t node : nodes_on_fct)
    {
      if (node == _nodei)
      {
        // Cell-local DOF
        dofmap(0, pcell_1, offs) = _inodes_local[pcell_1];
        dofmap(0, pcell_2, offs) = _inodes_local[pcell_2];

        // Patch-local DOF
        dofmap(2, pcell_1, offs) = 0;
        dofmap(2, pcell_2, offs) = 0;

        // Prefactor for construction of H(div=0) space
        dofmap(3, pcell_1, offs) = 1;
        dofmap(3, pcell_2, offs) = 1;
      }
      else
      {
        int offs_1 = offs + 1;
        int offs_2 = offs + 2;
        int pdof = a;

        // Cell-local DOF
        dofmap(0, pcell_1, offs_1) = (std::int32_t)node_local(cell_1, node);
        dofmap(0, pcell_2, offs_2) = (std::int32_t)node_local(cell_2, node);

        // Patch-local DOF
        dofmap(2, pcell_1, offs_1) = pdof;
        dofmap(2, pcell_2, offs_2) = pdof;

        // Prefactor for construction of H(div=0) space
        dofmap(3, pcell_1, offs_1) = 1;
        dofmap(3, pcell_2, offs_2) = 1;
      }
    }
  }

  void fctdofs_contraint_space(const int pfct, const int pcell,
                               base::mdspan_t<std::int32_t, 3> dofmap)
  {
    // Nodes on facet ii
    std::span<const std::int32_t> nodes_on_fct
        = _fct_to_node->links(_fcts[pfct]);
    const std::int32_t cell = _cells[pcell];

    /* Set DOFmap */
    // Offset and patch-local DOF
    int offs, pdof;

    if (pfct == 0)
    {
      offs = _offset_dofmap[3] + 2;
      pdof = _nfcts - 1;
    }
    else
    {
      offs = _offset_dofmap[3] + 1;
      pdof = _nfcts;
    }

    // Get additional node on facet
    std::int32_t node
        = (nodes_on_fct[0] == _nodei) ? nodes_on_fct[1] : nodes_on_fct[0];

    // Set cell-local DOF
    dofmap(0, pcell, offs) = (std::int32_t)node_local(cell, node);

    // Patch-local DOF
    dofmap(2, pcell, offs) = pdof;

    // Prefactor for construction of H(div=0) space
    dofmap(3, pcell, offs) = 1;
  }

  void set_assembly_informations(
      const std::vector<bool>& facet_orientation,
      base::mdspan_t<const std::uint8_t, 2> facet_reversion,
      std::span<const double> storage_detJ)
  {
    // Initialisation
    std::int8_t fctloc_ea, fctloc_eam1;
    std::int32_t prefactor_ea, prefactor_eam1;

    // Create mdspan of DOFmap
    base::mdspan_t<std::int32_t, 3> dofmap(_ddofmap.data(), _dofmap_shape);

    // Loop over all cells
    for (std::size_t a = 1; a < _ncells + 1; ++a)
    {
      // The cell id
      int id_a = a - 1;

      // The local facet IDs
      std::tie(fctloc_eam1, fctloc_ea) = fctid_local(a);

      // Prefactors
      if (storage_detJ[id_a] < 0)
      {
        if (facet_reversion(id_a, 0))
        {
          prefactor_eam1 = dofmap(3, a - 1, _ndof_flux_fct);
        }
        else
        {
          prefactor_eam1 = (facet_orientation[fctloc_eam1]) ? -1 : 1;
        }

        prefactor_ea = (facet_orientation[fctloc_ea]) ? 1 : -1;
      }
      else
      {
        if (facet_reversion(id_a, 0))
        {
          prefactor_eam1 = dofmap(3, a - 1, _ndof_flux_fct);
        }
        else
        {
          prefactor_eam1 = (facet_orientation[fctloc_eam1]) ? 1 : -1;
        }

        prefactor_ea = (facet_orientation[fctloc_ea]) ? -1 : 1;
      }

      /* Data to DOFmap */
      for (std::size_t i = 0; i < _ndof_flux_fct; ++i)
      {
        dofmap(3, a, i) = prefactor_eam1;
        dofmap(3, a, _ndof_flux_fct + i) = prefactor_ea;
      }
    }

    // Complete DOFmap
    if (is_internal())
    {
      if (facet_reversion(0, 0))
      {
        for (std::size_t i = 0; i < _ndof_flux_fct; ++i)
        {
          // DOFmap on T1, E0
          dofmap(3, 1, i) = dofmap(3, _ncells, _ndof_flux_fct + i);
        }
      }

      // Set DOFmap on cell 0
      for (std::size_t ii = 0; ii < dofmap.extent(2); ++ii)
      {
        // DOFmap on cell 0 (=ncells)
        dofmap(3, 0, ii) = dofmap(3, _ncells, ii);

        // DOFmap on cell ncells+1 (=1)
        dofmap(3, _ncells + 1, ii) = dofmap(3, 1, ii);
      }
    }
  }

  /* Overload functions from base-class */
  void create_subdofmap(const int node_i)
  {
    // Initialize patch
    this->initialize_patch(node_i);

    // Set DOF counters for (constraint) minimisation problem
    this->_ndof_min_flux = 1 + this->_degree_elmt_fluxhdiv * this->_nfcts
                           + this->_ndof_flux_add_cell * this->_ncells;

    if (this->_symconstr_required)
    {
      this->_ndof_min_cons = this->_nfcts + 1;
      this->_ndof_min = this->_ndof_min_flux + this->_ndof_min_cons;
    }
    else
    {
      this->_ndof_min_cons = 0;
      this->_ndof_min = this->_ndof_min_flux;
    }

    // Adjust pattern for mdspan of DOFmap
    this->_dofmap_shape[1] = this->_ncells + 2;

    // Create mdspan of DOFmap
    base::mdspan_t<std::int32_t, 3> dofmap(this->_ddofmap.data(),
                                           this->_dofmap_shape);
    base::mdspan_t<std::int32_t, 2> dofs_fluxdg(
        _list_fctdofs_fluxdg.data(), this->_ncells + 1, 2 * _ndof_fluxdg_fct);

    // Initialise DOFmap on each cell of the patch
    for (std::size_t a = 1; a < this->_ncells + 1; ++a)
    {
      // DOFmap H(div) flux
      this->flux_dofmap_cell(a, dofmap);

      // DOFs of projected fluxes on facets
      if constexpr (id_flux_order == 1)
      {
        // Cell-local DOFs of projected flux on facets
        dofs_fluxdg(a, 0) = 0;
        dofs_fluxdg(a, 1) = 0;
      }
      else
      {
        if (_degree_elmt_fluxdg > 0)
        {
          // Local facet IDs on cell
          std::int8_t lfct_ta_eam1, lfct_ta_ea;
          std::tie(lfct_ta_eam1, lfct_ta_ea) = this->fctid_local(a);

          // Loop over DOFs
          for (std::int8_t i = 0; i < _ndof_fluxdg_fct; ++i)
          {
            // Cell-local DOFs o projected flux on facets
            dofs_fluxdg(a - 1, i)
                = _entity_dofs_fluxcg[this->_dim_fct][lfct_ta_eam1][i];
            dofs_fluxdg(a, _ndof_fluxdg_fct + i)
                = _entity_dofs_fluxcg[this->_dim_fct][lfct_ta_ea][i];
          }
        }
        else
        {
          // Cell-local DOFs of projected flux on facets
          dofs_fluxdg(a, 0) = 0;
          dofs_fluxdg(a, 1) = 0;
        }
      }
    }

    // Complete DOFmap
    if (this->is_internal())
    {
      // Set DOFmap on cell 0, n+1
      for (std::size_t ii = 0; ii < dofmap.extent(2); ++ii)
      {
        // DOFmap on cell 0 (=ncells)
        dofmap(0, 0, ii) = dofmap(0, this->_ncells, ii);
        dofmap(1, 0, ii) = dofmap(1, this->_ncells, ii);
        dofmap(2, 0, ii) = dofmap(2, this->_ncells, ii);

        // DOFmap on cell ncells+1 (=1)
        dofmap(0, this->_ncells + 1, ii) = dofmap(0, 1, ii);
        dofmap(1, this->_ncells + 1, ii) = dofmap(1, 1, ii);
        dofmap(2, this->_ncells + 1, ii) = dofmap(2, 1, ii);
      }

      // Set DOFs of projected  flux on facet 0
      for (std::size_t ii = 0; ii < _ndof_fluxdg_fct; ++ii)
      {
        int offs = _ndof_fluxdg_fct + ii;

        dofs_fluxdg(this->_ncells, ii) = dofs_fluxdg(0, ii);
        dofs_fluxdg(0, offs) = dofs_fluxdg(this->_ncells, offs);
      }
    }
    else
    {
      // Handle facets 0 and _nfcts
      for (std::size_t ii = 0; ii < _ndof_fluxdg_fct; ++ii)
      {
        int offs = _ndof_fluxdg_fct + ii;

        dofs_fluxdg(0, offs) = dofs_fluxdg(0, ii);
        dofs_fluxdg(this->_ncells, ii) = dofs_fluxdg(this->_ncells, offs);
      }
    }
  }

  /* Getter functions (Geometry) */
  /// @return Number of mesh nodes on patch
  int npnts() const { return _nfcts + 1; }

  /// Get global node-ids on facet
  /// @param fct_i Patch-local facet-id
  /// @return List of nodes on facets
  std::span<const std::int32_t> nodes_on_fct(int fct_i)
  {
    return _fct_to_node->links(_fcts[fct_i]);
  }

  /// Return local facet id of E_a on T_a
  ///
  /// T_a has only two facets, that are relevant for current patch.
  /// Following the convention, T_a is adjacent to E_am1 and E_a!
  ///
  /// @param fct_i  Patch-local id of facet E_a (a>=0)
  ///               (inner patches: a=0 <-> a=nfcts)
  /// @param cell_i Patch-local id of cell T_a (a>0)
  /// @return Local facte id
  std::int8_t fctid_local(int fct_i, int cell_i)
  {
    int offst;

    if (_type[0] == base::PatchType::internal)
    {
      if (fct_i == 0 || fct_i == _ncells)
      {
        // Allows labeling cell 1 as cell _ncells + 1
        // and labeling cell _ncells as cell 0
        if (cell_i == 1 || cell_i == _ncells + 1)
        {
          offst = 1;
        }
        else if (cell_i == _ncells || cell_i == 0)
        {
          offst = 0;
        }
        else
        {
          throw std::runtime_error("Cell not adjacent to facet");
        }
      }
      else
      {
        if (cell_i == fct_i)
        {
          offst = 0;
        }
        else if (cell_i == fct_i + 1)
        {
          offst = 1;
        }
        else
        {
          throw std::runtime_error("Cell not adjacent to facet");
        }
      }
    }
    else
    {
      if (cell_i < 1)
      {
        throw std::runtime_error("No such cell on patch!");
      }

      if (fct_i == 0)
      {
        if (cell_i == 1)
        {
          offst = 0;
        }
        else
        {
          throw std::runtime_error("Cell not adjacent to facet");
        }
      }
      else
      {
        if (cell_i == fct_i)
        {
          offst = 0;
        }
        else if (cell_i == fct_i + 1)
        {
          offst = 1;
        }
        else
        {
          throw std::runtime_error("Cell not adjacent to facet");
        }
      }
    }

    return _fcts_local[2 * fct_i + offst];
  }

  /// Return local facet ids of E_am1 and E_a on cell T_a
  /// @param cell_i Patch-local id of cell T_a (a>0)
  /// @return Local facet ids of facets E_am1 and E_a
  std::pair<std::int8_t, std::int8_t> fctid_local(int cell_i)
  {
    assert(cell_i > 0);

    const int offs = 2 * cell_i;
    return {_fcts_local[offs - 1], _fcts_local[offs]};
  }

  /* Getter functions (DOFmap) */
  /// @return The element degree of the RT space
  int degree_raviart_thomas() { return _degree_elmt_fluxhdiv; }

  /// @return Number of RHS-DOFs on cell
  int ndofs_rhs_cell() { return _ndof_rhsdg; }

  /// @return Number of projected-flux-DOFs on cell
  int ndofs_fluxdg_cell() { return _ndof_fluxdg; }

  /// @return Number of projected-flux-DOFs on facet
  int ndofs_fluxdg_fct() { return _ndof_fluxdg_fct; }

  /// @return Number of flux-DOFs on element
  int ndofs_flux() { return _ndof_flux; }

  /// @return Number of flux-DOFs on facet
  int ndofs_flux_fct() { return _ndof_flux_fct; }

  /// @return Number of flux-DOFs on cell
  int ndofs_flux_cell() { return _ndof_flux_cell; }

  /// @return Number of divergence-dept. flux-DOFs on cell
  int ndofs_flux_cell_div() { return _ndof_flux_div_cell; }

  /// @return Number of additional flux-DOFs on cell
  int ndofs_flux_cell_add() { return _ndof_flux_add_cell; }

  /// @return Number of flux-DOFs on (patch-wise) H(div=0) space
  int ndofs_flux_hdiz_zero() { return _ndof_min_flux; }

  /// Extract assembly information for minnisation problem
  ///
  /// mdspan has the dimension id x cells x ndofs with
  ///   id = 0: local DOFs
  ///   id = 1: global DOFs
  ///   id = 2: patch-local DOFs
  ///   id = 3: prefactor for construction of H(div=0) space
  ///
  /// DOF ordering per cell: [d_TaEam1, d_TaEa, d_Ta-add, d_Ta-constr,
  /// d_Ta-div]
  ///
  /// @return List DOFs
  base::mdspan_t<const std::int32_t, 3> assembly_info_minimisation() const
  {
    return base::mdspan_t<const std::int32_t, 3>(_ddofmap.data(),
                                                 _dofmap_shape);
  }

  /// Offsets within DOFmap on each element
  ///
  /// id = 0: Offset dofs_TaEam1
  /// id = 1: Offset dofs_TaEa
  /// id = 2: Offset dofs_Ta-add
  /// id = 3: Offset dofs_Ta-constr
  /// id = 4: Offset dofs_Ta-div
  ///
  /// @return List of offsets
  std::span<const std::int32_t> offset_dofmap() const
  {
    return std::span<const std::int32_t>(_offset_dofmap.data(), 5);
  }

  /// Extract facet-DOFs (projected flux)
  /// @param fct_i Patch-local facet-id
  /// @return List DOFs [dof1_cip1, ..., dofn_cip1, dof1_ci, ..., dofn_ci]
  std::span<const std::int32_t> dofs_projflux_fct(const int fct_i)
  {
    const int offs = 2 * _ndof_fluxdg_fct * fct_i;

    return std::span<const std::int32_t>(_list_fctdofs_fluxdg.data() + offs,
                                         2 * _ndof_fluxdg_fct);
  }

protected:
  const bool _symconstr_required;

  /* Variables */
  // Element degree of fluxes (degree of RT elements starts with 0!)
  const int _degree_elmt_fluxdg;
  const int _degree_elmt_fluxhdiv;

  // The function space
  std::shared_ptr<const fem::FunctionSpace> _function_space_fluxdg;
  std::shared_ptr<const fem::FunctionSpace> _function_space_fluxhdiv;

  // Connectivity between entities and cell local DOFs
  const std::vector<std::vector<std::vector<int>>>& _entity_dofs_fluxcg;

  // Facet DOFs proj. flux
  // int: ([{ds_T1E0 ds_TnEn}, {ds_TapaEa ds_TaEa}, ...,{ds_T1E0 ds_TnEn}])
  // bndry: ([{ds_T1E0 ds_T1E0}, {ds_TapaEa ds_TaEa}, ...,{ds_TnEn ds_TnEn}])
  std::vector<std::int32_t> _list_fctdofs_fluxdg;

  // Number of DOFs on sub-elements (element definition)
  int _ndof_rhsdg;
  int _ndof_fluxdg_fct, _ndof_fluxdg;
  int _ndof_flux_fct, _ndof_flux_div_cell, _ndof_flux_add_cell, _ndof_flux_cell,
      _ndof_flux, _ndof_flux_nz;
  int _ndof_min, _ndof_min_flux, _ndof_min_cons;
};

} // namespace dolfinx_eqlb::se