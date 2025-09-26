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
#include <dolfinx_eqlb/base/equilibration.hpp>
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

template <std::floating_point U>
class Patch : public base::Patch<U>
{
public:
  /// Initialization
  ///
  /// Storage is designed for the maximum patch size occurring within
  /// the current mesh.
  ///
  /// @param nnodes_proc             Number of nodes on current processor
  /// @param mesh                    The current mesh
  /// @param bfct_type               List with type of all boundary facets
  /// @param function_space          The function spaced of the constrained
  ///                                minimisation problem
  /// @param function_space_fluxhdiv The function spaced of the constrained
  ///                                minimisation problem
  /// @param basix_element_flux      The Basix element of the flux space
  Patch(int nnodes_proc, std::shared_ptr<const mesh::Mesh<U>> mesh,
        base::mdspan_t<const std::int8_t, 2> bfct_type,
        const std::shared_ptr<const fem::FunctionSpace<U>> function_space,
        const std::shared_ptr<const fem::FunctionSpace<U>>
            function_space_fluxhdiv,
        const basix::FiniteElement<U>& basix_element_flux);

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
  void create_subdofmap(int node_i);

  /* Setter functions */

  /* Getter functions */
  /// @return Number of mesh nodes on patch
  int npnts() const { return this->_nfcts + 1; }

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
  /// Create a patch around a node
  ///
  /// Determine PatchType and determine order of patch facets.
  ///
  /// @param node_i Processor local id of patch-central node
  /// @return       First facet on patch,
  /// @return       Length of loop over facets
  std::pair<std::int32_t, std::int32_t> create_patch(int node_i);

  /// Determine connection on patch
  ///
  /// [1] Bertrand, F. et al.: https://doi.org/10.1007/s00211-023-01366-8, 2023
  ///
  /// @param id_r     Id of used RHS
  /// @param c_fct    Counter within loop over all facets of patch
  /// @param fct_i    Processor-local id of facet
  /// @param cell_in  Processor-local id of last cell on patch
  /// @return         Cell-local id of fct_i in cell_i,
  /// @return         Cell-local id of fct_i in cell_(i-1),
  /// @return         Next facet on patch
  std::tuple<std::int8_t, std::int8_t, std::int8_t, std::int32_t>
  fcti_to_celli(int id_r, int c_fct, std::int32_t fct_i, std::int32_t cell_in);

  /// Determine the next facet on a patch formed by triangular elements
  ///
  /// Triangle has three facets. Determine the two facets not already used.
  /// Check if smaller or larger facet is outside the range spanned by the
  /// facets of the patch. If not, conduct full search in the list of patch
  /// facets.
  ///
  /// @param cell_i     Processor local id of current cell
  /// @param fct_cell_i List of facets (processor local id) of current cell
  /// @param id_fct_loc Local facet id on current cell
  /// @return           The next facet
  std::int32_t next_facet(std::int32_t cell_i,
                          std::span<const std::int32_t> fct_cell_i,
                          std::int8_t id_fct_loc);

  /* Variables */
  // Intermediate storage
  std::vector<std::int32_t> _fcts_sorted;

  // The function space
  const std::shared_ptr<const fem::FunctionSpace<U>> _function_space;
  const std::shared_ptr<const fem::FunctionSpace<U>> _function_space_fluxhdiv;

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