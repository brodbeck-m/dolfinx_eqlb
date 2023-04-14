#pragma once

#include "Patch.hpp"
#include <algorithm>
#include <basix/finite-element.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <iostream>
#include <memory>
#include <span>
#include <vector>

using namespace dolfinx;

namespace dolfinx_adaptivity::equilibration
{
class PatchFluxEV : public Patch
{
public:
  /// Initialization
  ///
  /// Storage is designed for the maximum patch size occuring within
  /// the current mesh.
  ///
  /// @param nnodes_proc Numbe rof nodes on current processor
  /// @param mesh        The current mesh
  /// @param bfct_type   List with type of all boundary facets
  PatchFluxEV(
      int nnodes_proc, std::shared_ptr<const mesh::Mesh> mesh,
      graph::AdjacencyList<std::int8_t>& bfct_type,
      const std::shared_ptr<const fem::FunctionSpace> function_space,
      const std::shared_ptr<const fem::FunctionSpace> function_space_fluxhdiv,
      const basix::FiniteElement& basix_element_flux);

  /// Construction of a sub-DOFmap on each patch
  ///
  /// Determines type of patch (0-> internal, 1->bc_neumann, 2->bc_dirichlet
  /// 3->bc_mixed) and creats sorted DOFmap. Sorting of facets/elements/DOFs
  /// follows [1,2]. The sub-DOFmap is cearted for sub-problem 0. If patch-
  /// type differs between different patches, use recreate_subdofmap for
  /// sub-problem i>0.
  ///
  /// [1] Moldenhauer, M.: Stress reconstructionand a-posteriori error
  ///     estimationfor elasticity (PhdThesis)
  /// [2] Bertrand, F.; Carstensen, C.; Gräßle, B. & Tran, N. T.:
  ///     Stabilization-free HHO a posteriori error control, 2022
  ///
  /// @param node_i Processor-local id of current node
  void create_subdofmap(int node_i);

  void recreate_subdofmap(int index)
  {
    throw std::runtime_error("Equilibration: Multiple LHS not supported");
  }

  /* Setter functions */

  /* Getter functions */
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

  /// Extract DOFs (H(div) flux, mixed space) on patch
  /// @return Global flux-DOFs
  std::span<std::int32_t> dofs_fluxhdiv_mixed()
  {
    return std::span<std::int32_t>(_list_dofsnz_mixed_fluxhdiv.data(),
                                   _ndof_fluxhdiv);
  }

  /// Extract DOFs (H(div) flux, mixed space) on patch (const. version)
  /// @return Global flux-DOFs
  std::span<const std::int32_t> dofs_fluxhdiv_mixed() const
  {
    return std::span<const std::int32_t>(_list_dofsnz_mixed_fluxhdiv.data(),
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
  std::vector<std::int32_t> _dofsnz_elmt, _dofsnz_patch, _dofsnz_global,
      _offset_dofmap;

  // Storage DOFs H(div) flux (per patch)
  std::vector<std::int32_t> _list_dofsnz_patch_fluxhdiv,
      _list_dofsnz_global_fluxhdiv, _list_dofsnz_mixed_fluxhdiv;

  // Number of DOFs on element (element definition)
  int _ndof_elmt;

  // Number of DOFs on sub-elements (element definition)
  int _ndof_flux_fct, _ndof_flux_cell, _ndof_cons_cell, _ndof_flux, _ndof_cons;

  // Number on non-zero DOFs on element (on patch)
  int _ndof_elmt_nz, _ndof_flux_nz, _ndof_patch_nz, _ndof_fluxhdiv;
};

} // namespace dolfinx_adaptivity::equilibration