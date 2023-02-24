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
  /// @param ncells_max Maximum patch-size (number of elements)
  /// @param mesh       The current mesh
  /// @param bfct_type  List with type of all boundary facets
  PatchFluxEV(
      int ncells_max, std::shared_ptr<const dolfinx::mesh::Mesh> mesh,
      std::span<const std::int8_t> bfct_type,
      const std::shared_ptr<const dolfinx::fem::FunctionSpace> function_space,
      const basix::FiniteElement& basix_element_flux)
      : Patch(ncells_max, mesh, bfct_type), _function_space(function_space),
        _entity_dofs_flux(basix_element_flux.entity_dofs()),
        _ndof_elmt(function_space->element()->space_dimension())

  {
    /* Counter DOFs */
    // Number of DOFs on mixed-element
    _ndof_elmt = _function_space->element()->space_dimension();

    // Number of DOFs on subelements
    _ndof_flux_fct = _entity_dofs_flux[_dim_fct][0].size();
    _ndof_flux_cell = _entity_dofs_flux[_dim][0].size();
    _ndof_flux = _fct_per_cell * _ndof_flux_fct + _ndof_flux_cell;
    _ndof_cons_cell = _ndof_elmt - _ndof_flux;
    _ndof_cons = _ndof_cons_cell;

    // Number of non-zero DOFs
    _ndof_elmt_nz = _ndof_elmt - (_fct_per_cell - 2) * _ndof_flux_fct;
    _ndof_flux_nz = _ndof_flux - (_fct_per_cell - 2) * _ndof_flux_fct;

    /* Reserve storage of DOFmaps */
    int len_adjacency = _ncells_max * _ndof_elmt_nz;

    _dofsnz_elmt.resize(len_adjacency);
    _dofsnz_patch.resize(len_adjacency);
    _offset_dofmap.resize(_ncells_max + 1);
  }

  /// Construction of a sub-DOFmap on each patch
  ///
  /// Determines type of patch (0-> internal, 1->bc_neumann, 2->bc_dirichlet
  /// 3->bc_mixed) and creats sorted DOFmap. Sorting of facets/elements/DOFs
  /// follows [1,2].
  ///
  /// [1] Moldenhauer, M.: Stress reconstructionand a-posteriori error
  ///     estimationfor elasticity (PhdThesis)
  /// [2] Bertrand, F.; Carstensen, C.; Gräßle, B. & Tran, N. T.:
  ///     Stabilization-free HHO a posteriori error control, 2022
  ///
  /// @param node_i Processor-local id of current node
  void create_subdofmap(int node_i);

  /* Setter functions */

  /* Getter functions */
  /// @return Number of DOFs on element
  int ndof_elmt() { return _ndof_elmt; }

  /// @return Number of non-zero DOFs on element
  int ndof_elmt_nz() { return _ndof_elmt_nz; }

  /// @return Number of flux-DOFs on element
  int ndof_flux() { return _ndof_flux; }

  /// @return Number of non-zero flux-DOFs on element
  int ndof_flux_nz() { return _ndof_flux_nz; }

  /// @return Number of constrained-DOFs on element
  int ndof_cons() { return _ndof_cons; }

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
  const std::shared_ptr<const dolfinx::fem::FunctionSpace>& _function_space;

  // Connectivity between entities and cell local DOFs
  const std::vector<std::vector<std::vector<int>>>& _entity_dofs_flux;

  // Storage sub-dofmap
  std::vector<std::int32_t> _dofsnz_elmt, _dofsnz_patch, _offset_dofmap;

  // Number of DOFs on element (element definition)
  int _ndof_elmt;

  // Number of DOFs on sub-elements (element definition)
  int _ndof_flux_fct, _ndof_flux_cell, _ndof_cons_cell, _ndof_flux, _ndof_cons;

  // Number on non-zero DOFs on element (on patch)
  int _ndof_elmt_nz, _ndof_flux_nz;
};

} // namespace dolfinx_adaptivity::equilibration