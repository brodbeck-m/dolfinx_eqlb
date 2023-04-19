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
template <typename T, int flux_order>
class PatchFluxCstm : public Patch
{
public:
  /// Initialization
  ///
  /// Storage is designed for the maximum patch size occuring within
  /// the current mesh.
  ///
  /// @param nnodes_proc             Numbe rof nodes on current processor
  /// @param mesh                    The current mesh
  /// @param bfct_type               List with type of all boundary facets
  /// @param function_space_fluxhdiv Function space of H(div) flux
  /// @param function_space_fluxdg   Function space of projected flux
  /// @param function_space_rhsdg    Function space of projected RHS
  /// @param basix_element_fluxhdiv  BasiX element of H(div) flux
  /// @param basix_element_fluxdg    BasiX element of projected flux
  PatchFluxCstm(
      int nnodes_proc, std::shared_ptr<const mesh::Mesh> mesh,
      graph::AdjacencyList<std::int8_t>& bfct_type,
      const std::shared_ptr<const fem::FunctionSpace> function_space_fluxhdiv,
      const std::shared_ptr<const fem::FunctionSpace> function_space_fluxdg,
      const std::shared_ptr<const fem::FunctionSpace> function_space_rhsdg,
      const basix::FiniteElement& basix_element_fluxhdiv,
      const basix::FiniteElement& basix_element_fluxdg)
      : Patch(nnodes_proc, mesh, bfct_type),
        _function_space_fluxhdiv(function_space_fluxhdiv),
        _function_space_fluxdg(function_space_fluxdg),
        _function_space_rhsdg(function_space_rhsdg),
        _entity_dofs_fluxhdiv(basix_element_fluxhdiv.entity_dofs()),
        _entity_dofs_fluxcg(basix_element_fluxhdiv.entity_dofs())
  {
    /* Counter DOFs */
    const int degree_rt = basix_element_fluxhdiv.degree() - 1;

    // Number of DOFs (H(div) flux)
    _ndof_flux = _function_space_fluxhdiv->element()->space_dimension();
    _ndof_flux_fct = _entity_dofs_fluxhdiv[_dim_fct][0].size();
    _ndof_flux_cell = 0.5 * degree_rt * (degree_rt - 1);
    _ndof_flux_div_cell
        = _ndof_flux - this->_fct_per_cell * _ndof_flux_fct - _ndof_flux_cell;

    _ndof_flux_nz = _ndof_flux - (this->_fct_per_cell - 2) * _ndof_flux_fct;

    // Number of DOFs (projected flux)
    _ndof_fluxdg = _function_space_fluxdg->element()->space_dimension();
    _ndof_fluxdg_fct = _entity_dofs_fluxcg[_dim_fct][0].size();

    /* Reserve storage of DOFmaps */
    int len_adjacency_hflux_fct = _ncells_max * 2 * _ndof_flux_fct;
    int len_adjacency_hflux_cell
        = _ncells_max * (_ndof_flux_cell + _ndof_flux_div_cell);
    int len_adjacency_dflux_fct = _ncells_max * _ndof_fluxdg_fct;

    _dofsnz_elmt_fct.resize(len_adjacency_hflux_fct);
    _dofsnz_glob_fct.resize(len_adjacency_hflux_fct);
    _offset_dofmap_fct.resize(_ncells_max + 1, 0);

    _dofsnz_elmt_cell.resize(len_adjacency_hflux_fct);
    _dofsnz_glob_cell.resize(len_adjacency_hflux_fct);
    _offset_dofmap_cell.resize(_ncells_max + 1, 0);

    _list_fctdofs_elmt_fluxdg.resize((_ncells_max + 1) * _ndof_fluxdg_fct);
    _offset_list_fluxdg.resize(_ncells_max + 2, 0);
  }

  /// Initialization
  ///
  /// Storage is designed for the maximum patch size occuring within
  /// the current mesh.
  ///
  /// @param nnodes_proc             Numbe rof nodes on current processor
  /// @param mesh                    The current mesh
  /// @param bfct_type               List with type of all boundary facets
  /// @param function_space_fluxhdiv Function space of H(div) flux
  /// @param function_space_fluxdg   Function space of projected flux
  /// @param basix_element_fluxhdiv  BasiX element of H(div) flux
  /// @param basix_element_fluxdg    BasiX element of projected flux (cg!)
  PatchFluxCstm(
      int nnodes_proc, std::shared_ptr<const mesh::Mesh> mesh,
      graph::AdjacencyList<std::int8_t>& bfct_type,
      std::shared_ptr<const fem::FunctionSpace> function_space_fluxhdiv,
      std::shared_ptr<const fem::FunctionSpace> function_space_fluxdg,
      const basix::FiniteElement& basix_element_fluxhdiv,
      const basix::FiniteElement& basix_element_fluxdg)
      : PatchFluxCstm(nnodes_proc, mesh, bfct_type, function_space_fluxhdiv,
                      function_space_fluxdg, nullptr, basix_element_fluxhdiv,
                      basix_element_fluxdg)
  {
  }

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
  void create_subdofmap(int node_i) {}

  void recreate_subdofmap(int index)
  {
    throw std::runtime_error("Equilibration: Multiple LHS not supported");
  }

  /* Setter functions */

  /* Getter functions */

protected:
  /*Function (sub-) space*/
  // The function space
  std::shared_ptr<const fem::FunctionSpace> _function_space_fluxhdiv,
      _function_space_fluxdg, _function_space_rhsdg;

  // Connectivity between entities and cell local DOFs
  const std::vector<std::vector<std::vector<int>>>&_entity_dofs_fluxhdiv,
      _entity_dofs_fluxcg;

  // Storage sub-dofmaps (H(div) flux)
  std::vector<std::int32_t> _dofsnz_elmt_fct, _dofsnz_glob_fct,
      _offset_dofmap_fct;
  std::vector<std::int32_t> _dofsnz_elmt_cell, _dofsnz_glob_cell,
      _offset_dofmap_cell;

  // Facet DOFs projected flux
  std::vector<std::int32_t> _list_fctdofs_elmt_fluxdg, _offset_list_fluxdg;

  // Number of DOFs on sub-elements (element definition)
  int _ndof_flux_fct, _ndof_flux_div_cell, _ndof_flux_cell, _ndof_flux,
      _ndof_flux_nz;
  int _ndof_fluxdg_fct, _ndof_fluxdg, _ndof_rhsdg;
};

} // namespace dolfinx_adaptivity::equilibration