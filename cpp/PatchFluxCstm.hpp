#pragma once

#include "Patch.hpp"

#include <basix/finite-element.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/FunctionSpace.h>

#include <algorithm>
#include <cassert>

using namespace dolfinx;

namespace dolfinx_eqlb
{
template <typename T, int id_flux_order, bool constr_minms>
class PatchCstm : public PatchNew
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
  /// @param function_space_fluxhdiv Function space of H(div) flux
  PatchCstm(
      int nnodes_proc, std::shared_ptr<const mesh::Mesh> mesh,
      mdspan_t<const std::int8_t, 2> bfct_type,
      const std::shared_ptr<const fem::FunctionSpace> function_space_fluxhdiv)
      : PatchNew(nnodes_proc, mesh, bfct_type),
        _degree_elmt_fluxhdiv(
            function_space_fluxhdiv->element()->basix_element().degree() - 1),
        _function_space_fluxhdiv(function_space_fluxhdiv)
  {
    assert(id_flux_order < 0);

    // Set DOF counters of flux element
    _ndof_flux = _function_space_fluxhdiv->element()->space_dimension();

    _ndof_flux_fct = _degree_elmt_fluxhdiv + 1;

    _ndof_flux_cell = _ndof_flux - _fct_per_cell * _ndof_flux_fct;
    _ndof_flux_add_cell
        = 0.5 * _degree_elmt_fluxhdiv * (_degree_elmt_fluxhdiv - 1);
    _ndof_flux_div_cell = _ndof_flux_cell - _ndof_flux_add_cell;

    _ndof_flux_nz = _ndof_flux - (_fct_per_cell - 2) * _ndof_flux_fct;

    // Resize storage of DOFmap
    if constexpr (constr_minms)
    {
      const std::size_t ndofs_per_cell
          = std::max(2 * _ndof_flux_fct + _ndof_flux_add_cell + _fct_per_cell,
                     _ndof_flux_nz);
      _dofmap_shape = {4, (std::size_t)(_ncells_max + 2), ndofs_per_cell};
    }
    else
    {
      _dofmap_shape
          = {4, (std::size_t)(_ncells_max + 2), (std::size_t)_ndof_flux_nz};
    }

    _ddofmap.resize(_dofmap_shape[0] * _dofmap_shape[1] * _dofmap_shape[2], 0);
  }

  void flux_dofmap_cell(const int a, mdspan_t<std::int32_t, 3> dofmap)
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
      offs = _ndof_flux_fct;

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
        offs = 2 * _ndof_flux_fct;
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
          ldof += ii;
        }
      }

      /* Cell DOFs: Divergence */
      offs = 2 * _ndof_flux_fct + _ndof_flux_add_cell;
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
    if constexpr (constr_minms)
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

  void fctdofs_contraint_space(const int a, mdspan_t<std::int32_t, 3> dofmap)
  {
    // Nodes on facet ii
    std::span<const std::int32_t> nodes_on_fct = _fct_to_node->links(_fcts[a]);

    // Information of cells adjacent to facet a
    const int pcell_1 = a;
    const int pcell_2 = (a == _ncells) ? 1 : a + 1;

    const std::int32_t cell_1 = _cells[pcell_1];
    const std::int32_t cell_2 = _cells[pcell_2];

    // Determine the facet DOFs
    const int offs = 2 * _ndof_flux_fct + _ndof_flux_add_cell;
    for (std::int32_t node : nodes_on_fct)
    {
      if (node == _nodei)
      {
        // Cell-local DOF
        dofmap(0, pcell_1, offs) = _inodes_local[pcell_1];
        dofmap(0, pcell_2, offs) = _inodes_local[pcell_2];

        // Patch-local DOF
        dofmap(2, pcell_1, offs) = _ndof_min_flux;
        dofmap(2, pcell_2, offs) = _ndof_min_flux;

        // Prefactor for construction of H(div=0) space
        dofmap(3, pcell_1, offs) = 1;
        dofmap(3, pcell_2, offs) = 1;
      }
      else
      {
        int offs_1 = offs + 1;
        int offs_2 = offs + 2;
        int pdof = _ndof_min_flux + a;

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
                               mdspan_t<std::int32_t, 3> dofmap)
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
      offs = 2 * _ndof_flux_fct + _ndof_flux_add_cell + 2;
      pdof = _ndof_min_flux + _nfcts - 1;
    }
    else
    {
      offs = 2 * _ndof_flux_fct + _ndof_flux_add_cell + 1;
      pdof = _ndof_min_flux + _nfcts;
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

  void set_assembly_informations(const std::vector<bool>& facet_orientation,
                                 std::span<const double> storage_detJ)
  {
    // Initialisation
    std::int8_t fctloc_ea, fctloc_eam1;
    std::int32_t prefactor_ea, prefactor_eam1;

    // Create mdspan of DOFmap
    mdspan_t<std::int32_t, 3> dofmap(_ddofmap.data(), _dofmap_shape);

    // Loop over all cells
    for (std::size_t a = 1; a < _ncells + 1; ++a)
    {
      // Local facet IDs
      std::tie(fctloc_eam1, fctloc_ea) = fctid_local(a);

      // Prefactors
      if (storage_detJ[a - 1] < 0)
      {
        prefactor_eam1 = (facet_orientation[fctloc_eam1]) ? -1 : 1;
        prefactor_ea = (facet_orientation[fctloc_ea]) ? -1 : 1;
      }
      else
      {
        prefactor_eam1 = (facet_orientation[fctloc_eam1]) ? 1 : -1;
        prefactor_ea = (facet_orientation[fctloc_ea]) ? 1 : -1;
      }

      /* Set DOFmap */
      // DOFs associated with d_0
      dofmap(3, a, 0) = prefactor_eam1;
      dofmap(3, a, _ndof_flux_fct) = -prefactor_ea;

      // Higer order DOFs
      if constexpr (id_flux_order > 1)
      {
        // Extract type of patch 0
        const PatchType type_patch = _type[0];

        if constexpr (id_flux_order == 2)
        {
          dofmap(3, a, 1) = prefactor_eam1;
          dofmap(3, a, 3) = -prefactor_ea;
        }
        else
        {
          for (std::size_t i = 1; i < _ndof_flux_fct; ++i)
          {
            dofmap(3, a, i) = prefactor_eam1;
            dofmap(3, a, _ndof_flux_fct + i) = -prefactor_ea;
          }
        }
      }
    }
  }

  /* Overload functions from base-class */
  void create_subdofmap(const int node_i)
  {
    // Initialize patch
    initialize_patch(node_i);

    // Set DOF counters for (constraint) minimisation problem
    _ndof_min_flux
        = 1 + _degree_elmt_fluxhdiv * _nfcts + _ndof_flux_add_cell * _ncells;

    if constexpr (constr_minms)
    {
      _ndof_min_cons = _nfcts + 1;
      _ndof_min = _ndof_min_flux + _ndof_min_cons;
    }
    else
    {
      _ndof_min_cons = 0;
      _ndof_min = _ndof_min_flux;
    }

    // Adjust pattern for mdspan of DOFmap
    _dofmap_shape[1] = _ncells + 2;

    // Create mdspan of DOFmap
    mdspan_t<std::int32_t, 3> dofmap(_ddofmap.data(), _dofmap_shape);

    // Initialise DOFmap on each cell of the patch
    for (std::size_t a = 1; a < _ncells + 1; ++a)
    {
      flux_dofmap_cell(a, dofmap);
    }

    // Complete DOFmap
    if (is_internal())
    {
      // Set DOFmap on cell 0
      for (std::size_t ii = 0; ii < dofmap.extent(2); ++ii)
      {
        // DOFmap on cell 0 (=ncells)
        dofmap(0, 0, ii) = dofmap(0, _ncells, ii);
        dofmap(1, 0, ii) = dofmap(1, _ncells, ii);
        dofmap(2, 0, ii) = dofmap(2, _ncells, ii);

        // DOFmap on cell ncells+1 (=1)
        dofmap(0, _ncells + 1, ii) = dofmap(0, 1, ii);
        dofmap(1, _ncells + 1, ii) = dofmap(1, 1, ii);
        dofmap(2, _ncells + 1, ii) = dofmap(2, 1, ii);
      }
    }
  }

  /* Setter functions */

  /* Getter functions (Geometry) */

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

    if (_type[0] == PatchType::internal)
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

  /// Extract assembly information for minnisation problem
  ///
  /// mdspan has the dimension id x cells x ndofs with
  ///   id = 0: local DOFs
  ///   id = 1: global DOFs
  ///   id = 2: patch-local DOFs
  ///   id = 3: prefactor for construction of H(div=0) space
  ///
  /// DOF ordering per cell: [dofs_TaEam1, dofs_TaEa, dofs_Ta-add, dofs_Ta-div]
  ///
  /// @return List DOFs
  mdspan_t<const int32_t, 3> assembly_info_minimisation() const
  {
    return mdspan_t<const int32_t, 3>(_ddofmap.data(), _dofmap_shape);
  }

protected:
  /* Variables */
  // Element degree of fluxes (degree of RT elements starts with 0!)
  const int _degree_elmt_fluxhdiv;

  // The function space
  std::shared_ptr<const fem::FunctionSpace> _function_space_fluxhdiv;

  // Number of DOFs on sub-elements (element definition)
  int _ndof_flux_fct, _ndof_flux_div_cell, _ndof_flux_add_cell, _ndof_flux_cell,
      _ndof_flux, _ndof_flux_nz;
  int _ndof_min, _ndof_min_flux, _ndof_min_cons;
};

template <typename T, int id_flux_order, bool constr_minms>
class PatchFluxCstmNew : public PatchCstm<T, id_flux_order, constr_minms>
{
public:
  /// Initialization
  ///
  /// Storage is designed for the maximum patch size occurring within
  /// the current mesh. (Cell-IDs start at 1, facet-IDs at 0!)
  ///
  /// @param nnodes_proc             Number of nodes on current processor
  /// @param mesh                    The current mesh
  /// @param bfct_type               List with type of all boundary facets
  /// @param function_space_fluxhdiv Function space of H(div) flux
  /// @param function_space_fluxdg   Function space of projected flux
  /// @param basix_element_fluxdg    BasiX element of projected flux
  ///                                (continuous version for required
  ///                                entity_closure_dofs)
  PatchFluxCstmNew(
      int nnodes_proc, std::shared_ptr<const mesh::Mesh> mesh,
      mdspan_t<const std::int8_t, 2> bfct_type,
      const std::shared_ptr<const fem::FunctionSpace> function_space_fluxhdiv,
      const std::shared_ptr<const fem::FunctionSpace> function_space_fluxdg,
      const basix::FiniteElement& basix_element_fluxdg)
      : PatchCstm<T, id_flux_order, constr_minms>(nnodes_proc, mesh, bfct_type,
                                                  function_space_fluxhdiv),
        _degree_elmt_fluxdg(basix_element_fluxdg.degree()),
        _function_space_fluxdg(function_space_fluxdg),
        _entity_dofs_fluxcg(basix_element_fluxdg.entity_closure_dofs())
  {
    assert(id_flux_order < 0);

    /* Counter DOFs */
    // Number of DOFs (projected flux)
    _ndof_fluxdg = _function_space_fluxdg->element()->space_dimension();

    // Number of DOFs (projected RHS)
    _ndof_rhsdg = _ndof_fluxdg / this->_dim;

    if (_degree_elmt_fluxdg == 0)
    {
      _ndof_fluxdg_fct = 1;
    }
    else
    {
      _ndof_fluxdg_fct = _entity_dofs_fluxcg[this->_dim_fct][0].size();
    }

    /* Reserve storage */
    // DOFs of projected flux on facets
    _list_fctdofs_fluxdg.resize(2 * (this->_ncells_max + 1) * _ndof_fluxdg_fct,
                                0);
  }

  /* Overload functions from base-class */
  void create_subdofmap(const int node_i)
  {
    // Initialize patch
    this->initialize_patch(node_i);

    // Set DOF counters for (constraint) minimisation problem
    this->_ndof_min_flux = 1 + this->_degree_elmt_fluxhdiv * this->_nfcts
                           + this->_ndof_flux_add_cell * this->_ncells;

    if constexpr (constr_minms)
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
    mdspan_t<std::int32_t, 3> dofmap(this->_ddofmap.data(),
                                     this->_dofmap_shape);
    mdspan_t<std::int32_t, 2> dofs_fluxdg(
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

  /* Getter functions (DOFmap) */
  /// @return Number of RHS-DOFs on cell
  int ndofs_rhs_cell() { return _ndof_rhsdg; }

  /// @return Number of projected-flux-DOFs on cell
  int ndofs_fluxdg_cell() { return _ndof_fluxdg; }

  /// @return Number of projected-flux-DOFs on facet
  int ndofs_fluxdg_fct() { return _ndof_fluxdg_fct; }

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
  /* Variables */
  // Element degree of fluxes (degree of RT elements starts with 0!)
  const int _degree_elmt_fluxdg;

  // The function space
  std::shared_ptr<const fem::FunctionSpace> _function_space_fluxdg;

  // Connectivity between entities and cell local DOFs
  const std::vector<std::vector<std::vector<int>>>& _entity_dofs_fluxcg;

  // Facet DOFs proj. flux
  // int: ([{ds_T1E0 ds_TnEn}, {ds_TapaEa ds_TaEa}, ...,{ds_T1E0 ds_TnEn}])
  // bndry: ([{ds_T1E0 ds_T1E0}, {ds_TapaEa ds_TaEa}, ...,{ds_TnEn ds_TnEn}])
  std::vector<std::int32_t> _list_fctdofs_fluxdg;

  // Number of DOFs on sub-elements (element definition)
  int _ndof_fluxdg_fct, _ndof_fluxdg, _ndof_rhsdg;
};

template <typename T, int id_flux_order = 3>
class PatchFluxCstm : public Patch
{
public:
  /// Initialization
  ///
  /// Storage is designed for the maximum patch size occurring within
  /// the current mesh. (Cell-IDs start at 1, facet-IDs at 0!)
  ///
  /// @param nnodes_proc             Number of nodes on current processor
  /// @param mesh                    The current mesh
  /// @param bfct_type               List with type of all boundary facets
  /// @param function_space_fluxhdiv Function space of H(div) flux
  /// @param function_space_fluxdg   Function space of projected flux
  /// @param basix_element_fluxdg    BasiX element of projected flux
  ///                                (continuous version for required
  ///                                entity_closure_dofs)
  PatchFluxCstm(
      int nnodes_proc, std::shared_ptr<const mesh::Mesh> mesh,
      mdspan_t<const std::int8_t, 2> bfct_type,
      const std::shared_ptr<const fem::FunctionSpace> function_space_fluxhdiv,
      const std::shared_ptr<const fem::FunctionSpace> function_space_fluxdg,
      const basix::FiniteElement& basix_element_fluxdg)
      : Patch(nnodes_proc, mesh, bfct_type),
        _degree_elmt_fluxhdiv(
            function_space_fluxhdiv->element()->basix_element().degree() - 1),
        _degree_elmt_fluxdg(basix_element_fluxdg.degree()),
        _function_space_fluxhdiv(function_space_fluxhdiv),
        _function_space_fluxdg(function_space_fluxdg),
        _entity_dofs_fluxcg(basix_element_fluxdg.entity_closure_dofs())
  {
    assert(id_flux_order < 0);

    /* Counter DOFs */
    // Number of DOFs (H(div) flux)
    _ndof_flux = _function_space_fluxhdiv->element()->space_dimension();
    _ndof_flux_fct = _degree_elmt_fluxhdiv + 1;
    _ndof_flux_cell = _ndof_flux - this->_fct_per_cell * _ndof_flux_fct;
    _ndof_flux_add_cell
        = 0.5 * _degree_elmt_fluxhdiv * (_degree_elmt_fluxhdiv - 1);
    _ndof_flux_div_cell = _ndof_flux_cell - _ndof_flux_add_cell;

    _ndof_flux_nz = _ndof_flux - (this->_fct_per_cell - 2) * _ndof_flux_fct;

    // Number of DOFs (projected flux)
    _ndof_fluxdg = _function_space_fluxdg->element()->space_dimension();

    // Number of DOFs (projected RHS)
    _ndof_rhsdg = _ndof_fluxdg / _dim;

    if (_degree_elmt_fluxdg == 0)
    {
      _ndof_fluxdg_fct = 1;
    }
    else
    {
      _ndof_fluxdg_fct = _entity_dofs_fluxcg[_dim_fct][0].size();
    }

    /* Reserve storage */
    // DOFMaps
    int len_adjacency_hflux_fct = _ncells_max * 2 * _ndof_flux_fct;
    int len_adjacency_hflux_cell = _ncells_max * _ndof_flux_cell;
    int len_adjacency_dflux_fct = _ncells_max * _ndof_fluxdg_fct;

    _dofsnz_elmt_fct.resize(len_adjacency_hflux_fct);
    _dofsnz_glob_fct.resize(len_adjacency_hflux_fct);
    _offset_dofmap_fct.resize(_ncells_max + 1, 0);

    _dofsnz_elmt_cell.resize(len_adjacency_hflux_cell);
    _dofsnz_glob_cell.resize(len_adjacency_hflux_cell);
    _offset_dofmap_cell.resize(_ncells_max + 1, 0);

    // DOFs of projected flux on facets
    _list_fctdofs_fluxdg.resize(2 * (_ncells_max + 1) * _ndof_fluxdg_fct, 0);
    _offset_list_fluxdg.resize(_ncells_max + 2, 0);

    // Local facet IDs
    _localid_fct.resize(2 * (_ncells_max + 1));
  }

  /* Overload functions from base-class */
  void create_subdofmap(int node_i)
  {
    // Initialize patch
    auto [fct_i, c_fct_loop] = initialize_patch(node_i);

    const bool patch_on_boundary = is_on_boundary();

    // Set number of DOFs on patch
    const int bs_fluxdg = _function_space_fluxdg->dofmap()->bs();

    const int ndof_fct = _fct_per_cell * _ndof_flux_fct;
    const int ndof_fluxdg_fct = _ndof_fluxdg_fct;

    /* Create DOFmap on patch */
    // Initialisation
    std::int32_t cell_i = -1, cell_im1 = 0;

    // Loop over all facets on patch
    for (std::size_t ii = 0; ii < c_fct_loop; ++ii)
    {
      // Set next cell on patch
      auto [id_fct_loc_ci, id_fct_loc_cim1, id_cell_plus, fct_next]
          = fcti_to_celli(0, ii, fct_i, cell_i);

      // Store local facet ids
      int offs_fct = 2 * ii;
      _localid_fct[offs_fct] = id_fct_loc_cim1;
      _localid_fct[offs_fct + 1] = id_fct_loc_ci;

      // Offset flux DOFs ond second facet elmt_(i-1)
      std::int32_t offs_f, offs_fhdiv_fct, offs_fhdiv_cell, offs_fdg_fct;

      // Extract data, set offsets for data storage and set +/- cells on
      // facet
      if (patch_on_boundary)
      {
        // Extract cell_i
        cell_i = _cells[ii];

        if (ii == 0)
        {
          // Offsets cell_(i-1), DOFs facet
          offs_f = _ndof_flux_fct;

          // Set cell_im1
          cell_im1 = _cells[0];
        }
        else
        {
          // Offsets cell_(i-1), DOFs facet
          offs_f = _offset_dofmap_fct[ii - 1] + _ndof_flux_fct;

          // Set cell_im1
          cell_im1 = _cells[ii - 1];
        }

        int iip1 = ii + 1;

        // Offset H(div)-flux DOFmap (facet)
        offs_fhdiv_fct = 2 * ii * _ndof_flux_fct;
        _offset_dofmap_fct[iip1] = 2 * iip1 * _ndof_flux_fct;

        // Offset H(div)-flux DOFmap (cell)
        offs_fhdiv_cell = ii * _ndof_flux_cell;
        _offset_dofmap_cell[iip1] = iip1 * _ndof_flux_cell;

        // Offset DOFs projected flux on facet
        offs_fdg_fct = 2 * ii * ndof_fluxdg_fct;
        _offset_list_fluxdg[iip1] = 2 * iip1 * ndof_fluxdg_fct;
      }
      else
      {
        int iip1 = ii + 1;

        // Offset H(div)-flux DOFmap (facet)
        offs_fhdiv_fct = 2 * iip1 * _ndof_flux_fct;
        _offset_dofmap_fct[iip1] = offs_fhdiv_fct;

        // Offset H(div)-flux DOFmap (cell)
        offs_fhdiv_cell = iip1 * _ndof_flux_cell;
        _offset_dofmap_cell[iip1] = offs_fhdiv_cell;

        // Offset DOFs projected flux on facet
        offs_fdg_fct = 2 * ii * ndof_fluxdg_fct;
        _offset_list_fluxdg[iip1] = 2 * iip1 * ndof_fluxdg_fct;

        // Current cell and offset cell_(i-1)
        if (ii < _ncells - 1)
        {
          // Extract cell_i
          cell_i = _cells[ii + 1];
          cell_im1 = _cells[ii];

          // Offsets cell_(i-1), flux DOFs facet
          offs_f = _offset_dofmap_fct[ii] + _ndof_flux_fct;
        }
        else
        {
          // Extract cell_i
          cell_i = _cells[0];
          cell_im1 = _cells[_ncells - 1];

          // Offsets cell_(i-1), DOFs facet
          offs_f = _offset_dofmap_fct[ii] + _ndof_flux_fct;
          offs_fhdiv_fct = 0;
          offs_fhdiv_cell = 0;
        }
      }

      // Extract DOFmaps
      if constexpr (id_flux_order == 1)
      {
        /* Get DOFS of H(div) flux on fct_i */
        // Precalculations
        int ldof_cell_i = id_fct_loc_ci * _ndof_flux_fct;
        int ldof_cell_im1 = id_fct_loc_cim1 * _ndof_flux_fct;

        // Add cell-local DOFs
        _dofsnz_elmt_fct[offs_fhdiv_fct] = ldof_cell_i;
        _dofsnz_elmt_fct[offs_f] = ldof_cell_im1;

        // Add global DOFs
        _dofsnz_glob_fct[offs_fhdiv_fct] = cell_i * _ndof_flux + ldof_cell_i;
        _dofsnz_glob_fct[offs_f] = cell_im1 * _ndof_flux + ldof_cell_im1;

        /* Get DOFs of projected flux on fct_i */
        _list_fctdofs_fluxdg[offs_fdg_fct] = 0;
        _list_fctdofs_fluxdg[offs_fdg_fct + 1] = 0;
      }
      else
      {
        /* Get DOFS of H(div) flux on fct_i */
        for (std::int8_t jj = 0; jj < _ndof_flux_fct; ++jj)
        {
          // Precalculations
          int ldof_cell_i = id_fct_loc_ci * _ndof_flux_fct + jj;
          int ldof_cell_im1 = id_fct_loc_cim1 * _ndof_flux_fct + jj;

          // Add cell-local DOFs
          _dofsnz_elmt_fct[offs_fhdiv_fct] = ldof_cell_i;
          _dofsnz_elmt_fct[offs_f + jj] = ldof_cell_im1;

          // Add global DOFs
          _dofsnz_glob_fct[offs_fhdiv_fct] = cell_i * _ndof_flux + ldof_cell_i;
          _dofsnz_glob_fct[offs_f + jj] = cell_im1 * _ndof_flux + ldof_cell_im1;

          // Increment offset
          offs_fhdiv_fct += 1;
        }

        /* Get DOFS of projected flux on fct_i */
        if (_degree_elmt_fluxdg > 0)
        {
          for (std::int8_t jj = 0; jj < _ndof_fluxdg_fct; ++jj)
          {
            // Precalculations
            int ldof_cell_i = _entity_dofs_fluxcg[_dim_fct][id_fct_loc_ci][jj];
            int ldof_cell_im1
                = _entity_dofs_fluxcg[_dim_fct][id_fct_loc_cim1][jj];

            int offs_fdg = offs_fdg_fct + _ndof_fluxdg_fct;
            _list_fctdofs_fluxdg[offs_fdg_fct] = ldof_cell_i;
            _list_fctdofs_fluxdg[offs_fdg] = ldof_cell_im1;

            // Increment offset
            offs_fdg_fct += 1;
          }
        }
        else
        {
          // Store DOFs
          _list_fctdofs_fluxdg[offs_fdg_fct] = 0;
          _list_fctdofs_fluxdg[offs_fdg_fct + 1] = 0;
        }

        /* Get cell-wise DOFs on cell_i */
        for (std::int8_t jj = 0; jj < _ndof_flux_cell; ++jj)
        {
          // Precalculations
          int ldof_cell_i = ndof_fct + jj;

          // Add cell-local DOFs
          _dofsnz_elmt_cell[offs_fhdiv_cell] = ldof_cell_i;

          // Add global DOFs
          _dofsnz_glob_cell[offs_fhdiv_cell]
              = cell_i * _ndof_flux + ldof_cell_i;

          // Increment offset
          offs_fhdiv_cell += 1;
        }
      }

      // Set next facet
      _fcts[ii] = fct_i;
      fct_i = fct_next;
    }

    // Handle last boundary facet (boundary patches)
    if (patch_on_boundary)
    {
      // Get local id of facet
      std::int8_t id_fct_loc = get_fctid_local(fct_i, cell_i);

      // Store local facet ids
      int offs_fct = 2 * _nfcts - 2;
      _localid_fct[offs_fct] = id_fct_loc;
      _localid_fct[offs_fct + 1] = id_fct_loc;

      // Initialize offsets
      std::int32_t offs_fhdiv_fct = (2 * _ncells - 1) * _ndof_flux_fct;
      std::int32_t offs_fdg_fct = 2 * _ncells * ndof_fluxdg_fct;

      _offset_dofmap_fct[_nfcts] = 2 * _nfcts * _ndof_flux_fct;
      _offset_list_fluxdg[_nfcts] = 2 * _nfcts * ndof_fluxdg_fct;

      if constexpr (id_flux_order == 1)
      {
        /* Get DOFS of H(div) flux on fct_i */
        // Precalculations
        int ldof_cell_i = id_fct_loc * _ndof_flux_fct;

        // Add cell-local DOFs
        _dofsnz_elmt_fct[offs_fhdiv_fct] = ldof_cell_i;

        // Add global DOFs
        _dofsnz_glob_fct[offs_fhdiv_fct] = cell_i * _ndof_flux + ldof_cell_i;

        /* Get DOFS of projected flux on fct_i */
        _list_fctdofs_fluxdg[offs_fdg_fct] = 0;
        _list_fctdofs_fluxdg[offs_fdg_fct + 1] = 0;
      }
      else
      {
        /* Get DOFs of H(div) flux on fct_i */
        for (std::int8_t jj = 0; jj < _ndof_flux_fct; ++jj)
        {
          // Precalculations
          int ldof_cell_i = id_fct_loc * _ndof_flux_fct + jj;

          // Add cell-local DOFs
          _dofsnz_elmt_fct[offs_fhdiv_fct] = ldof_cell_i;

          // Add global DOFs
          _dofsnz_glob_fct[offs_fhdiv_fct] = cell_i * _ndof_flux + ldof_cell_i;

          // Increment offset
          offs_fhdiv_fct += 1;
        }

        // Get DOFs of projected flux on fct_i
        if (_degree_elmt_fluxdg > 0)
        {
          for (std::int8_t jj = 0; jj < _ndof_fluxdg_fct; ++jj)
          {
            // Precalculations
            int ldof_cell_i = _entity_dofs_fluxcg[_dim_fct][id_fct_loc][jj];

            // Add cell-local DOFs
            _list_fctdofs_fluxdg[offs_fdg_fct] = ldof_cell_i;
            _list_fctdofs_fluxdg[offs_fdg_fct + ndof_fluxdg_fct] = ldof_cell_i;

            // Increment offset
            offs_fdg_fct += 1;
          }
        }
        else
        {
          _list_fctdofs_fluxdg[offs_fdg_fct] = 0;
          _list_fctdofs_fluxdg[offs_fdg_fct + 1] = 0;
        }
      }

      // Set next facet
      _fcts[_nfcts - 1] = fct_i;
    }
  }

  /* Setter functions */

  /* Getter functions (Geometry) */
  /// Return processor-local cell id. For inner patches a=0 resp.
  /// a=ncells+1 point to last resp. first cell on patch.
  /// @param cell_i Patch-local cell id
  /// @return cell
  std::int32_t cell(int cell_i)
  {
    if (_type[0] != PatchType::internal)
    {
      int celli = cellid_patch_to_data(cell_i);
      return _cells[celli];
    }
    else
    {
      if (cell_i == 0)
      {
        return _cells[_ncells - 1];
      }
      else if (cell_i == _ncells + 1)
      {
        return _cells[0];
      }
      else
      {
        int celli = cellid_patch_to_data(cell_i);
        return _cells[celli];
      }
    }
  }

  /// Return processor-local facet id
  /// @param fct_i Patch-local facet id
  /// @return facet
  std::int32_t fct(int fct_i)
  {
    int fcti = fctid_patch_to_data(fct_i);
    return _fcts[fcti];
  }

  /// Return cell-local node id of patch-central node
  /// @param cell_i Patch-local cell id
  /// @return inode_local
  std::int8_t inode_local(int cell_i)
  {
    int celli = cellid_patch_to_data(cell_i);
    return _inodes_local[celli];
  }

  /// Get global node-ids on facet
  /// @param fct_i Patch-local facet-id
  /// @return List of nodes on facets
  std::span<const std::int32_t> nodes_on_fct(int fct_i)
  {
    int fcti = fctid_patch_to_data(fct_i);

    return _fct_to_node->links(_fcts[fcti]);
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
    int ifct, offst;

    if (_type[0] == PatchType::internal)
    {
      if (fct_i == 0 || fct_i == _ncells)
      {
        ifct = _ncells - 1;

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
        ifct = fct_i - 1;

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

      ifct = fct_i;

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

    return _localid_fct[2 * ifct + offst];
  }

  /// Return local facet ids of E_a and E_am1 on cell T_a
  /// @param cell_i Patch-local id of cell T_a (a>0)
  /// @return Local facet ids of facets E_am1 and E_a
  std::pair<std::int8_t, std::int8_t> fctid_local(int cell_i)
  {
    assert(cell_i > 0);

    int fcti, fctim1;

    if ((cell_i == 1) && (_type[0] == PatchType::internal))
    {
      fcti = 0;
      fctim1 = _ncells - 1;
    }
    else
    {
      fcti = fctid_patch_to_data(cell_i);
      fctim1 = fcti - 1;
    }

    return {_localid_fct[2 * fctim1 + 1], _localid_fct[2 * fcti]};
  }

  /* Getter functions (DOFmap) */
  /// @return The element degree of the RT space
  int degree_raviart_thomas() { return _degree_elmt_fluxhdiv; }

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

  /// @return Number of RHS-DOFs on cell
  int ndofs_rhs_cell() { return _ndof_rhsdg; }

  /// @return Number of projected-flux-DOFs on cell
  int ndofs_fluxdg_cell() { return _ndof_fluxdg; }

  /// @return Number of projected-flux-DOFs on facet
  int ndofs_fluxdg_fct() { return _ndof_fluxdg_fct; }

  /// Extract global facet-DOFs (H(div) flux)
  /// @param cell_i Patch-local cell-id
  /// @return List DOFs (zero DOFs excluded)
  std::span<const std::int32_t> dofs_flux_fct_global(int cell_i)
  {
    int celli = cellid_patch_to_data(cell_i);

    return std::span<const std::int32_t>(
        _dofsnz_glob_fct.data() + _offset_dofmap_fct[celli],
        _offset_dofmap_fct[celli + 1] - _offset_dofmap_fct[celli]);
  }

  /// Extract global facet-DOFs (H(div) flux)
  /// @param cell_i Patch-local cell-id
  /// @param fct_i Patch-local facet-id
  /// @return List DOFs (zero DOFs excluded)
  std::span<const std::int32_t> dofs_flux_fct_global(int cell_i, int fct_i)
  {
    // Get cell-id in data-format
    int celli = cellid_patch_to_data(cell_i);

    // Determine offset based on facet
    int offs;

    if (fct_i == cell_i)
    {
      offs = _ndof_flux_fct + _offset_dofmap_fct[celli];
    }
    else if (fct_i == cell_i - 1)
    {
      offs = _offset_dofmap_fct[celli];
    }
    else
    {
      throw std::runtime_error("Cell not adjacent to facet");
    }

    return std::span<const std::int32_t>(_dofsnz_glob_fct.data() + offs,
                                         _ndof_flux_fct);
  }

  /// Extract global cell-DOFs (H(div) flux)
  /// @param cell_i Patch-local cell-id
  /// @return List DOFs (zero DOFs excluded)
  std::span<const std::int32_t> dofs_flux_cell_global(int cell_i)
  {
    int celli = cellid_patch_to_data(cell_i);

    return std::span<const std::int32_t>(
        _dofsnz_glob_cell.data() + _offset_dofmap_cell[celli],
        _offset_dofmap_cell[celli + 1] - _offset_dofmap_cell[celli]);
  }

  /// Extract cell-local facet-DOFs (H(div) flux)
  /// @param cell_i Patch-local cell-id
  /// @return List DOFs (zero DOFs excluded)
  std::span<const std::int32_t> dofs_flux_fct_local(int cell_i)
  {
    int celli = cellid_patch_to_data(cell_i);

    return std::span<const std::int32_t>(
        _dofsnz_elmt_fct.data() + _offset_dofmap_fct[celli],
        _offset_dofmap_fct[celli + 1] - _offset_dofmap_fct[celli]);
  }

  /// Extract cell-local facet-DOFs (H(div) flux)
  /// @param cell_i Patch-local cell-id
  /// @param fct_i Patch-local facet-id
  /// @return List DOFs (zero DOFs excluded)
  std::span<const std::int32_t> dofs_flux_fct_local(int cell_i, int fct_i)
  {
    // Get cell-id in data-format
    int celli = cellid_patch_to_data(cell_i);

    // Determine offset based on facet
    int offs;

    if (fct_i == cell_i)
    {
      offs = _ndof_flux_fct + _offset_dofmap_fct[celli];
    }
    else if (fct_i == cell_i - 1)
    {
      offs = _offset_dofmap_fct[celli];
    }
    else
    {
      throw std::runtime_error("Cell not adjacent to facet");
    }

    return std::span<const std::int32_t>(_dofsnz_elmt_fct.data() + offs,
                                         _ndof_flux_fct);
  }

  /// Extract cell-local cell-DOFs (H(div) flux)
  /// @param cell_i Patch-local cell-id
  /// @return List DOFs (zero DOFs excluded)
  std::span<const std::int32_t> dofs_flux_cell_local(int cell_i)
  {
    int celli = cellid_patch_to_data(cell_i);

    return std::span<const std::int32_t>(
        _dofsnz_elmt_cell.data() + _offset_dofmap_cell[celli],
        _offset_dofmap_cell[celli + 1] - _offset_dofmap_cell[celli]);
  }

  /// Extract facet-DOFs (projected flux)
  /// @param cell_i Patch-local facet-id
  /// @return List DOFs [dof1_cip1, ..., dofn_cip1, dof1_ci, ..., dofn_ci]
  std::span<const std::int32_t> dofs_projflux_fct(int fct_i)
  {
    int fcti = fctid_patch_to_data(fct_i);

    return std::span<const std::int32_t>(
        _list_fctdofs_fluxdg.data() + _offset_list_fluxdg[fcti],
        _offset_list_fluxdg[fcti + 1] - _offset_list_fluxdg[fcti]);
  }

protected:
  /// Get facet id within data structure from patch id
  /// @param fct_i Patch-local id of facet Ea
  /// @return Id of facet within data structure
  int fctid_patch_to_data(int fct_i)
  {
    int ifct = fct_i;

    if (_type[0] == PatchType::internal)
    {
      ifct = (fct_i == 0) ? _ncells - 1 : fct_i - 1;
    }

    return ifct;
  }

  /// Get cell id within data structure from patch id
  /// @param cell_i Patch-local id of cell Ta
  /// @return Id of cell within data structure
  int cellid_patch_to_data(int cell_i)
  {
    // ID of patch cells is always greater zero!
    assert(cell_i > 0);

    int celli = cell_i - 1;

    return celli;
  }

  /* Variables */
  // Element degree of fluxes (degree of RT elements starts with 0!)
  const int _degree_elmt_fluxhdiv, _degree_elmt_fluxdg;

  // The function space
  std::shared_ptr<const fem::FunctionSpace> _function_space_fluxhdiv,
      _function_space_fluxdg;

  // Connectivity between entities and cell local DOFs
  const std::vector<std::vector<std::vector<int>>>& _entity_dofs_fluxcg;

  // Storage sub-dofmaps (H(div) flux)
  std::vector<std::int32_t> _dofsnz_elmt_fct, _dofsnz_glob_fct,
      _offset_dofmap_fct;
  std::vector<std::int32_t> _dofsnz_elmt_cell, _dofsnz_glob_cell,
      _offset_dofmap_cell;

  // Facet DOFs projected flux (fct_a: [flux_E+, flux_E-], fct_ap1: []
  // ...)
  std::vector<std::int32_t> _list_fctdofs_fluxdg, _offset_list_fluxdg;

  // Number of DOFs on sub-elements (element definition)
  int _ndof_flux_fct, _ndof_flux_div_cell, _ndof_flux_add_cell, _ndof_flux_cell,
      _ndof_flux, _ndof_flux_nz;
  int _ndof_fluxdg_fct, _ndof_fluxdg, _ndof_rhsdg;

  // Local facet IDs (fct_a: [id_fct_Ta, id_fct_Tap1])
  std::vector<std::int8_t> _localid_fct;
};

} // namespace dolfinx_eqlb