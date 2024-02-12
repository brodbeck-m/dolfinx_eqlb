#pragma once

#include "Patch.hpp"

#include <basix/finite-element.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/FunctionSpace.h>

#include <cassert>

using namespace dolfinx;

namespace dolfinx_eqlb
{
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
  PatchFluxCstm(int nnodes_proc, std::shared_ptr<const mesh::Mesh> mesh,
                mdspan_t<const std::int8_t, 2> bfct_type,
                const std::shared_ptr<const fem::FunctionSpace> function_space)
      : Patch(nnodes_proc, mesh, bfct_type),
        _degree_elmt(function_space->element()->basix_element().degree() - 1),
        _function_space(function_space)
  {
    assert(id_flux_order < 0);

    /* Counter DOFs */
    // Number of DOFs (H(div) flux)
    _ndof = _function_space->element()->space_dimension();
    _ndof_fct = _degree_elmt + 1;
    _ndof_cell = _ndof - this->_fct_per_cell * _ndof_fct;
    _ndof_add_cell = 0.5 * _degree_elmt * (_degree_elmt - 1);
    _ndof_div_cell = _ndof_cell - _ndof_add_cell;

    _ndof_nz = _ndof - (this->_fct_per_cell - 2) * _ndof_fct - _ndof_div_cell;
    _ndof_cmin = _ndof_nz + this->_fct_per_cell;

    /* Reserve storage */
    // DOFMaps
    _gdofs_stress.resize(_ncells_max * _ndof);
    _asmbl_info.resize(4 * _ncells_max * _ndof_cmin);

    // Local facet IDs
    _localid_fct.resize(2 * (_ncells_max + 1));
  }

  void set_assembly_informations(const std::vector<bool>& facet_orientation,
                                 std::span<const double> storage_detJ)
  {
    for (std::size_t id_a = 0; id_a < this->_ncells; ++id_a)
    {
      // Local facet IDs

      // Prefactors

      // Set informations
    }
  }

  /* Overload functions from base-class */
  void create_subdofmap(int node_i)
  {
    // Initialize patch
    auto [fct_i, c_fct_loop] = initialize_patch(node_i);
    const bool patch_on_boundary = is_on_boundary();

    // Create mdspans
    mdspan_t<std::int32_t, 2> dofmap_stress(
        _gdofs_stress.data(), (std::size_t)_ncells, (std::size_t)_ndof);
    mdspan_t<std::int32_t, 2> asmbl_info(
        _asmbl_info.data(), 4, (std::size_t)_ncells, (std::size_t)_ndof_cmin);

    /* Create DOFmap on patch */
    // Initialisation
    std::int32_t cell_i = -1, cell_im1 = 0;
    std::int32_t pcell_i, pcell_im1;

    const int ndofs_all_fcts = this->_fct_per_cell * _ndof_fct;
    const int ndofs_hdivz = 1 + _degree_elmt * _nfcts
                            + 0.5 * _degree_elmt * (_degree_elmt - 1) * _ncells;

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

      // Cells Ti and Tim1
      if (patch_on_boundary)
      {
        pcell_i = ii;
        pcell_im1 = (ii == 0) ? 0 : ii - 1;
      }
      else
      {
        if (ii < _ncells - 1)
        {
          pcell_i = ii + 1;
          pcell_im1 = ii
        }
        else
        {
          pcell_i = 0;
          pcell_im1 = _ncells - 1;
        }
      }

      cell_i = _cells[pcell_i];
      cell_im1 = _cells[pcell_im1];

      /* DOFmap: Stress DOFs */
      // Stress DOFs
      if constexpr (id_flux_order == 1)
      {
        int ldof_cell_i = id_fct_loc_ci * _ndof_fct;
        int ldof_cell_im1 = id_fct_loc_cim1 * _ndof_fct;

        // Cell-local DOFs
        asmbl_info(0, pcell_i, 0) = ldof_cell_i;
        asmbl_info(0, pcell_im1, 1) = ldof_cell_im1;

        // Patch-local DOFs
        asmbl_info(1, pcell_i, 0) = 0;
        asmbl_info(1, pcell_im1, 1) = 0;

        // Global DOFs
        asmbl_info(2, pcell_i, 0) = cell_i * _ndof + ldof_cell_i;
        asmbl_info(2, pcell_im1, 1) = cell_im1 * _ndof + ldof_cell_im1;
      }
      else
      {
        // DOFs on facet
        for (std::size_t jj = 0; jj < _ndof_fct; ++jj)
        {
          int offs = _ndof_fct + jj;
          int ldof_cell_i = id_fct_loc_ci * _ndof_fct + jj;
          int ldof_cell_im1 = id_fct_loc_cim1 * _ndof_fct + jj;
          int pdof = (jj = 0) ? 0 : ii * (_ndof_fct + _ndof_add_cell) + jj;

          // Cell-local DOFs
          asmbl_info(0, pcell_i, jj) = ldof_cell_i;
          asmbl_info(0, pcell_im1, offs) = ldof_cell_im1;

          // Patch-local DOFs
          asmbl_info(1, pcell_i, jj) = pdof;
          asmbl_info(1, pcell_im1, offs) = pdof;

          // Global DOFs
          asmbl_info(2, pcell_i, jj) = cell_i * _ndof + ldof_cell_i;
          asmbl_info(2, pcell_im1, offs) = cell_im1 * _ndof + ldof_cell_im1;
        }

        // DOFs on cell
        for (std::size_t jj = 0; jj < _ndof_add_cell; ++jj)
        {
          int offs = 2 * _ndof_fct + jj;
          int ldof = ndofs_all_fcts + jj;

          // Cell-local DOFs
          asmbl_info(0, pcell_i, offs) = ldof;

          // Patch-local DOFs
          asmbl_info(1, pcell_i, offs)
              = (ii + 1) * _ndof_fct + ii * _ndof_add_cell + jj;

          // Global DOFs
          asmbl_info(2, pcell_i, offs) = cell_i * _ndof + ldof;
        }
      }

      /* DOFmap: Constraint DOFs */
      // Nodes on facet ii
      std::span<const std::int32_t> nodes_on_fct
          = this->_fct_to_node->links(fct_i);

      for (std::int32_t node : nodes_on_fct)
      {
        if (node == this->_nodei)
        {
          // Cell-local DOF
          asmbl_info(0, pcell_i, _ndof_nz)
              = (std::int32_t)this->node_local(cell_i, node);
          asmbl_info(0, pcell_im1, _ndof_nz)
              = (std::int32_t)this->node_local(cell_im1, node);

          // Patch-local DOF
          asmbl_info(1, pcell_i, _ndof_nz) = ndofs_hdivz;
          asmbl_info(1, pcell_im1, _ndof_nz) = ndofs_hdivz;
        }
        else
        {
          int offs_i = _ndof_nz + 1;
          int offs_im1 = _ndof_nz + 2;
          int pdof = ndofs_hdivz + ii + 1;

          // Cell-local DOF
          asmbl_info(0, pcell_i, offs_i)
              = (std::int32_t)this->node_local(cell_i, node);
          asmbl_info(0, pcell_im1, offs_im1)
              = (std::int32_t)this->node_local(cell_im1, node);

          // Patch-local DOF
          asmbl_info(1, pcell_i, offs_i) = pdof;
          asmbl_info(1, pcell_im1, offs_im1) = pdof;
        }
      }
    }

    // Handle last boundary facet (boundary patches)
    if (patch_on_boundary)
    {
      // Cell identifiers
      const std::int32_t cell = cell_i, pcell = _ncells - 1;

      // Get local id of facet
      std::int8_t id_fct_loc = get_fctid_local(fct_i, cell_i);

      // Store local facet ids
      int offs_fct = 2 * _nfcts - 2;
      _localid_fct[offs_fct] = id_fct_loc;
      _localid_fct[offs_fct + 1] = id_fct_loc;

      /* DOFmap: Stress DOFs */
      if constexpr (id_flux_order == 1)
      {
        int ldof_cell = id_fct_loc * _ndof_fct;

        // Cell-local DOFs
        asmbl_info(0, pcell, 1) = ldof_cell;

        // Patch-local DOFs
        asmbl_info(1, pcell, 1) = 0;

        // Global DOFs
        asmbl_info(2, pcell, 1) = cell * _ndof + ldof_cell;
      }
      else
      {
        // DOFs on facet
        for (std::size_t jj = 0; jj < _ndof_fct; ++jj)
        {
          int offs = _ndof_fct + jj;
          int ldof_cell = id_fct_loc * _ndof_fct + jj;

          // Cell-local DOFs
          asmbl_info(0, pcell, offs) = ldof_cell;

          // Patch-local DOFs
          asmbl_info(1, pcell, offs)
              = (jj = 0) ? 0 : _nfcts * (_ndof_fct + _ndof_add_cell) + jj;

          // Global DOFs
          asmbl_info(2, pcell, offs) = cell * _ndof + ldof_cell;
        }
      }

      /* DOFmap: Constraint DOFs */
      // Nodes on facet ii
      std::span<const std::int32_t> nodes_on_fct
          = this->_fct_to_node->links(fct_i);

      std::int32_t bnode
          = (nodes_on_fct[0] == this->_nodei) nodes_on_fct[1] : nodes_on_fct[0];

      // Offset within cell DOFmap
      int offs = _ndof_nz + 2;
      int pdof = ndofs_hdivz + ii + 1;

      // Cell-local DOF
      asmbl_info(0, pcell_im1, offs)
          = (std::int32_t)this->node_local(cell, bnode);

      // Patch-local DOF
      asmbl_info(1, pcell, offs) = ndofs_hdivz + ii + 1;
    }
  }

  /* Setter functions */

  /* Getter functions (Geometry) */
  /// Return processor-local cell id. For inner patches a=0 resp. a=ncells+1
  /// point to last resp. first cell on patch.
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
  const int _degree_elmt;

  // The function space
  std::shared_ptr<const fem::FunctionSpace> _function_space;

  // Connectivity between entities and cell local DOFs
  const std::vector<std::vector<std::vector<int>>>& _entity_dofs_fluxcg;

  // Cell-wise DOFs of the stress without symmetry condition
  std::vector<std::int32_t> _gdofs_stress;

  // DOFmap and assembly infos for constrained minimisation
  std::vector<std::int32_t> _asmbl_info;

  // Number of DOFs on sub-elements (element definition)
  int _ndof_fct, _ndof_div_cell, _ndof_add_cell, _ndof_cell, _ndof, _ndof_nz,
      _ndof_cmin;

  // Local facet IDs (fct_a: [id_fct_Ta, id_fct_Tap1])
  std::vector<std::int8_t> _localid_fct;
};

} // namespace dolfinx_eqlb