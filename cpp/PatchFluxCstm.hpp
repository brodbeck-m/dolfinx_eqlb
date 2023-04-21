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
template <typename T, int id_flux_order = -1>
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
  /// @param basix_element_fluxdg    BasiX element of projected flux
  PatchFluxCstm(
      int nnodes_proc, std::shared_ptr<const mesh::Mesh> mesh,
      graph::AdjacencyList<std::int8_t>& bfct_type,
      const std::shared_ptr<const fem::FunctionSpace> function_space_fluxhdiv,
      const std::shared_ptr<const fem::FunctionSpace> function_space_fluxdg,
      const std::shared_ptr<const fem::FunctionSpace> function_space_rhsdg,
      const basix::FiniteElement& basix_element_fluxdg)
      : Patch(nnodes_proc, mesh, bfct_type),
        _degree_elmt_fluxhdiv(
            function_space_fluxhdiv->element()->basix_element().degree() - 1),
        _degree_elmt_fluxdg(basix_element_fluxdg.degree()),
        _function_space_fluxhdiv(function_space_fluxhdiv),
        _function_space_fluxdg(function_space_fluxdg),
        _function_space_rhsdg(function_space_rhsdg),
        _entity_dofs_fluxcg(basix_element_fluxdg.entity_closure_dofs())
  {
    assert(flux_order < 0);

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

    if constexpr (id_flux_order == 1)
    {
      _ndof_fluxdg_fct = 1;
    }
    else
    {
      if (_degree_elmt_fluxdg == 0)
      {
        _ndof_fluxdg_fct = 1;
      }
      else
      {
        _ndof_fluxdg_fct = _entity_dofs_fluxcg[_dim_fct][0].size();
      }
    }

    /* Reserve storage of DOFmaps */
    int len_adjacency_hflux_fct = _ncells_max * 2 * _ndof_flux_fct;
    int len_adjacency_hflux_cell = _ncells_max * _ndof_flux_cell;
    int len_adjacency_dflux_fct = _ncells_max * _ndof_fluxdg_fct;

    _dofsnz_elmt_fct.resize(len_adjacency_hflux_fct);
    _dofsnz_glob_fct.resize(len_adjacency_hflux_fct);
    _offset_dofmap_fct.resize(_ncells_max + 1, 0);

    _dofsnz_elmt_cell.resize(len_adjacency_hflux_cell);
    _dofsnz_glob_cell.resize(len_adjacency_hflux_cell);
    _offset_dofmap_cell.resize(_ncells_max + 1, 0);

    if constexpr (id_flux_order == 1)
    {
      _list_fctdofs_fluxdg.resize(4 * (_ncells_max + 1), 0);
      _offset_list_fluxdg.resize(_ncells_max + 2, 0);
    }
    else
    {
      _list_fctdofs_fluxdg.resize(2 * (_ncells_max + 1) * _ndof_fluxdg_fct, 0);
      _offset_list_fluxdg.resize(_ncells_max + 2, 0);
    }
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
      const basix::FiniteElement& basix_element_fluxdg)
      : PatchFluxCstm(nnodes_proc, mesh, bfct_type, function_space_fluxhdiv,
                      function_space_fluxdg, nullptr, basix_element_fluxdg)
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
  void create_subdofmap(int node_i)
  {
    // Initialize patch
    auto [fct_i, c_fct_loop] = initialize_patch(node_i);

    // Set number of DOFs on patch
    const int bs_fluxdg = _function_space_fluxdg->dofmap()->bs();

    const int ndof_fct = _fct_per_cell * _ndof_flux_fct;
    int ndof_fluxdg_fct;

    if constexpr (id_flux_order == 1)
    {
      ndof_fluxdg_fct = _ndof_fluxdg_fct * bs_fluxdg;
    }
    else
    {
      ndof_fluxdg_fct = _ndof_fluxdg_fct;
    }

    /* Create DOFmap on patch */
    // Initialisation
    std::int32_t cell_i = -1, cell_im1 = 0;

    // Loop over all facets on patch
    for (std::size_t ii = 0; ii < c_fct_loop; ++ii)
    {
      // Set next cell on patch
      auto [id_fct_loc_ci, id_fct_loc_cim1, fct_next]
          = fcti_to_celli(0, ii, fct_i, cell_i);

      // Offset flux DOFs ond second facet elmt_(i-1)
      std::int32_t offs_f, offs_fhdiv_fct, offs_fhdiv_cell, offs_fdg_fct;

      if (_type[0] > 0)
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
        ;
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

        /* Get DOFS of projected flux on fct_i */
        // TODO - Consider facet orientation (evaluation jump operator)
        int gdof_cell_i = cell_i * _ndof_fluxdg;
        int gdof_cell_im1 = cell_im1 * _ndof_fluxdg;

        _list_fctdofs_fluxdg[offs_fdg_fct] = gdof_cell_i;
        _list_fctdofs_fluxdg[offs_fdg_fct + 1] = gdof_cell_i + 1;
        _list_fctdofs_fluxdg[offs_fdg_fct + 2] = gdof_cell_im1;
        _list_fctdofs_fluxdg[offs_fdg_fct + 3] = gdof_cell_im1 + 1;
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
        // TODO - Consider facet orientation (evaluation jump operator)
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
    if (_type[0] > 0)
    {
      // Get local id of facet
      std::int8_t id_fct_loc = get_fctid_local(fct_i, cell_i);

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
        // Precalculations
        int gdof_cell_i = cell_i * _ndof_fluxdg;

        // Add global DOFs
        _list_fctdofs_fluxdg[offs_fdg_fct] = gdof_cell_i;
        _list_fctdofs_fluxdg[offs_fdg_fct + 1] = gdof_cell_i + 1;
        _list_fctdofs_fluxdg[offs_fdg_fct + 2] = gdof_cell_i;
        _list_fctdofs_fluxdg[offs_fdg_fct + 3] = gdof_cell_i + 1;
      }
      else
      {
        /* Get DOFS of H(div) flux on fct_i */
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

        // Get DOFS of projected flux on fct_i
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
      }
    }

    // // Debug
    // std::cout << "ndof_flux: " << _ndof_flux << std::endl;
    // std::cout << "ndof_flux_fct: " << _ndof_flux_fct << std::endl;
    // std::cout << "ndof_flux_cell (div): " << _ndof_flux_div_cell <<
    // std::endl; std::cout << "ndof_flux_cell (add): " << _ndof_flux_add_cell
    // << std::endl;

    // std::cout << "ndof_flux_dg: " << _ndof_fluxdg << std::endl;
    // std::cout << "ndof_flux_dg (fct): " << _ndof_fluxdg_fct << std::endl;

    // std::cout << "Offset fct-dofs (flux-dg): " << std::endl;
    // for (auto e : _offset_list_fluxdg)
    // {
    //   std::cout << e << " ";
    // }
    // std::cout << "\n";

    // std::cout << "Cells: " << std::endl;
    // for (auto e : _cells)
    // {
    //   std::cout << e << " ";
    // }
    // std::cout << "\n";

    // std::cout << "Global DOFs flux (H(div)) facet:" << std::endl;
    // for (std::int8_t i = 0; i < _ncells; ++i)
    // {
    //   auto dofs_facet = dofs_flux_fct_global(i);
    //   for (auto dof : dofs_facet)
    //   {
    //     std::cout << dof << " ";
    //   }
    //   std::cout << "\n";
    // }

    // std::cout << "Global DOFs flux (H(div)) cell:" << std::endl;
    // for (std::int8_t i = 0; i < _ncells; ++i)
    // {
    //   auto dofs_cell = dofs_flux_cell_global(i);
    //   for (auto dof : dofs_cell)
    //   {
    //     std::cout << dof << " ";
    //   }
    //   std::cout << "\n";
    // }

    // std::cout << "Global DOFs flux (projected) facet:" << std::endl;
    // std::cout << "n_fcts: " << _nfcts << std::endl;
    // for (std::int8_t i = 0; i < _nfcts; ++i)
    // {
    //   auto dofs_cell = dofs_projflux_fct(i);
    //   for (auto dof : dofs_cell)
    //   {
    //     std::cout << dof << " ";
    //   }
    //   std::cout << "\n";
    // }
  }

  void recreate_subdofmap(int index)
  {
    throw std::runtime_error("Equilibration: Multiple LHS not supported");
  }

  /* Setter functions */

  /* Getter functions */
  /// Extract global facet-DOFs (H(div) flux)
  /// @param cell_i Patch-local cell-id
  /// @return List DOFs (zero DOFs excluded)
  std::span<std::int32_t> dofs_flux_fct_global(int cell_i)
  {
    return std::span<std::int32_t>(
        _dofsnz_glob_fct.data() + _offset_dofmap_fct[cell_i],
        _offset_dofmap_fct[cell_i + 1] - _offset_dofmap_fct[cell_i]);
  }

  /// Extract global cell-DOFs (H(div) flux)
  /// @param cell_i Patch-local cell-id
  /// @return List DOFs (zero DOFs excluded)
  std::span<std::int32_t> dofs_flux_cell_global(int cell_i)
  {
    return std::span<std::int32_t>(
        _dofsnz_glob_cell.data() + _offset_dofmap_cell[cell_i],
        _offset_dofmap_cell[cell_i + 1] - _offset_dofmap_cell[cell_i]);
  }

  /// Extract facet-DOFs (projected flux)
  /// @param cell_i Patch-local facet-id
  /// @return List DOFs (zero DOFs excluded)
  std::span<std::int32_t> dofs_projflux_fct(int fct_i)
  {
    return std::span<std::int32_t>(
        _list_fctdofs_fluxdg.data() + _offset_list_fluxdg[fct_i],
        _offset_list_fluxdg[fct_i + 1] - _offset_list_fluxdg[fct_i]);
  }

protected:
  /*Function (sub-) space*/
  // Element degree of fluxes (degree of RT elements starts with 0!)
  const int _degree_elmt_fluxhdiv, _degree_elmt_fluxdg;

  // The function space
  std::shared_ptr<const fem::FunctionSpace> _function_space_fluxhdiv,
      _function_space_fluxdg, _function_space_rhsdg;

  // Connectivity between entities and cell local DOFs
  const std::vector<std::vector<std::vector<int>>>& _entity_dofs_fluxcg;

  // Storage sub-dofmaps (H(div) flux)
  std::vector<std::int32_t> _dofsnz_elmt_fct, _dofsnz_glob_fct,
      _offset_dofmap_fct;
  std::vector<std::int32_t> _dofsnz_elmt_cell, _dofsnz_glob_cell,
      _offset_dofmap_cell;

  // Facet DOFs projected flux
  std::vector<std::int32_t> _list_fctdofs_fluxdg, _offset_list_fluxdg;

  // Number of DOFs on sub-elements (element definition)
  int _ndof_flux_fct, _ndof_flux_div_cell, _ndof_flux_add_cell, _ndof_flux_cell,
      _ndof_flux, _ndof_flux_nz;
  int _ndof_fluxdg_fct, _ndof_fluxdg, _ndof_rhsdg;
};

} // namespace dolfinx_adaptivity::equilibration