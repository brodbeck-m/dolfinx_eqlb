#include "PatchFluxEV.hpp"
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <iostream>

namespace dolfinx_adaptivity::equilibration
{
void PatchFluxEV::create_subdofmap(int node_i)
{
  // Initialize patch
  auto [fct_i, c_fct_loop] = initialize_patch(node_i);

  /* Create DOFmap on patch */
  // Initialisation
  std::int32_t cell_i = -1, dof_patch = 0;
  const int ndof_cell = _ndof_flux_cell + _ndof_cons_cell;

  // Loop over all facets on patch
  for (std::size_t ii = 0; ii < c_fct_loop; ++ii)
  {
    // Set next cell on patch
    auto [id_fct_loc_ci, id_fct_loc_cim1, fct_next]
        = fcti_to_celli(ii, fct_i, cell_i);

    // Offset first DOF (flux DOFs on facet 1) on elmt_i
    std::int32_t offs_p = (ii + 1) * _ndof_elmt_nz;
    _offset_dofmap[ii + 1] = offs_p;

    // Offset flux DOFs ond second facet elmt_(i-1)
    std::int32_t offs_f;

    if (_type > 0)
    {
      // Extract cell_i
      cell_i = _cells[ii];

      // Offsets for DOFmap creation
      offs_f = (ii == 0) ? _ndof_flux_fct
                         : _offset_dofmap[ii - 1] + _ndof_flux_fct;
      offs_p = _offset_dofmap[ii];
    }
    else
    {
      // Extract cell_i
      cell_i = _cells[ii + 1];

      // Offsets for DOFmap creation
      if (ii < _nfcts - 1)
      {
        offs_f = _offset_dofmap[ii] + _ndof_flux_fct;
      }
      else
      {
        offs_f = _offset_dofmap[ii] + _ndof_flux_fct;
        offs_p = 0;
      }
    }

    // Get flux-DOFs on fct_i
    for (std::int8_t jj = 0; jj < _ndof_flux_fct; ++jj)
    {
      // Add cell-local DOFs
      _dofsnz_elmt[offs_p] = _entity_dofs_flux[_dim_fct][id_fct_loc_ci][jj];
      _dofsnz_elmt[offs_f + jj]
          = _entity_dofs_flux[_dim_fct][id_fct_loc_cim1][jj];

      // Calculate patch-local DOFs
      _dofsnz_patch[offs_p] = dof_patch;
      _dofsnz_patch[offs_f + jj] = dof_patch;

      // Increment id of patch-local DOFs
      dof_patch += 1;
      offs_p += 1;
    }

    // Get cell-wise DOFs on cell_i
    offs_p += _ndof_flux_fct;

    for (std::int8_t jj = 0; jj < ndof_cell; ++jj)
    {
      // Add cell-local DOFs
      _dofsnz_elmt[offs_p] = _ndof_flux_nz + jj;

      // Calculate patch-local DOFs
      _dofsnz_patch[offs_p] = dof_patch;

      // Increment id of patch-local DOFs
      dof_patch += 1;
      offs_p += 1;
    }

    // Set next facet
    _fcts[ii] = fct_i;
    fct_i = fct_next;
  }

  // Handle last boundary facet (boundary patches)
  if (_type > 0)
  {
    // Get local id of facet
    std::int8_t id_fct_loc = get_fctid_local(fct_i, cell_i);

    // Add DOFs to DOFmap
    std::int32_t offs_p = (_ncells - 1) * _ndof_elmt_nz + _ndof_flux_fct;

    for (std::int8_t jj = 0; jj < _ndof_flux_fct; ++jj)
    {
      // Add cell-local DOFs
      _dofsnz_elmt[offs_p] = _entity_dofs_flux[_dim_fct][id_fct_loc][jj];

      // Calculate patch-local DOFs
      _dofsnz_patch[offs_p] = dof_patch;

      // Increment id of patch-local DOFs
      dof_patch += 1;
      offs_p += 1;

      // Store last facet to facte-list
      _fcts[_nfcts - 1] = fct_i;
    }
  }

  // // Output Debug
  // std::cout << "\n DOFs patch: " << std::endl;
  // for (std::int8_t jj = 0; jj < _ncells; ++jj)
  // {
  //   auto list = dofs_patch(jj);

  //   for (auto e : list)
  //   {
  //     std::cout << e << " ";
  //   }
  //   std::cout << "\n";
  // }
  // std::cout << "\n DOFs global: " << std::endl;
  // const dolfinx::graph::AdjacencyList<std::int32_t>& dofs0
  //     = _function_space->dofmap()->list();
  // for (std::int8_t jj = 0; jj < _ncells; ++jj)
  // {
  //   auto list = dofs_elmt(jj);
  //   auto cell_i = _cells[jj];

  //   for (auto e : list)
  //   {
  //     std::cout << dofs0.links(cell_i)[e] << " ";
  //   }
  //   std::cout << "\n";
  // }
  // throw std::exception();
}
} // namespace dolfinx_adaptivity::equilibration