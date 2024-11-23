// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "PatchFluxEV.hpp"

using namespace dolfinx;
using namespace dolfinx_eqlb;

Patch::Patch(int nnodes_proc, std::shared_ptr<const mesh::Mesh> mesh,
             base::mdspan_t<const std::int8_t, 2> bfct_type)
    : _mesh(mesh), _bfct_type(bfct_type), _dim(mesh->geometry().dim()),
      _dim_fct(mesh->geometry().dim() - 1),
      _type(bfct_type.extent(0), base::PatchType::internal),
      _nrhs(bfct_type.extent(0))
{
  // Initialize connectivities
  _node_to_cell = _mesh->topology().connectivity(0, _dim);
  _node_to_fct = _mesh->topology().connectivity(0, _dim_fct);
  _fct_to_node = _mesh->topology().connectivity(_dim_fct, 0);
  _fct_to_cell = _mesh->topology().connectivity(_dim_fct, _dim);
  _cell_to_fct = _mesh->topology().connectivity(_dim, _dim_fct);
  _cell_to_node = _mesh->topology().connectivity(_dim, 0);

  // Determine maximum patch size
  set_max_patch_size(nnodes_proc);

  // Reserve storage for patch geometry
  _cells.resize(_ncells_max);
  _fcts.resize(_ncells_max + 1);
  _fcts_sorted_data.resize(_ncells_max + 1);
  _inodes_local.resize(_ncells_max);

  // Initialize number of facets per cell
  _fct_per_cell = _cell_to_fct->links(0).size();
}

bool Patch::reversion_required(int index)
{
  // Initialise output
  bool patch_reversed = false;

  if ((index > 0) && requires_flux_bcs(index))
  {
    // Get type of the subsequent RHSs
    const base::PatchType type_i = _type[index];
    const base::PatchType type_im1 = _type[index - 1];

    // Check if patch has to be reversed
    if ((type_i != type_im1) || type_i == base::PatchType::bound_mixed)
    {
      if (_bfct_type(index, _fcts[0]) != base::PatchFacetType::essnt_dual)
      {
        patch_reversed = true;
      }
    }
  }

  return patch_reversed;
}

void Patch::set_max_patch_size(int nnodes_proc)
{
  // Initialization
  _ncells_max = 0;

  // Loop over all nodes
  for (std::size_t i_node = 0; i_node < nnodes_proc; ++i_node)
  {
    // Get number of cells on patch
    int n_cells = _node_to_cell->links(i_node).size();

    if (n_cells > _ncells_max)
    {
      _ncells_max = n_cells;
    }
  }
}

std::pair<std::int32_t, std::int32_t> Patch::initialize_patch(int node_i)
{
  // Set central node
  _nodei = node_i;

  // Get cells and facets of patch
  std::span<const std::int32_t> cells = _node_to_cell->links(node_i);
  std::span<const std::int32_t> fcts = _node_to_fct->links(node_i);

  // Set size of current patch
  _ncells = cells.size();
  _nfcts = fcts.size();

  // Creat sorted list of facets
  set_fcts_sorted(fcts);

  // Initialize type of patch
  std::fill(_type.begin(), _type.end(), base::PatchType::internal);
  _equal_patches = true;

  // Initialize first facet
  std::int32_t fct_first = fcts[0];

  // Initialize loop over facets
  std::int32_t c_fct_loop = _ncells;

  if (_nfcts > _ncells)
  {
    /* Determine patch type: i_rhs = 0 */
    std::array<std::int32_t, 2> fct_ef = {-1, -1};
    std::array<std::int32_t, 2> fct_ep = {-1, -1};

    // Check for boundary facets
    for (std::int32_t id_fct : fcts)
    {
      if (_bfct_type(0, id_fct) == base::PatchFacetType::essnt_primal)
      {
        // Mark first facet for DOFmap construction
        if (fct_ep[0] < 0)
        {
          fct_ep[0] = id_fct;
        }
        else
        {
          fct_ep[1] = id_fct;
        }
      }
      else if (_bfct_type(0, id_fct) == base::PatchFacetType::essnt_dual)
      {
        // Mark first facet for DOFmap construction
        if (fct_ef[0] < 0)
        {
          fct_ef[0] = id_fct;
        }
        else
        {
          fct_ef[1] = id_fct;
        }
      }
    }

    // Set patch type
    if (fct_ef[0] < 0)
    {
      _type[0] = base::PatchType::bound_essnt_primal;

      // Start patch construction on dirichlet facet
      fct_first = fct_ep[0];
    }
    else
    {
      if (fct_ep[0] < 0)
      {
        _type[0] = base::PatchType::bound_essnt_dual;
      }
      else
      {
        _type[0] = base::PatchType::bound_mixed;
      }

      // Start patch construction on neumann facet
      fct_first = fct_ef[0];
    }

    /* Set types of following RHS */
    if (_bfct_type.extent(0) > 1)
    {
      for (std::size_t i_rhs = 1; i_rhs < _bfct_type.extent(0); ++i_rhs)
      {
        // Extract (global) Ids of first and last facet
        std::int32_t fct_0, fct_n;

        if (_type[0] == base::PatchType::bound_essnt_primal)
        {
          fct_0 = fct_ep[0];
          fct_n = fct_ep[1];
        }
        else if (_type[0] == base::PatchType::bound_essnt_dual)
        {
          fct_0 = fct_ef[0];
          fct_n = fct_ef[1];
        }
        else
        {
          fct_0 = fct_ef[0];
          fct_n = fct_ep[0];
        }

        // Identify patch type of current RHS
        if (_bfct_type(i_rhs, fct_0) == _bfct_type(i_rhs, fct_n))
        {
          if (_bfct_type(i_rhs, fct_0) == base::PatchFacetType::essnt_primal)
          {
            _type[i_rhs] = base::PatchType::bound_essnt_primal;
          }
          else
          {
            _type[i_rhs] = base::PatchType::bound_essnt_dual;
          }
        }
        else
        {
          _type[i_rhs] = base::PatchType::bound_mixed;
        }
      }

      // Check if all patches have the same type
      if (std::adjacent_find(_type.begin(), _type.end(), std::not_equal_to<>())
          == _type.end())
      {
        _equal_patches = false;
      }
    }
  }

  return {fct_first, c_fct_loop};
}

std::tuple<std::int8_t, std::int8_t, std::int8_t, std::int32_t>
Patch::fcti_to_celli(int id_l, int c_fct, std::int32_t fct_i,
                     std::int32_t cell_in)
{
  // Initialize local facet_ids
  std::int8_t id_fct_loc_ci = 0;
  std::int8_t id_fct_loc_cim1 = 0;

  // Initialize id of next cell
  std::int32_t cell_i, cell_im1;

  // Initialize +/- ids of cells on facet
  std::int8_t id_cell_plus;

  // Get cells adjacent to fct_i
  std::span<const std::int32_t> cell_fct_i = _fct_to_cell->links(fct_i);

  // Initialize facet-lists
  std::span<const std::int32_t> fct_cell_i, fct_cell_im1;

  // Cells adjacent to current facet and local facet IDs
  if (_type[id_l] != base::PatchType::internal && c_fct == 0)
  {
    cell_i = cell_fct_i[0];
    cell_im1 = cell_i;

    // Facets of cells
    fct_cell_i = _cell_to_fct->links(cell_i);

    // Local facet id
    id_fct_loc_ci = get_fctid_local(fct_i, fct_cell_i);
    id_fct_loc_cim1 = id_fct_loc_ci;
  }
  else
  {
    if (cell_fct_i[0] == cell_in)
    {
      cell_i = cell_fct_i[1];
      cell_im1 = cell_fct_i[0];
    }
    else
    {
      cell_i = cell_fct_i[0];
      cell_im1 = cell_fct_i[1];
    }

    // Facets of cells
    fct_cell_i = _cell_to_fct->links(cell_i);
    fct_cell_im1 = _cell_to_fct->links(cell_im1);

    // Get id (cell_local) of fct_i
    id_fct_loc_ci = get_fctid_local(fct_i, fct_cell_i);
    id_fct_loc_cim1 = get_fctid_local(fct_i, fct_cell_im1);
  }

  // Get local id of node_i on cell_i
  std::int8_t id_node_loc_ci = nodei_local(cell_i);

  // Determine next facet on patch
  std::int32_t fct_next
      = next_facet_triangle(cell_i, fct_cell_i, id_fct_loc_ci);

  // Store relevant data
  if (_type[id_l] != base::PatchType::internal)
  {
    _cells[c_fct] = cell_i;
    _inodes_local[c_fct] = id_node_loc_ci;
    id_cell_plus = (cell_i == cell_fct_i[0]) ? 1 : 0;
  }
  else
  {
    if (c_fct < _nfcts - 1)
    {
      _cells[c_fct + 1] = cell_i;
      _cells[c_fct] = cell_im1;
      _inodes_local[c_fct + 1] = id_node_loc_ci;
      id_cell_plus = (cell_i == cell_fct_i[0]) ? 1 : 0;
    }
    else
    {
      _cells[0] = cell_i;
      _inodes_local[0] = id_node_loc_ci;
      id_cell_plus = (cell_i == cell_fct_i[0]) ? 1 : 0;
    }
  }

  return {id_fct_loc_ci, id_fct_loc_cim1, id_cell_plus, fct_next};
}

std::int8_t Patch::get_fctid_local(std::int32_t fct_i, std::int32_t cell_i)
{
  // Get facets on cell
  std::span<const std::int32_t> fct_cell_i = _cell_to_fct->links(cell_i);

  return get_fctid_local(fct_i, fct_cell_i);
}

std::int8_t Patch::get_fctid_local(std::int32_t fct_i,
                                   std::span<const std::int32_t> fct_cell_i)
{
  // Initialize local id
  std::int8_t fct_loc = 0;

  // Get id (cell_local) of fct_i
  while (fct_cell_i[fct_loc] != fct_i && fct_loc < _fct_per_cell)
  {
    fct_loc += 1;
  }

  // Check for face not on cell
  assert(fct_loc < _fct_per_cell);

  return fct_loc;
}

std::int8_t Patch::node_local(std::int32_t cell_i, std::int32_t node_i)
{
  // Initialize cell-local node-id
  std::int8_t id_node_loc_ci = 0;

  // Get nodes on cell_i
  std::span<const std::int32_t> node_cell_i = _cell_to_node->links(cell_i);

  while (node_cell_i[id_node_loc_ci] != node_i)
  {
    id_node_loc_ci += 1;
  }

  return id_node_loc_ci;
}

std::int8_t Patch::nodei_local(std::int32_t cell_i)
{
  return node_local(cell_i, _nodei);
}

std::int32_t
Patch::next_facet_triangle(std::int32_t cell_i,
                           std::span<const std::int32_t> fct_cell_i,
                           std::int8_t id_fct_loc)
{
  // Get remaining factes in correct order
  std::vector<std::int32_t> fct_es(2);
  std::int32_t fct_next;

  switch (id_fct_loc)
  {
  case 0:
    if (fct_cell_i[1] < fct_cell_i[2])
    {
      fct_es[0] = fct_cell_i[1];
      fct_es[1] = fct_cell_i[2];
    }
    else
    {
      fct_es[0] = fct_cell_i[2];
      fct_es[1] = fct_cell_i[1];
    }
    break;
  case 1:
    if (fct_cell_i[0] < fct_cell_i[2])
    {
      fct_es[0] = fct_cell_i[0];
      fct_es[1] = fct_cell_i[2];
    }
    else
    {
      fct_es[0] = fct_cell_i[2];
      fct_es[1] = fct_cell_i[0];
    }
    break;
  case 2:
    if (fct_cell_i[0] < fct_cell_i[1])
    {
      fct_es[0] = fct_cell_i[0];
      fct_es[1] = fct_cell_i[1];
    }
    else
    {
      fct_es[0] = fct_cell_i[1];
      fct_es[1] = fct_cell_i[0];
    }
    break;
  default:
    assert(id_fct_loc < 2);
  }

  // Get next facet
  if (fct_es[0] < _fcts_sorted[0])
  {
    fct_next = fct_es[1];
  }
  else
  {
    if (fct_es[1] > _fcts_sorted.back())
    {
      fct_next = fct_es[0];
    }
    else
    {
      // Full search
      if (std::count(_fcts_sorted.begin(), _fcts_sorted.end(), fct_es[0]))
      {
        fct_next = fct_es[0];
      }
      else
      {
        fct_next = fct_es[1];
      }
    }
  }

  return fct_next;
}

// ---------------------------------------------------------------------------------------------------
PatchFluxEV::PatchFluxEV(
    int nnodes_proc, std::shared_ptr<const mesh::Mesh> mesh,
    base::mdspan_t<const std::int8_t, 2> bfct_type,
    const std::shared_ptr<const fem::FunctionSpace> function_space,
    const std::shared_ptr<const fem::FunctionSpace> function_space_fluxhdiv,
    const basix::FiniteElement& basix_element_flux)
    : Patch(nnodes_proc, mesh, bfct_type), _function_space(function_space),
      _function_space_fluxhdiv(function_space_fluxhdiv),
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
  const int len_adjacency = _ncells_max * _ndof_elmt_nz;
  const int len_adjacency_flux
      = _ncells_max * _ndof_flux_cell + (_ncells_max + 1) * _ndof_flux_fct;

  _dofsnz_elmt.resize(len_adjacency);
  _dofsnz_patch.resize(len_adjacency);
  _dofsnz_global.resize(len_adjacency);
  _offset_dofmap.resize(_ncells_max + 1);

  _list_dofsnz_patch_fluxhdiv.resize(len_adjacency_flux);
  _list_dofsnz_global_fluxhdiv.resize(len_adjacency_flux);
}

void PatchFluxEV::create_subdofmap(int node_i)
{
  // Initialize patch
  auto [fct_i, c_fct_loop] = initialize_patch(node_i);

  const bool patch_on_boundary = is_on_boundary();

  // Set number of DOFs on patch
  const int ndof_cell = _ndof_flux_cell + _ndof_cons_cell;
  const int ndof_fct = _fct_per_cell * _ndof_flux_fct;
  _ndof_patch_nz = _nfcts * _ndof_flux_fct + _ncells * ndof_cell;
  _ndof_fluxhdiv = _nfcts * _ndof_flux_fct + _ncells * _ndof_flux_cell;

  /* Create DOFmap on patch */
  // Initialisation
  std::int32_t cell_i = -1, dof_patch = 0;
  const graph::AdjacencyList<std::int32_t>& gdofmap
      = _function_space->dofmap()->list();
  const graph::AdjacencyList<std::int32_t>& fdofmap
      = _function_space_fluxhdiv->dofmap()->list();
  std::span<const std::int32_t> gdofs;
  std::span<const std::int32_t> fdofs;

  // Loop over all facets on patch
  std::int32_t offs_l = 0;

  for (std::size_t ii = 0; ii < c_fct_loop; ++ii)
  {
    // Set next cell on patch
    auto [id_fct_loc_ci, id_fct_loc_cim1, id_cell_plus, fct_next]
        = fcti_to_celli(0, ii, fct_i, cell_i);

    // Offset first DOF (flux DOFs on facet 1) on elmt_i
    std::int32_t offs_p = (ii + 1) * _ndof_elmt_nz;
    _offset_dofmap[ii + 1] = offs_p;

    // Offset flux DOFs ond second facet elmt_(i-1)
    std::int32_t offs_f;

    if (patch_on_boundary)
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
      // Offsets for DOFmap creation
      if (ii < _nfcts - 1)
      {
        // Extract cell_i
        cell_i = _cells[ii + 1];

        // Offsets
        offs_f = _offset_dofmap[ii] + _ndof_flux_fct;
      }
      else
      {
        // Extract cell_i
        cell_i = _cells[0];

        // Offsets
        offs_f = _offset_dofmap[ii] + _ndof_flux_fct;
        offs_p = 0;
      }
    }

    // Get global DOFs on current element
    gdofs = gdofmap.links(cell_i);
    fdofs = fdofmap.links(cell_i);

    // Get flux-DOFs on fct_i
    for (std::int8_t jj = 0; jj < _ndof_flux_fct; ++jj)
    {
      // Precalculations
      int ldof_cell_i = _entity_dofs_flux[_dim_fct][id_fct_loc_ci][jj];
      int gdof_cell_i = gdofs[ldof_cell_i];

      // Add cell-local DOFs
      _dofsnz_elmt[offs_p] = ldof_cell_i;
      _dofsnz_elmt[offs_f + jj]
          = _entity_dofs_flux[_dim_fct][id_fct_loc_cim1][jj];

      // Calculate patch-local DOFs
      _dofsnz_patch[offs_p] = dof_patch;
      _dofsnz_patch[offs_f + jj] = dof_patch;

      // Calculate global DOFs
      _dofsnz_global[offs_p] = gdof_cell_i;
      _dofsnz_global[offs_f + jj] = gdof_cell_i;

      // Calculate global DOFs of H(div) confomring flux
      _list_dofsnz_patch_fluxhdiv[offs_l] = dof_patch;
      _list_dofsnz_global_fluxhdiv[offs_l] = fdofs[ldof_cell_i];

      // Increment id of patch-local DOFs
      dof_patch += 1;
      offs_p += 1;
      offs_l += 1;
    }

    // Get cell-wise DOFs on cell_i
    offs_p += _ndof_flux_fct;

    for (std::int8_t jj = 0; jj < _ndof_flux_cell; ++jj)
    {
      // Precalculations
      int ldof_cell_i = ndof_fct + jj;
      int gdof_cell_i = gdofs[ldof_cell_i];

      // Add cell-local DOFs
      _dofsnz_elmt[offs_p] = ldof_cell_i;

      // Calculate patch-local DOFs
      _dofsnz_patch[offs_p] = dof_patch;

      // Calculate global DOFs
      _dofsnz_global[offs_p] = gdof_cell_i;

      // Calculate global DOFs of H(div) conforming flux
      _list_dofsnz_patch_fluxhdiv[offs_l] = dof_patch;
      _list_dofsnz_global_fluxhdiv[offs_l] = fdofs[ndof_fct + jj];

      // Increment id of patch-local DOFs
      dof_patch += 1;
      offs_p += 1;
      offs_l += 1;
    }

    for (std::int8_t jj = _ndof_flux_cell; jj < ndof_cell; ++jj)
    {
      // Precalculations
      int ldof_cell_i = ndof_fct + jj;

      // Add cell-local DOFs
      _dofsnz_elmt[offs_p] = ldof_cell_i;

      // Calculate patch-local DOFs
      _dofsnz_patch[offs_p] = dof_patch;

      // Calculate global DOFs
      _dofsnz_global[offs_p] = gdofs[ldof_cell_i];

      // Increment id of patch-local DOFs
      dof_patch += 1;
      offs_p += 1;
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

    // Add DOFs to DOFmap
    std::int32_t offs_p = (_ncells - 1) * _ndof_elmt_nz + _ndof_flux_fct;

    for (std::int8_t jj = 0; jj < _ndof_flux_fct; ++jj)
    {
      // Precalculations
      const int ldof_cell_i = _entity_dofs_flux[_dim_fct][id_fct_loc][jj];
      int gdof_cell_i = gdofs[ldof_cell_i];

      // Add cell-local DOFs
      _dofsnz_elmt[offs_p] = _entity_dofs_flux[_dim_fct][id_fct_loc][jj];

      // Calculate patch-local DOFs
      _dofsnz_patch[offs_p] = dof_patch;

      // Calculate global DOFs
      _dofsnz_global[offs_p] = gdof_cell_i;

      // Calculate global DOFs of H(div) confomring flux
      _list_dofsnz_patch_fluxhdiv[offs_l] = dof_patch;
      _list_dofsnz_global_fluxhdiv[offs_l] = fdofs[ldof_cell_i];

      // Increment id of patch-local DOFs
      dof_patch += 1;
      offs_p += 1;
      offs_l += 1;

      // Store last facet to facte-list
      _fcts[_nfcts - 1] = fct_i;
    }
  }
}