// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Patch.hpp"

using namespace dolfinx;
using namespace dolfinx_eqlb::se;

// ---------------------------------------------------------------------------------------------------
OrientedPatch::OrientedPatch(std::shared_ptr<const mesh::Mesh> mesh,
                             base::mdspan_t<const std::int8_t, 2> bfct_type,
                             std::span<const std::int8_t> pnts_on_bndr,
                             const int ncells_min, const int ncells_crit)
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

  // The patch size
  set_max_patch_size(mesh->topology().index_map(0)->size_local(), pnts_on_bndr,
                     ncells_min, ncells_crit);

  // Initialize number of facets per cell
  _fct_per_cell = _cell_to_fct->links(0).size();

  // Reserve storage for patch geometry
  if (_dim == 2)
  {
    const int sp1 = _ncells_max + 1, sp2 = _ncells_max + 2;

    _cells.resize(sp2);
    _fcts.resize(sp2);
    _fcts_sorted.resize(sp1);
    _fcts_local.resize(2 * sp1);
    _inodes_local.resize(sp2);

    // Check for triangular elements
    if (_fct_per_cell != 3)
    {
      throw std::runtime_error("Only triangular elements supported!");
    }
  }
  else
  {
    throw std::runtime_error("Only 2D meshes supported!");
  }
}

std::vector<std::int32_t>
OrientedPatch::group_boundary_patches(const std::int32_t node_i,
                                      std::span<const std::int8_t> pnts_on_bndr,
                                      const int ncells_crit) const
{
  // Initialisation
  std::vector<std::int32_t> grouped_patches;

  // Check the actual boundary patch
  if ((pnts_on_bndr[node_i]) && (ncells(node_i) == ncells_crit))
  {
    // Get adjacent inner patch
    std::int32_t node_i_internal = adjacent_internal_patch(node_i);
    grouped_patches.push_back(node_i_internal);

    // Cells on inner patch
    std::span<const std::int32_t> pcells
        = _node_to_cell->links(node_i_internal);

    // Find boundary nodes of inner patch
    for (std::int32_t cell : pcells)
    {
      // Nodes of current cell
      std::span<const std::int32_t> cpnts = _cell_to_node->links(cell);

      // Check boundary patches
      for (std::int32_t pnt : cpnts)
      {
        if (pnts_on_bndr[pnt])
        {
          if (std::find(grouped_patches.begin(), grouped_patches.end(), pnt)
              == grouped_patches.end())
          {
            if (ncells(pnt) == ncells_crit)
            {
              grouped_patches.push_back(pnt);
            }
          }
        }
      }
    }
  }

  return std::move(grouped_patches);
}

bool OrientedPatch::reversion_required(int index) const
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

double OrientedPatch::estimate_squared_korn_constant() const
{
  // The spatial node positions
  std::span<const double> x = _mesh->geometry().x();

  // The patch central node
  const int node_patch = node_i();

  // The minimal angle
  double theta_min;

  if (is_internal())
  {
    // Coordinates patch-central node
    std::array<const double, 2> x_i
        = {x[3 * node_patch], x[3 * node_patch + 1]};

    // List of cell nodes on patch boundary
    std::array<std::int32_t, 2> bnode_id;

    // Auxilary variables
    std::array<double, 2> v1, v2;
    double abs_v1, abs_v2, v1_t_v2;

    // Loop over cells
    theta_min = 0.5 * M_PI;

    for (std::int32_t cell : cells())
    {
      // Get nodes on boundary of patch
      int count_bnodes = 0;

      for (std::int32_t node : _cell_to_node->links(cell))
      {
        if (node != node_patch)
        {
          bnode_id[count_bnodes] = 3 * node;
          count_bnodes += 1;
        }
      }

      // Vector between boundary nodes
      v2[0] = x[bnode_id[1]] - x[bnode_id[0]];
      v2[1] = x[bnode_id[1] + 1] - x[bnode_id[0] + 1];
      abs_v2 = std::sqrt(v2[0] * v2[0] + v2[1] * v2[1]);

      // Evaluate phi^1_i
      v1[0] = x_i[0] - x[bnode_id[0]];
      v1[1] = x_i[1] - x[bnode_id[0] + 1];
      abs_v1 = std::sqrt(v1[0] * v1[0] + v1[1] * v1[1]);

      v1_t_v2 = v1[0] * v2[0] + v1[1] * v2[1];
      theta_min = std::min(theta_min, std::acos(v1_t_v2 / (abs_v1 * abs_v2)));

      // Evaluate phi^2_i
      v1[0] = x_i[0] - x[bnode_id[1]];
      v1[1] = x_i[1] - x[bnode_id[1] + 1];
      abs_v1 = std::sqrt(v1[0] * v1[0] + v1[1] * v1[1]);

      v1_t_v2 = v1[0] * v2[0] + v1[1] * v2[1];
      theta_min = std::min(theta_min, std::acos(-v1_t_v2 / (abs_v1 * abs_v2)));
    }
  }
  else
  {
    // Coordinates stencil-central node
    std::array<double, 6> data_cnodes;
    std::fill(data_cnodes.begin(), data_cnodes.end(), 0);

    base::mdspan_t<double, 2> cnodes(data_cnodes.data(), 3, 2);

    // Minimal angles for different stencils
    std::array<double, 3> phi_min = {M_PI, M_PI, M_PI};

    // Auxilary variables
    std::array<double, 2> v1, v2, v3;
    double abs_v1, abs_v2, abs_v3, v1_t_vn;

    std::int32_t node_im1, node_i, node_ip1, idn_im1, idn_i, idn_ip1;

    // Get possible center nodes
    if (_ncells % 2 == 0)
    {
      const int ncells_by_two = _ncells / 2;

      // Centroids of the 2 middle cells
      for (int i = 0; i < 2; ++i)
      {
        std::span<const std::int32_t> enodes
            = _cell_to_node->links(_cells[ncells_by_two + i]);

        for (std::int32_t j = 0; j < 3; ++j)
        {
          std::int32_t idn = 3 * enodes[j];

          cnodes(i, 0) += x[idn] / 3;
          cnodes(i, 1) += x[idn + 1] / 3;
        }
      }

      // Mid point of patch-central facet
      std::span<const std::int32_t> enodes
          = _fct_to_node->links(_fcts[ncells_by_two]);

      for (std::int32_t j = 0; j < 2; ++j)
      {
        std::int32_t idn = 3 * enodes[j];

        cnodes(2, 0) += 0.5 * x[idn];
        cnodes(2, 1) += 0.5 * x[idn + 1];
      }
    }
    else
    {
      // Mid points of the 2 middle facets
      const int nfcts_by_two = _nfcts / 2;

      for (int i = 0; i < 2; ++i)
      {
        std::span<const std::int32_t> enodes
            = _fct_to_node->links(_fcts[nfcts_by_two - i]);

        for (std::int32_t j = 0; j < 2; ++j)
        {
          std::int32_t idn = 3 * enodes[j];

          cnodes(i, 0) += 0.5 * x[idn];
          cnodes(i, 1) += 0.5 * x[idn + 1];
        }
      }

      // Centroid of the middle cell
      std::span<const std::int32_t> enodes
          = _cell_to_node->links(_cells[nfcts_by_two]);

      for (std::int32_t j = 0; j < 3; ++j)
      {
        std::int32_t idn = 3 * enodes[j];

        cnodes(2, 0) += x[idn] / 3;
        cnodes(2, 1) += x[idn + 1] / 3;
      }
    }

    // Start stencil at patch-central node
    node_i = node_patch;
    idn_i = 3 * node_i;

    std::span<const std::int32_t> enodes = _fct_to_node->links(_fcts[_ncells]);
    node_im1 = (enodes[0] == node_i) ? enodes[1] : enodes[0];
    idn_im1 = 3 * node_im1;

    v2[0] = x[idn_im1] - x[idn_i];
    v2[1] = x[idn_im1 + 1] - x[idn_i + 1];
    abs_v2 = std::sqrt(v2[0] * v2[0] + v2[1] * v2[1]);

    // Loop over stencil
    for (std::size_t i = 0; i < _nfcts; ++i)
    {
      // Evaluate v3
      std::span<const std::int32_t> enodes = _fct_to_node->links(_fcts[i]);
      node_ip1 = (enodes[0] == node_patch) ? enodes[1] : enodes[0];
      idn_ip1 = 3 * node_ip1;

      v3[0] = x[idn_ip1] - x[idn_i];
      v3[1] = x[idn_ip1 + 1] - x[idn_i + 1];
      abs_v3 = std::sqrt(v3[0] * v3[0] + v3[1] * v3[1]);

      // Evaluate angles for different stencil centers
      for (std::size_t j = 0; j < 3; ++j)
      {
        // Evaluate v1
        v1[0] = cnodes(j, 0) - x[idn_i];
        v1[1] = cnodes(j, 1) - x[idn_i + 1];
        abs_v1 = std::sqrt(v1[0] * v1[0] + v1[1] * v1[1]);

        // Evaluate angles
        v1_t_vn = v1[0] * v2[0] + v1[1] * v2[1];
        phi_min[j]
            = std::min(phi_min[j], std::acos(v1_t_vn / (abs_v1 * abs_v2)));

        v1_t_vn = v1[0] * v3[0] + v1[1] * v3[1];
        phi_min[j]
            = std::min(phi_min[j], std::acos(v1_t_vn / (abs_v1 * abs_v3)));
      }

      // Update for next node
      node_i = node_ip1;
      idn_i = idn_ip1;
      v2[0] = -v3[0];
      v2[1] = -v3[1];
      abs_v2 = abs_v3;
    }

    // Get theta_min
    theta_min = 0.0;

    for (double phi : phi_min)
    {
      theta_min = std::max(theta_min, phi);
    }
  }

  return 2 * std::pow(std::sin(theta_min / 2), -2);
}

// --- Protected methods
void OrientedPatch::set_max_patch_size(
    const int nnodes_proc, std::span<const std::int8_t> pnts_on_bndr,
    const int ncells_min, const int ncells_crit)
{
  // Initialization
  _ncells_max = 0;
  _groupsize_max = 1;

  if (ncells_crit == 1)
  {
    for (std::size_t i_node = 0; i_node < nnodes_proc; ++i_node)
    {
      // Get number of cells on patch
      int n_cells = _node_to_cell->links(i_node).size();

      // Check patch size
      if (n_cells == ncells_min)
      {
        std::string error_msg = "Patch around node " + std::to_string(i_node)
                                + " has only " + std::to_string(ncells_min)
                                + " cells.";
        throw std::runtime_error(error_msg);
      }
      else
      {
        if (n_cells > _ncells_max)
        {
          _ncells_max = n_cells;
        }
      }
    }
  }
  else
  {
    // Loop over all nodes
    for (std::size_t node_i = 0; node_i < nnodes_proc; ++node_i)
    {
      // Get number of cells on patch
      int n_cells = _node_to_cell->links(node_i).size();

      // Check patch size
      if (n_cells == ncells_min)
      {
        std::string error_msg = "Patch around node " + std::to_string(node_i)
                                + " has only " + std::to_string(ncells_min)
                                + " cells.";
        throw std::runtime_error(error_msg);
      }
      else
      {
        if (n_cells > _ncells_max)
        {
          _ncells_max = n_cells;
        }
      }

      // Check if patches have to be grouped
      std::vector<std::int32_t> grouped_bndr_patches
          = group_boundary_patches(node_i, pnts_on_bndr, ncells_crit);
      int groupsize = grouped_bndr_patches.size() - 1;

      if (groupsize > _groupsize_max)
      {
        _groupsize_max = groupsize;
      }
    }
  }
}

void OrientedPatch::initialize_patch(const int node_i)
{
  // Set central node
  _nodei = node_i;

  /* Get cells/facet on current patch */
  std::span<const std::int32_t> cells = _node_to_cell->links(node_i);
  std::span<const std::int32_t> fcts = _node_to_fct->links(node_i);

  _ncells = cells.size();
  _nfcts = fcts.size();

  // Creat sorted list of facets
  std::span<std::int32_t> fcts_sorted(_fcts_sorted.data(), _nfcts);
  std::copy(fcts.begin(), fcts.end(), _fcts_sorted.begin());
  std::sort(_fcts_sorted.begin(), std::next(_fcts_sorted.begin(), _nfcts));

  /* Initialise patch type */
  std::fill(_type.begin(), _type.end(), base::PatchType::internal);
  _equal_patches = true;

  std::int32_t fct_first = fcts[0];
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

  // Store first facet
  if (is_internal())
  {
    _fcts[1] = fct_first;
  }
  else
  {
    _fcts[0] = fct_first;
  }

  /* Structure sub-mesh */
  // Extract data
  base::PatchType type = _type[0];
  base::mdspan_t<std::int8_t, 2> fcts_local(_fcts_local.data(), _nfcts, 2);

  // Initialisation
  int lloop = _ncells + 1;

  if (type == base::PatchType::internal)
  {
    _cells[1] = _fct_to_cell->links(_fcts[1])[1];
  }
  else
  {
    // Set cell a=1
    _cells[1] = _fct_to_cell->links(_fcts[0])[0];

    // Get local facet ids
    std::int8_t lfct
        = get_fctid_local(fct_first, _cell_to_fct->links(_cells[1]));
    fcts_local(0, 0) = lfct;
    fcts_local(0, 1) = lfct;

    // Get next facet
    _fcts[1] = next_facet(_cells[1], _cell_to_fct->links(_cells[1]), lfct);

    // Adjust loop counter
    lloop = _ncells;
  }

  // Loop over internal facets
  for (std::size_t a = 1; a < lloop; ++a)
  {
    std::size_t ap1 = a + 1;
    std::int32_t fct_a = _fcts[a], cell_a = _cells[a];

    // Get cells adjacent to facet E_a
    std::span<const std::int32_t> cells_fct = _fct_to_cell->links(fct_a);

    // Set next cell T_(a+1) on patch
    std::int32_t cell_ap1
        = (cells_fct[0] == cell_a) ? cells_fct[1] : cells_fct[0];
    _cells[ap1] = cell_ap1;

    // Get local facet IDs
    std::span<const std::int32_t> fcts_cell_ap1 = _cell_to_fct->links(cell_ap1);
    std::int8_t lfct_cell_ap1 = get_fctid_local(fct_a, fcts_cell_ap1);

    fcts_local(a, 0) = get_fctid_local(fct_a, _cell_to_fct->links(cell_a));
    fcts_local(a, 1) = lfct_cell_ap1;

    // Get local node ID on patch-central node
    _inodes_local[a] = node_local(cell_a, _nodei);

    // Determine next facet
    _fcts[ap1] = next_facet(cell_ap1, fcts_cell_ap1, lfct_cell_ap1);
  }

  // Complete Definition of patch
  if (is_on_boundary())
  {
    // Get local node ID on patch-central node
    _inodes_local[_ncells] = node_local(_cells[_ncells], _nodei);

    // Get local facet IDs
    std::int8_t lfct_cell
        = get_fctid_local(_fcts[_ncells], _cell_to_fct->links(_cells[_ncells]));

    fcts_local(_ncells, 0) = lfct_cell;
    fcts_local(_ncells, 1) = lfct_cell;
  }
  else
  {
    // Set cell 0 and cell n+1
    _cells[0] = _cells[_ncells];
    _cells[_ncells + 1] = _cells[1];

    // Set ID of patch-central node in cell 0 and cell n+1
    _inodes_local[0] = _inodes_local[_ncells];
    _inodes_local[_ncells + 1] = _inodes_local[1];

    // Set facet 0
    _fcts[0] = _fcts[_nfcts];

    // Local facets
    _fcts_local[0] = _fcts_local[2 * _nfcts];
    _fcts_local[1] = _fcts_local[2 * _nfcts + 1];
  }
}

std::int8_t OrientedPatch::get_fctid_local(std::int32_t fct_i,
                                           std::int32_t cell_i) const
{
  // Get facets on cell
  std::span<const std::int32_t> fct_cell_i = _cell_to_fct->links(cell_i);

  return get_fctid_local(fct_i, fct_cell_i);
}

std::int8_t
OrientedPatch::get_fctid_local(std::int32_t fct_i,
                               std::span<const std::int32_t> fct_cell_i) const
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

std::int8_t OrientedPatch::node_local(std::int32_t cell_i,
                                      std::int32_t node_i) const
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

std::int32_t OrientedPatch::next_facet(std::int32_t cell_i,
                                       std::span<const std::int32_t> fct_cell_i,
                                       std::int8_t id_fct_loc) const
{
  // Get remaining facets in correct order
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
    if (fct_es[1] > _fcts_sorted[_nfcts - 1])
    {
      fct_next = fct_es[0];
    }
    else
    {
      // Full search
      if (std::count(_fcts_sorted.begin(),
                     std::next(_fcts_sorted.begin(), _nfcts), fct_es[0]))
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

std::int32_t
OrientedPatch::adjacent_internal_patch(const std::int32_t node_i) const
{
  // Facets on patch
  std::span<const std::int32_t> fcts_patch = _node_to_fct->links(node_i);

  // Adjacent internal patch
  std::int32_t inner_node;

  for (std::int32_t fct : fcts_patch)
  {
    if (_bfct_type(0, fct) == base::PatchFacetType::internal)
    {
      // Nodes on facet
      std::span<const std::int32_t> nodes_fct = _fct_to_node->links(fct);

      // Output patch-central node
      inner_node = (nodes_fct[0] == node_i) ? nodes_fct[1] : nodes_fct[0];
      break;
    }
  }

  return inner_node;
}
