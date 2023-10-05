#include "Patch.hpp"

using namespace dolfinx;
using namespace dolfinx_eqlb;

Patch::Patch(int nnodes_proc, std::shared_ptr<const mesh::Mesh> mesh,
             mdspan_t<const std::int8_t, 2> bfct_type)
    : _mesh(mesh), _bfct_type(bfct_type), _dim(mesh->geometry().dim()),
      _dim_fct(mesh->geometry().dim() - 1),
      _type(bfct_type.extent(0), PatchType::internal),
      _npatches(bfct_type.extent(0))
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

bool Patch::recreate_subdofmap(int index)
{
  if (index == 0)
  {
    // Patch was cerated for index == 0 --> Do nothing
    return false;
  }
  else
  {
    if (_equal_patches)
    {
      // All RHS have equal patch-types --> Do nothing
      return false;
    }
    else if (_type[index - 1] == _type[index])
    {
      // Two sequential RHS have equal patch-types --> Do nothing
      return false;
    }
    else if (requires_flux_bcs(index - 1) && requires_flux_bcs(index))
    {
      // Two sequential RHS require flux BCs --> Do nothing
      return false;
    }
    else if ((requires_flux_bcs(index - 1))
             && (_type[index] == PatchType::bound_essnt_primal))
    {
      // Switch to patch with primal BCs (orientation arbitrary) --> Do nothing
      return false;
    }
    else
    {
      // Check if facet[0] has flux BCs
      if (_bfct_type(index, _fcts[0]) == PatchFacetType::essnt_dual)
      {
        // facet zero requires flux BCs --> Do nothing
        return false;
      }
      else
      {
        reverse_patch_orientation();

        return true;
      }
    }
  }
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
  std::fill(_type.begin(), _type.end(), PatchType::internal);
  _equal_patches = true;

  // Initialize first facet
  std::int32_t fct_first = fcts[0];

  // Initialize loop over facets
  std::int32_t c_fct_loop = _ncells;

  if (_nfcts > _ncells)
  {
    // Determine patch types
    for (int i = _npatches - 1; i >= 0; --i)
    {
      // Initializations
      std::int32_t fct_ef = -1;
      std::int32_t fct_ep = -1;

      // Check for boundary facets (id=1->esnt_prime, id=2, esnt_flux)
      for (std::int32_t id_fct : fcts)
      {
        if (_bfct_type(i, id_fct) == PatchFacetType::essnt_primal)
        {
          // Mark first facet for DOFmap construction
          fct_ep = id_fct;
        }
        else if (_bfct_type(i, id_fct) == PatchFacetType::essnt_dual)
        {
          // Mark first facet for DOFmap construction
          fct_ef = id_fct;
        }
      }

      // Set patch type
      if (fct_ef < 0)
      {
        _type[i] = PatchType::bound_essnt_primal;

        // Start patch construction on dirichlet facet
        fct_first = fct_ep;
      }
      else
      {
        if (fct_ep < 0)
        {
          _type[i] = PatchType::bound_essnt_dual;
        }
        else
        {
          _type[i] = PatchType::bound_mixed;
        }

        // Start patch construction on neumann facet
        fct_first = fct_ef;
      }
    }

    // Check if all patches have the same type
    if (std::adjacent_find(_type.begin(), _type.end(), std::not_equal_to<>())
        == _type.end())
    {
      _equal_patches = false;
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
  if (_type[id_l] != PatchType::internal && c_fct == 0)
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
  if (_type[id_l] != PatchType::internal)
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

  // Initialize local id
  std::int8_t fct_loc;

  // Get id (cell_local) of fct_i
  while (fct_cell_i[fct_loc] != fct_i && fct_loc < _fct_per_cell)
  {
    fct_loc += 1;
  }

  // Check for face not on cell
  assert(fct_loc > _fct_per_cell - 1);

  return fct_loc;
}

std::int8_t Patch::get_fctid_local(std::int32_t fct_i,
                                   std::span<const std::int32_t> fct_cell_i)
{
  // Initialize local id
  std::int8_t fct_loc;

  // Get id (cell_local) of fct_i
  while (fct_cell_i[fct_loc] != fct_i && fct_loc < _fct_per_cell)
  {
    fct_loc += 1;
  }

  // Check for face not on cell
  assert(fct_loc > _fct_per_cell - 1);

  return fct_loc;
}

std::int8_t Patch::nodei_local(std::int32_t cell_i)
{
  // Initialize cell-local node-id
  std::int8_t id_node_loc_ci = 0;

  // Get nodes on cell_i
  std::span<const std::int32_t> node_cell_i = _cell_to_node->links(cell_i);

  while (node_cell_i[id_node_loc_ci] != _nodei)
  {
    id_node_loc_ci += 1;
  }

  return id_node_loc_ci;
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