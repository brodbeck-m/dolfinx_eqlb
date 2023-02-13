#include <basix/finite-element.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <iostream>
#include <reconstruction.hpp>
#include <span>
#include <stdexcept>

namespace dolfinx_eqlb
{

/// Determine the next facet on a patch formed by triangular elements
///
/// Triangle has three facets. Determine the two facets not already used.
/// Check if smaller or larger facet is outside the range spanned by the
/// facets of the patch. If not, conduct full search in the list of patch
/// facets.
///
/// @param id_fct_i (loacl facet id on current cell)
/// @param fct_elmt (list of factes (global ID) of current cell)
/// @param fct_patch (list of factes (global ID) forming the patch, sorted!)
/// @return next facet
std::int32_t next_facet_triangle(std::int8_t id_fct_i,
                                 std::span<const std::int32_t>& fct_elmt,
                                 std::vector<std::int32_t>& fct_patch)
{
  // Get remaining factes in correct order
  std::vector<std::int32_t> fct_es(2);
  std::int32_t fct_next;

  switch (id_fct_i)
  {
  case 0:
    if (fct_elmt[1] < fct_elmt[2])
    {
      fct_es[0] = fct_elmt[1];
      fct_es[1] = fct_elmt[2];
    }
    else
    {
      fct_es[0] = fct_elmt[2];
      fct_es[1] = fct_elmt[1];
    }
    break;
  case 1:
    if (fct_elmt[0] < fct_elmt[2])
    {
      fct_es[0] = fct_elmt[0];
      fct_es[1] = fct_elmt[2];
    }
    else
    {
      fct_es[0] = fct_elmt[2];
      fct_es[1] = fct_elmt[0];
    }
    break;
  case 2:
    if (fct_elmt[0] < fct_elmt[1])
    {
      fct_es[0] = fct_elmt[0];
      fct_es[1] = fct_elmt[1];
    }
    else
    {
      fct_es[0] = fct_elmt[1];
      fct_es[1] = fct_elmt[0];
    }
    break;
  default:
    assert(id_fct_i < 2);
  }

  // Get next facet
  if (fct_es[0] < fct_patch[0])
  {
    fct_next = fct_es[1];
  }
  else
  {
    if (fct_es[1] > fct_patch.back())
    {
      fct_next = fct_es[0];
    }
    else
    {
      // Full search
      if (std::count(fct_patch.begin(), fct_patch.end(), fct_es[0]))
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

std::tuple<int, std::vector<std::int32_t>> submap_equilibration_patch(
    int i_node, std::shared_ptr<const fem::FunctionSpace> function_space,
    const std::vector<std::vector<std::vector<int>>>& entity_dofs0,
    const std::vector<std::vector<std::vector<int>>>& entity_dofs1,
    const std::vector<std::int8_t>& fct_type)
{
  /* Geometry data */
  // Mesh
  std::shared_ptr<const mesh::Mesh> mesh = function_space->mesh();
  const mesh::Topology& topology = mesh->topology();

  // Spacial dimensions
  int dim = mesh->geometry().dim();
  int dim_fct = dim - 1;

  // Connectivities
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> node_to_cell
      = topology.connectivity(0, dim);
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> node_to_fct
      = topology.connectivity(0, dim_fct);
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> fct_to_cell
      = topology.connectivity(dim_fct, dim);
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> cell_to_fct
      = topology.connectivity(dim, dim_fct);

  // Counters
  int fct_per_cell = cell_to_fct->links(0).size();

  // Extract facets/cells patch
  std::span<const std::int32_t> fct_patch_aux = node_to_fct->links(i_node);
  const int n_fct_patch = fct_patch_aux.size();
  std::vector<std::int32_t> fct_patch(n_fct_patch);
  std::copy(fct_patch_aux.begin(), fct_patch_aux.end(), fct_patch.begin());
  std::sort(fct_patch.begin(), fct_patch.end());
  std::span<const std::int32_t> cell_patch = node_to_cell->links(i_node);
  const int n_cell_patch = cell_patch.size();

  /* Informations about FunctionSpace */
  // // BasiX elements of subspaces
  // std::vector<int> sub0(1, 0);
  // std::vector<int> sub1(1, 1);
  // const basix::FiniteElement& basix_element0
  //     = function_space->sub(sub0)->element()->basix_element();
  // const basix::FiniteElement& basix_element1
  //     = function_space->sub(sub1)->element()->basix_element();

  // Local DOFmap facets (sorted by entity)
  // const std::vector<std::vector<std::vector<int>>>& entity_dofs0
  //     = basix_element0.entity_dofs();
  // const std::vector<std::vector<std::vector<int>>>& entity_dofs1
  //     = basix_element1.entity_dofs();

  // DOF counters
  const int ndof_flux_fct = entity_dofs0[dim_fct][0].size();
  const int ndof_flux_cell = entity_dofs0[dim][0].size();
  const int ndof_cons_cell = entity_dofs1[dim][0].size();
  const int ndof_cell = ndof_flux_cell + ndof_cons_cell;
  const int ndof_flux_elmt = 2 * ndof_flux_fct + ndof_flux_cell;
  const int ndof_elmt = 2 * ndof_flux_fct + ndof_cell;

  /* Determine type of patch */
  // Initialize patch marker
  int type_patch = 0;

  // Initialize counters (on current patch)
  std::int32_t fct_i = fct_patch_aux[0];
  std::int32_t cell_i = 0;
  std::int32_t c_fct_loop = n_fct_patch;

  // Set type of patch (0->internal, 1->neumann, 2->dirichlet, 3->mixed_bound)
  if (n_fct_patch > n_cell_patch)
  {
    std::int32_t fct_ef = -1;
    std::int32_t fct_ep = -1;

    // Check for boundary facets (id=1->esnt_prime, id=2, esnt_flux)
    for (std::int32_t id_fct : fct_patch)
    {
      if (fct_type[id_fct] == 1)
      {
        // Mark first facet for DOFmap construction
        fct_ep = id_fct;
      }
      else if (fct_type[id_fct] == 2)
      {
        // Mark first facet for DOFmap construction
        fct_ef = id_fct;
      }
    }

    // Set patch type
    if (fct_ef < 0)
    {
      type_patch = 2;

      // Start patch construction on dirichlet facet
      fct_i = fct_ep;
    }
    else
    {
      type_patch = (fct_ep < 0) ? 1 : 3;

      // Start patch construction on neumann facet
      fct_i = fct_ef;
    }

    // Set counter for loop over facets
    c_fct_loop = n_fct_patch - 1;
  }

  /* Create DOFmaps (local, element-wise) for patch */
  // Initialize data for adjacency-lists
  const int len_adjacency = n_cell_patch * ndof_elmt;
  std::vector<std::int32_t> data_adjacency_elmt(len_adjacency),
      data_adjacency_patch(len_adjacency), adjacency_offset(n_cell_patch + 1),
      cells_patch(n_cell_patch);

  adjacency_offset[0] = 0;

  // Initialize cell adjacent to fct_i
  cell_i = -1;

  // Loop over facets
  std::int32_t dof_patch = 0;

  for (std::size_t ii = 0; ii < c_fct_loop; ++ii)
  {
    // Get cell
    std::span<const std::int32_t> cell_fct_i = fct_to_cell->links(fct_i);
    cell_i = (cell_fct_i[0] == cell_i) ? cell_fct_i[1] : cell_fct_i[0];

    // Get facets of cell_i
    std::span<const std::int32_t> fct_cell_i = cell_to_fct->links(cell_i);

    // Get id (cell_local) of fct_i
    std::int8_t id_facet_loc = 0;

    while (fct_cell_i[id_facet_loc] != fct_i)
    {
      id_facet_loc += 1;
    }

    // Offset first DOF (flus DOFs on facet 1) on elmt_i
    std::int32_t offs_p = (ii + 1) * ndof_elmt;
    adjacency_offset[ii + 1] = offs_p;

    // Offset flux DOFs ond second facet elmt_(i-1)
    std::int32_t offs_f;

    if (type_patch > 0)
    {
      offs_f = (ii == 0) ? ndof_flux_fct
                         : adjacency_offset[ii - 1] + ndof_flux_fct;
      offs_p = adjacency_offset[ii];
      cells_patch[ii] = cell_i;
    }
    else
    {
      if (ii < n_fct_patch - 1)
      {
        offs_f = adjacency_offset[ii] + ndof_flux_fct;
        cells_patch[ii + 1] = cell_i;
      }
      else
      {
        offs_f = adjacency_offset[ii] + ndof_flux_fct;
        offs_p = 0;
        cells_patch[0] = cell_i;
      }
    }

    // Offset of DOF n on elmt_i
    std::int32_t offs_e = offs_p;

    // Get flux-DOFs on fct_i
    for (std::int8_t jj = 0; jj < ndof_flux_fct; ++jj)
    {
      // Local index
      int dofl_fct = entity_dofs0[dim_fct][id_facet_loc][jj];

      // Add cell-local DOFs
      data_adjacency_elmt[offs_p] = dofl_fct;
      data_adjacency_elmt[offs_f + jj] = dofl_fct;

      // Calculate patch-local DOFs
      data_adjacency_patch[offs_p] = dof_patch;
      data_adjacency_patch[offs_f + jj] = dof_patch;

      // Increment id of patch-local DOFs
      dof_patch += 1;
      offs_p += 1;
    }

    // Get flux-DOFs on cell_i
    offs_p += ndof_flux_fct;

    for (std::int8_t jj = 0; jj < ndof_flux_cell; ++jj)
    {
      // Local index
      int dofl_cell = entity_dofs0[dim][0][jj];

      // Add cell-local DOFs
      data_adjacency_elmt[offs_p] = dofl_cell;

      // Calculate patch-local DOFs
      data_adjacency_patch[offs_p] = dof_patch;

      // Increment id of patch-local DOFs
      dof_patch += 1;
      offs_p += 1;
    }

    // Get constraining-DOFs on cell_i
    for (std::int8_t jj = 0; jj < ndof_cons_cell; ++jj)
    {
      // Local index
      int dofl_cell = entity_dofs1[dim][0][jj];

      // Add cell-local DOFs
      data_adjacency_elmt[offs_p] = dofl_cell;

      // Calculate patch-local DOFs
      data_adjacency_patch[offs_p] = dof_patch;

      // Increment id of patch-local DOFs
      dof_patch += 1;
      offs_p += 1;
    }

    // Get next cell and facet
    fct_i = next_facet_triangle(id_facet_loc, fct_cell_i, fct_patch);
  }

  // Handle last facet (boundary patches)
  if (type_patch > 0)
  {
    // Get facets of cell_i
    std::span<const std::int32_t> fct_cell_i = cell_to_fct->links(cell_i);

    // Get last facet on patch
    std::int8_t id_facet_loc = 0;

    while (fct_cell_i[id_facet_loc] != fct_i)
    {
      id_facet_loc += 1;
    }

    // Add DOFs to DOFmap
    std::int32_t offs_p = (n_cell_patch - 1) * ndof_elmt + ndof_flux_fct;

    for (std::int8_t jj = 0; jj < ndof_flux_fct; ++jj)
    {
      // Local index
      int dofl_fct = entity_dofs0[dim_fct][id_facet_loc][jj];

      // Add cell-local DOFs
      data_adjacency_elmt[offs_p] = dofl_fct;

      // Calculate patch-local DOFs
      data_adjacency_patch[offs_p] = dof_patch;

      // Increment id of patch-local DOFs
      dof_patch += 1;
      offs_p += 1;
    }
  }

  // Create copy from adjacency_offset
  std::vector<std::int32_t> copy_adjacency_offset(adjacency_offset);

  // Create adjacency lists
  graph::AdjacencyList<std::int32_t> lpatch_local
      = graph::AdjacencyList<std::int32_t>(std::move(data_adjacency_elmt),
                                           std::move(copy_adjacency_offset));
  graph::AdjacencyList<std::int32_t> lpatch_dofmap
      = graph::AdjacencyList<std::int32_t>(std::move(data_adjacency_patch),
                                           std::move(adjacency_offset));

  // return {type_patch, std::move(cells_patch)};
  return {0, std::move(cells_patch)};
}
} // namespace dolfinx_eqlb