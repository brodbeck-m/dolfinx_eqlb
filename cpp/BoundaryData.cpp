#include "BoundaryData.hpp"

using namespace dolfinx;
using namespace dolfinx_eqlb;

template <typename T>
BoundaryData<T>::BoundaryData(
    int flux_degree,
    std::vector<std::vector<std::shared_ptr<FluxBC<T>>>>& list_bcs,
    std::vector<std::shared_ptr<fem::Function<T>>>& boundary_flux,
    std::shared_ptr<const fem::FunctionSpace> V_flux_hdiv,
    std::shared_ptr<const fem::FunctionSpace> V_flux_l2,
    const std::vector<std::vector<std::int32_t>>& fct_esntbound_prime)
    : _boundary_flux(boundary_flux),
      _quadrature_rule(QuadratureRule(
          V_flux_hdiv->mesh()->topology().cell_type(), 2 * flux_degree,
          V_flux_hdiv->mesh()->geometry().dim() - 1)),
      _kernel_data(
          KernelDataBC<T>(V_flux_hdiv->mesh(),
                          {std::make_shared<QuadratureRule>(_quadrature_rule)},
                          V_flux_hdiv->element()->basix_element(),
                          V_flux_l2->element()->basix_element())),
      _num_rhs(list_bcs.size()), _gdim(V_flux_hdiv->mesh()->geometry().dim()),
      _num_fcts(
          V_flux_hdiv->mesh()->topology().index_map(_gdim - 1)->size_local()),
      _nfcts_per_cell(_kernel_data.nfacets_cell()), _V_flux_hdiv(V_flux_hdiv),
      _flux_is_discontinous(
          V_flux_hdiv->element()->basix_element().discontinuous()),
      _flux_degree(V_flux_hdiv->element()->basix_element().degree()),
      _ndofs_per_cell(V_flux_hdiv->element()->space_dimension()),
      _ndofs_per_fct(_gdim == 2 ? _flux_degree
                                : 0.5 * _flux_degree * (_flux_degree + 1)),
      _num_dofs(V_flux_hdiv->dofmap()->index_map->size_local()
                + V_flux_hdiv->dofmap()->index_map->num_ghosts())
{
  // Resize storage
  _num_bcfcts.resize(_num_rhs);

  std::int32_t size_dofdata = _num_rhs * _num_dofs;
  _boundary_values.resize(size_dofdata, 0.0);
  _boundary_markers.resize(size_dofdata, false);
  _offset_dofdata.resize(_num_rhs + 1);

  std::int32_t size_fctdata = _num_rhs * _num_fcts;
  _local_fct_id.resize(size_fctdata);
  _facet_type.resize(size_fctdata, facte_type::internal);
  _offset_fctdata.resize(_num_rhs + 1);

  // Set offsets
  std::int32_t num_dofs = _num_dofs;
  std::generate(_offset_dofdata.begin(), _offset_dofdata.end(),
                [n = 0, num_dofs]() mutable { return num_dofs * (n++); });

  std::int32_t num_fcts = _num_fcts;
  std::generate(_offset_fctdata.begin(), _offset_fctdata.end(),
                [n = 0, num_fcts]() mutable { return num_fcts * (n++); });

  // Extract required data
  std::shared_ptr<const mesh::Mesh> mesh = V_flux_hdiv->mesh();

  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> fct_to_cell
      = mesh->topology().connectivity(_gdim - 1, _gdim);
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> cell_to_fct
      = mesh->topology().connectivity(_gdim, _gdim - 1);

  const graph::AdjacencyList<std::int32_t>& dofmap_hdiv
      = V_flux_hdiv->dofmap()->list();
  const fem::ElementDofLayout& doflayout_hdiv
      = V_flux_hdiv->dofmap()->element_dof_layout();

  std::vector<std::int32_t> boundary_dofs_fct(_ndofs_per_fct);

  // Loop over all facets
  for (int i_rhs = 0; i_rhs < _num_rhs; ++i_rhs)
  {
    // Storage of current RHS
    std::span<std::int8_t> facet_type_i = facet_type(i_rhs);
    std::span<std::int8_t> local_fct_id_i = local_facet_id(i_rhs);
    std::span<std::int8_t> boundary_markers_i = boundary_markers(i_rhs);

    std::span<T> x_bvals = _boundary_flux[i_rhs]->x()->mutable_array();

    // Handle facets with essential BCs on primal problem
    for (std::int32_t fct : fct_esntbound_prime[i_rhs])
    {
      // Set facet type
      facet_type_i[fct] = facte_type::essnt_primal;
    }

    // Handle facets with essential BCs on dual problem
    for (std::shared_ptr<FluxBC<T>> bc : list_bcs[i_rhs])
    {
      // Set number of boundary facets
      _num_bcfcts[i_rhs] += bc->num_facets();

      // Loop over all facets
      for (std::int32_t fct = 0; fct < _num_fcts; ++fct)
      {
        // Get cell adjacent to facet
        std::int32_t cell = fct_to_cell->links(fct)[0];

        // Get cell-local facet id
        std::span<const std::int32_t> fcts_cell = cell_to_fct->links(cell);
        std::size_t fct_loc
            = std::distance(fcts_cell.begin(),
                            std::find(fcts_cell.begin(), fcts_cell.end(), fct));

        // Get ids of boundary DOFs
        boundary_dofs(cell, fct_loc, boundary_dofs_fct);

        // Calculate boundary DOFs
        if (bc->projection_required())
        {
          throw std::runtime_error(
              "Projection for boundary conditions not implemented");
        }
        else
        {
          throw std::runtime_error(
              "Evaluation of boundary conditions not implemented");
        }

        // Set values and markers
        facet_type_i[fct] = facte_type::essnt_dual;
        local_fct_id_i[fct] = fct_loc;

        for (std::size_t i = 0; i < _ndofs_per_fct; ++i)
        {
          boundary_markers_i[boundary_dofs_fct[i]] = true;
          x_bvals[boundary_dofs_fct[i]] = 0;
        }
      }
    }
  }
}

template <typename T>
void BoundaryData<T>::boundary_dofs(std::int32_t cell, std::int8_t fct_loc,
                                    std::span<std::int32_t> boundary_dofs)
{
  if (_flux_is_discontinous)
  {
    // Get offset of facet DOFs in current cell
    const int offs = cell * _ndofs_per_cell + fct_loc * _ndofs_per_fct;

    // Calculate DOFs on boundary facet
    for (std::size_t i = 0; i < _ndofs_per_fct; ++i)
    {
      boundary_dofs[i] = offs + i;
    }
  }
  else
  {
    // Extract DOFs of current cell
    std::span<const std::int32_t> dofs_cell
        = _V_flux_hdiv->dofmap()->list().links(cell);

    // Local DOF-ids on facet
    const std::vector<int>& entity_dofs
        = _V_flux_hdiv->dofmap()->element_dof_layout().entity_dofs(_gdim - 1,
                                                                   fct_loc);

    // Calculate DOFs on boundary facet
    for (std::size_t i = 0; i < _ndofs_per_fct; ++i)
    {
      boundary_dofs[i] = dofs_cell[entity_dofs[i]];
    }
  }
}

// ------------------------------------------------------------------------------
template class BoundaryData<float>;
template class BoundaryData<double>;
// ------------------------------------------------------------------------------
