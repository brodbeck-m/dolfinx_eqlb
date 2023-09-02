#include "BoundaryData.hpp"

using namespace dolfinx;
using namespace dolfinx_eqlb;

template <typename T>
BoundaryData<T>::BoundaryData(
    std::vector<std::vector<std::shared_ptr<FluxBC<T>>>>& list_bcs,
    std::vector<std::shared_ptr<fem::Function<T>>>& boundary_flux,
    std::shared_ptr<const fem::FunctionSpace> V_flux_hdiv,
    bool rtflux_is_custom, std::shared_ptr<const fem::FunctionSpace> V_flux_l2,
    const std::vector<std::vector<std::int32_t>>& fct_esntbound_prime)
    : _flux_degree(V_flux_hdiv->element()->basix_element().degree()),
      _boundary_flux(boundary_flux),
      _quadrature_rule(QuadratureRule(
          V_flux_hdiv->mesh()->topology().cell_type(), 2 * _flux_degree,
          V_flux_hdiv->mesh()->geometry().dim() - 1)),
      _kernel_data(KernelDataBC<T>(
          V_flux_hdiv->mesh(),
          {std::make_shared<QuadratureRule>(_quadrature_rule)},
          V_flux_hdiv->element()->basix_element(),
          V_flux_l2->element()->basix_element(), rtflux_is_custom)),
      _num_rhs(list_bcs.size()), _gdim(V_flux_hdiv->mesh()->geometry().dim()),
      _num_fcts(
          V_flux_hdiv->mesh()->topology().index_map(_gdim - 1)->size_local()),
      _nfcts_per_cell((_gdim == 2) ? 3 : 4), _V_flux_hdiv(V_flux_hdiv),
      _flux_is_discontinous(
          V_flux_hdiv->element()->basix_element().discontinuous()),
      _ndofs_per_cell(V_flux_hdiv->element()->space_dimension()),
      _ndofs_per_fct((_gdim == 2) ? _flux_degree
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

  _J = mdspan_t<double, 2>(_data_J.data(), _gdim, _gdim);
  _K = mdspan_t<double, 2>(_data_K.data(), _gdim, _gdim);

  /* Extract required data */
  // The mesh
  std::shared_ptr<const mesh::Mesh> mesh = V_flux_hdiv->mesh();

  // Required conductivities
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> fct_to_cell
      = mesh->topology().connectivity(_gdim - 1, _gdim);
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> cell_to_fct
      = mesh->topology().connectivity(_gdim, _gdim - 1);

  // Geometry DOFmap and nodal coordinates
  const graph::AdjacencyList<std::int32_t>& dofmap_geom
      = mesh->geometry().dofmap();
  std::span<const double> node_coordinates = mesh->geometry().x();

  // Flux DOFmap
  const graph::AdjacencyList<std::int32_t>& dofmap_hdiv
      = V_flux_hdiv->dofmap()->list();
  const fem::ElementDofLayout& doflayout_hdiv
      = V_flux_hdiv->dofmap()->element_dof_layout();

  // Transformation information flux space
  const auto apply_inverse_dof_transform
      = _V_flux_hdiv->element()->get_dof_transformation_function<T>(true, true,
                                                                    false);

  std::span<const std::uint32_t> cell_info;
  if (_V_flux_hdiv->element()->needs_dof_transformations())
  {
    mesh->topology_mutable().create_entity_permutations();
    cell_info = std::span(mesh->topology().get_cell_permutation_info());
  }

  /* Initialise storage */
  // DOF ids and values on boundary facets
  // (values on entire element to enable application of DOF transformations)
  std::vector<std::int32_t> boundary_dofs_fct(_ndofs_per_fct);
  std::vector<T> boundary_values(_ndofs_per_cell);

  // Storage for extraction of cell geometry
  std::vector<double> coordinates_data(mesh->geometry().cmap().dim() * 3);
  mdspan_t<const double, 2> coordinates(coordinates_data.data(),
                                        mesh->geometry().cmap().dim(), 3);

  // Number of interpolation points per facet
  const int nipoints = _kernel_data.num_interpolation_points();
  const int nipoints_per_fct
      = _kernel_data.num_interpolation_points_per_facet();

  // The boundary kernel
  std::vector<T> values_bkernel(nipoints);

  // Constants/coefficients for evaluation of boundary kernel
  std::vector<T> constants_belmts, coefficients_belmts;
  std::int32_t cstride_coefficients;

  /* Calculate boundary DOFs */
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
      // Adjust number of boundary facets
      _num_bcfcts[i_rhs] += bc->num_facets();

      // Extract list of boundary facets
      std::span<const std::int32_t> boundary_fcts = bc->facets();

      // Extract constants and coefficients
      constants_belmts = bc->extract_constants();

      std::tie(cstride_coefficients, coefficients_belmts)
          = bc->extract_coefficients();

      // Extract boundary kernel
      const auto& boundary_kernel = bc->boundary_kernel();

      // Loop over all facets
      for (std::int32_t i_fct = 0; i_fct < bc->num_facets(); ++i_fct)
      {
        // Get facet
        std::int32_t fct = boundary_fcts[i_fct];

        // Get cell adjacent to facet
        std::int32_t cell = fct_to_cell->links(fct)[0];

        // Get cell-local facet id
        std::span<const std::int32_t> fcts_cell = cell_to_fct->links(cell);
        std::size_t fct_loc
            = std::distance(fcts_cell.begin(),
                            std::find(fcts_cell.begin(), fcts_cell.end(), fct));

        // Get ids of boundary DOFs
        boundary_dofs(cell, fct_loc, boundary_dofs_fct);

        /* Calculate boundary values on current cell */
        // Extract cell geometry
        std::span<const std::int32_t> nodes_cell = dofmap_geom.links(cell);
        for (std::size_t j = 0; j < nodes_cell.size(); ++j)
        {
          std::copy_n(std::next(node_coordinates.begin(), 3 * nodes_cell[j]), 3,
                      std::next(coordinates_data.begin(), 3 * j));
        }

        // Extract DOFs on boundary facet
        std::span<T> boundary_values_fct(
            boundary_values.data() + fct_loc * _ndofs_per_fct, _ndofs_per_fct);

        // Calculate mapping tensors
        double detJ
            = _kernel_data.compute_jacobian(_J, _K, _detJ_scratch, coordinates);

        // Calculate boundary DOFs
        if (bc->projection_required())
        {
          throw std::runtime_error(
              "Projection for boundary conditions not implemented");
        }
        else
        {
          // Evaluate boundary condition at interpolation points
          std::fill(values_bkernel.begin(), values_bkernel.end(), 0.0);
          boundary_kernel(values_bkernel.data(),
                          coefficients_belmts.data()
                              + i_fct * cstride_coefficients,
                          constants_belmts.data(), coordinates_data.data(),
                          nullptr, nullptr);

          // Perform interpolation
          std::span<T> flux_ntrace(values_bkernel.data()
                                       + fct_loc * nipoints_per_fct,
                                   nipoints_per_fct);

          _kernel_data.interpolate_flux(flux_ntrace, boundary_values_fct,
                                        fct_loc, _J, detJ, _K);
        }

        // Apply DOF transformations
        apply_inverse_dof_transform(boundary_values, cell_info, cell, 1);

        // Set values and markers
        facet_type_i[fct] = facte_type::essnt_dual;
        local_fct_id_i[fct] = fct_loc;

        for (std::size_t i = 0; i < _ndofs_per_fct; ++i)
        {
          boundary_markers_i[boundary_dofs_fct[i]] = true;
          x_bvals[boundary_dofs_fct[i]] = boundary_values_fct[i];
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
template class BoundaryData<double>;
// ------------------------------------------------------------------------------
