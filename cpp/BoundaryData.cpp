#include "BoundaryData.hpp"

using namespace dolfinx;
using namespace dolfinx_eqlb;

template <typename T>
BoundaryData<T>::BoundaryData(
    int flux_degree,
    std::vector<std::vector<std::shared_ptr<FluxBC<T>>>>& bcs_flux,
    std::shared_ptr<const mesh::Mesh> mesh,
    const std::vector<std::vector<std::int32_t>>& fct_esntbound_prime)
    : _flux_bcs(bcs_flux),
      _quadrature_rule(QuadratureRule(mesh->topology().cell_type(),
                                      2 * flux_degree,
                                      mesh->geometry().dim() - 1)),
      _kernel_data(KernelData<T>(
          mesh, {std::make_shared<QuadratureRule>(_quadrature_rule)})),
      _belmt_hat(basix::element::create_lagrange(
          mesh::cell_type_to_basix_type(mesh->topology().cell_type()), 1,
          basix::element::lagrange_variant::equispaced, false))

{
  // Create Basix element of projected flux
  basix::FiniteElement belmt_flux_proj = basix::element::create_lagrange(
      mesh::cell_type_to_basix_type(mesh->topology().cell_type()), 1,
      basix::element::lagrange_variant::equispaced, false);

  initialise_boundary_data(flux_degree, mesh, fct_esntbound_prime,
                           belmt_flux_proj);
}

template <typename T>
BoundaryData<T>::BoundaryData(
    int flux_degree,
    std::vector<std::vector<std::shared_ptr<FluxBC<T>>>>& bcs_flux,
    std::shared_ptr<const mesh::Mesh> mesh,
    const std::vector<std::vector<std::int32_t>>& fct_esntbound_prime,
    const basix::FiniteElement& belmt_flux_proj)
    : _flux_bcs(bcs_flux),
      _quadrature_rule(QuadratureRule(mesh->topology().cell_type(),
                                      2 * flux_degree,
                                      mesh->geometry().dim() - 1)),
      _kernel_data(KernelData<T>(
          mesh, {std::make_shared<QuadratureRule>(_quadrature_rule)})),
      _belmt_hat(basix::element::create_lagrange(
          mesh::cell_type_to_basix_type(mesh->topology().cell_type()), 1,
          basix::element::lagrange_variant::equispaced, false))

{
  initialise_boundary_data(flux_degree, mesh, fct_esntbound_prime,
                           belmt_flux_proj);
}

template <typename T>
void BoundaryData<T>::initialise_boundary_data(
    int flux_degree, std::shared_ptr<const mesh::Mesh> mesh,
    const std::vector<std::vector<std::int32_t>>& fct_esntbound_prime,
    const basix::FiniteElement& belmt_flux_proj)
{
  /* Check Input */
  _projection_required = false;
  _num_rhs = _flux_bcs.size();

  std::shared_ptr<const fem::FunctionSpace> function_space_flux;

  for (std::vector<std::shared_ptr<FluxBC<T>>> list_bc_rhsi : _flux_bcs)
  {
    for (std::shared_ptr<FluxBC<T>> bc : list_bc_rhsi)
    {
      // Check if bc_rhsi is a valid FluxBC
      if (bc->projection_required())
      {
        // Set projection id
        _projection_required = true;

        // Extract function space
        function_space_flux = bc->function_space();

        break;
      }

      if (_projection_required)
      {
        break;
      }
    }
  }

  /* Extract relevant data */
  // Mesh: Geometry/ Topology
  const mesh::Topology& topology = mesh->topology();
  const mesh::Geometry& geometry = mesh->geometry();

  // Storage
  _flux_degree = function_space_flux->element()->basix_element().degree();
  _gdim = geometry.dim();

  /* Extract interpolation data */
  std::array<std::size_t, 4> shape_intpl = interpolation_data_facet_rt(
      function_space_flux->element()->basix_element(), geometry.dim(),
      _kernel_data.nfacets_cell(), _ipoints, _data_M);

  _M = mdspan_t<const double, 4>(_data_M.data(), shape_intpl);

  std::tie(_num_ipoints_per_fct, _num_ipoints)
      = size_interpolation_data_facet_rt(shape_intpl);

  /* Tabulate hat-function */
  std::array<std::size_t, 5> shape_hat = _kernel_data.tabulate_basis(
      _belmt_hat, _ipoints, _basis_hat_values, false, false);

  _basis_hat = mdspan_t<const double, 5>(_basis_hat_values.data(), shape_hat);

  /* Tabulate projected flux */
  std::vector<double> basis_bc_values_proj;
  mdspan_t<const double, 5> basis_bc_proj;

  if (_projection_required)
  {
    // Tabulate basis at interpolation points
    std::array<std::size_t, 5> shape_flux = _kernel_data.tabulate_basis(
        belmt_flux_proj, _ipoints, _basis_bc_values, false, false);

    _basis_bc = mdspan_t<const double, 5>(_basis_bc_values.data(), shape_flux);

    // Tabulate basis at quadrature points on surfaces
    shape_flux = _kernel_data.tabulate_basis(
        belmt_flux_proj, _kernel_data.quadrature_points_flattened(0),
        basis_bc_values_proj, false, false);
    basis_bc_proj
        = mdspan_t<const double, 5>(basis_bc_values_proj.data(), shape_flux);
  }

  /* Initialise boundary information */
  initialise_boundary_conditions(mesh, fct_esntbound_prime, basis_bc_proj,
                                 function_space_flux->dofmap()->list());
}

template <typename T>
void BoundaryData<T>::initialise_boundary_conditions(
    std::shared_ptr<const mesh::Mesh> mesh,
    const std::vector<std::vector<std::int32_t>>& fct_esntbound_prime,
    mdspan_t<const double, 5> basis_flux,
    const graph::AdjacencyList<std::int32_t>& dofmap_flux)
{
  // Number of facets
  const std::int32_t num_facets
      = mesh->topology().index_map(_gdim - 1)->size_local();

  // Connectivity facet->cell
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> fct_to_cell
      = mesh->topology().connectivity(_gdim - 1, _gdim);

  // Storage of facet data
  const std::int32_t size_fctstrg = num_facets * _num_rhs;

  _data_facet_type.resize(size_fctstrg);
  _data_dof_to_fluxbc.resize(size_fctstrg);
  _data_dof_to_fluxbcid.resize(size_fctstrg);

  // Offset vector
  _offset_facetdata.resize(_num_rhs + 1);
  std::generate(_offset_facetdata.begin(), _offset_facetdata.end(),
                [n = 0, num_facets]() mutable { return num_facets * (n++); });

  /* Perform initialisation */
  for (int i_rhs = 0; i_rhs < _num_rhs; ++i_rhs)
  {
    // Extract data for current rhs
    std::span<std::int8_t> fct_type_i = facet_type(i_rhs);
    std::span<std::int32_t> dof_to_fluxbc_i = dof_to_fluxbc(i_rhs);
    std::span<std::int32_t> dof_to_fluxbcid_i = dof_to_fluxbcid(i_rhs);

    // Mark facets with essential BC on primal problem
    for (std::int32_t fct : fct_esntbound_prime[i_rhs])
    {
      // Set facet type
      fct_type_i[fct] = 1;
    }

    // Handle facets with essential BC on flux
    for (std::int32_t i_bc = 0; i_bc < _flux_bcs[i_rhs].size(); ++i_bc)
    {
      // Get boundary function
      std::shared_ptr<FluxBC<T>> bc = _flux_bcs[i_rhs][i_bc];

      // Add up number of boundary facets
      const std::int32_t num_facets = bc->num_facets();
      _num_fcts_fluxbc[i_rhs] += num_facets;

      // Extract list of boundary facets
      std::span<const std::int32_t> list_fcts = bc->facets();

      // Loop over boundary facets
      for (std::size_t i_fct = 0; i_fct < num_facets; ++i_fct)
      {
        // Global facet id
        std::int32_t fct = list_fcts[i_fct];

        // Cell adjacent to facet
        std::int32_t c = fct_to_cell->links(fct)[0];

        // Extract DOFs on facet
        std::span<const std::int32_t> dofs = dofmap_flux.links(c);

        /* Project boundary data */
        if (bc->projection_required())
        {
          throw std::runtime_error("Projection of BCs not implemented");
        }

        /* Set facet type and connectivity */
        // Set connectivity onto global dof
        // TODO - Adjust for parallel computation --> How is global vector
        // partitioned?

        // TODO - Extract DOFs on facet --> see locate DOFs topological?

        // Set facet type
        if (fct_type_i[i_fct] != 0)
        {
          throw std::runtime_error(
              "Dirichlet- and Neumann BC set on same facet");
        }
        else
        {
          fct_type_i[i_fct] = 2;
        }
      }
    }
  }
}

// ------------------------------------------------------------------------------
template class BoundaryData<float>;
template class BoundaryData<double>;
// ------------------------------------------------------------------------------
