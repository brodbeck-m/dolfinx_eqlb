#include "BoundaryData.hpp"

using namespace dolfinx;
using namespace dolfinx_eqlb;

template <typename T>
BoundaryData<T>::BoundaryData(
    int flux_degree,
    const std::vector<std::vector<std::shared_ptr<const FluxBC<T>>>>& bcs_flux,
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

  setup_boundary_data(flux_degree, mesh, fct_esntbound_prime, belmt_flux_proj);
}

template <typename T>
BoundaryData<T>::BoundaryData(
    int flux_degree,
    const std::vector<std::vector<std::shared_ptr<const FluxBC<T>>>>& bcs_flux,
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
  setup_boundary_data(flux_degree, mesh, fct_esntbound_prime, belmt_flux_proj);
}

template <typename T>
void BoundaryData<T>::setup_boundary_data(
    int flux_degree, std::shared_ptr<const mesh::Mesh> mesh,
    const std::vector<std::vector<std::int32_t>>& fct_esntbound_prime,
    const basix::FiniteElement& belmt_flux_proj)
{
  /* Check Input */
  bool _projection_required = false;
  std::shared_ptr<const fem::FunctionSpace> function_space_flux;

  for (auto list_bc_rhsi : _flux_bcs)
  {
    for (auto bc : list_bc_rhsi)
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

  // The flux degree
  const int degree_flux
      = function_space_flux->element()->basix_element().degree();

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
  if (_projection_required)
  {
    std::array<std::size_t, 5> shape_flux = _kernel_data.tabulate_basis(
        belmt_flux_proj, _ipoints, _basis_bc_values, false, false);

    _basis_bc = mdspan_t<const double, 5>(_basis_bc_values.data(), shape_flux);
  }

  /* Initialise boundary information */
}

// ------------------------------------------------------------------------------
template class BoundaryData<float>;
template class BoundaryData<double>;
// ------------------------------------------------------------------------------
