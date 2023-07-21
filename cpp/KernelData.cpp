#include "KernelData.hpp"
#include "QuadratureRule.hpp"

using namespace dolfinx;

namespace dolfinx_adaptivity::equilibration
{
KernelData::KernelData(std::shared_ptr<const mesh::Mesh> mesh,
                       std::shared_ptr<const QuadratureRule> qrule,
                       const basix::FiniteElement& basix_element_fluxpw,
                       const basix::FiniteElement& basix_element_rhs,
                       const basix::FiniteElement& basix_element_hat)
    : _quadrature_rule(qrule)
{
  const mesh::Topology& topology = mesh->topology();
  const mesh::Geometry& geometry = mesh->geometry();
  const fem::CoordinateElement& cmap = geometry.cmap();

  // Check if mesh is affine
  _is_affine = cmap.is_affine();

  if (!_is_affine)
  {
    throw std::runtime_error("Equilibration on no-affine meshes not supported");
  }

  // Set dimensions
  _gdim = geometry.dim();
  _tdim = topology.dim();

  // Tabulate basis function of geometry element
  _num_coordinate_dofs = cmap.dim();

  _g_basis_shape = cmap.tabulate_shape(1, 1);
  _g_basis_values = std::vector<double>(std::reduce(
      _g_basis_shape.begin(), _g_basis_shape.end(), 1, std::multiplies{}));

  std::vector<double> points(_gdim, 0);
  cmap.tabulate(1, points, {1, _gdim}, _g_basis_values);

  // Get facet normals of reference element
  basix::cell::type basix_cell
      = mesh::cell_type_to_basix_type(topology.cell_type());

  std::tie(_fct_normals, _normals_shape)
      = basix::cell::facet_outward_normals(basix_cell);

  _fct_normal_out = basix::cell::facet_orientations(basix_cell);

  // Number of quadrature points
  std::size_t n_qpoints_cell = _quadrature_rule->npoints_cell();
  std::size_t n_qpoints_fct = _quadrature_rule->npoints_fct();

  // Tabulate shape functions of pice-wise H(div) flux space
  std::array<std::size_t, 4> flux_basis_shape
      = basix_element_fluxpw.tabulate_shape(0, n_qpoints_cell);

  _flux_basis_values = std::vector<double>(std::reduce(
      flux_basis_shape.begin(), flux_basis_shape.end(), 1, std::multiplies{}));
  _flux_basis_current_values = std::vector<double>(std::reduce(
      flux_basis_shape.begin(), flux_basis_shape.end(), 1, std::multiplies{}));

  basix_element_fluxpw.tabulate(0, _quadrature_rule->points_cell(),
                                {n_qpoints_cell, _gdim}, _flux_basis_values);

  _flux_fullbasis = dolfinx_adaptivity::cmdspan4_t(_flux_basis_values.data(),
                                                   flux_basis_shape);
  _flux_fullbasis_current = dolfinx_adaptivity::mdspan4_t(
      _flux_basis_current_values.data(), flux_basis_shape);

  // Tabulate shape functions of right-hand side space
  // (Assumption: Same order for projected flux and RHS)
  std::array<std::size_t, 4> rhs_basis_shape_cell
      = basix_element_rhs.tabulate_shape(1, n_qpoints_cell);
  std::array<std::size_t, 4> rhs_basis_shape_fct
      = basix_element_rhs.tabulate_shape(0, n_qpoints_fct);

  _rhs_basis_cell_values = std::vector<double>(
      std::reduce(rhs_basis_shape_cell.begin(), rhs_basis_shape_cell.end(), 1,
                  std::multiplies{}));
  _rhs_basis_current_values = std::vector<double>(
      std::reduce(rhs_basis_shape_cell.begin(), rhs_basis_shape_cell.end(), 1,
                  std::multiplies{}));

  _rhs_basis_fct_values = std::vector<double>(
      std::reduce(rhs_basis_shape_fct.begin(), rhs_basis_shape_fct.end(), 1,
                  std::multiplies{}));

  basix_element_rhs.tabulate(1, _quadrature_rule->points_cell(),
                             {n_qpoints_cell, _gdim}, _rhs_basis_cell_values);
  basix_element_rhs.tabulate(0, _quadrature_rule->points_fct(),
                             {n_qpoints_fct, _gdim}, _rhs_basis_fct_values);

  _rhs_cell_fullbasis = dolfinx_adaptivity::cmdspan4_t(
      _rhs_basis_cell_values.data(), rhs_basis_shape_cell);
  _rhs_fct_fullbasis = dolfinx_adaptivity::cmdspan4_t(
      _rhs_basis_fct_values.data(), rhs_basis_shape_fct);

  _rhs_fullbasis_current = dolfinx_adaptivity::mdspan4_t(
      _rhs_basis_current_values.data(), rhs_basis_shape_cell);

  // Move shape functions from reference to current
  // (Lagrangian elements --> no mapping required)
  for (std::size_t i = 0; i < _rhs_cell_fullbasis.extent(1); ++i)
  {
    for (std::size_t j = 0; j < _rhs_cell_fullbasis.extent(2); ++j)
    {
      _rhs_fullbasis_current(0, i, j, 0) = _rhs_cell_fullbasis(0, i, j, 0);
    }
  }

  // Tabulate hat function
  std::array<std::size_t, 4> hat_basis_shape_cell
      = basix_element_hat.tabulate_shape(0, n_qpoints_cell);
  std::array<std::size_t, 4> hat_basis_shape_fct
      = basix_element_hat.tabulate_shape(0, n_qpoints_fct);

  _hat_basis_cell_values = std::vector<double>(
      std::reduce(hat_basis_shape_cell.begin(), hat_basis_shape_cell.end(), 1,
                  std::multiplies{}));

  _hat_basis_fct_values = std::vector<double>(
      std::reduce(hat_basis_shape_fct.begin(), hat_basis_shape_fct.end(), 1,
                  std::multiplies{}));

  basix_element_hat.tabulate(0, _quadrature_rule->points_cell(),
                             {n_qpoints_cell, _gdim}, _hat_basis_cell_values);
  basix_element_hat.tabulate(0, _quadrature_rule->points_fct(),
                             {n_qpoints_fct, _gdim}, _hat_basis_fct_values);

  _hat_cell_fullbasis = dolfinx_adaptivity::cmdspan4_t(
      _hat_basis_cell_values.data(), hat_basis_shape_cell);
  _hat_fct_fullbasis = dolfinx_adaptivity::cmdspan4_t(
      _hat_basis_fct_values.data(), hat_basis_shape_fct);
}

double KernelData::compute_jacobian(dolfinx_adaptivity::mdspan2_t J,
                                    dolfinx_adaptivity::mdspan2_t K,
                                    std::span<double> detJ_scratch,
                                    dolfinx_adaptivity::cmdspan2_t coords)
{
  // Reshape basis functions (geometry)
  dolfinx_adaptivity::cmdspan4_t full_basis(_g_basis_values.data(),
                                            _g_basis_shape);

  // Basis functions evaluated at first gauss-point
  dolfinx_adaptivity::s_cmdspan2_t dphi
      = stdex::submdspan(full_basis, std::pair{1, (std::size_t)_tdim + 1}, 0,
                         stdex::full_extent, 0);

  // Compute Jacobian
  for (std::size_t i = 0; i < J.extent(0); ++i)
  {
    for (std::size_t j = 0; j < J.extent(1); ++j)
    {
      J(i, j) = 0;
      K(i, j) = 0;
    }
  }

  fem::CoordinateElement::compute_jacobian(dphi, coords, J);
  fem::CoordinateElement::compute_jacobian_inverse(J, K);

  return fem::CoordinateElement::compute_jacobian_determinant(J, detJ_scratch);
}

double KernelData::compute_jacobian(dolfinx_adaptivity::mdspan2_t J,
                                    std::span<double> detJ_scratch,
                                    dolfinx_adaptivity::cmdspan2_t coords)
{
  // Reshape basis functions (geometry)
  dolfinx_adaptivity::cmdspan4_t full_basis(_g_basis_values.data(),
                                            _g_basis_shape);

  // Basis functions evaluated at first gauss-point
  dolfinx_adaptivity::s_cmdspan2_t dphi
      = stdex::submdspan(full_basis, std::pair{1, (std::size_t)_tdim + 1}, 0,
                         stdex::full_extent, 0);

  // Compute Jacobian
  for (std::size_t i = 0; i < J.extent(0); ++i)
  {
    for (std::size_t j = 0; j < J.extent(1); ++j)
    {
      J(i, j) = 0;
    }
  }

  fem::CoordinateElement::compute_jacobian(dphi, coords, J);

  return fem::CoordinateElement::compute_jacobian_determinant(J, detJ_scratch);
}

void KernelData::physical_fct_normal(std::span<double> normal_phys,
                                     dolfinx_adaptivity::mdspan2_t K,
                                     std::int8_t fct_id)
{
  // Set physical normal to zero
  std::fill(normal_phys.begin(), normal_phys.end(), 0);

  // Extract normal on reference cell
  std::span<const double> normal_ref = fct_normal(fct_id);

  // n_phys = F^(-T) * n_ref
  for (int i = 0; i < _gdim; ++i)
  {
    for (int j = 0; j < _gdim; ++j)
    {
      normal_phys[i] += K(j, i) * normal_ref[j];
    }
  }

  // Normalize vector
  double norm = 0;
  std::for_each(normal_phys.begin(), normal_phys.end(),
                [&norm](auto ni) { norm += std::pow(ni, 2); });
  norm = std::sqrt(norm);
  std::for_each(normal_phys.begin(), normal_phys.end(),
                [norm](auto& ni) { ni = ni / norm; });
}

dolfinx_adaptivity::s_cmdspan3_t
KernelData::shapefunctions_flux(dolfinx_adaptivity::mdspan2_t J, double detJ)
{
  // Loop over all evaluation points
  for (std::size_t i = 0; i < _flux_fullbasis.extent(1); ++i)
  {
    // Loop over all basis functions
    for (std::size_t j = 0; j < _flux_fullbasis.extent(2); ++j)
    {
      double inv_detJ = 1.0 / detJ;

      // Evaluate (1/detj) * J * phi^j(x_i)
      _flux_fullbasis_current(0, i, j, 0)
          = inv_detJ * J(0, 0) * _flux_fullbasis(0, i, j, 0)
            + inv_detJ * J(0, 1) * _flux_fullbasis(0, i, j, 1);
      _flux_fullbasis_current(0, i, j, 1)
          = inv_detJ * J(1, 0) * _flux_fullbasis(0, i, j, 0)
            + inv_detJ * J(1, 1) * _flux_fullbasis(0, i, j, 1);
    }
  }

  return stdex::submdspan(_flux_fullbasis_current, 0, stdex::full_extent,
                          stdex::full_extent, stdex::full_extent);
}

dolfinx_adaptivity::s_cmdspan3_t
KernelData::shapefunctions_cell_rhs(dolfinx_adaptivity::cmdspan2_t K)
{
  // Loop over all evaluation points
  for (std::size_t i = 0; i < _rhs_cell_fullbasis.extent(1); ++i)
  {
    // Loop over all basis functions
    for (std::size_t j = 0; j < _rhs_cell_fullbasis.extent(2); ++j)
    {
      // Evaluate (J^-1)^T * phi^j(x_i)
      _rhs_fullbasis_current(1, i, j, 0)
          = K(0, 0) * _rhs_cell_fullbasis(1, i, j, 0)
            + K(1, 0) * _rhs_cell_fullbasis(2, i, j, 0);
      _rhs_fullbasis_current(2, i, j, 0)
          = K(0, 1) * _rhs_cell_fullbasis(1, i, j, 0)
            + K(1, 1) * _rhs_cell_fullbasis(2, i, j, 0);
    }
  }

  return stdex::submdspan(_rhs_fullbasis_current, stdex::full_extent,
                          stdex::full_extent, stdex::full_extent, 0);
}

} // namespace dolfinx_adaptivity::equilibration