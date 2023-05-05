#include "KernelData.hpp"

using namespace dolfinx;

namespace dolfinx_adaptivity::equilibration
{
KernelData::KernelData(std::shared_ptr<const mesh::Mesh> mesh,
                       std::shared_ptr<const QuadratureRule> qrule,
                       const basix::FiniteElement& basix_element_fluxpw)
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

  // Tabulate shape functions of pice-wise flux space
  std::size_t n_qpoints_cell = qrule->npoints_cell();

  _flux_basis_shape = basix_element_fluxpw.tabulate_shape(0, n_qpoints_cell);
  _flux_basis_values = std::vector<double>(
      std::reduce(_flux_basis_shape.begin(), _flux_basis_shape.end(), 1,
                  std::multiplies{}));

  basix_element_fluxpw.tabulate(0, qrule->points_cell(),
                                {n_qpoints_cell, _gdim}, _flux_basis_values);

  _flux_fullbasis = dolfinx_adaptivity::cmdspan4_t(_flux_basis_values.data(),
                                                   _flux_basis_shape);
}

KernelData::KernelData(std::shared_ptr<const mesh::Mesh> mesh,
                       std::shared_ptr<const QuadratureRule> qrule,
                       const basix::FiniteElement& basix_element_fluxpw,
                       int degree_flux_proj, int degree_rhs_proj)
    : KernelData(mesh, qrule, basix_element_fluxpw)
{
  // Tabulate shape functions of projected flux

  // Tabulate shape-functions of projected RHS

  throw std::runtime_error("Kernel data for higer-order not implemented");
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

  return std::fabs(
      fem::CoordinateElement::compute_jacobian_determinant(J, detJ_scratch));
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

  return std::fabs(
      fem::CoordinateElement::compute_jacobian_determinant(J, detJ_scratch));
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

} // namespace dolfinx_adaptivity::equilibration