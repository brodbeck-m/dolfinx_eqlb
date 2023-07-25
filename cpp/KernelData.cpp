#include "KernelData.hpp"
#include "QuadratureRule.hpp"

using namespace dolfinx;
using namespace dolfinx_adaptivity::equilibration;

template <typename T>
KernelData<T>::KernelData(std::shared_ptr<const mesh::Mesh> mesh,
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

  if (_gdim == 2)
  {
    _nfcts_per_cell = 3;
  }
  else
  {
    _nfcts_per_cell = 4;
  }

  /* Geometry element */
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

  /* Interpolation points on facets */
  // Extract interpolation points
  auto [X, Xshape] = basix_element_fluxpw.points();
  const auto [Mdata, Mshape] = basix_element_fluxpw.interpolation_matrix();
  dolfinx_adaptivity::cmdspan2_t M(Mdata.data(), Mshape);

  // Determine number of pointe per facet
  _nipoints_per_fct = 0;

  double x_fctpoint = X[0];

  while (x_fctpoint > 0.0)
  {
    // Increment number of points per facet
    _nipoints_per_fct++;

    // Get next x-coordinate
    x_fctpoint = X[_nipoints_per_fct * _gdim];
  }

  _nipoints_fct = _nipoints_per_fct * _nfcts_per_cell;

  // Initialise storage for interpolation data
  const int degree = basix_element_fluxpw.degree();
  std::size_t ndofs_fct
      = (_gdim == 2) ? degree : (degree + 1) * (degree + 2) / 2;

  _ipoints_fct.resize(_nipoints_fct * _gdim, 0);
  _data_M_fct.resize(ndofs_fct * _nipoints_fct * _gdim, 0);

  std::array<std::size_t, 4> M_shape
      = {_nfcts_per_cell, ndofs_fct, _gdim, _nipoints_per_fct};
  _M_fct = dolfinx_adaptivity::mdspan4_t(_data_M_fct.data(), M_shape);

  // Copy interpolation points (on facets)
  std::copy_n(X.begin(), _nipoints_fct * _gdim, _ipoints_fct.begin());

  // Copy interpolation matrix (on facets)
  const int offs_drvt = (degree > 1) ? _gdim + 1 : 1;

  int id_dof = 0;
  int offs_pnt = 0;

  for (std::size_t i = 0; i < _gdim; ++i)
  {
    for (std::size_t j = 0; j < _nfcts_per_cell; ++j)
    {
      for (std::size_t k = 0; k < ndofs_fct; ++k)
      {
        // Determine cell-local DOF id
        id_dof = j * ndofs_fct + k;
        offs_pnt = (i * Xshape[0] + j * _nipoints_per_fct) * offs_drvt;

        // Copy interpolation coefficients
        for (std::size_t l = 0; l < _nipoints_per_fct; ++l)
        {
          _M_fct(j, k, i, l) = M(id_dof, offs_pnt + offs_drvt * l);
        }
      }
    }
  }

  /* Tabulate required shape-functions */
  // Tabulate pice-wise H(div) flux
  tabulate_flux_basis(basix_element_fluxpw);

  // Tabulate right-hand side
  // (Assumption: Same order for projected flux and RHS)
  tabulate_rhs_basis(basix_element_rhs);

  // Tabulate hat-function
  tabulate_hat_basis(basix_element_hat);

  /* H(div)-flux: Pull back into reference */
  using V_t = stdex::mdspan<T, stdex::dextents<std::size_t, 2>>;
  using v_t = stdex::mdspan<const T, stdex::dextents<std::size_t, 2>>;
  using J_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
  using K_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;

  _pull_back_fluxspace = basix_element_fluxpw.map_fn<V_t, v_t, K_t, J_t>();
}

/* Tabulation of shape-functions */
template <typename T>
void KernelData<T>::tabulate_flux_basis(
    const basix::FiniteElement& basix_element_fluxpw)
{
  // Number of surface quadrature points
  std::size_t n_qpoints_cell = _quadrature_rule->npoints_cell();

  // Get shape of storage for tabulated functions
  std::array<std::size_t, 4> flux_basis_shape
      = basix_element_fluxpw.tabulate_shape(0, n_qpoints_cell);

  // Initialise storage of reference- and current cell
  _flux_basis_values = std::vector<double>(std::reduce(
      flux_basis_shape.begin(), flux_basis_shape.end(), 1, std::multiplies{}));
  _flux_basis_current_values = std::vector<double>(std::reduce(
      flux_basis_shape.begin(), flux_basis_shape.end(), 1, std::multiplies{}));

  // Tabulate functions on reference cell
  basix_element_fluxpw.tabulate(0, _quadrature_rule->points_cell(),
                                {n_qpoints_cell, _gdim}, _flux_basis_values);

  // Recast functions into mdspans for later usage
  _flux_fullbasis = dolfinx_adaptivity::cmdspan4_t(_flux_basis_values.data(),
                                                   flux_basis_shape);
  _flux_fullbasis_current = dolfinx_adaptivity::mdspan4_t(
      _flux_basis_current_values.data(), flux_basis_shape);
}

template <typename T>
void KernelData<T>::tabulate_rhs_basis(
    const basix::FiniteElement& basix_element_rhs)
{
  // Number of surface quadrature points
  std::size_t n_qpoints_cell = _quadrature_rule->npoints_cell();

  // Get shape of storage for tabulated functions
  std::array<std::size_t, 4> rhs_basis_shape_cell
      = basix_element_rhs.tabulate_shape(1, n_qpoints_cell);
  std::array<std::size_t, 4> rhs_basis_shape_fct
      = basix_element_rhs.tabulate_shape(0, _nipoints_fct);

  // Initialise storage of reference- and current cell
  _rhs_basis_cell_values = std::vector<double>(
      std::reduce(rhs_basis_shape_cell.begin(), rhs_basis_shape_cell.end(), 1,
                  std::multiplies{}));
  _rhs_basis_current_values = std::vector<double>(
      std::reduce(rhs_basis_shape_cell.begin(), rhs_basis_shape_cell.end(), 1,
                  std::multiplies{}));

  _rhs_basis_fct_values = std::vector<double>(
      std::reduce(rhs_basis_shape_fct.begin(), rhs_basis_shape_fct.end(), 1,
                  std::multiplies{}));

  // Tabulate functions on reference cell
  basix_element_rhs.tabulate(1, _quadrature_rule->points_cell(),
                             {n_qpoints_cell, _gdim}, _rhs_basis_cell_values);
  basix_element_rhs.tabulate(0, _ipoints_fct, {_nipoints_fct, _gdim},
                             _rhs_basis_fct_values);

  // Recast functions into mdspans for later usage
  _rhs_cell_fullbasis = dolfinx_adaptivity::cmdspan4_t(
      _rhs_basis_cell_values.data(), rhs_basis_shape_cell);
  _rhs_fct_fullbasis = dolfinx_adaptivity::cmdspan4_t(
      _rhs_basis_fct_values.data(), rhs_basis_shape_fct);

  _rhs_fullbasis_current = dolfinx_adaptivity::mdspan4_t(
      _rhs_basis_current_values.data(), rhs_basis_shape_cell);

  // Apply identity-map (ref->cur) on shape-functions (on cell)
  for (std::size_t i = 0; i < _rhs_cell_fullbasis.extent(1); ++i)
  {
    for (std::size_t j = 0; j < _rhs_cell_fullbasis.extent(2); ++j)
    {
      _rhs_fullbasis_current(0, i, j, 0) = _rhs_cell_fullbasis(0, i, j, 0);
    }
  }
}

template <typename T>
void KernelData<T>::tabulate_hat_basis(
    const basix::FiniteElement& basix_element_hat)
{
  // Number of surface quadrature points
  std::size_t n_qpoints_cell = _quadrature_rule->npoints_cell();

  // Get shape of storage for tabulated functions
  std::array<std::size_t, 4> hat_basis_shape_cell
      = basix_element_hat.tabulate_shape(0, n_qpoints_cell);
  std::array<std::size_t, 4> hat_basis_shape_fct
      = basix_element_hat.tabulate_shape(0, _nipoints_fct);

  // Initialise storage on reference cell
  _hat_basis_cell_values = std::vector<double>(
      std::reduce(hat_basis_shape_cell.begin(), hat_basis_shape_cell.end(), 1,
                  std::multiplies{}));

  _hat_basis_fct_values = std::vector<double>(
      std::reduce(hat_basis_shape_fct.begin(), hat_basis_shape_fct.end(), 1,
                  std::multiplies{}));

  // Tabulate functions on reference cell
  basix_element_hat.tabulate(0, _quadrature_rule->points_cell(),
                             {n_qpoints_cell, _gdim}, _hat_basis_cell_values);
  basix_element_hat.tabulate(0, _ipoints_fct, {_nipoints_fct, _gdim},
                             _hat_basis_fct_values);

  // Recast functions into mdspans for later usage
  _hat_cell_fullbasis = dolfinx_adaptivity::cmdspan4_t(
      _hat_basis_cell_values.data(), hat_basis_shape_cell);
  _hat_fct_fullbasis = dolfinx_adaptivity::cmdspan4_t(
      _hat_basis_fct_values.data(), hat_basis_shape_fct);
}

/* Compute isoparametric mapping */
template <typename T>
double KernelData<T>::compute_jacobian(dolfinx_adaptivity::mdspan2_t J,
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

template <typename T>
double KernelData<T>::compute_jacobian(dolfinx_adaptivity::mdspan2_t J,
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

template <typename T>
void KernelData<T>::physical_fct_normal(std::span<double> normal_phys,
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

/* Push-forward of shape-functions */
template <typename T>
dolfinx_adaptivity::s_cmdspan3_t
KernelData<T>::shapefunctions_flux(dolfinx_adaptivity::mdspan2_t J, double detJ)
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

template <typename T>
dolfinx_adaptivity::s_cmdspan3_t
KernelData<T>::shapefunctions_cell_rhs(dolfinx_adaptivity::cmdspan2_t K)
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

// ------------------------------------------------------------------------------
template class dolfinx_adaptivity::equilibration::KernelData<float>;
template class dolfinx_adaptivity::equilibration::KernelData<double>;
// ------------------------------------------------------------------------------