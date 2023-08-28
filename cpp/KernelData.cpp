#include "KernelData.hpp"

using namespace dolfinx;
using namespace dolfinx_eqlb;

// ------------------------------------------------------------------------------
/* KernelData */
// ------------------------------------------------------------------------------
template <typename T>
KernelData<T>::KernelData(
    std::shared_ptr<const mesh::Mesh> mesh,
    std::vector<std::shared_ptr<const QuadratureRule>> quadrature_rule)
    : _quadrature_rule(quadrature_rule)
{
  const mesh::Topology& topology = mesh->topology();
  const mesh::Geometry& geometry = mesh->geometry();
  const fem::CoordinateElement& cmap = geometry.cmap();

  // Check if mesh is affine
  _is_affine = cmap.is_affine();

  if (!_is_affine)
  {
    throw std::runtime_error("KernelData limited to affine meshes!");
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

  std::array<std::size_t, 4> g_basis_shape = cmap.tabulate_shape(1, 1);
  _g_basis_values = std::vector<double>(std::reduce(
      g_basis_shape.begin(), g_basis_shape.end(), 1, std::multiplies{}));
  _g_basis = mdspan_t<const double, 4>(_g_basis_values.data(), g_basis_shape);

  std::vector<double> points(_gdim, 0);
  cmap.tabulate(1, points, {1, _gdim}, _g_basis_values);

  // Get facet normals of reference element
  basix::cell::type basix_cell
      = mesh::cell_type_to_basix_type(topology.cell_type());

  std::tie(_fct_normals, _normals_shape)
      = basix::cell::facet_outward_normals(basix_cell);

  _fct_normal_out = basix::cell::facet_orientations(basix_cell);
}

/* Basic transformations */
template <typename T>
double KernelData<T>::compute_jacobian(mdspan_t<double, 2> J,
                                       mdspan_t<double, 2> K,
                                       std::span<double> detJ_scratch,
                                       mdspan_t<const double, 2> coords)
{
  // Basis functions evaluated at first gauss-point
  smdspan_t<const double, 2> dphi = stdex::submdspan(
      _g_basis, std::pair{1, (std::size_t)_tdim + 1}, 0, stdex::full_extent, 0);

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
double KernelData<T>::compute_jacobian(mdspan_t<double, 2> J,
                                       std::span<double> detJ_scratch,
                                       mdspan_t<const double, 2> coords)
{
  // Basis functions evaluated at first gauss-point
  smdspan_t<const double, 2> dphi = stdex::submdspan(
      _g_basis, std::pair{1, (std::size_t)_tdim + 1}, 0, stdex::full_extent, 0);

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
                                        mdspan_t<double, 2> K,
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

/* Tabulate shape function */
template <typename T>
std::array<std::size_t, 5>
KernelData<T>::tabulate_basis(const basix::FiniteElement& basix_element,
                              const std::vector<double>& points,
                              std::vector<double>& storage,
                              bool tabulate_gradient, bool stoarge_elmtcur)
{
  // Number of tabulated points
  std::size_t num_points = points.size() / _gdim;

  // Get shape of tabulated data
  int id_grad = (tabulate_gradient) ? 1 : 0;
  std::array<std::size_t, 4> shape
      = basix_element.tabulate_shape(id_grad, num_points);

  // Resize storage
  std::size_t size_storage
      = std::reduce(shape.begin(), shape.end(), 1, std::multiplies{});

  if (stoarge_elmtcur)
  {
    storage.resize(2 * size_storage, 0);
  }
  else
  {
    storage.resize(size_storage, 0);
  }

  // Tabulate basis
  std::span<double> storage_ref(storage.data(), size_storage);
  basix_element.tabulate(id_grad, points, {num_points, _gdim}, storage_ref);

  // Create shape of final mdspan
  std::array<std::size_t, 5> shape_final
      = {{1, shape[0], shape[1], shape[2], shape[3]}};

  if (stoarge_elmtcur)
  {
    shape_final[0] = 2;
  }

  return std::move(shape_final);
}

// ------------------------------------------------------------------------------
/* KernelDataEqlb */
// ------------------------------------------------------------------------------
template <typename T>
KernelDataEqlb<T>::KernelDataEqlb(
    std::shared_ptr<const mesh::Mesh> mesh,
    std::shared_ptr<const QuadratureRule> quadrature_rule_cell,
    const basix::FiniteElement& basix_element_fluxpw,
    const basix::FiniteElement& basix_element_rhs,
    const basix::FiniteElement& basix_element_hat)
    : KernelData<T>(mesh, {quadrature_rule_cell})
{
  /* Interpolation points on facets */
  std::array<std::size_t, 4> shape_intpl = interpolation_data_facet_rt(
      basix_element_fluxpw, this->_gdim, this->_nfcts_per_cell, _ipoints_fct,
      _data_M_fct);

  _M_fct = mdspan_t<const double, 4>(_data_M_fct.data(), shape_intpl);

  std::tie(_nipoints_per_fct, _nipoints_fct)
      = size_interpolation_data_facet_rt(shape_intpl);

  /* Tabulate required shape-functions */
  // Tabulate pice-wise H(div) flux
  tabulate_flux_basis(basix_element_fluxpw);

  // Tabulate right-hand side
  // (Assumption: Same order for projected flux and RHS)
  tabulate_rhs_basis(basix_element_rhs);

  // Tabulate hat-function
  tabulate_hat_basis(basix_element_hat);

  /* H(div)-flux: Pull back into reference */
  using V_t = mdspan_t<T, 2>;
  using v_t = mdspan_t<const T, 2>;
  using J_t = mdspan_t<const double, 2>;
  using K_t = mdspan_t<const double, 2>;

  _pull_back_fluxspace = basix_element_fluxpw.map_fn<V_t, v_t, K_t, J_t>();
}

/* Tabulation of shape-functions */
template <typename T>
void KernelDataEqlb<T>::tabulate_flux_basis(
    const basix::FiniteElement& basix_element_fluxpw)
{
  // Number of surface quadrature points
  std::size_t n_qpoints_cell = this->_quadrature_rule[0]->num_points();

  // Get shape of storage for tabulated functions
  std::array<std::size_t, 4> flux_basis_shape
      = basix_element_fluxpw.tabulate_shape(0, n_qpoints_cell);

  // Initialise storage of reference- and current cell
  _flux_basis_values = std::vector<double>(std::reduce(
      flux_basis_shape.begin(), flux_basis_shape.end(), 1, std::multiplies{}));
  _flux_basis_current_values = std::vector<double>(std::reduce(
      flux_basis_shape.begin(), flux_basis_shape.end(), 1, std::multiplies{}));

  // Tabulate functions on reference cell
  basix_element_fluxpw.tabulate(0, this->_quadrature_rule[0]->points(),
                                {n_qpoints_cell, this->_gdim},
                                _flux_basis_values);

  // Recast functions into mdspans for later usage
  _flux_fullbasis = cmdspan4_t(_flux_basis_values.data(), flux_basis_shape);
  _flux_fullbasis_current
      = mdspan4_t(_flux_basis_current_values.data(), flux_basis_shape);
}

template <typename T>
void KernelDataEqlb<T>::tabulate_rhs_basis(
    const basix::FiniteElement& basix_element_rhs)
{
  // Number of surface quadrature points
  std::size_t n_qpoints_cell = this->_quadrature_rule[0]->num_points();

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
  basix_element_rhs.tabulate(1, this->_quadrature_rule[0]->points(),
                             {n_qpoints_cell, this->_gdim},
                             _rhs_basis_cell_values);
  basix_element_rhs.tabulate(0, _ipoints_fct, {_nipoints_fct, this->_gdim},
                             _rhs_basis_fct_values);

  // Recast functions into mdspans for later usage
  _rhs_cell_fullbasis
      = cmdspan4_t(_rhs_basis_cell_values.data(), rhs_basis_shape_cell);
  _rhs_fct_fullbasis
      = cmdspan4_t(_rhs_basis_fct_values.data(), rhs_basis_shape_fct);

  _rhs_fullbasis_current
      = mdspan4_t(_rhs_basis_current_values.data(), rhs_basis_shape_cell);

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
void KernelDataEqlb<T>::tabulate_hat_basis(
    const basix::FiniteElement& basix_element_hat)
{
  // Number of surface quadrature points
  std::size_t n_qpoints_cell = this->_quadrature_rule[0]->num_points();

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
  basix_element_hat.tabulate(0, this->_quadrature_rule[0]->points(),
                             {n_qpoints_cell, this->_gdim},
                             _hat_basis_cell_values);
  basix_element_hat.tabulate(0, _ipoints_fct, {_nipoints_fct, this->_gdim},
                             _hat_basis_fct_values);

  // Recast functions into mdspans for later usage
  _hat_cell_fullbasis
      = cmdspan4_t(_hat_basis_cell_values.data(), hat_basis_shape_cell);
  _hat_fct_fullbasis
      = cmdspan4_t(_hat_basis_fct_values.data(), hat_basis_shape_fct);
}

/* Mapping routines */
template <typename T>
void KernelDataEqlb<T>::contravariant_piola_mapping(
    smdspan_t<double, 3> phi_cur, smdspan_t<const double, 3> phi_ref,
    mdspan2_t J, double detJ)
{
  // Loop over all evaluation points
  for (std::size_t i = 0; i < phi_ref.extent(0); ++i)
  {
    // Loop over all basis functions
    for (std::size_t j = 0; j < phi_ref.extent(1); ++j)
    {
      double inv_detJ = 1.0 / detJ;

      // Evaluate (1/detj) * J * phi^j(x_i)
      phi_cur(i, j, 0) = inv_detJ * J(0, 0) * phi_ref(i, j, 0)
                         + inv_detJ * J(0, 1) * phi_ref(i, j, 1);
      phi_cur(i, j, 1) = inv_detJ * J(1, 0) * phi_ref(i, j, 0)
                         + inv_detJ * J(1, 1) * phi_ref(i, j, 1);
    }
  }
}

/* Push-forward of shape-functions */
template <typename T>
s_cmdspan3_t KernelDataEqlb<T>::shapefunctions_cell_rhs(cmdspan2_t K)
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
template class KernelData<float>;
template class KernelData<double>;

template class KernelDataEqlb<float>;
template class KernelDataEqlb<double>;
// ------------------------------------------------------------------------------