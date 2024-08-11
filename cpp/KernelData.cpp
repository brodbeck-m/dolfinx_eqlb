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
                                        mdspan_t<const double, 2> K,
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
    const basix::FiniteElement& basix_element_hat)
    : KernelData<T>(mesh, {quadrature_rule_cell})
{
  /* Interpolation points on facets */
  std::array<std::size_t, 4> shape_intpl = interpolation_data_facet_rt(
      basix_element_fluxpw, true, this->_gdim, this->_nfcts_per_cell,
      _ipoints_fct, _data_M_fct);

  _M_fct = mdspan_t<const double, 4>(_data_M_fct.data(), shape_intpl);

  std::tie(_nipoints_per_fct, _nipoints_fct)
      = size_interpolation_data_facet_rt(shape_intpl);

  /* Tabulate required shape-functions */
  // Tabulate pice-wise H(div) flux
  tabulate_flux_basis(basix_element_fluxpw);

  // Tabulate hat-function
  tabulate_hat_basis(basix_element_hat);

  /* H(div)-flux: Mapping */
  // Mapping between reference and current cell
  using V_t = mdspan_t<T, 2>;
  using v_t = mdspan_t<const T, 2>;
  using J_t = mdspan_t<const double, 2>;
  using K_t = mdspan_t<const double, 2>;

  _pull_back_fluxspace = basix_element_fluxpw.map_fn<V_t, v_t, K_t, J_t>();

  // Transformation of shape function on reversed facets
  const int ndofs_flux_fct = basix_element_fluxpw.degree();

  _data_transform_shpfkt.resize(ndofs_flux_fct * ndofs_flux_fct, 0.0);
  _shape_transform_shpfkt
      = {(std::size_t)ndofs_flux_fct, (std::size_t)ndofs_flux_fct};
  mdspan_t<double, 2> data_transform_shpfk(_data_transform_shpfkt.data(),
                                           ndofs_flux_fct, ndofs_flux_fct);

  for (int line = 0; line < ndofs_flux_fct; line++)
  {
    int val = 1;

    for (int i = 0; i <= line; i++)
    {
      data_transform_shpfk(i, line) = ((i % 2) == 0) ? -val : val;
      val = val * (line - i) / (i + 1);
    }
  }
}

template <typename T>
KernelDataEqlb<T>::KernelDataEqlb(
    std::shared_ptr<const mesh::Mesh> mesh,
    std::shared_ptr<const QuadratureRule> quadrature_rule_cell,
    const basix::FiniteElement& basix_element_fluxpw,
    const basix::FiniteElement& basix_element_rhs,
    const basix::FiniteElement& basix_element_hat)
    : KernelDataEqlb<T>(mesh, quadrature_rule_cell, basix_element_fluxpw,
                        basix_element_hat)
{
  // Tabulate right-hand side
  // (Assumption: Same order for projected flux and RHS)
  tabulate_rhs_basis(basix_element_rhs);
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
  _flux_fullbasis
      = mdspan_t<const double, 4>(_flux_basis_values.data(), flux_basis_shape);
  _flux_fullbasis_current = mdspan_t<double, 4>(
      _flux_basis_current_values.data(), flux_basis_shape);
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
  _rhs_cell_fullbasis = mdspan_t<const double, 4>(_rhs_basis_cell_values.data(),
                                                  rhs_basis_shape_cell);
  _rhs_fct_fullbasis = mdspan_t<const double, 4>(_rhs_basis_fct_values.data(),
                                                 rhs_basis_shape_fct);

  _rhs_fullbasis_current = mdspan_t<double, 4>(_rhs_basis_current_values.data(),
                                               rhs_basis_shape_cell);

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
  _hat_cell_fullbasis = mdspan_t<const double, 4>(_hat_basis_cell_values.data(),
                                                  hat_basis_shape_cell);
  _hat_fct_fullbasis = mdspan_t<const double, 4>(_hat_basis_fct_values.data(),
                                                 hat_basis_shape_fct);
}

/* Mapping routines */
template <typename T>
void KernelDataEqlb<T>::contravariant_piola_mapping(
    smdspan_t<double, 3> phi_cur, smdspan_t<const double, 3> phi_ref,
    mdspan_t<const double, 2> J, const double detJ)
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
smdspan_t<const double, 3>
KernelDataEqlb<T>::shapefunctions_cell_rhs(mdspan_t<const double, 2> K)
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
/* KernelDataBC */
// ------------------------------------------------------------------------------
template <typename T>
KernelDataBC<T>::KernelDataBC(
    std::shared_ptr<const mesh::Mesh> mesh,
    std::shared_ptr<const QuadratureRule> quadrature_rule_fct,
    std::shared_ptr<const fem::FiniteElement> element_flux_hdiv,
    const int nfluxdofs_per_fct, const int nfluxdofs_cell,
    const bool flux_is_custom)
    : KernelData<T>(mesh, {quadrature_rule_fct}),
      _ndofs_per_fct(nfluxdofs_per_fct),
      _ndofs_fct(this->_nfcts_per_cell * _ndofs_per_fct),
      _ndofs_cell(nfluxdofs_cell),
      _basix_element_hat(basix::element::create_lagrange(
          mesh::cell_type_to_basix_type(mesh->topology().cell_type()), 1,
          basix::element::lagrange_variant::equispaced, false))
{
  // Extract the baisx flux element
  const basix::FiniteElement& basix_element_flux_hdiv
      = element_flux_hdiv->basix_element();

  /* Interpolation points on facets */
  std::array<std::size_t, 4> shape_intpl = interpolation_data_facet_rt(
      basix_element_flux_hdiv, flux_is_custom, this->_gdim,
      this->_nfcts_per_cell, _ipoints, _data_M);

  _M = mdspan_t<const double, 4>(_data_M.data(), shape_intpl);

  std::tie(_nipoints_per_fct, _nipoints)
      = size_interpolation_data_facet_rt(shape_intpl);

  /* Tabulate required shape-functions */
  std::array<std::size_t, 5> shape;

  // Tabulate H(div) flux at interpolation points
  shape = this->tabulate_basis(basix_element_flux_hdiv, _ipoints,
                               _basis_flux_values, false, false);
  _basis_flux = mdspan_t<const double, 5>(_basis_flux_values.data(), shape);

  _mbasis_flux_values.resize(_ndofs_cell * this->_gdim);
  _mbasis_scratch_values.resize(_ndofs_cell * this->_gdim);
  _mbasis_flux = mdspan_t<double, 2>(_mbasis_flux_values.data(), _ndofs_cell,
                                     this->_gdim);
  _mbasis_scratch = mdspan_t<double, 2>(_mbasis_scratch_values.data(),
                                        _ndofs_cell, this->_gdim);

  // Tabulate hat-function at interpolation points
  shape = this->tabulate_basis(_basix_element_hat, _ipoints, _basis_hat_values,
                               false, false);
  _basis_hat = mdspan_t<const double, 5>(_basis_hat_values.data(), shape);

  /* H(div)-flux: Pull back into reference */
  // Initialise scratch
  _size_flux_scratch
      = std::max(_nipoints_per_fct, this->_quadrature_rule[0]->num_points(0));
  _flux_scratch_data.resize(_size_flux_scratch * this->_gdim);
  _mflux_scratch_data.resize(_size_flux_scratch * this->_gdim);

  _flux_scratch = mdspan_t<T, 2>(_flux_scratch_data.data(), _size_flux_scratch,
                                 this->_gdim);
  _mflux_scratch = mdspan_t<T, 2>(_mflux_scratch_data.data(),
                                  _size_flux_scratch, this->_gdim);

  /* Extract mapping functions */
  using J_t = mdspan_t<const double, 2>;
  using K_t = mdspan_t<const double, 2>;

  // Push-forward for shape-functions
  using u_t = mdspan_t<double, 2>;
  using U_t = mdspan_t<const double, 2>;

  _push_forward_flux = basix_element_flux_hdiv.map_fn<u_t, U_t, K_t, J_t>();

  // Pull-back for values
  using V_t = mdspan_t<T, 2>;
  using v_t = mdspan_t<const T, 2>;

  _pull_back_flux = basix_element_flux_hdiv.map_fn<V_t, v_t, K_t, J_t>();

  // DOF-transformation function (shape functions)
  _apply_dof_transformation
      = element_flux_hdiv->get_dof_transformation_function<double>(false, false,
                                                                   false);
}

template <typename T>
void KernelDataBC<T>::map_shapefunctions_flux(std::int8_t lfct_id,
                                              mdspan_t<double, 3> phi_cur,
                                              mdspan_t<const double, 5> phi_ref,
                                              mdspan_t<const double, 2> J,
                                              double detJ)
{
  const int offs_pnt = lfct_id * this->_quadrature_rule[0]->num_points(0);
  const int offs_dof = lfct_id * _ndofs_per_fct;

  // Loop over all evaluation points
  for (std::size_t i = 0; i < phi_cur.extent(0); ++i)
  {
    // Index of the reference shape-functions
    int i_ref = offs_pnt + i;

    // Loop over all basis functions
    for (std::size_t j = 0; j < phi_cur.extent(1); ++j)
    {
      // Index of the reference shape-functions
      int j_ref = offs_dof + j;

      // Inverse of the determinenant
      double inv_detJ = 1.0 / detJ;

      // Evaluate (1/detj) * J * phi^j(x_i)
      double acc = 0;

      if (phi_cur.extent(2) == 2)
      {
        acc = J(0, 0) * phi_ref(0, 0, i_ref, j_ref, 0)
              + J(0, 1) * phi_ref(0, 0, i_ref, j_ref, 1);
        phi_cur(i, j, 0) = inv_detJ * acc;

        acc = J(1, 0) * phi_ref(0, 0, i_ref, j_ref, 0)
              + J(1, 1) * phi_ref(0, 0, i_ref, j_ref, 1);
        phi_cur(i, j, 1) = inv_detJ * acc;
      }
      else
      {
        acc = J(0, 0) * phi_ref(0, 0, i_ref, j_ref, 0)
              + J(0, 1) * phi_ref(0, 0, i_ref, j_ref, 1)
              + J(0, 2) * phi_ref(0, 0, i_ref, j_ref, 2);
        phi_cur(i, j, 0) = inv_detJ * acc;

        acc = J(1, 0) * phi_ref(0, 0, i_ref, j_ref, 0)
              + J(1, 1) * phi_ref(0, 0, i_ref, j_ref, 1)
              + J(1, 2) * phi_ref(0, 0, i_ref, j_ref, 2);
        phi_cur(i, j, 1) = inv_detJ * acc;

        acc = J(2, 0) * phi_ref(0, 0, i_ref, j_ref, 0)
              + J(2, 1) * phi_ref(0, 0, i_ref, j_ref, 1)
              + J(2, 2) * phi_ref(0, 0, i_ref, j_ref, 2);
        phi_cur(i, j, 1) = inv_detJ * acc;
      }
    }
  }
}

template <typename T>
void KernelDataBC<T>::interpolate_flux(std::span<const T> flux_ntrace_cur,
                                       std::span<T> flux_dofs,
                                       std::int8_t lfct_id,
                                       mdspan_t<const double, 2> J, double detJ,
                                       mdspan_t<const double, 2> K)
{
  // Calculate flux within current cell
  normaltrace_to_vector(flux_ntrace_cur, lfct_id, K);

  // Calculate DOFs based on values at interpolation points
  interpolate_flux(_flux_scratch, flux_dofs, lfct_id, J, detJ, K);
}

template <typename T>
void KernelDataBC<T>::interpolate_flux(std::span<const T> flux_dofs_bc,
                                       std::span<T> flux_dofs_patch,
                                       std::int32_t cell_id,
                                       std::int8_t lfct_id, std::int8_t hat_id,
                                       std::span<const std::uint32_t> cell_info,
                                       mdspan_t<const double, 2> J, double detJ,
                                       mdspan_t<const double, 2> K)
{
  /* Map shape functions*/
  // Copy shape-functions from reference element
  const int offs_ipnt = lfct_id * _nipoints_per_fct;
  const int offs_dof = lfct_id * _ndofs_per_fct;

  for (std::size_t i_pnt = 0; i_pnt < _nipoints_per_fct; ++i_pnt)
  {
    // Copy shape-functions at current point
    // TODO - Check if copy on only functions on current facet is enough!
    for (std::size_t i_dof = 0; i_dof < _ndofs_fct; ++i_dof)
    {
      for (std::size_t i_dim = 0; i_dim < this->_gdim; ++i_dim)
      {
        _mbasis_scratch(i_dof, i_dim)
            = _basis_flux(0, 0, offs_ipnt + i_pnt, i_dof, i_dim);
      }
    }

    // Apply dof transformation
    _apply_dof_transformation(_mbasis_scratch_values, cell_info, cell_id,
                              this->_gdim);

    // Apply push-foreward function
    _push_forward_flux(_mbasis_flux, _mbasis_scratch, J, detJ, K);

    // Evaluate flux
    for (std::size_t i_dim = 0; i_dim < this->_gdim; ++i_dim)
    {
      // Evaluate flux at current point
      T acc = 0;
      for (std::size_t i_dof = 0; i_dof < _ndofs_per_fct; ++i_dof)
      {
        acc += flux_dofs_bc[i_dof] * _mbasis_flux(offs_dof + i_dof, i_dim);
      }

      // Multiply flux with hat function
      _flux_scratch(i_pnt, i_dim)
          = acc * _basis_hat(0, 0, offs_ipnt + i_pnt, hat_id, 0);
    }
  }

  // Calculate DOF of scaled flux
  interpolate_flux(_flux_scratch, flux_dofs_patch, lfct_id, J, detJ, K);
}

template <typename T>
void KernelDataBC<T>::interpolate_flux(mdspan_t<const T, 2> flux_cur,
                                       std::span<T> flux_dofs,
                                       std::int8_t lfct_id,
                                       mdspan_t<const double, 2> J, double detJ,
                                       mdspan_t<const double, 2> K)
{
  // Map flux to reference cell
  _pull_back_flux(_mflux_scratch, flux_cur, K, 1 / detJ, J);

  // Apply interpolation operator
  for (std::size_t i = 0; i < flux_dofs.size(); ++i)
  {
    T dof = 0;
    for (std::size_t j = 0; j < _nipoints_per_fct; ++j)
    {
      for (std::size_t k = 0; k < this->_gdim; ++k)
      {
        {
          dof += _M(lfct_id, i, k, j) * _mflux_scratch(j, k);
        }
      }
    }

    flux_dofs[i] = dof;
  }
}

template <typename T>
void KernelDataBC<T>::normaltrace_to_vector(std::span<const T> normaltrace_cur,
                                            std::int8_t lfct_id,
                                            mdspan_t<const double, 2> K)
{
  // Calculate physical facet normal
  std::span<double> normal_cur(_normal_scratch.data(), this->_gdim);
  this->physical_fct_normal(normal_cur, K, lfct_id);

  // Calculate flux within current cell
  for (std::size_t i = 0; i < normaltrace_cur.size(); ++i)
  {
    int offs = this->_gdim * i;

    // Set flux
    for (std::size_t j = 0; j < this->_gdim; ++j)
    {
      _flux_scratch(i, j) = normal_cur[j] * normaltrace_cur[i];
    }
  }
}

// ------------------------------------------------------------------------------
template class KernelData<float>;
template class KernelData<double>;

template class KernelDataEqlb<float>;
template class KernelDataEqlb<double>;

template class KernelDataBC<float>;
template class KernelDataBC<double>;
// ------------------------------------------------------------------------------