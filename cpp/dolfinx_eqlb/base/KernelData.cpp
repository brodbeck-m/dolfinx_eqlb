// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "KernelData.hpp"

using namespace dolfinx;
using namespace dolfinx_eqlb::base;

template <std::floating_point U>
KernelData<U>::KernelData(std::shared_ptr<const mesh::Mesh<U>> mesh,
                          std::vector<std::tuple<int, int>> quadrature_rules)
{
  // Create quadrature rules
  for (auto& qrule : quadrature_rules)
  {
    _quadrature_rule.push_back(QuadratureRule<U>(
        mesh->topology()->cell_type(), std::get<0>(qrule), std::get<1>(qrule)));
  }

  // Extract mesh data
  std::shared_ptr<const mesh::Topology> topology = mesh->topology();
  const mesh::Geometry<U>& geometry = mesh->geometry();
  const fem::CoordinateElement<U>& cmap = geometry.cmap();

  // Check if mesh is affine
  _is_affine = cmap.is_affine();

  if (!_is_affine)
  {
    throw std::runtime_error("KernelData limited to affine meshes!");
  }

  // Set dimensions
  _gdim = geometry.dim();
  _tdim = topology->dim();

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
  _g_basis_values = std::vector<U>(std::reduce(
      g_basis_shape.begin(), g_basis_shape.end(), 1, std::multiplies{}));
  _g_basis = mdspan_t<const U, 4>(_g_basis_values.data(), g_basis_shape);

  std::vector<U> points(_gdim, 0);
  cmap.tabulate(1, points, {1, _gdim}, _g_basis_values);

  // Get facet normals of reference element
  basix::cell::type basix_cell
      = mesh::cell_type_to_basix_type(topology->cell_type());

  std::tie(_fct_normals, _normals_shape)
      = basix::cell::facet_outward_normals<U>(basix_cell);

  _fct_normal_out = basix::cell::facet_orientations(basix_cell);
}

/* Basic transformations */
template <std::floating_point U>
U KernelData<U>::compute_jacobian(mdspan_t<U, 2> J, mdspan_t<U, 2> K,
                                  std::span<U> detJ_scratch,
                                  mdspan_t<const U, 2> coords)
{
  // Basis functions evaluated at first gauss-point
  base::smdspan_t<const U, 2> dphi = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      _g_basis, std::pair{1, (std::size_t)_tdim + 1}, 0,
      MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);

  // Compute Jacobian
  for (std::size_t i = 0; i < J.extent(0); ++i)
  {
    for (std::size_t j = 0; j < J.extent(1); ++j)
    {
      J(i, j) = 0;
      K(i, j) = 0;
    }
  }

  fem::CoordinateElement<U>::compute_jacobian(dphi, coords, J);
  fem::CoordinateElement<U>::compute_jacobian_inverse(J, K);

  return fem::CoordinateElement<U>::compute_jacobian_determinant(J,
                                                                 detJ_scratch);
}

template <std::floating_point U>
U KernelData<U>::compute_jacobian(mdspan_t<U, 2> J, std::span<U> detJ_scratch,
                                  mdspan_t<const U, 2> coords)
{
  // Basis functions evaluated at first gauss-point
  base::smdspan_t<const U, 2> dphi = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      _g_basis, std::pair{1, (std::size_t)_tdim + 1}, 0,
      MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);

  // Compute Jacobian
  for (std::size_t i = 0; i < J.extent(0); ++i)
  {
    for (std::size_t j = 0; j < J.extent(1); ++j)
    {
      J(i, j) = 0;
    }
  }

  fem::CoordinateElement<U>::compute_jacobian(dphi, coords, J);

  return fem::CoordinateElement<U>::compute_jacobian_determinant(J,
                                                                 detJ_scratch);
}

template <std::floating_point U>
void KernelData<U>::physical_fct_normal(std::span<U> normal_phys,
                                        mdspan_t<const U, 2> K,
                                        std::int8_t fct_id)
{
  // Set physical normal to zero
  std::fill(normal_phys.begin(), normal_phys.end(), 0);

  // Extract normal on reference cell
  std::span<const U> normal_ref = fct_normal(fct_id);

  // n_phys = F^(-T) * n_ref
  for (int i = 0; i < _gdim; ++i)
  {
    for (int j = 0; j < _gdim; ++j)
    {
      normal_phys[i] += K(j, i) * normal_ref[j];
    }
  }

  // Normalize vector
  U norm = 0;
  std::for_each(normal_phys.begin(), normal_phys.end(),
                [&norm](auto ni) { norm += std::pow(ni, 2); });
  norm = std::sqrt(norm);
  std::for_each(normal_phys.begin(), normal_phys.end(),
                [norm](auto& ni) { ni = ni / norm; });
}

/* Tabulate shape function */
template <std::floating_point U>
std::array<std::size_t, 5> KernelData<U>::tabulate_basis(
    const basix::FiniteElement<U>& basix_element, const std::vector<U>& points,
    std::vector<U>& storage, bool tabulate_gradient, bool stoarge_elmtcur)
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
  std::span<U> storage_ref(storage.data(), size_storage);
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

template <std::floating_point U>
std::array<std::size_t, 4> KernelData<U>::interpolation_data_facet_rt(
    const basix::FiniteElement<U>& basix_element, const bool flux_is_custom,
    const std::size_t gdim, const std::size_t nfcts_per_cell,
    std::vector<U>& ipoints_fct, std::vector<U>& data_M_fct)
{
  // Extract interpolation points
  auto [X, Xshape] = basix_element.points();
  const auto [Mdata, Mshape] = basix_element.interpolation_matrix();
  mdspan_t<const U, 2> M(Mdata.data(), Mshape);

  // Determine number of pointe per facet
  std::size_t nipoints_per_fct = 0;

  U x_fctpoint = X[0];

  while (x_fctpoint > 0.0)
  {
    // Increment number of points per facet
    nipoints_per_fct++;

    // Get next x-coordinate
    x_fctpoint = X[nipoints_per_fct * gdim];
  }

  std::size_t nipoints_fct = nipoints_per_fct * nfcts_per_cell;

  // Resize storage of interpolation data
  const int degree = basix_element.degree();
  std::size_t ndofs_fct
      = (gdim == 2) ? degree : (degree + 1) * (degree + 2) / 2;

  ipoints_fct.resize(nipoints_fct * gdim, 0);
  data_M_fct.resize(ndofs_fct * nipoints_fct * gdim, 0);

  // Cast Interpolation matrix into mdspan
  std::array<std::size_t, 4> M_fct_shape
      = {nfcts_per_cell, ndofs_fct, gdim, nipoints_per_fct};
  mdspan_t<U, 4> M_fct(data_M_fct.data(), M_fct_shape);

  // Copy interpolation points (on facets)
  std::copy_n(X.begin(), nipoints_fct * gdim, ipoints_fct.begin());

  // Copy interpolation matrix (on facets)
  int offs_drvt;

  if (flux_is_custom)
  {
    offs_drvt = (degree > 1) ? gdim + 1 : 1;
  }
  else
  {
    offs_drvt = 1;
  }

  int id_dof = 0;
  int offs_pnt = 0;

  for (std::size_t i = 0; i < gdim; ++i)
  {
    for (std::size_t j = 0; j < nfcts_per_cell; ++j)
    {
      for (std::size_t k = 0; k < ndofs_fct; ++k)
      {
        // Determine cell-local DOF id
        id_dof = j * ndofs_fct + k;
        offs_pnt = (i * Xshape[0] + j * nipoints_per_fct) * offs_drvt;

        // Copy interpolation coefficients
        for (std::size_t l = 0; l < nipoints_per_fct; ++l)
        {
          M_fct(j, k, i, l) = M(id_dof, offs_pnt + offs_drvt * l);
        }
      }
    }
  }

  return std::move(M_fct_shape);
}

// ------------------------------------------------------------------------------
template class KernelData<double>;
// ------------------------------------------------------------------------------