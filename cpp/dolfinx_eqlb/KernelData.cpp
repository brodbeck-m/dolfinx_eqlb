// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "KernelData.hpp"

using namespace dolfinx;
using namespace dolfinx_eqlb;

template <typename T>
KernelDataEqlb<T>::KernelDataEqlb(
    std::shared_ptr<const mesh::Mesh> mesh,
    std::shared_ptr<const base::QuadratureRule> quadrature_rule_cell,
    const basix::FiniteElement& basix_element_fluxpw,
    const basix::FiniteElement& basix_element_hat)
    : base::KernelData<T>(mesh, {quadrature_rule_cell})
{
  /* Interpolation points on facets */
  std::array<std::size_t, 4> shape_intpl = this->interpolation_data_facet_rt(
      basix_element_fluxpw, true, this->_gdim, this->_nfcts_per_cell,
      _ipoints_fct, _data_M_fct);

  _M_fct = base::mdspan_t<const double, 4>(_data_M_fct.data(), shape_intpl);

  std::tie(_nipoints_per_fct, _nipoints_fct)
      = this->size_interpolation_data_facet_rt(shape_intpl);

  /* Tabulate required shape-functions */
  // Tabulate pice-wise H(div) flux
  tabulate_flux_basis(basix_element_fluxpw);

  // Tabulate hat-function
  tabulate_hat_basis(basix_element_hat);

  /* H(div)-flux: Mapping */
  // Mapping between reference and current cell
  using V_t = base::mdspan_t<T, 2>;
  using v_t = base::mdspan_t<const T, 2>;
  using J_t = base::mdspan_t<const double, 2>;
  using K_t = base::mdspan_t<const double, 2>;

  _pull_back_fluxspace = basix_element_fluxpw.map_fn<V_t, v_t, K_t, J_t>();

  // Transformation of shape function on reversed facets
  const int ndofs_flux_fct = basix_element_fluxpw.degree();

  _data_transform_shpfkt.resize(ndofs_flux_fct * ndofs_flux_fct, 0.0);
  _shape_transform_shpfkt
      = {(std::size_t)ndofs_flux_fct, (std::size_t)ndofs_flux_fct};
  base::mdspan_t<double, 2> data_transform_shpfk(
      _data_transform_shpfkt.data(), ndofs_flux_fct, ndofs_flux_fct);

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
    std::shared_ptr<const base::QuadratureRule> quadrature_rule_cell,
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
  _flux_fullbasis = base::mdspan_t<const double, 4>(_flux_basis_values.data(),
                                                    flux_basis_shape);
  _flux_fullbasis_current = base::mdspan_t<double, 4>(
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
  _rhs_cell_fullbasis = base::mdspan_t<const double, 4>(
      _rhs_basis_cell_values.data(), rhs_basis_shape_cell);
  _rhs_fct_fullbasis = base::mdspan_t<const double, 4>(
      _rhs_basis_fct_values.data(), rhs_basis_shape_fct);

  _rhs_fullbasis_current = base::mdspan_t<double, 4>(
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
  _hat_cell_fullbasis = base::mdspan_t<const double, 4>(
      _hat_basis_cell_values.data(), hat_basis_shape_cell);
  _hat_fct_fullbasis = base::mdspan_t<const double, 4>(
      _hat_basis_fct_values.data(), hat_basis_shape_fct);
}

/* Mapping routines */
template <typename T>
void KernelDataEqlb<T>::contravariant_piola_mapping(
    base::smdspan_t<double, 3> phi_cur,
    base::smdspan_t<const double, 3> phi_ref, base::mdspan_t<const double, 2> J,
    const double detJ)
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
base::smdspan_t<const double, 3>
KernelDataEqlb<T>::shapefunctions_cell_rhs(base::mdspan_t<const double, 2> K)
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

  return std::experimental::submdspan(
      _rhs_fullbasis_current, std::experimental::full_extent,
      std::experimental::full_extent, std::experimental::full_extent, 0);
}

// ------------------------------------------------------------------------------
template class KernelDataEqlb<float>;
template class KernelDataEqlb<double>;
// ------------------------------------------------------------------------------