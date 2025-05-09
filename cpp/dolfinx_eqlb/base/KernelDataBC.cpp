// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "BoundaryData.hpp"

using namespace dolfinx;
using namespace dolfinx_eqlb::base;

template <dolfinx::scalar T, std::floating_point U>
KernelDataBC<T, U>::KernelDataBC(const basix::FiniteElement<U>& element_geom,
                                 std::tuple<int, int> quadrature_rule_fct,
                                 const basix::FiniteElement<U>& element_hat,
                                 const basix::FiniteElement<U>& element_flux,
                                 const EqStrategy equilibration_strategy)
    : KernelData<U>(element_geom, {quadrature_rule_fct}),
      _ndofs_per_fct((this->_dim == 2) ? element_flux.degree()
                                       : 0.5 * element_flux.degree()
                                             * (element_flux.degree() + 1)),
      _ndofs_fct(this->_nfcts_per_cell * _ndofs_per_fct),
      _ndofs_cell(element_flux.dim())
{
  // Create a fem.FiniteElement
  fem::FiniteElement<U> element_flux_hdiv
      = fem::FiniteElement<U>(element_flux, 1, false);

  /* Interpolation points on facets */
  const bool flux_is_custom
      = (equilibration_strategy == EqStrategy::constrained_minimisation) ? false
                                                                         : true;
  std::array<std::size_t, 4> shape_intpl = this->interpolation_data_facet_rt(
      element_flux, flux_is_custom, this->_dim, this->_nfcts_per_cell, _ipoints,
      _data_M);

  _M = mdspan_t<const U, 4>(_data_M.data(), shape_intpl);

  std::tie(_nipoints_per_fct, _nipoints)
      = this->size_interpolation_data_facet_rt(shape_intpl);

  /* Tabulate required shape-functions */
  std::array<std::size_t, 5> shape;

  // Tabulate H(div) flux at interpolation points
  shape = this->tabulate_basis(element_flux, _ipoints, _basis_flux_values,
                               false, false);
  _basis_flux = mdspan_t<const U, 5>(_basis_flux_values.data(), shape);

  _mbasis_flux_values.resize(_ndofs_cell * this->_dim);
  _mbasis_scratch_values.resize(_ndofs_cell * this->_dim);
  _mbasis_flux
      = mdspan_t<U, 2>(_mbasis_flux_values.data(), _ndofs_cell, this->_dim);
  _mbasis_scratch
      = mdspan_t<U, 2>(_mbasis_scratch_values.data(), _ndofs_cell, this->_dim);

  // Tabulate hat-function at interpolation points
  shape = this->tabulate_basis(element_hat, _ipoints, _basis_hat_values, false,
                               false);
  _basis_hat = mdspan_t<const U, 5>(_basis_hat_values.data(), shape);

  /* H(div)-flux: Pull back into reference */
  // Initialise scratch
  _size_flux_scratch
      = std::max(_nipoints_per_fct, this->_quadrature_rule[0].num_points(0));
  _flux_scratch_data.resize(_size_flux_scratch * this->_dim);
  _mflux_scratch_data.resize(_size_flux_scratch * this->_dim);

  _flux_scratch = mdspan_t<U, 2>(_flux_scratch_data.data(), _size_flux_scratch,
                                 this->_dim);
  _mflux_scratch = mdspan_t<U, 2>(_mflux_scratch_data.data(),
                                  _size_flux_scratch, this->_dim);

  /* Extract mapping functions */
  using J_t = mdspan_t<const U, 2>;
  using K_t = mdspan_t<const U, 2>;

  // Push-forward for shape-functions
  using u_t = mdspan_t<U, 2>;
  using U_t = mdspan_t<const U, 2>;

  _push_forward_flux = element_flux.template map_fn<u_t, U_t, K_t, J_t>();

  // Pull-back for values
  using V_t = mdspan_t<T, 2>;
  using v_t = mdspan_t<const T, 2>;

  _pull_back_flux = element_flux.template map_fn<V_t, v_t, K_t, J_t>();

  // DOF-transformation function (shape functions)
  _apply_dof_transformation
      = element_flux_hdiv.template dof_transformation_fn<U>(
          fem::doftransform::standard, false);
}

template <dolfinx::scalar T, std::floating_point U>
void KernelDataBC<T, U>::map_shapefunctions_flux(std::int8_t lfct_id,
                                                 mdspan_t<U, 3> phi_cur,
                                                 mdspan_t<const U, 5> phi_ref,
                                                 mdspan_t<const U, 2> J, U detJ)
{
  const int offs_pnt = lfct_id * this->_quadrature_rule[0].num_points(0);
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
      U inv_detJ = 1.0 / detJ;

      // Evaluate (1/detj) * J * phi^j(x_i)
      U acc = 0;

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

template <dolfinx::scalar T, std::floating_point U>
void KernelDataBC<T, U>::interpolate_flux(std::span<const T> flux_ntrace_cur,
                                          std::span<T> flux_dofs,
                                          std::int8_t lfct_id,
                                          mdspan_t<const U, 2> J, U detJ,
                                          mdspan_t<const U, 2> K)
{
  // Calculate flux within current cell
  normaltrace_to_vector(flux_ntrace_cur, lfct_id, K);

  // Calculate DOFs based on values at interpolation points
  interpolate_flux(_flux_scratch, flux_dofs, lfct_id, J, detJ, K);
}

template <dolfinx::scalar T, std::floating_point U>
void KernelDataBC<T, U>::interpolate_flux(
    std::span<const T> flux_dofs_bc, std::span<T> flux_dofs_patch,
    std::int32_t cell_id, std::int8_t lfct_id, std::int8_t hat_id,
    std::span<const std::uint32_t> cell_info, mdspan_t<const U, 2> J, U detJ,
    mdspan_t<const U, 2> K)
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
      for (std::size_t i_dim = 0; i_dim < this->_dim; ++i_dim)
      {
        _mbasis_scratch(i_dof, i_dim)
            = _basis_flux(0, 0, offs_ipnt + i_pnt, i_dof, i_dim);
      }
    }

    // Apply dof transformation
    _apply_dof_transformation(_mbasis_scratch_values, cell_info, cell_id,
                              this->_dim);

    // Apply push-foreward function
    _push_forward_flux(_mbasis_flux, _mbasis_scratch, J, detJ, K);

    // Evaluate flux
    for (std::size_t i_dim = 0; i_dim < this->_dim; ++i_dim)
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

template <dolfinx::scalar T, std::floating_point U>
void KernelDataBC<T, U>::interpolate_flux(mdspan_t<const T, 2> flux_cur,
                                          std::span<T> flux_dofs,
                                          std::int8_t lfct_id,
                                          mdspan_t<const U, 2> J, U detJ,
                                          mdspan_t<const U, 2> K)
{
  // Map flux to reference cell
  _pull_back_flux(_mflux_scratch, flux_cur, K, 1 / detJ, J);

  // Apply interpolation operator
  for (std::size_t i = 0; i < flux_dofs.size(); ++i)
  {
    T dof = 0;
    for (std::size_t j = 0; j < _nipoints_per_fct; ++j)
    {
      for (std::size_t k = 0; k < this->_dim; ++k)
      {
        {
          dof += _M(lfct_id, i, k, j) * _mflux_scratch(j, k);
        }
      }
    }

    flux_dofs[i] = dof;
  }
}

template <dolfinx::scalar T, std::floating_point U>
void KernelDataBC<T, U>::normaltrace_to_vector(
    std::span<const T> normaltrace_cur, std::int8_t lfct_id,
    mdspan_t<const U, 2> K)
{
  // Calculate physical facet normal
  std::span<U> normal_cur(_normal_scratch.data(), this->_dim);
  this->physical_fct_normal(normal_cur, K, lfct_id);

  // Calculate flux within current cell
  for (std::size_t i = 0; i < normaltrace_cur.size(); ++i)
  {
    int offs = this->_dim * i;

    // Set flux
    for (std::size_t j = 0; j < this->_dim; ++j)
    {
      _flux_scratch(i, j) = normal_cur[j] * normaltrace_cur[i];
    }
  }
}

// ------------------------------------------------------------------------------
template class KernelDataBC<double, double>;
// ------------------------------------------------------------------------------