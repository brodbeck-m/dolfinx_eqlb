// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <basix/cell.h>
#include <basix/e-lagrange.h>
#include <basix/finite-element.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx_eqlb/base/KernelData.hpp>
#include <dolfinx_eqlb/base/QuadratureRule.hpp>
#include <dolfinx_eqlb/base/mdspan.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <numeric>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

using namespace dolfinx;

namespace base = dolfinx_eqlb::base;

namespace dolfinx_eqlb::se
{

template <typename T>
class KernelData : public base::KernelData<T>
{
public:
  /// Kernel data basic constructor
  ///
  /// Generates data required for isoparametric mapping between reference and
  /// actual element and tabulates elements for flux and hat-function.
  ///
  /// @param[in] mesh                 The mesh
  /// @param[in] quadrature_rule_cell The quadrature rule on the cell
  /// @param[in] basix_element_fluxpw The basix-element for the H(div) flux
  /// @param[in] basix_element_hat    The basix-element for the hat-function
  KernelData(std::shared_ptr<const mesh::Mesh> mesh,
             std::shared_ptr<const base::QuadratureRule> quadrature_rule_cell,
             const basix::FiniteElement& basix_element_fluxpw,
             const basix::FiniteElement& basix_element_hat);

  /// Kernel data constructor
  ///
  /// Generates data required for isoparametric mapping between reference and
  /// actual element and tabulates elements for flux, projected RHS and
  /// hat-function.
  ///
  /// @param[in] mesh                 The mesh
  /// @param[in] quadrature_rule_cell The quadrature rule on the cell
  /// @param[in] basix_element_fluxpw The basix-element for the H(div) flux
  /// @param[in] basix_element_rhs    The basix-element for RHS and proj. flux
  ///                                 (continuous Pk element)
  /// @param[in] basix_element_hat    The basix-element for the hat-function
  ///                                 (continuous P1 element)
  KernelData(std::shared_ptr<const mesh::Mesh> mesh,
             std::shared_ptr<const base::QuadratureRule> quadrature_rule_cell,
             const basix::FiniteElement& basix_element_fluxpw,
             const basix::FiniteElement& basix_element_rhs,
             const basix::FiniteElement& basix_element_hat);

  /// Pull back of flux-data from current to reference cell
  /// @param flux_ref The flux data on reference cell
  /// @param flux_cur The flux data on current cell
  /// @param J        The Jacobian
  /// @param detJ     The determinant of the Jacobian
  /// @param K        The inverse of the Jacobian
  void pull_back_flux(base::mdspan_t<T, 2> flux_ref,
                      base::mdspan_t<const T, 2> flux_cur,
                      base::mdspan_t<const double, 2> J, double detJ,
                      base::mdspan_t<const double, 2> K)
  {
    _pull_back_fluxspace(flux_ref, flux_cur, K, 1.0 / detJ, J);
  }

  /* Setter functions */

  /* Getter functions (Shape functions) */

  /// Extract shape functions (H(div) flux)
  /// Array with indexes i, j and k: phi_j(x_i)[k] is the
  /// shape-function j at point i within direction k.
  /// @return Array of shape functions (reference cell)
  base::smdspan_t<const double, 3> shapefunctions_flux() const
  {
    return std::experimental::submdspan(
        _flux_fullbasis, 0, std::experimental::full_extent,
        std::experimental::full_extent, std::experimental::full_extent);
  }

  /// Extract mapped shape functions (H(div) flux)
  /// Array with indexes i, j and k: phi_j(x_i)[k] is the
  /// shape-function j at point i within direction k.
  /// @param J     The Jacobian
  /// @param detJ  The determinant of the Jacobian
  /// @return Array of shape functions (current cell)
  base::smdspan_t<double, 3>
  shapefunctions_flux(base::mdspan_t<const double, 2> J, const double detJ)
  {
    // Map shape functions
    contravariant_piola_mapping(
        std::experimental::submdspan(
            _flux_fullbasis_current, 0, std::experimental::full_extent,
            std::experimental::full_extent, std::experimental::full_extent),
        std::experimental::submdspan(
            _flux_fullbasis, 0, std::experimental::full_extent,
            std::experimental::full_extent, std::experimental::full_extent),
        J, detJ);

    return std::experimental::submdspan(
        _flux_fullbasis_current, 0, std::experimental::full_extent,
        std::experimental::full_extent, std::experimental::full_extent);
  }

  /// Extract transformation data for shape functions (H(div) flux)
  /// @return Array of transformation data
  base::mdspan_t<const double, 2> entity_transformations_flux() const
  {
    return base::mdspan_t<const double, 2>(_data_transform_shpfkt.data(),
                                           _shape_transform_shpfkt);
  }

  /// Extract shape functions on cell (RHS, projected flux)
  /// Array with indexes i, j and k:
  /// phi_k(x_j) is the shape-function k at point j while i determins
  /// if function or the derivative is returned.
  /// @return Array of shape functions (reference cell)
  base::smdspan_t<const double, 3> shapefunctions_cell_rhs() const
  {
    return std::experimental::submdspan(
        _rhs_cell_fullbasis, std::experimental::full_extent,
        std::experimental::full_extent, std::experimental::full_extent, 0);
  }

  /// Extract mapped shape functions on cell (RHS, projected flux)
  /// Array with indexes i, j and k:
  /// phi_k(x_j) is the shape-function k at point j while k determins
  /// if function or the derivative is returned.
  /// @param K The inverse Jacobian
  /// @return Array of shape functions (current cell)
  base::smdspan_t<const double, 3>
  shapefunctions_cell_rhs(base::mdspan_t<const double, 2> K);

  /// Extract shape functions on facet (RHS, projected flux)
  /// Array with indexes i, j: phi_j(x_i) is the shape-function j
  /// at point i.
  /// @return Array of shape functions (current cell)
  base::smdspan_t<const double, 2> shapefunctions_fct_rhs(std::int8_t fct_id)
  {
    // Offset of shpfkt for current facet
    std::size_t obgn = fct_id * _nipoints_per_fct;
    std::size_t oend = obgn + _nipoints_per_fct;

    return std::experimental::submdspan(_rhs_fct_fullbasis, 0,
                                        std::pair{obgn, oend},
                                        std::experimental::full_extent, 0);
  }

  /// Extract shape functions on cell (hat-function)
  /// Array with indexes i and j: phi_k(x_j) is the shape-function k
  /// at point j
  /// @return Array of shape functions (reference cell)
  base::smdspan_t<const double, 2> shapefunctions_cell_hat() const
  {
    return std::experimental::submdspan(_hat_cell_fullbasis, 0,
                                        std::experimental::full_extent,
                                        std::experimental::full_extent, 0);
  }

  /// Extract shape functions on facet (hat-function)
  /// Array with indexes i, j: phi_j(x_i) is the shape-function j
  /// at point i.
  /// @return Array of shape functions (reference cell)
  base::smdspan_t<const double, 2> shapefunctions_fct_hat(std::int8_t fct_id)
  {
    // Offset of shpfkt for current facet
    std::size_t obgn = fct_id * _nipoints_per_fct;
    std::size_t oend = obgn + _nipoints_per_fct;

    return std::experimental::submdspan(_hat_fct_fullbasis, 0,
                                        std::pair{obgn, oend},
                                        std::experimental::full_extent, 0);
  }

  /// Extract shape functions on facet (hat-function)
  /// Array with indexe i: phi_j(x_i) is the shape-function j
  /// at point i.
  /// @return Array of shape functions (reference cell)
  base::smdspan_t<const double, 1> shapefunctions_fct_hat(std::int8_t fct_id,
                                                          std::size_t j)
  {
    // Offset of shpfkt for current facet
    std::size_t obgn = fct_id * _nipoints_per_fct;
    std::size_t oend = obgn + _nipoints_per_fct;

    return std::experimental::submdspan(_hat_fct_fullbasis, 0,
                                        std::pair{obgn, oend}, j, 0);
  }

  /* Getter functions (Interpolation) */
  // Extract number of interpolation points per facet
  /// @return Number of interpolation points
  int nipoints_facet() const { return _nipoints_per_fct; }

  /// Extract interpolation matrix on facet for single DOF
  /// Indices of M: nfct x ndofs x spacial dimension x points
  /// @param fct_id The cell-local facet id
  /// @return The interpolation matrix M
  base::mdspan_t<const double, 4> interpl_matrix_facte() { return _M_fct; }

  /// Extract interpolation matrix on facet for single DOF
  /// Indices of M: ndofs x spacial dimension x points
  /// @param fct_id The cell-local facet id
  /// @return The interpolation matrix M
  base::mdspan_t<const double, 3> interpl_matrix_facte(std::int8_t fct_id)
  {
    return std::experimental::submdspan(
        _M_fct, (std::size_t)fct_id, std::experimental::full_extent,
        std::experimental::full_extent, std::experimental::full_extent);
  }

  /// Extract interpolation matrix on facet
  /// Indices of M: spacial dimension x points
  /// @param fct_id The cell-local facet id
  /// @param dof_id The facet-local DOF id
  /// @return The interpolation matrix M
  base::mdspan_t<const double, 2> interpl_matrix_facte(std::int8_t fct_id,
                                                       std::int8_t dof_id)
  {
    return std::experimental::submdspan(
        _M_fct, (std::size_t)fct_id, (std::size_t)dof_id,
        std::experimental::full_extent, std::experimental::full_extent);
  }

protected:
  /// Tabulate shape functions of piecewise H(div) flux
  /// @param basix_element_fluxpw The basix-element for the H(div) flux
  void tabulate_flux_basis(const basix::FiniteElement& basix_element_fluxpw);

  /// Tabulate shape functions of RHS and projected flux
  /// @param basix_element_rhs The basix-element for RHS and projected flux
  void tabulate_rhs_basis(const basix::FiniteElement& basix_element_rhs);

  /// Tabulate shape functions of hat-function
  /// @param basix_element_hat The basix-element for the hat-function
  void tabulate_hat_basis(const basix::FiniteElement& basix_element_hat);

  /// Contravariant Piola mapping
  /// Shape of the basis: points x dofs x spacial dimension
  /// @param phi_cur The current basis function values
  /// @param phi_ref The reference basis function values
  /// @param J    The Jacobian matrix
  /// @param detJ The determinant of the Jacobian matrix
  void contravariant_piola_mapping(base::smdspan_t<double, 3> phi_cur,
                                   base::smdspan_t<const double, 3> phi_ref,
                                   base::mdspan_t<const double, 2> J,
                                   const double detJ);

  /* Variable definitions */
  // Interpolation data
  std::size_t _nipoints_per_fct, _nipoints_fct;
  std::vector<double> _ipoints_fct, _data_M_fct;
  base::mdspan_t<const double, 4> _M_fct; // Indices: facet, dof, gdim, points

  // Tabulated shape-functions (pice-wise H(div) flux)
  std::vector<double> _flux_basis_values, _flux_basis_current_values;
  base::mdspan_t<const double, 4> _flux_fullbasis;
  base::mdspan_t<double, 4> _flux_fullbasis_current;

  // Tabulated shape-functions (projected flux, RHS)
  std::vector<double> _rhs_basis_cell_values, _rhs_basis_fct_values,
      _rhs_basis_current_values;
  base::mdspan_t<const double, 4> _rhs_cell_fullbasis, _rhs_fct_fullbasis;
  base::mdspan_t<double, 4> _rhs_fullbasis_current;

  // Tabulated shape-functions (hat-function)
  std::vector<double> _hat_basis_cell_values, _hat_basis_fct_values;
  base::mdspan_t<const double, 4> _hat_cell_fullbasis, _hat_fct_fullbasis;

  // Push-back H(div) data
  std::function<void(base::mdspan_t<T, 2>&, const base::mdspan_t<const T, 2>&,
                     const base::mdspan_t<const double, 2>&, double,
                     const base::mdspan_t<const double, 2>&)>
      _pull_back_fluxspace;

  // Transformation infos for reversed facets
  std::array<std::size_t, 2> _shape_transform_shpfkt;
  std::vector<double> _data_transform_shpfkt;
};

} // namespace dolfinx_eqlb::se