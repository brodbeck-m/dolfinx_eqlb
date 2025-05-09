// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "KernelData.hpp"
#include "QuadratureRule.hpp"
#include "equilibration.hpp"
#include "mdspan.hpp"

#include <basix/finite-element.h>
#include <dolfinx/fem/FiniteElement.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <span>
#include <tuple>
#include <vector>

using namespace dolfinx;

namespace dolfinx_eqlb::base
{

template <dolfinx::scalar T, std::floating_point U>
class KernelDataBC : public KernelData<U>
{
  // TODO - Implement general casting between T and U in interpolation routines!

public:
  /// Kernel data for calculation of patch boundary-conditions
  ///
  /// Holds the required surface-quadrature rule, calculates required mappings,
  /// interpolates from a flux function and calculates DOFs of a flux-function.
  ///
  /// @param[in] mesh                 The mesh
  /// @param[in] quadrature_rule_fct  The quadrature rule on the cell facets
  /// @param[in] basix_element_fluxpw The basix-element for the H(div) flux
  /// @param[in] basix_element_rhs    The basix-element for RHS and proj. flux
  KernelDataBC(const basix::FiniteElement<U>& element_geom,
               std::tuple<int, int> quadrature_rule_fct,
               const basix::FiniteElement<U>& element_hat,
               const basix::FiniteElement<U>& element_flux,
               const EqlbStrategy equilibration_strategy);

  /* Tabulate/map shape-function sof the flux-space */

  /// Tabulates a basix_element at the surface-quadrature points
  /// @param[in] basix_element The Basix element
  /// @param[in,out] storage   Stoarge vector for the tabulates shape-functions
  /// @param[out] shape_vector The shape for creating an mdspan of the tabulated
  /// functions
  std::array<std::size_t, 5>
  shapefunctions_flux_qpoints(const basix::FiniteElement<U>& basix_element,
                              std::vector<U>& storage)
  {
    return this->tabulate_basis(basix_element,
                                this->_quadrature_rule[0].points(), storage,
                                false, false);
  }

  /// Map flux-functions from reference to current cell
  ///
  /// Applies the contra-variant Piola mapping to the shape-functions of the
  /// flux. The here performed mapping is restricted to the shape-functions
  /// of one cell facet. All other functions will be neglected.
  ///
  /// @param[in] lfct_id      The cell-local Id of the facet
  /// @param[in, out] phi_cur The shape-function on the current cell
  /// @param[in] phi_ref      The shape-functions on the reference cell
  /// @param[in] J            The Jacobian
  /// @param[in] detJ         The determinant of the Jacobian
  void map_shapefunctions_flux(std::int8_t lfct_id, mdspan_t<U, 3> phi_cur,
                               mdspan_t<const U, 5> phi_ref,
                               mdspan_t<const U, 2> J, U detJ);

  /* Calculate flux DOFs based on different inputs */

  /// Calculates a flux (vector) from a given normal-trace on a cell-facet
  ///
  /// Takes over a list of normal-traces at (multiple) points and return the
  /// flux-vectors calculates from facet-normal x trace, where the facet-normal
  /// has a magnitude of 1.
  ///
  /// @param[in] flux_ntrace_cur The vector of flux normal-traces within the
  ///                            current cell
  /// @param[in] lfct_id         The cell-local Id of the facet
  /// @param[in] K               The inverse of the Jacobian
  /// @param[out] flux_vector    A List of flux-vectors recovered from the
  ///                            normal-trace
  mdspan_t<const T, 2> normaltrace_to_flux(std::span<const T> flux_ntrace_cur,
                                           std::int8_t lfct_id,
                                           mdspan_t<const U, 2> K)
  {
    // Perform calculation
    normaltrace_to_vector(flux_ntrace_cur, lfct_id, K);

    return mdspan_t<const T, 2>(_flux_scratch_data.data(),
                                flux_ntrace_cur.size(),
                                (std::size_t)this->_dim);
  }

  /// Calculates flux DOFs based on the normal-trace
  ///
  /// Claculates flux DOFs on a cell facet based on given normal-trace,
  /// evaluated at the required interpolation points of the flux-space.
  ///
  /// @param[in] flux_ntrace_cur The flux normal-trace on the interpolation
  ///                            points
  /// @param[in, out] flux_dofs  The calculated flux DOFs on the facet
  /// @param[in] lfct_id         The (cell-local) Id of the facet
  /// @param[in] J               The Jacobi matrix of the mapping function
  /// @param[in] detJ            The determinant of the Jacobi matrix
  /// @param[in] K               The inverse of the Jacobi matrix
  void interpolate_flux(std::span<const T> flux_ntrace_cur,
                        std::span<T> flux_dofs, std::int8_t lfct_id,
                        mdspan_t<const U, 2> J, U detJ, mdspan_t<const U, 2> K);

  /// Calculates boundary DOFs for a patch problem
  ///
  /// Calculates the boundary DOFs for a patrch problem from the boundary DOFs
  /// where the hat-function is neglected by inteprolating boundary_function x
  /// hat_function.
  ///
  /// @param[in] flux_dofs_bc         The boundary DOFs without consideration of
  ///                                 the hat-function
  /// @param[in] cell_id              Vector with Ids of the boundary cells
  /// @param[in] lfct_id              Vector with the (cell-local) Ids of the
  ///                                 boundary facets
  /// @param[in] hat_id               Vector of (cell-local) Ids of the
  ///                                 patch-central node
  /// @param[in] cell_info            The informatios required for DOF
  ///                                 transformations on a cell
  /// @param[in] J                    The Jacobi matrix of the mapping function
  /// @param[in] detJ                 The determinant of the Jacobi matrix
  /// @param[in] K                    The inverse of the Jacobi matrix
  /// @param[out] flux_dofs_patch     The boundary DOFs for a patch-problem
  std::vector<T> interpolate_flux(std::span<const T> flux_dofs_bc,
                                  std::int32_t cell_id, std::int8_t lfct_id,
                                  std::int8_t hat_id,
                                  std::span<const std::uint32_t> cell_info,
                                  mdspan_t<const U, 2> J, U detJ,
                                  mdspan_t<const U, 2> K)
  {
    // Initialise storage
    std::vector<T> flux_dofs_patch(flux_dofs_bc.size());

    // Interpolaate flux
    interpolate_flux(flux_dofs_bc, flux_dofs_patch, cell_id, lfct_id, hat_id,
                     cell_info, J, detJ, K);

    return std::move(flux_dofs_patch);
  }

  /// Calculates boundary DOFs for a patch problem
  ///
  /// Calculates the boundary DOFs for a patrch problem from the boundary DOFs
  /// where the hat-function is neglected by inteprolating boundary_function x
  /// hat_function.
  ///
  /// This routine should be called where performace is relevant!
  ///
  /// @param[in] flux_dofs_bc         The boundary DOFs without consideration of
  ///                                 the hat-function
  /// @param[in, out] flux_dofs_patch The boundary DOFs for a patch-problem
  /// @param[in] cell_id              Vector with Ids of the boundary cells
  /// @param[in] lfct_id              Vector with the (cell-local) Ids of the
  ///                                 boundary facets
  /// @param[in] hat_id               Vector of (cell-local) Ids of the
  ///                                 patch-central node
  /// @param[in] cell_info            The informatios required for DOF
  ///                                 transformations on a cell
  /// @param[in] J                    The Jacobi matrix of the mapping function
  /// @param[in] detJ                 The determinant of the Jacobi matrix
  /// @param[in] K                    The inverse of the Jacobi matrix
  void interpolate_flux(std::span<const T> flux_dofs_bc,
                        std::span<T> flux_dofs_patch, std::int32_t cell_id,
                        std::int8_t lfct_id, std::int8_t hat_id,
                        std::span<const std::uint32_t> cell_info,
                        mdspan_t<const U, 2> J, U detJ, mdspan_t<const U, 2> K);

  /// Calculates flux-DOFs from flux-vector at the interpolation points
  ///
  /// Calculates flux DOFs on one cell facet from the flux-values (on the
  /// current cell) on the interpolation points of the flux-space.
  ///
  /// @param[in] flux_cur             The flux (current cell) on the
  ///                                 interpolation points
  /// @param[in, out] flux_dofs_patch The calculated flux DOFs
  /// @param[in] lfct_id              The (cell-local) Id of the facet on which
  ///                                 the DOFs are calculated
  /// @param[in] J                    The Jacobi matrix of the mapping function
  /// @param[in] detJ                 The determinant of the Jacobi matrix
  /// @param[in] K                    The inverse of the Jacobi matrix
  void interpolate_flux(mdspan_t<const T, 2> flux_cur,
                        std::span<T> flux_dofs_patch, std::int8_t lfct_id,
                        mdspan_t<const U, 2> J, U detJ, mdspan_t<const U, 2> K);

  /* Getter methods: Interpolation */

  /// Extract number of interpolation points
  /// @return Number of interpolation points
  int num_interpolation_points() const { return _nipoints; }

  /// Extract number of interpolation points per facet
  /// @return Number of interpolation points
  int num_interpolation_points_per_facet() const { return _nipoints_per_fct; }

protected:
  void normaltrace_to_vector(std::span<const T> normaltrace_cur,
                             std::int8_t lfct_id, mdspan_t<const U, 2> K);

  /* Variable definitions */

  // Interpolation data
  std::size_t _nipoints_per_fct, _nipoints;
  std::vector<U> _ipoints, _data_M;
  mdspan_t<const U, 4> _M; // Indices: facet, dof, gdim, points

  // Tabulated shape-functions H(div) flux (integration points)
  std::vector<U> _basis_flux_values;
  mdspan_t<const U, 5> _basis_flux;
  const int _ndofs_per_fct, _ndofs_fct, _ndofs_cell;

  // Tabulated shape-functions (hat-function)
  std::vector<U> _basis_hat_values;
  mdspan_t<const U, 5> _basis_hat;

  // Pull-back H(div) data
  std::size_t _size_flux_scratch;
  std::vector<T> _flux_scratch_data, _mflux_scratch_data;
  mdspan_t<T, 2> _flux_scratch, _mflux_scratch;

  std::array<U, 3> _normal_scratch;

  std::function<void(mdspan_t<T, 2>&, const mdspan_t<const T, 2>&,
                     const mdspan_t<const U, 2>&, U,
                     const mdspan_t<const U, 2>&)>
      _pull_back_flux;

  // Push-forward H(div) shape-functions
  std::vector<U> _mbasis_flux_values, _mbasis_scratch_values;
  mdspan_t<U, 2> _mbasis_flux, _mbasis_scratch;

  std::function<void(mdspan_t<U, 2>&, const mdspan_t<const U, 2>&,
                     const mdspan_t<const U, 2>&, U,
                     const mdspan_t<const U, 2>&)>
      _push_forward_flux;

  std::function<void(const std::span<T>&, const std::span<const std::uint32_t>&,
                     std::int32_t, int)>
      _apply_dof_transformation;
};

} // namespace dolfinx_eqlb::base