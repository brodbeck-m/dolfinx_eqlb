// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "QuadratureRule.hpp"

#include <basix/cell.h>
#include <basix/e-lagrange.h>
#include <basix/finite-element.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx_eqlb/base/mdspan.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <memory>
#include <numeric>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

using namespace dolfinx;

namespace dolfinx_eqlb::base
{

template <std::floating_point U>
class KernelData
{
public:
  KernelData(std::shared_ptr<const mesh::Mesh<U>> mesh,
             std::vector<std::tuple<int, int>> qtypes);

  /// Compute isogeometric mapping for a given cell
  /// @param[in,out] J            The Jacobian
  /// @param[in,out] K            The inverse Jacobian
  /// @param[in,out] detJ_scratch Storage for determinant calculation
  /// @param[in] coords           The cell coordinates
  /// @return                     The determinant of the Jacobian
  U compute_jacobian(mdspan_t<U, 2> J, mdspan_t<U, 2> K,
                     std::span<U> detJ_scratch, mdspan_t<const U, 2> coords);

  /* Basic transformations */
  /// Compute isogeometric mapping for a given cell
  /// @param[in,out] J            The Jacobian
  /// @param[in,out] detJ_scratch Storage for determinant calculation
  /// @param[in] coords           The cell coordinates
  /// @return                     The determinant of the Jacobian
  U compute_jacobian(mdspan_t<U, 2> J, std::span<U> detJ_scratch,
                     mdspan_t<const U, 2> coords);

  /// Calculate physical normal of facet
  /// @param[in,out] normal_phys The physical normal
  /// @param[in] K               The inverse Jacobi-Matrix
  /// @param[in] fct_id          The cell-local facet id
  void physical_fct_normal(std::span<U> normal_phys, mdspan_t<const U, 2> K,
                           std::int8_t fct_id);

  /* Getter functions (Cell geometry) */
  /// Returns number of nodes, forming a reference cell
  /// @param[out] n The number of nodes, forming the cell
  int nnodes_cell() { return _num_coordinate_dofs; }

  /// Returns number of facets, forming a reference cell
  /// @param[out] n The number of facets, forming the cell
  int nfacets_cell() { return _nfcts_per_cell; }

  /// Returns facet normal on reference facet (const. version)
  /// @param[in] id_fct The cell-local facet id
  /// @param[out] normal_ref The reference facet normal
  std::span<const U> fct_normal(std::int8_t fct_id) const
  {
    return std::span<const U>(_fct_normals.data() + fct_id * _tdim, _tdim);
  }

  /// Returns id if cell-normal points outward
  /// @param[out] is_outward Direction indicator (true->outward)
  const std::vector<bool>& fct_normal_is_outward() const
  {
    return _fct_normal_out;
  }

  /// Returns id if cell-normal points outward
  /// @param[in] id_fct      The cell-local facet id
  /// @param[out] is_outward Direction indicator (true->outward)
  bool fct_normal_is_outward(std::int8_t id_fct)
  {
    return _fct_normal_out[id_fct];
  }

  /// Returns id if cell-normal points outward
  /// @param id_fct1 The cell-local facet id
  /// @param id_fct2 The cell-local facet id
  /// @return Direction indicator (true->outward)
  std::pair<bool, bool> fct_normal_is_outward(std::int8_t id_fct1,
                                              std::int8_t id_fct2)
  {
    return {_fct_normal_out[id_fct1], _fct_normal_out[id_fct2]};
  }

  /* Getter functions (Quadrature) */
  /// @param[in] id_qspace The id of the quadrature space
  // Return the number of quadrature points
  std::size_t num_points(int id_qspace) const
  {
    return _quadrature_rule[id_qspace].num_points();
  }

  /// Return the number of quadrature points per entity
  /// @param[in] id_qspace The id of the quadrature space
  /// @param[in] i         The local entity index
  std::size_t num_points(int id_qspace, std::int8_t i) const
  {
    return _quadrature_rule[id_qspace].num_points(i);
  }

  /// Extract quadrature points on all sub-entity of cell
  /// @param[in] id_qspace The id of the quadrature space
  /// @param[out] points   The quadrature points (flattened storage)
  const std::vector<U>& quadrature_points_flattened(int id_qspace) const
  {
    return _quadrature_rule[id_qspace].points();
  }

  /// Extract quadrature points on all sub-entity of cell
  /// @param[in] id_qspace The id of the quadrature space
  /// @param[out] points   The quadrature points
  mdspan_t<const U, 2> quadrature_points(int id_qspace)
  {
    // Extract quadrature rule
    QuadratureRule<U> qrule = _quadrature_rule[id_qspace];

    // Cast points to mdspan
    return mdspan_t<const U, 2>(qrule.points().data(), qrule.num_points(),
                                qrule.tdim());
  }

  /// Extract quadrature points on one sub-entity of cell
  /// @param[in] id_qspace    The id of the quadrature space
  /// @param[in] id_subentity The id of the sub-entity
  /// @param[out] points      The quadrature points
  mdspan_t<const U, 2> quadrature_points(int id_qspace,
                                         std::int8_t id_subentity)
  {
    return _quadrature_rule[id_qspace].points(id_subentity);
  }

  /// Extract quadrature weights on all sub-entity of cell
  /// @param[in] id_qspace The id of the quadrature space
  /// @param[out] weights  The quadrature weights
  std::span<const U> quadrature_weights(int id_qspace)
  {
    return _quadrature_rule[id_qspace].weights();
  }

  /// Extract quadrature weights on one sub-entity of cell
  /// @param[in] id_qspace    The id of the quadrature space
  /// @param[in] id_subentity The id of the sub-entity
  /// @param[out] weights     The quadrature weights
  std::span<const U> quadrature_weights(int id_qspace, std::int8_t id_subentity)
  {
    return _quadrature_rule[id_qspace].weights(id_subentity);
  }

protected:
  /* Tabulate shape function */
  /// Extract interpolation data of an RT-space on facets
  /// @param[in] basix_element     The Basix element (has to be RT!)
  /// @param[in] points            The tabulation points
  /// @param[in,out] storage       The storage for the tabulated basis functions
  /// @param[in] tabulate_gradient True, if gradient is tabulated
  /// @param[in] stoarge_elmtcur   ???
  /// @return The Shape M for creation of an mdspan
  std::array<std::size_t, 5>
  tabulate_basis(const basix::FiniteElement<U>& basix_element,
                 const std::vector<U>& points, std::vector<U>& storage,
                 bool tabulate_gradient, bool stoarge_elmtcur);

  /// Extract interpolation data of an RT-space on facets
  /// @param[in] basix_element   The Basix element (has to be RT!)
  /// @param[in] flux_is_custom  Flag, if custom flux space is used
  /// @param[in] gdim            The geometric dimension of the problem
  /// @param[in] nfcts_per_cell  The number of facets per cell
  /// @param[in,out] ipoints_fct Storage for interpolation points
  /// @param[in,out] data_M_fct  Storage for interpolation matrix
  /// @return The Shape M for creation of an mdspan
  std::array<std::size_t, 4> interpolation_data_facet_rt(
      const basix::FiniteElement<U>& basix_element, const bool flux_is_custom,
      const std::size_t gdim, const std::size_t nfcts_per_cell,
      std::vector<U>& ipoints_fct, std::vector<U>& data_M_fct);

  /// Get shape of facet interpolation data of an RT-space on facets
  /// @param[in] shape Shape, used for creation of the mdspan
  /// @return std::tuple(nipoints_per_fct, nipoints_all_fcts)
  std::pair<std::size_t, std::size_t>
  size_interpolation_data_facet_rt(std::array<std::size_t, 4> shape)
  {
    return {shape[3], shape[0] * shape[3]};
  }

  /* Variable definitions */
  // Dimensions
  std::uint32_t _gdim, _tdim;

  // Description mesh element
  int _num_coordinate_dofs, _nfcts_per_cell;
  bool _is_affine;

  // Facet normals (reference element)
  std::vector<U> _fct_normals;
  std::array<std::size_t, 2> _normals_shape;
  std::vector<bool> _fct_normal_out;

  // Quadrature rule
  std::vector<QuadratureRule<U>> _quadrature_rule;

  // Tabulated shape-functions (geometry)
  std::vector<U> _g_basis_values;
  mdspan_t<const U, 4> _g_basis;
};

} // namespace dolfinx_eqlb::base