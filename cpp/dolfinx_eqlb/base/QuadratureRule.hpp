// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "mdspan.hpp"

#include <basix/finite-element.h>
#include <basix/polyset.h>
#include <basix/quadrature.h>
#include <dolfinx/common/math.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/utils.h>

#include <algorithm>
#include <array>
#include <iostream>
#include <span>
#include <vector>

using namespace dolfinx;

namespace dolfinx_eqlb::base
{
template <std::floating_point U>
class QuadratureRule
{
  // Contains quadrature points and weights on a cell on a set of entities
  // Implementation taken from https://github.com/Wells-Group/asimov-contact/

public:
  /// Constructor
  /// @param[in] ct        The cell type
  /// @param[in] degree    Degree of quadrature rule
  /// @param[in] dim       The dimension of the entity
  /// @param[in] type      Type of quadrature rule
  QuadratureRule(basix::cell::type ct, int degree, int dim,
                 basix::quadrature::type type
                 = basix::quadrature::type::Default)
      : _cell_type(ct), _degree(degree), _type(type), _dim(dim)
  {
    _num_sub_entities = basix::cell::num_sub_entities(ct, dim);
    _tdim = basix::cell::topological_dimension(ct);
    assert(dim <= 3);
    // If cell dimension no pushing forward
    if (_tdim == std::size_t(dim))
    {
      std::array<std::vector<U>, 2> quadrature
          = basix::quadrature::make_quadrature<U>(
              type, ct, basix::polyset::type::standard, degree);
      std::vector<U>& q_weights = quadrature.back();
      std::size_t num_points = q_weights.size();
      std::size_t pt_shape = quadrature.front().size() / num_points;
      mdspan_t<const U, 2> qp(quadrature.front().data(), num_points, pt_shape);

      _points = std::vector<U>(num_points * _num_sub_entities * _tdim);
      _entity_offset = std::vector<std::size_t>(_num_sub_entities + 1, 0);
      _weights = std::vector<U>(num_points * _num_sub_entities);
      for (std::int32_t i = 0; i < _num_sub_entities; i++)
      {
        _entity_offset[i + 1] = (i + 1) * q_weights.size();
        for (std::size_t j = 0; j < num_points; ++j)
        {
          _weights[i * num_points * _num_sub_entities + j] = q_weights[j];
          for (std::size_t k = 0; k < _tdim; ++k)
            _points[i * num_points * _num_sub_entities + j * _tdim + k]
                = qp(j, k);
        }
      }
    }
    else
    {
      // Create reference topology and geometry
      auto entity_topology = basix::cell::topology(ct)[dim];

      // Create map for each facet type to the local index
      std::vector<std::size_t> num_points_per_entity(_num_sub_entities);
      for (std::int32_t i = 0; i < _num_sub_entities; i++)
      {
        // Create reference element to map facet quadrature to
        basix::cell::type et = basix::cell::sub_entity_type(ct, dim, i);
        basix::FiniteElement entity_element = basix::create_element<U>(
            basix::element::family::P, et, 1,
            basix::element::lagrange_variant::gll_warped,
            basix::element::dpc_variant::unset, false);
        // Create quadrature and tabulate on entity
        std::array<std::vector<U>, 2> quadrature
            = basix::quadrature::make_quadrature<U>(
                type, et, basix::polyset::type::standard, degree);
        const std::vector<U>& q_weights = quadrature.back();
        const std::vector<U>& q_points = quadrature.front();
        const std::size_t num_points = q_weights.size();
        const std::size_t tdim = q_points.size() / q_weights.size();
        num_points_per_entity[i] = num_points;

        const std::array<std::size_t, 4> e_tab_shape
            = entity_element.tabulate_shape(0, num_points);
        std::vector<U> reference_entity_b(std::reduce(
            e_tab_shape.cbegin(), e_tab_shape.cend(), 1, std::multiplies{}));

        entity_element.tabulate(0, q_points, {num_points, tdim},
                                reference_entity_b);

        mdspan_t<const U, 4> basis_full(reference_entity_b.data(), e_tab_shape);
        auto phi = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
            basis_full, 0, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
            MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);

        auto [sub_geomb, sub_geom_shape]
            = basix::cell::sub_entity_geometry<U>(ct, dim, i);
        mdspan_t<const U, 2> coords(sub_geomb.data(), sub_geom_shape);

        // Push forward quadrature point from reference entity to reference
        // entity on cell
        const std::size_t offset = _points.size();
        _points.resize(_points.size() + num_points * coords.extent(1));
        mdspan_t<U, 2> entity_qp(_points.data() + offset, num_points,
                                 coords.extent(1));
        assert(coords.extent(1) == _tdim);
        dolfinx::math::dot(phi, coords, entity_qp);
        const std::size_t weights_offset = _weights.size();
        _weights.resize(_weights.size() + q_weights.size());
        std::copy(q_weights.cbegin(), q_weights.cend(),
                  std::next(_weights.begin(), weights_offset));
      }
      _entity_offset = std::vector<std::size_t>(_num_sub_entities + 1, 0);
      std::partial_sum(num_points_per_entity.begin(),
                       num_points_per_entity.end(),
                       std::next(_entity_offset.begin()));
    }
  }

  /// Constructor
  /// @param[in] ct        The cell type
  /// @param[in] degree    Degree of quadrature rule
  /// @param[in] dim       The dimension of the entity
  /// @param[in] type      Type of quadrature rule
  QuadratureRule(mesh::CellType ct, int degree, int dim,
                 basix::quadrature::type type
                 = basix::quadrature::type::Default)
      : QuadratureRule(dolfinx::mesh::cell_type_to_basix_type(ct), degree, dim,
                       type)
  {
  }

  /* Getter methods */
  /// Return degree of quadrature rule
  int degree() const { return _degree; };

  // Return the number of quadrature points
  std::size_t num_points() const { return _weights.size(); }

  /// Return the number of quadrature points per entity
  /// @param[in] i The local entity index
  std::size_t num_points(std::int8_t i) const
  {
    assert(i < _num_sub_entities);
    return _entity_offset[i + 1] - _entity_offset[i];
  }

  /// Return a list of quadrature points for each entity in the cell
  const std::vector<U>& points() const { return _points; }

  /// Return the quadrature points for the ith entity
  /// @param[in] i The local entity index
  mdspan_t<const U, 2> points(std::int8_t i) const
  {
    assert(i < _num_sub_entities);
    mdspan_t<const U, 2> all_points(_points.data(), _weights.size(), _tdim);
    return MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        all_points, std::pair(_entity_offset[i], _entity_offset[i + 1]),
        MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
  }

  /// Return a list of quadrature weights for each entity in the cell
  /// (using local entity index as in DOLFINx/Basix)
  const std::vector<U>& weights() const { return _weights; }

  /// Return the quadrature weights for the ith entity
  /// @param[in] i The local entity index
  std::span<const U> weights(std::int8_t i) const
  {
    assert(i < _num_sub_entities);
    return std::span(_weights.data() + _entity_offset[i],
                     _entity_offset[i + 1] - _entity_offset[i]);
  }

  /// Return dimension of entity in the quadrature rule
  int dim() const { return _dim; }

  /// Return the topological dimension of the quadrature rule
  std::size_t tdim() const { return _tdim; };

private:
  /* Variable definition */
  // Spacial dimensions
  std::size_t _tdim;
  int _dim;
  int _num_sub_entities; // Number of sub entities

  // Cell type
  basix::cell::type _cell_type;

  // Quadrature type
  basix::quadrature::type _type;

  // Quadrature degree
  int _degree;

  // Quadrature points
  // (for each entity. Shape (entity, num_points, tdim). Flattened row-major)
  std::vector<U> _points;

  // Quadrature weights
  // (for each entity on the cell)
  std::vector<U> _weights;

  // Offset for each entity
  std::vector<std::size_t> _entity_offset;
};

} // namespace dolfinx_eqlb::base