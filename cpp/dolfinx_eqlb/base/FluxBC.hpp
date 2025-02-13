// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "mdspan.hpp"

#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/fem/Expression.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/graph/AdjacencyList.h>

#include <algorithm>
#include <array>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <span>
#include <vector>

using namespace dolfinx;

namespace dolfinx_eqlb::base
{

template <dolfinx::scalar T, std::floating_point U>
class FluxBC
{
public:
  /// Storage of boundary conditions (no quadrature required)
  ///
  /// Pass a boundary condition for the flux-space from the python
  /// interface to the c++ level. Its precompiled normal-trace is stored,
  /// together with the boundary facets and information, required for the
  /// calculation the actual boundary DOFs. If negative values are passed
  /// for the quadrature degree, no projection is performed.
  ///
  /// @param boundary_expression The boundary expression
  /// @param constants           The constants
  /// @param coefficients        The coefficients
  /// @param n_eval_per_fct      The number of point-evaluations per facet
  /// @param quadrature_degree   The quadrature degree used for the projection
  ///                            of the normal-trace into a polynomial space
  /// @param is_timedependent    True, if the boundary value is time-dependent
  /// @param has_time_function   True, if the time-dependency is covered by a
  ///                            multiplicative time function
  /// @param boundary_facets     The boundary facets
  /// @param n_bdofs             The number of boundary DOFs
  FluxBC(std::shared_ptr<const fem::Expression<T, U>> value,
         const std::vector<std::int32_t>& facets,
         const fem::FunctionSpace<U>& V)
      : _expression(value), _nfcts(facets.size()), _is_zero(false),
        _projection_required(false), _facets(facets), _quadrature_degree(0)
  {
    // Set up entities (pair of cell and local facet id)
    if (!(_is_zero))
    {
      // The mesh
      std::shared_ptr<const mesh::Mesh<U>> mesh = V.mesh();

      // The spatial dimension
      const int gdim = mesh->geometry().dim();

      // The connectivity between facets and cells
      std::shared_ptr<const graph::AdjacencyList<std::int32_t>> fct_to_cell
          = mesh->topology()->connectivity(gdim - 1, gdim);
      std::shared_ptr<const graph::AdjacencyList<std::int32_t>> cell_to_fct
          = mesh->topology()->connectivity(gdim, gdim - 1);

      // Resize storage
      _entities.resize(2 * _nfcts);

      for (int i = 0; i < _nfcts; ++i)
      {
        // Facet and adjacent cell
        std::int32_t f = facets[i];
        std::int32_t c = fct_to_cell->links(f)[0];

        // All facets on cell
        std::span<const std::int32_t> fs = cell_to_fct->links(c);

        // Add to entity list
        _entities[2 * i] = c;
        _entities[2 * i + 1]
            = std::distance(fs.begin(), std::find(fs.begin(), fs.end(), f));
      }
    }

    // Create entities

    // if ((_is_timedependent) && (_has_time_function))
    // {
    //   _boundary_values.resize(n_bdofs);
    // }
    // std::cout << "Test boundary kernel: " << std::endl;
    // std::vector<T> values(3);
    // std::vector<U> coordinates(9);

    // // _boundary_kernel(values.data(), nullptr, nullptr, coordinates.data(),
    // //                  nullptr, nullptr);
    // std::vector<std::int32_t> entities{0, 1, 1, 0, 2, 1};
    // _expression->eval(*V.mesh().get(), entities, values, {3, 1});

    // std::cout << "Calculated values: " << values[0] << ", " << values[1] <<
    // ", "
    //           << values[2] << std::endl;
  }

  /* Getter functions */
  /// Return if the BC is a function of time
  /// @param[out] is_timedependent True, if the BC is a function of time
  bool is_timedependent() const
  {
    // return _is_timedependent;
    throw std::runtime_error("Not implemented!");
  }

  /// Return if the BC has a time function
  /// @param[out] is_timedependent True, if the BC has a time function
  bool has_time_function() const
  {
    // return _has_time_function;
    throw std::runtime_error("Not implemented!");
  }

  /// Return the number of boundary facets
  /// @param[out] nfcts The number of boundary facets
  std::int32_t num_facets() const { return _nfcts; }

  /// Return the number of point evaluations per facet
  /// @param[out] num_eval The number of point evaluations per facet
  int num_eval_per_facet() const
  {
    return _expression->X().second[0] * _expression->value_size();
  }

  /// Check if projection is required
  /// @param[out] projection_id The projection id
  bool projection_required() const { return _projection_required; }

  /// Get quadrature degree for projection
  /// @param[out] projection_id The projection id
  int quadrature_degree() const { return _quadrature_degree; }

  /// Return list of boundary entities

  /// Each entity constains the boundary cell and the cell-local facet id of the
  /// boundary facet.

  /// @param[out] entities The list of boundary entities
  std::span<const std::int32_t> entities() const
  {
    return std::span<const std::int32_t>(_entities.data(), _entities.size());
  }

  std::span<const std::int32_t> facets() const
  {
    return std::span<const std::int32_t>(_facets.data(), _facets.size());
  }

  /// Pack constants to evaluate boundary values
  /// @returns The constants vector
  std::vector<T> pack_constants()
  {
    return fem::pack_constants(*_expression.get());
  }

  /// Extract coefficients for each boundary cell
  ///
  /// Extract coefficients for each cell and store values in flattened array.
  /// The array structure is relative to the boundary facets/cells affected by
  /// this condition.
  ///
  /// @returns The coefficients and its cstride
  std::pair<std::vector<T>, int> pack_coefficients()
  {
    return fem::pack_coefficients(*_expression.get(), _entities, 2);
  }

  /// Extract the boundary kernel
  /// @return The boundary kernel
  const std::function<void(T*, const T*, const T*, const U*, const int*,
                           const uint8_t*)>&
  get_tabulate_expression() const
  {
    return _expression->get_tabulate_expression();
  }

protected:
  /* Variable definitions */
  // Identifiers
  // const bool _is_zero, _is_timedependent, _has_time_function,
  //     _projection_required;
  const bool _is_zero, _projection_required;

  // Quadrature degree
  const int _quadrature_degree;

  // Boundary facets
  std::vector<std::int32_t> _entities;
  const std::vector<std::int32_t> _facets;
  const std::int32_t _nfcts;

  // The boundary expression
  std::shared_ptr<const fem::Expression<T, U>> _expression;

  // The boundary values
  std::vector<T> _values;
};

} // namespace dolfinx_eqlb::base