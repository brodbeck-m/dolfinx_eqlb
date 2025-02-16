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
#include <stdexcept>
#include <vector>

using namespace dolfinx;

namespace dolfinx_eqlb::base
{

enum class TimeType : int
{
  stationary = 0,
  timefunction = 1,
  timedependent = 2,
};

template <dolfinx::scalar T, std::floating_point U>
class FluxBC
{
public:
  /// Storage of boundary conditions
  ///
  /// Pass a boundary condition for the flux-space from the python
  /// interface to the c++ level. Its precompiled normal-trace is stored,
  /// together with the boundary facets and information, required for the
  /// calculation the actual boundary DOFs. If negative values are passed
  /// for the quadrature degree, no projection is performed.
  ///
  /// @param value              The boundary expression
  /// @param facets             The boundary facets
  /// @param V                  The function space
  /// @param quadrature_degree  The quadrature degree used for the projection
  /// @param tbehaviour         The behaviour in time
  FluxBC(std::shared_ptr<const fem::Expression<T, U>> value,
         const std::vector<std::int32_t>& facets,
         std::shared_ptr<const fem::FunctionSpace<U>> V,
         const int quadrature_degree, const TimeType tbehaviour)
      : _expression(value), _facets(facets), _nfcts(facets.size()), _V(V),
        _quadrature_degree(quadrature_degree),
        _projection_required((quadrature_degree < 0) ? false : true),
        _is_zero(false), _tbehaviour(tbehaviour)
  {
    if (_tbehaviour == TimeType::timefunction)
    {
      // The spatial dimension
      const int gdim = _V->mesh()->geometry().dim();

      // The degree of the function space
      const int deg = _V->element()->basix_element().degree();
      const int ndofs_per_fct = (gdim == 2) ? deg : 0.5 * deg * (deg + 1);

      // Initialise storage of initial boundary values
      _initial_boundary_dofs.resize(_nfcts * ndofs_per_fct);
    }
  }

  /// Storage of boundary conditions (no quadrature required)
  ///
  /// This constructor is used for homogenous BCs, where no boundary expression
  /// is required.
  ///
  /// @param facets             The boundary facets
  FluxBC(const std::vector<std::int32_t>& facets)
      : _expression(nullptr), _facets(facets), _nfcts(facets.size()),
        _V(nullptr), _quadrature_degree(-1), _projection_required(false),
        _is_zero(true), _tbehaviour(TimeType::stationary)
  {
  }

  /* Setter methods */
  /// Store the initial boundary DOFs
  ///
  /// These initial values are required, such that the time-function can later
  /// be applied!
  ///
  /// @param[in] fct  The (BC local) facet id
  /// @param[in] dofs The boundary DOFs of the flux field on fct
  void set_initial_boundary_dofs(const int fct, std::span<const T> dofs)
  {
    if (_tbehaviour == TimeType::timefunction)
    {
      // Number of DOFs per facet
      const int ndofs_per_fct = dofs.size();

      // Store dofs
      const std::int32_t offs = fct * ndofs_per_fct;

      for (int i = 0; i < ndofs_per_fct; ++i)
      {
        _initial_boundary_dofs[offs + i] = dofs[i];
      }
    }
  }

  /* Getter functions */
  /// Marker for homogenous BC
  /// @return True, if the BC is homogenous
  bool is_zero() const { return _is_zero; }

  /// The transient behaviour
  /// @return The transient behaviour of the boundary expression
  TimeType transient_behaviour() const { return _tbehaviour; }

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

  /// Return list of boundary facets
  /// @param[out] entities The list of boundary entities
  std::span<const std::int32_t> facets() const
  {
    return std::span<const std::int32_t>(_facets.data(), _facets.size());
  }

  /// Get storage of the initial boundary DOFs
  /// @param[in]  ndofs_per_fct         The number of DOFs per facet
  /// @param[out] initial_boundary_dofs The initial boundary DOFs
  mdspan_t<const T, 2> initial_boundary_dofs(const int ndofs_per_fct) const
  {
    if (_tbehaviour == TimeType::timefunction)
    {
      return mdspan_t<const T, 2>(_initial_boundary_dofs.data(), _nfcts,
                                  static_cast<std::size_t>(ndofs_per_fct));
    }
    else
    {
      throw std::runtime_error(
          "Initial boundary values are not available for this BC.");
    }
  }

  /// Get storage of the initial boundary DOFs
  /// @param[out] initial_boundary_dofs The initial boundary DOFs
  std::span<T> initial_boundary_dofs()
  {
    if (_tbehaviour == TimeType::timefunction)
    {
      return std::span<T>(_initial_boundary_dofs.data(),
                          _initial_boundary_dofs.size());
    }
    else
    {
      throw std::runtime_error(
          "Initial boundary values are not available for this BC.");
    }
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
  /// @param[in] local_factet_ids The local facet ids
  /// @returns The coefficients and their stride
  std::pair<std::vector<T>, int>
  pack_coefficients(std::span<const std::int8_t> local_factet_ids)
  {
    // --- Create entities
    std::vector<std::int32_t> entities(2 * _nfcts, 0);

    // The spatial dimension
    const int gdim = _V->mesh()->geometry().dim();

    // The connectivity between facets and cells
    std::shared_ptr<const graph::AdjacencyList<std::int32_t>> fct_to_cell
        = _V->mesh()->topology()->connectivity(gdim - 1, gdim);

    for (int i = 0; i < _nfcts; ++i)
    {
      // The facet
      std::int32_t f = _facets[i];

      // Add to entity list
      entities[2 * i] = fct_to_cell->links(f)[0];
      entities[2 * i + 1] = local_factet_ids[f];
    }

    // Flatten coefficients
    return fem::pack_coefficients(*_expression.get(), entities, 2);
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
  const bool _is_zero, _projection_required;

  // Behaviour in time
  const TimeType _tbehaviour;

  // The Quadrature degree
  const int _quadrature_degree;

  // The boundary facets
  const std::vector<std::int32_t> _facets;
  const std::int32_t _nfcts;

  // The boundary expression
  std::shared_ptr<const fem::Expression<T, U>> _expression;
  std::vector<T> _initial_boundary_dofs;

  // The function space
  std::shared_ptr<const fem::FunctionSpace<U>> _V;
};

} // namespace dolfinx_eqlb::base