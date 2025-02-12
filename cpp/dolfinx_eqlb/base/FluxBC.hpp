// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
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
  /// @param is_zero             True, if the boundary value is zero
  /// @param is_timedependent    True, if the boundary value is time-dependent
  /// @param has_time_function   True, if the time-dependency is covered by a
  ///                            multiplicative time function
  /// @param boundary_facets     The boundary facets
  /// @param n_bdofs             The number of boundary DOFs
  FluxBC(std::function<void(T*, const T*, const T*, const U*, const int*,
                            const std::uint8_t*)>
             boundary_expression,
         std::vector<std::shared_ptr<const fem::Constant<T>>> constants,
         std::vector<std::shared_ptr<const fem::Function<T, U>>> coefficients,
         int n_eval_per_fct, int quadrature_degree, bool is_zero,
         bool is_timedependent, bool has_time_function,
         const std::vector<std::int32_t>& boundary_facets, int n_bdofs)
      : _boundary_kernel(boundary_expression), _constants(constants),
        _coefficients(coefficients), _quadrature_degree(quadrature_degree),
        _projection_required((quadrature_degree < 0) ? false : true),
        _cstide_eval(n_eval_per_fct), _is_zero(is_zero),
        _is_timedependent(is_timedependent),
        _has_time_function(has_time_function), _fcts(boundary_facets),
        _nfcts(boundary_facets.size()), _nbdofs(n_bdofs)
  {
    if ((_is_timedependent) && (_has_time_function))
    {
      _boundary_values.resize(n_bdofs);
    }
  }

  /* Getter functions */
  /// Return the number of boundary facets
  /// @param[out] nfcts The number of boundary facets
  std::int32_t num_facets() const { return _nfcts; }

  /// Return the number of point evaluations per facet
  /// @param[out] num_eval The number of point evaluations per facet
  int num_eval_per_facet() const { return _cstide_eval; }

  /// Check if projection is required
  /// @param[out] projection_id The projection id
  bool projection_required() const { return _projection_required; }

  /// Get quadrature degree for projection
  /// @param[out] projection_id The projection id
  int quadrature_degree() const { return _quadrature_degree; }

  /// Return list of boundary facets
  /// @param[out] fcts The list of boundary facets
  std::span<const std::int32_t> facets() const
  {
    return std::span<const std::int32_t>(_fcts.data(), _fcts.size());
  }

  /// Extract constants for evaluation of boundary kernel
  /// @returns The constants vector
  std::vector<T> extract_constants()
  {
    // Initialise constants
    std::vector<T> constants;

    if (_constants.size() > 0)
    {
      // Calculate the number of constants
      std::int32_t size_constants = std::accumulate(
          _constants.cbegin(), _constants.cend(), 0,
          [](std::int32_t sum, auto& c) { return sum + c->value.size(); });

      // Initialise storage
      constants.resize(size_constants);

      // Extract coefficients
      std::int32_t offset = 0;

      for (auto& cnst : _constants)
      {
        const std::vector<T>& value = cnst->value;
        std::copy(value.begin(), value.end(),
                  std::next(constants.begin(), offset));
        offset += value.size();
      }
    }

    return std::move(constants);
  }

  /// Extract coefficients for each boundary cell
  ///
  /// Extract coefficients for each boundary cell and store values in flattened
  /// array. Array structure relative to counter on boundary cells for this
  /// boundary.
  ///
  /// @returns std::pair<cstride, The coefficient vector>
  std::pair<std::int32_t, std::vector<T>> extract_coefficients()
  {
    // Initialise storage
    std::int32_t cstride = 0;
    std::vector<T> coefficients;

    if (_coefficients.size() > 0)
    {
      // Number of constants
      cstride = std::accumulate(
          _coefficients.cbegin(), _coefficients.cend(), 0, [](int sum, auto& f)
          { return sum + f->function_space()->element()->space_dimension(); });

      // Initialise storage
      coefficients.resize(_nfcts * cstride);

      // The mesh
      std::shared_ptr<mesh::Mesh<U>> mesh
          = _coefficients[0]->function_space()->mesh();

      // Extract connectivity facets->cell
      int dim = mesh->geometry().dim();
      std::shared_ptr<const graph::AdjacencyList<std::int32_t>> fct_to_cell
          = mesh->topology().connectivity(dim - 1, dim);

      // Extract coefficients
      std::int32_t offs_cstride = 0;

      for (std::shared_ptr<const fem::Function<T>> function : _coefficients)
      {
        // Data storage
        std::span<const T> x_coeff = function->x()->array();

        // Function space
        std::shared_ptr<const fem::FunctionSpace<U>> function_space
            = function->function_space();

        // cstride of current coefficients
        const int bs = function_space->element()->block_size();
        const int space_dimension
            = function_space->element()->space_dimension();
        const int map_dimension = space_dimension / bs;

        for (std::size_t i = 0; i < _nfcts; ++i)
        {
          // Global facet id
          std::int32_t fct = _fcts[i];

          // Get cell, adjacent to facet
          std::int32_t c = fct_to_cell->links(fct)[0];

          // DOFmap of cell
          std::span<const int32_t> dofs
              = function_space->dofmap()->list().links(c);

          // Flattened storage
          std::int32_t offs_coef = i * cstride + offs_cstride;

          // Extract coefficient data
          for (std::size_t j = 0; j < map_dimension; ++j)
          {
            // DOF id
            std::int32_t offs_dof = bs * dofs[j];
            std::int32_t offs = offs_coef + j * bs;

            // Copy DOF
            for (std::size_t k = 0; k < bs; ++k)
            {
              coefficients[offs + k] = x_coeff[offs_dof + k];
            }
          }
        }

        // Update cstride
        offs_cstride += space_dimension;
      }
    }

    return {cstride, std::move(coefficients)};
  }

  /// Extract the boundary kernel
  /// @return The boundary kernel
  const std::function<void(T*, const T*, const T*, const U*, const int*,
                           const std::uint8_t*)>&
  boundary_kernel() const
  {
    return _boundary_kernel;
  }

protected:
  /* Variable definitions */
  // Identifiers
  const bool _is_zero, _is_timedependent, _has_time_function,
      _projection_required;

  // Quadrature degree
  const int _quadrature_degree;

  // Boundary facets
  const std::vector<std::int32_t> _fcts;
  const std::int32_t _nfcts, _nbdofs;

  // Kernel (executable c++ code)
  std::function<void(T*, const T*, const T*, const U*, const int*,
                     const std::uint8_t*)>
      _boundary_kernel;

  // Number of evaluation-points per facet
  const int _cstide_eval;

  // Coefficients associated with the BCs
  std::vector<std::shared_ptr<const fem::Function<T, U>>> _coefficients;

  // Constants associated with the BCs
  std::vector<std::shared_ptr<const fem::Constant<T>>> _constants;

  // Storage for the boundary values (only for BCs with time-function)
  std::vector<T> _boundary_values;
};

} // namespace dolfinx_eqlb::base