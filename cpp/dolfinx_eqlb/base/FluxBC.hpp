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

template <typename T>
class FluxBC
{

  template <typename X, typename = void>
  struct scalar_value_type
  {
    typedef X value_type;
  };
  template <typename X>
  struct scalar_value_type<X, std::void_t<typename X::value_type>>
  {
    typedef typename X::value_type value_type;
  };
  using scalar_value_type_t = typename scalar_value_type<T>::value_type;

public:
  /// Temporary storage of boundary conditions (no quadrature required)
  ///
  /// Passes the boundary conditions for the flux-space from the python
  /// interface to the c++ level. The precompiled normal-trace of the flux is
  /// stored, together with the boundary facets an some informations, required
  /// for calculation the boundary DOFs therefrom.
  ///
  /// @param function_space            The collapsed flux FunctionSpace
  /// @param boundary_facets           The mesh-facets on which the flux is
  ///                                  prescribed
  /// @param boundary_value            The pre-complied normal-trace of the flux
  /// @param n_bceval_per_fct          Number of evaluations per facet
  /// @param coefficients              The vector of coefficient data
  /// @param positions_of_coefficients The positions each data set within the
  ///                                  extracted vector of coefficients.
  /// @param constants                 The constants
  FluxBC(std::shared_ptr<const fem::FunctionSpace> function_space,
         const std::vector<std::int32_t>& boundary_facets,
         std::function<void(T*, const T*, const T*, const scalar_value_type_t*,
                            const int*, const std::uint8_t*)>
             boundary_value,
         int n_bceval_per_fct,
         std::vector<std::shared_ptr<const fem::Function<T>>> coefficients,
         std::vector<int> positions_of_coefficients,
         std::vector<std::shared_ptr<const fem::Constant<T>>> constants)
      : _function_space(function_space), _fcts(boundary_facets),
        _nfcts(boundary_facets.size()), _boundary_kernel(boundary_value),
        _cstide_eval(n_bceval_per_fct), _projection_required(false),
        _quadrature_degree(0), _coefficients(coefficients),
        _coefficient_positions(positions_of_coefficients), _constants(constants)
  {
  }

  /// Temporary storage of boundary conditions (quadrature required)
  ///
  /// Passes the boundary conditions for the flux-space from the python
  /// interface to the c++ level. The precompiled normal-trace of the flux is
  /// stored, together with the boundary facets an some informations, required
  /// for calculation the boundary DOFs therefrom.
  ///
  /// @param function_space            The collapsed flux FunctionSpace
  /// @param boundary_facets           The mesh-facets on which the flux is
  ///                                  prescribed
  /// @param boundary_value            The pre-complied normal-trace of the flux
  /// @param n_bceval_per_fct          Number of evaluations per facet
  /// @param quadrature_degree         The quadrature degree for the projection
  ///                                  of the boundary condition
  /// @param coefficients              The vector of coefficient data
  /// @param positions_of_coefficients The positions each data set within the
  ///                                  extracted vector of coefficients.
  /// @param constants                 The constants
  FluxBC(std::shared_ptr<const fem::FunctionSpace> function_space,
         const std::vector<std::int32_t>& boundary_facets,
         std::function<void(T*, const T*, const T*, const scalar_value_type_t*,
                            const int*, const std::uint8_t*)>
             boundary_value,
         int n_bceval_per_fct, int quadrature_degree,
         std::vector<std::shared_ptr<const fem::Function<T>>> coefficients,
         std::vector<int> positions_of_coefficients,
         std::vector<std::shared_ptr<const fem::Constant<T>>> constants)
      : _function_space(function_space), _fcts(boundary_facets),
        _nfcts(boundary_facets.size()), _boundary_kernel(boundary_value),
        _cstide_eval(n_bceval_per_fct), _projection_required(true),
        _quadrature_degree(quadrature_degree), _coefficients(coefficients),
        _coefficient_positions(positions_of_coefficients), _constants(constants)
  {
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

  /// Return the flux function-space
  /// @param[out] function_space The flux function-space
  std::shared_ptr<const fem::FunctionSpace> function_space() const
  {
    return _function_space;
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

      // Extract connectivity facets->cell
      int dim = _function_space->mesh()->geometry().dim();
      std::shared_ptr<const graph::AdjacencyList<std::int32_t>> fct_to_cell
          = _function_space->mesh()->topology().connectivity(dim - 1, dim);

      // Extract coefficients
      std::int32_t offs_cstride = 0;

      for (int i : _coefficient_positions)
      {
        // Extract function
        std::shared_ptr<const fem::Function<T>> function = _coefficients[i];

        // Data storage
        std::span<const T> x_coeff = function->x()->array();

        // Function space
        std::shared_ptr<const fem::FunctionSpace> function_space
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
  const std::function<void(T*, const T*, const T*, const scalar_value_type_t*,
                           const int*, const std::uint8_t*)>&
  boundary_kernel() const
  {
    return _boundary_kernel;
  }

protected:
  /* Variable definitions */
  // Boundary facets
  const std::int32_t _nfcts;
  const std::vector<std::int32_t> _fcts;

  // The flux function space
  std::shared_ptr<const fem::FunctionSpace> _function_space;

  // Kernel (executable c++ code)
  std::function<void(T*, const T*, const T*, const scalar_value_type_t*,
                     const int*, const std::uint8_t*)>
      _boundary_kernel;

  // Coefficients associated with the BCs
  std::vector<std::shared_ptr<const fem::Function<T>>> _coefficients;
  const std::vector<int> _coefficient_positions;

  // Constants associated with the BCs
  std::vector<std::shared_ptr<const fem::Constant<T>>> _constants;

  // Number of data-points per facet
  const int _cstide_eval;

  // Projection
  const bool _projection_required;
  const int _quadrature_degree;
};

} // namespace dolfinx_eqlb::base