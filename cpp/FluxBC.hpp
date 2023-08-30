#pragma once

#include "utils.hpp"

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

namespace dolfinx_eqlb
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
  FluxBC(std::shared_ptr<const fem::FunctionSpace> function_space,
         const std::vector<std::int32_t>& boundary_facets,
         std::function<void(T*, const T*, const T*, const scalar_value_type_t*,
                            const int*, const std::uint8_t*)>
             boundary_value,
         int n_bceval_per_fct, bool projection_required,
         std::vector<std::shared_ptr<const fem::Function<T>>> coefficients,
         std::vector<int> positions_of_coefficients,
         std::vector<std::shared_ptr<const fem::Constant<T>>> constants)
      : _function_space(function_space), _fcts(boundary_facets),
        _nfcts(boundary_facets.size()), _boundary_kernel(boundary_value),
        _cstide_eval(n_bceval_per_fct),
        _projection_required(projection_required), _coefficients(coefficients),
        _coefficient_positions(positions_of_coefficients), _constants(constants)
  {
  }

  /* Getter functions */
  /// Return the flux function space
  /// @param[out] projection_id The projection id
  bool projection_required() const { return _projection_required; }

  /// Return the number of boundary facets
  /// @param[out] nfcts The number of boundary facets
  std::int32_t num_facets() const { return _nfcts; }

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
      std::int32_t size_constants = size_constants_data<T>(_constants);

      // Initialise storage
      constants.resize(size_constants);

      // Extract coefficients
      extract_constants_data<T>(_constants, constants);
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
          _coefficients.cbegin(), _coefficients.cend(), 0,
          [](int sum, auto& f)
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
          std::span<T> data
              = std::span<T>(coefficients.data() + offs_coef, space_dimension);

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

  // Projection id (true, if projection is required)
  const bool _projection_required;
};

} // namespace dolfinx_eqlb