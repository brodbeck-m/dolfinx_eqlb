#pragma once

#include "utils.hpp"

#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/graph/AdjacencyList.h>

#include <algorithm>
#include <array>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <span>
#include <vector>

using namespace dolfinx;

namespace dolfinx_adaptivity::equilibration
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
        _projection_required(projection_required),
        _c_positions(positions_of_coefficients)
  {
    /* Initialise constants */
    if (constants.size() > 0)
    {
      // Number of constants
      std::int32_t size_constants
          = dolfinx_adaptivity::size_constants_data<T>(constants);

      // Resize storage vector
      _constants.resize(size_constants, 0);

      // Extract constants
      dolfinx_adaptivity::extract_constants_data<T>(constants, _constants);
    }

    /* Initialise coefficients */
    if (coefficients.size() > 0)
    {
      // Number of coefficients per element
      _cstride_coefficients = std::accumulate(
          coefficients.cbegin(), coefficients.cend(), 0,
          [](int sum, auto& f)
          { return sum + f->function_space()->element()->space_dimension(); });

      // Resize storage vector
      _coefficients.resize(_nfcts * _cstride_coefficients, 0);

      // Extract coefficients
      extract_coefficients_data(coefficients);
    }

    /* Initialise storage of projection */
    if (_projection_required)
    {
      // Number of projected DOFs per facet
      if (_function_space->mesh()->geometry().dim() == 2)
      {
        _cstride_proj = _function_space->element()->basix_element().degree();
      }
      else if (_function_space->mesh()->geometry().dim() == 3)
      {
        const int degree = _function_space->element()->basix_element().degree();
        _cstride_proj = 0.5 * (degree + 1) * (degree + 2);
      }
      else
      {
        throw std::runtime_error("Unsupported dimension");
      }

      // Resize storage vector
      _coefficients.resize(_nfcts * _cstride_proj, 0);
    }
  }

protected:
  /// Compute size of coefficient for each boundary cell
  ///
  /// Extract coefficients for each boundary cell and store values in flattened
  /// array. Array structure relative to counter on boundary cells for this
  /// boundary.
  ///
  /// @param coefficients Vector with fem::Function objects
  void extract_coefficients_data(
      std::vector<std::shared_ptr<const fem::Function<T>>> coefficients)
  {
    // Extract connectivity facets->cell
    int dim = _function_space->mesh()->geometry().dim();
    std::shared_ptr<const graph::AdjacencyList<std::int32_t>> fct_to_cell
        = _function_space->mesh()->topology().connectivity(dim - 1, dim);

    // Extract coefficients
    std::int32_t offs_cstride = 0;

    for (int i : _c_positions)
    {
      // Extract function
      std::shared_ptr<const fem::Function<T>> function = coefficients[i];

      // Data storage
      std::span<const T> x_coeff = function->x()->array();

      // Function space
      std::shared_ptr<const fem::FunctionSpace> function_space
          = function->function_space();

      // cstride of current coefficients
      const int bs = function_space->element()->block_size();
      const int space_dimension = function_space->element()->space_dimension();
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
        std::int32_t offs_coef = i * _cstride_coefficients + offs_cstride;
        std::span<T> data
            = std::span<T>(_coefficients.data() + offs_coef, space_dimension);

        // Extract coefficient data
        for (std::size_t j = 0; j < map_dimension; ++j)
        {
          // DOF id
          std::int32_t offs_dof = bs * dofs[j];
          std::int32_t offs = offs_coef + j * bs;

          // Copy DOF
          for (std::size_t k = 0; k < bs; ++k)
          {
            _coefficients[offs + k] = x_coeff[offs_dof + k];
          }
        }
      }

      // Update cstride
      offs_cstride += space_dimension;
    }
  }

  /* Variable definitions */
  // The flux function space
  std::shared_ptr<const fem::FunctionSpace> _function_space;

  // Boundary facets
  const int _nfcts;
  const std::vector<std::int32_t> _fcts;

  // Kernel (executable c++ code)
  std::function<void(T*, const T*, const T*, const scalar_value_type_t*,
                     const int*, const std::uint8_t*)>
      _boundary_kernel;

  // Coefficients associated with the BCs
  std::vector<T> _coefficients;
  int _cstride_coefficients;

  const std::vector<int> _c_positions;

  // Constants associated with the BCs
  std::vector<T> _constants;

  // Number of data-points per facet
  const int _cstide_eval;

  // Projection id (true, if projection is required)
  const bool _projection_required;

  // Storage of projected BC
  std::vector<T> _projected_bc;
  int _cstride_proj;
};

} // namespace dolfinx_adaptivity::equilibration