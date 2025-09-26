// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/utils.h>

#include <algorithm>
#include <memory>
#include <numeric>
#include <span>
#include <vector>

using namespace dolfinx;

namespace dolfinx_eqlb::base
{
template <dolfinx::scalar T, std::floating_point U>
class ProblemData
{
public:
  /// Initialize storage of data of (multiple) RHS
  ///
  /// Initializes storage of all forms, constants, the solution functions
  /// and linear forms of all simultaneously solved problems.
  ///
  /// @param solutions List of solution functions
  /// @param ls        The linear forms
  ProblemData(std::vector<std::shared_ptr<fem::Function<T, U>>>& solutions,
              const std::vector<std::shared_ptr<const fem::Form<T, U>>>& ls)
      : _nrhs(solutions.size()), _solutions(solutions), _ls(ls),
        _kernels(ls.size()), _offset_cnsts(ls.size() + 1, 0),
        _offset_coeffs(ls.size() + 1, 0), _cstrides(ls.size(), 0)
  {
    // Extract function space on which BCs are set
    const std::shared_ptr<const fem::FunctionSpace<U>> function_space
        = ls[0]->function_spaces().at(0);

    /* Initialize constants */
    std::int32_t size_cnst = 0, size_bc = 0;

    for (std::size_t i = 0; i < _nrhs; ++i)
    {
      // Get current linear form
      const fem::Form<T, U>& l_i = *(ls[i]);

      // Get sizes (constants, boundary conditions)
      std::int32_t size_cnst_i = std::accumulate(
          l_i.constants().cbegin(), l_i.constants().cend(), 0,
          [](std::int32_t sum, auto& c) { return sum + c->value.size(); });

      // Increment overall size
      size_cnst += size_cnst_i;

      // Set offset
      _offset_cnsts[i + 1] = size_cnst;
    }

    // Resize storage and set values
    _data_cnsts.resize(size_cnst, 0.0);
    set_data_constants();
  }

  /// Initialize integration kernels and related coefficients
  ///
  /// Set vector of function pointers onto integration kernels (LHS)
  /// over a given subdomain and initialize storage of related coefficients
  ///
  /// @param integral_type Integral type
  /// @param id            Id of integration-subdomain
  void initialize_kernel(fem::IntegralType integral_type, int id)
  {
    // Determine size of all coefficients
    std::int32_t size_coef = 0;

    // Loop over all RHS
    for (std::size_t i = 0; i < _nrhs; ++i)
    {
      /* Get Kernel */
      const fem::Form<T, U>& l_i = *(_ls[i]);
      _kernels[i] = _ls[i]->kernel(integral_type, id);

      /* Initialize data-structure coefficients */
      const std::vector<std::shared_ptr<const fem::Function<T, U>>>& coeffs_i
          = l_i.coefficients();
      const std::vector<int> offsets_i = l_i.coefficient_offsets();

      // Determine number of coefficients
      std::size_t num_entities = 0;
      int cstride = 0;

      if (!coeffs_i.empty())
      {
        cstride = offsets_i.back();

        num_entities = l_i.domain(integral_type, id).size();
        if (integral_type == fem::IntegralType::exterior_facet
            or integral_type == fem::IntegralType::interior_facet)
        {
          num_entities /= 2;
        }
      }

      // Set offset
      size_coef += cstride * num_entities;
      _offset_coeffs[i + 1] = size_coef;
    }

    // Extract coefficients
    _data_coeffs.resize(size_coef);
    set_data_coefficients(integral_type, id);
  }

  /* Setter functions */

  /* Getter functions */
  /// @return Number of linearforms
  int nrhs() const { return _nrhs; }

  /// Extract linearform l_i
  /// @param[in] index Id of the sub-problem
  /// @return The linearform
  const fem::Form<T, U>& l(int index) const { return *(_ls[index]); }

  /// Extract integration kernel of l_i
  /// @param index Id of the sub-problem
  /// @return The integration kernel
  const std::function<void(T*, const T*, const T*, const U*, const int*,
                           const std::uint8_t*)>&
  kernel(int index) const
  {
    return _kernels[index];
  }

  /// Extract a solution function
  /// @param index Id of the sub-problem
  /// @return The solution function
  fem::Function<T, U>& solution_function(int index) const
  {
    return *(_solutions[index]);
  }

  /// Extract constants of l_i (constant version)
  /// @param index Id of the sub-problem
  /// @return Constants of linearform l_i
  std::span<const T> constants(int index) const
  {
    return std::span<const T>(_data_cnsts.data() + _offset_cnsts[index],
                              _offset_cnsts[index + 1] - _offset_cnsts[index]);
  }

  /// Extract coefficients of l_i
  /// @param index Id of the sub-problem
  /// @return Coefficients of linearform l_i
  std::span<T> coefficients(int index)
  {
    return std::span<T>(_data_coeffs.data() + _offset_coeffs[index],
                        _offset_coeffs[index + 1] - _offset_coeffs[index]);
  }

  /// Extract coefficients of l_i (constant version)
  /// @param index Id of the sub-problem
  /// @return Coefficients of linearform l_i
  std::span<const T> coefficients(int index) const
  {
    return std::span<const T>(_data_coeffs.data() + _offset_coeffs[index],
                              _offset_coeffs[index + 1]
                                  - _offset_coeffs[index]);
  }

  /// Extract cstride (coefficients) of l_i
  /// @param index Id of the sub-problem
  /// @return cstride of linearform l_i
  int cstride(int index) { return _cstrides[index]; }

protected:
  /* Handle constants */
  void set_data_constants()
  {
    for (std::size_t i = 0; i < _nrhs; ++i)
    {
      // The linear form l_i
      const fem::Form<T, U>& l_i = *(_ls[i]);

      // Extract data
      std::int32_t offset = 0;

      for (auto& cnst : l_i.constants())
      {
        const std::vector<T>& value = cnst->value;
        std::span<T> cnst_values
            = std::span<T>(_data_cnsts.data() + _offset_cnsts[i],
                           _offset_cnsts[i + 1] - _offset_cnsts[i]);

        std::ranges::copy(value, std::next(cnst_values.begin(), offset));
        offset += value.size();
      }
    }
  }

  /* Handle coefficients */
  void set_data_coefficients(fem::IntegralType integral_type, int id)
  {
    for (std::size_t i = 0; i < _nrhs; ++i)
    {
      // Get current linear form
      const fem::Form<T, U>& l_i = *(_ls[i]);

      const std::vector<std::shared_ptr<const fem::Function<T, U>>>&
          coefficients_i
          = l_i.coefficients();
      const std::vector<int> offsets_i = l_i.coefficient_offsets();

      // Storage for current coefficients
      std::span<T> data_coef = coefficients(i);

      // Extract coefficients by default methode
      auto interm_coefficients = fem::allocate_coefficient_storage(l_i);
      fem::pack_coefficients(l_i, interm_coefficients);

      auto& [coeffs_i, cstride_i] = interm_coefficients.at({integral_type, id});

      // Move data into storage
      std::move(coeffs_i.begin(), coeffs_i.end(), data_coef.begin());
      _cstrides[i] = cstride_i;
    }
  }

  // Number of equilibrations
  const int _nrhs;

  // Linearforms
  std::vector<std::shared_ptr<const fem::Form<T>>> _ls;

  // Integration kernels
  std::vector<std::function<void(T*, const T*, const T*, const U*, const int*,
                                 const std::uint8_t*)>>
      _kernels;

  // Solution functions
  std::vector<std::shared_ptr<fem::Function<T>>>& _solutions;

  /* Storage coefficients and constants */
  // Informations
  std::vector<int> _cstrides;

  // Data
  std::vector<T> _data_coeffs, _data_cnsts;

  // Offset
  std::vector<std::int32_t> _offset_coeffs, _offset_cnsts;
};
} // namespace dolfinx_eqlb::base