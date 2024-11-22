// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <basix/finite-element.h>
#include <basix/mdspan.hpp>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/CoordinateElement.h>

#include <algorithm>
#include <array>
#include <functional>
#include <iostream>
#include <memory>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

using namespace dolfinx;

namespace dolfinx_eqlb
{
// ------------------------------------------------------------------------------

/* Definition of mdspans */
namespace stdex = std::experimental;

template <typename T, std::size_t d>
using mdspan_t = stdex::mdspan<T, stdex::dextents<std::size_t, d>>;

template <typename T, std::size_t d>
using smdspan_t
    = stdex::mdspan<T, stdex::dextents<std::size_t, d>, stdex::layout_stride>;

// ------------------------------------------------------------------------------

// ------------------------------------------------------------------------------

/* Interface kernel functions */
template <typename T, bool asmbl_systmtrx>
using kernel_fn = std::function<void(
    stdex::mdspan<T, stdex::dextents<std::size_t, 2>>, std::span<const T>,
    stdex::mdspan<const std::int32_t, stdex::dextents<std::size_t, 2>,
                  stdex::layout_stride>,
    const std::uint8_t, const double,
    stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>)>;

template <typename T>
using kernel_fn_schursolver = std::function<void(
    stdex::mdspan<T, stdex::dextents<std::size_t, 2>>,
    stdex::mdspan<T, stdex::dextents<std::size_t, 2>>, std::span<T>,
    std::span<T>, std::span<const T>,
    stdex::mdspan<const std::int32_t, stdex::dextents<std::size_t, 2>,
                  stdex::layout_stride>,
    const std::uint8_t, const double,
    stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>, const bool)>;

// ------------------------------------------------------------------------------

// ------------------------------------------------------------------------------

/* Extract coefficients of a form */

/// Compute size of coefficient data
/// @tparam T Scalar value type
/// @param constants Vector with fem::Constant objects
/// @return Size of flattened storage vector
template <typename T>
std::int32_t size_constants_data(
    const std::vector<std::shared_ptr<const fem::Constant<T>>>& constants)
{
  std::int32_t size = std::accumulate(constants.cbegin(), constants.cend(), 0,
                                      [](std::int32_t sum, auto& c)
                                      { return sum + c->value.size(); });
  return size;
}

/// Extract coefficients into flattened storage
/// @tparam T Scalar value type
/// @param constants      Vector with fem::Constant objects
/// @param data_constants Flattened storage for coefficients
template <typename T>
void extract_constants_data(
    const std::vector<std::shared_ptr<const fem::Constant<T>>>& constants,
    std::span<T> data_constants)
{
  std::int32_t offset = 0;

  for (auto& cnst : constants)
  {
    const std::vector<T>& value = cnst->value;
    std::copy(value.begin(), value.end(),
              std::next(data_constants.begin(), offset));
    offset += value.size();
  }
}
// ------------------------------------------------------------------------------

} // namespace dolfinx_eqlb