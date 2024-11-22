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

} // namespace dolfinx_eqlb