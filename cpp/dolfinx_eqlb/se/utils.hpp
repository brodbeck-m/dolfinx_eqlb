// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <basix/mdspan.hpp>

#include <functional>
#include <span>

namespace dolfinx_eqlb::se
{

/* Interface kernel functions */
template <typename T, bool asmbl_systmtrx>
using kernel_fn = std::function<void(
    std::experimental::mdspan<T, std::experimental::dextents<std::size_t, 2>>,
    std::span<const T>,
    std::experimental::mdspan<const std::int32_t,
                              std::experimental::dextents<std::size_t, 2>,
                              std::experimental::layout_stride>,
    const std::uint8_t, const double,
    std::experimental::mdspan<const double,
                              std::experimental::dextents<std::size_t, 2>>)>;

template <typename T>
using kernel_fn_schursolver = std::function<void(
    std::experimental::mdspan<T, std::experimental::dextents<std::size_t, 2>>,
    std::experimental::mdspan<T, std::experimental::dextents<std::size_t, 2>>,
    std::span<T>, std::span<T>, std::span<const T>,
    std::experimental::mdspan<const std::int32_t,
                              std::experimental::dextents<std::size_t, 2>,
                              std::experimental::layout_stride>,
    const std::uint8_t, const double,
    std::experimental::mdspan<const double,
                              std::experimental::dextents<std::size_t, 2>>,
    const bool)>;

} // namespace dolfinx_eqlb