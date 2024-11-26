// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <basix/mdspan.hpp>

namespace dolfinx_eqlb::base
{

template <typename T, std::size_t d>
using mdspan_t
    = std::experimental::mdspan<T, std::experimental::dextents<std::size_t, d>>;

template <typename T, std::size_t d>
using smdspan_t
    = std::experimental::mdspan<T, std::experimental::dextents<std::size_t, d>,
                                std::experimental::layout_stride>;

} // namespace dolfinx_eqlb::base