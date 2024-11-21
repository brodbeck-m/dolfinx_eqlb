// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cinttypes>

namespace dolfinx_eqlb::base
{

enum class PatchType : std::int8_t
{
  internal = 0,
  bound_essnt_dual = 1,
  bound_essnt_primal = 2,
  bound_mixed = 3
};

enum PatchFacetType : std::int8_t
{
  internal = 0,
  essnt_primal = 1,
  essnt_dual = 2
};

} // namespace dolfinx_eqlb::base