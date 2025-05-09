// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>

namespace dolfinx_eqlb::base
{

/// @brief The considered problem type
enum class ProblemType : int
{
  flux = 0,
  stress = 1,
  stress_and_flux = 2,
};

/// @brief The considered equilibration strategy
enum class EqStrategy : int
{
  semi_explicit = 0,
  constrained_minimisation = 1,
};

/// @brief The type of a patch
enum class PatchType : std::int8_t
{
  internal = 0,
  bound_essnt_dual = 1,
  bound_essnt_primal = 2,
  bound_mixed = 3
};

/// @brief The type of a facet (on a patch)
enum FacetType : std::int8_t
{
  internal = 0,
  essnt_primal = 1,
  essnt_dual = 2
};

} // namespace dolfinx_eqlb::base