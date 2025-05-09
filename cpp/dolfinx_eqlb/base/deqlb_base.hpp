#pragma once

/// @brief Miscellaneous classes, functions and types.
///
/// This namespace provides the basic the basic definitions of a patch, boundary
/// conditions for equilibration problems an a local solver e.g. for
/// projections.
namespace dolfinx_eqlb::base
{

} // namespace dolfinx_eqlb::base

// dolfinx_eqlb base interface
#include "BoundaryData.hpp"
#include "EquilibrationProblem.hpp"
#include "Equilibrator.hpp"
#include "FluxBC.hpp"
#include "KernelDataBC.hpp"
#include "equilibration.hpp"
#include "mdspan.hpp"