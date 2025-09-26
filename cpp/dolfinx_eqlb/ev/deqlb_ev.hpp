#pragma once

/// @brief Flux equilibration functionalities.
///
/// Classes and algorithms for the equilibration of fluxes and stresses based on
/// the direct solution of a constrained minimisation problem [1] while symmetry
/// is enforced weakly following [2].
///
/// [1] Ern, A. and Vohralík, M.: https://doi.org/10.1137/130950100, 2015
/// [2] Bertrand, F. et al.: https://doi.org/10.1002/num.22741, 2021

namespace dolfinx_eqlb::ev
{
}

// dolfinx_eqlb ev interface
#include "reconstruction.hpp"