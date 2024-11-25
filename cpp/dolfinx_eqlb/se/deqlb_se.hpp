#pragma once

/// @brief Flux equilibration functionalities based on a semi-explicit approach.
/// Classes and algorithms for the equilibration of fluxes [1, 2] and
/// stresses [3] based on the semi-explicit approach .
///
/// [1] Cai, Z. and Zhang, S.: https://doi.org/10.1137/100803857, 2012
/// [2] Bertrand, F. et al.: https://doi.org/10.1007/s00211-023-01366-8, 2023
/// [3] Bertrand, F. et al.: https://doi.org/10.1002/num.22741, 2021

namespace dolfinx_eqlb::ev
{
}

// dolfinx_eqlb se interface
#include "Patch.hpp"
#include "reconstruction.hpp"
