#pragma once

#include "PatchFluxCstm.hpp"
#include "ProblemDataFluxCstm.hpp"
#include "eigen3/Eigen/Dense"
#include <algorithm>
#include <cmath>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <functional>
#include <iostream>
#include <iterator>
#include <span>
#include <vector>

using namespace dolfinx;

namespace dolfinx_adaptivity::equilibration
{
template <typename T, int id_flux_order = -1>
void equilibrate_flux_constrmin(const mesh::Geometry& geometry,
                                PatchFluxCstm<T, id_flux_order>& patch,
                                ProblemDataFluxCstm<T>& problem_data)
{
  assert(flux_order < 0);

  /* Initialize solution process */

  if constexpr (id_flux_order == 1)
  {
    /* Solution step 1: Jump and divergence condition */

    /* Solution step 2: Minimisation */
  }
  else if constexpr (id_flux_order == 2)
  {
    throw std::runtime_error("Equilibartion for fluxorder=2 not implemented");
  }
  else
  {
    throw std::runtime_error(
        "Equilibartion for general fluxorder not implemented");
  }
}
} // namespace dolfinx_adaptivity::equilibration