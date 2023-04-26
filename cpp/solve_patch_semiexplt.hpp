#pragma once

#include "PatchFluxCstm.hpp"
#include "ProblemDataFluxCstm.hpp"
#include "eigen3/Eigen/Dense"
#include "utils.hpp"
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

  /* Extract patch data */
  // Elements on patch
  std::span<const std::int32_t> cells = patch.cells();
  int ncells = patch.ncells();

  /* Initialize solution process */

  // Solve equilibration
  for (std::size_t i_rhs = 0; i_rhs < problem_data.nlhs(); ++i_rhs)
  {
    // Data dependent on RHS
    int type_patch = patch.type(i_rhs);

    // Initialize storage of coefficients
    T coeffs_flux[ncells][2];

    /* Solution step 1: Jump and divergence condition */
    int loop_end;
    if (type_patch == 0 | type_patch == 2)
    {
      loop_end = ncells + 1;

      // Cell 1 and determinant
      std::int32_t cell_i = cells[0];
      double detj_i = 1;

      // Set DOFs for cell 1
      coeffs_flux[0][0] = 0;
      coeffs_flux[0][1] = detj_i / 6;
    }
    else if (type_patch == 1)
    {
      throw std::runtime_error("Equilibration: Neumann BCs not supported!");
    }
    else
    {
      throw std::runtime_error("Equilibration: Neumann BCs not supported!");
    }

    for (std::size_t a = 2; a < loop_end; ++a)
    {
      // Set id for acessing storage
      int id_a = a - 1;

      // Determine cell and Piola mapping
      std::int32_t cell_i = cells[id_a];
      double detj_i = 1;

      // Set DOFs for cell
      // FIXME - Add jump contribution
      T jump = 0;
      coeffs_flux[id_a][0] = jump - coeffs_flux[id_a - 1][1];
      coeffs_flux[id_a][1] = 0;
    }

    /* Solution step 2: Minimisation */
  }
}
} // namespace dolfinx_adaptivity::equilibration