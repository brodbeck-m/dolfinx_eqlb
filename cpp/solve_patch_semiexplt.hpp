#pragma once

#include "KernelData.hpp"
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
                                ProblemDataFluxCstm<T>& problem_data,
                                KernelData& kernel_data)
{
  assert(flux_order < 0);

  /* Geometry data */
  const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();
  std::span<const fem::impl::scalar_value_type_t<T>> x = geometry.x();

  /* Extract patch data */
  // Elements on patch
  std::span<const std::int32_t> cells = patch.cells();
  int ncells = patch.ncells();

  /* Initialize solution process */
  // Jacobian J, inverse K and determinant detJ
  std::array<double, 9> Jb;
  dolfinx_adaptivity::mdspan2_t J(Jb.data(), 2, 2);
  std::array<double, 9> Kb;
  dolfinx_adaptivity::mdspan2_t K(Kb.data(), 2, 2);
  double detJ = 0;
  std::array<double, 18> detJ_scratch;

  // Physical normal
  std::array<double, 2> n_phys;

  // Extract geometry data
  const int cstride_geom = 3 * geometry.cmap().dim();
  std::vector<fem::impl::scalar_value_type_t<T>> coordinate_dofs(
      ncells * cstride_geom, 0);

  for (std::size_t index = 0; index < ncells; ++index)
  {
    // Get current cell
    std::int32_t c = cells[index];

    // Copy cell geometry
    std::span<fem::impl::scalar_value_type_t<T>> coordinate_dofs_e(
        coordinate_dofs.data() + index * cstride_geom, cstride_geom);

    auto x_dofs = x_dofmap.links(c);
    for (std::size_t j = 0; j < x_dofs.size(); ++j)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs[j]), 3,
                  std::next(coordinate_dofs_e.begin(), 3 * j));
    }
  }

  /* Solve equilibration */
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