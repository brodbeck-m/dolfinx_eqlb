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
void get_cell_coordinates(std::span<const double> x_g,
                          std::span<const std::int32_t> x_dofs,
                          std::vector<double>& coordinate_dofs)
{
  for (std::size_t j = 0; j < x_dofs.size(); ++j)
  {
    std::copy_n(std::next(x_g.begin(), 3 * x_dofs[j]), 3,
                std::next(coordinate_dofs.begin(), 3 * j));
  }
}

template <typename T, int id_flux_order = -1>
void equilibrate_flux_constrmin(const mesh::Geometry& geometry,
                                PatchFluxCstm<T, id_flux_order>& patch,
                                ProblemDataFluxCstm<T>& problem_data,
                                KernelData& kernel_data)
{
  assert(flux_order < 0);

  /* Geometry data */
  const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();
  std::span<const double> x = geometry.x();

  /* Extract patch data */
  // Elements on patch
  std::span<const std::int32_t> cells = patch.cells();
  int ncells = patch.ncells();

  /* Initialize solution process */
  // Number of nodes on reference cell
  int nnodes_cell = kernel_data.nnodes_cell();

  // Jacobian J, inverse K and determinant detJ
  std::array<double, 9> Jb;
  dolfinx_adaptivity::mdspan2_t J(Jb.data(), 2, 2);
  std::array<double, 9> Kb;
  dolfinx_adaptivity::mdspan2_t K(Kb.data(), 2, 2);
  double detJ = 0;
  std::array<double, 18> detJ_scratch;

  // Physical normal
  std::array<double, 2> normal_phys;

  // Storage cell geometries
  const int cstride_geom = 3 * nnodes_cell;
  std::vector<double> coordinate_dofs(ncells * cstride_geom, 0);

  for (std::size_t index = 0; index < ncells; ++index)
  {
    // Get current cell
    std::int32_t c = cells[index];

    // Copy cell geometry
    std::span<double> coordinate_dofs_e(
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

      // Physical coordinates of cell 1
      std::span<double> coordinate_dofs_e(coordinate_dofs.data(), cstride_geom);
      dolfinx_adaptivity::cmdspan2_t coords(coordinate_dofs_e.data(),
                                            nnodes_cell, 3);

      // Isoparametric mappring cell 1
      const double detJ
          = kernel_data.compute_jacobian(J, K, detJ_scratch, coords);

      // Set DOFs for cell 1
      coeffs_flux[0][0] = 0;
      coeffs_flux[0][1] = detJ / 6;
    }
    else if (type_patch == 1)
    {
      loop_end = ncells;
      throw std::runtime_error("Equilibration: Neumann BCs not supported!");
    }
    else
    {
      loop_end = ncells + 1;
      throw std::runtime_error("Equilibration: Neumann BCs not supported!");
    }

    for (std::size_t a = 2; a < loop_end; ++a)
    {
      // Set id for acessing storage
      int id_a = a - 1;

      // Physical coordinates of cell
      std::span<double> coordinate_dofs_e(
          coordinate_dofs.data() + id_a * cstride_geom, cstride_geom);

      dolfinx_adaptivity::cmdspan2_t coords(coordinate_dofs_e.data(),
                                            nnodes_cell, 3);

      // Isoparametric mappring cell
      const double detJ
          = kernel_data.compute_jacobian(J, K, detJ_scratch, coords);

      // Compute physical normal
      std::int8_t fctid_loc_plus = 0;

      kernel_data.physical_fct_normal(normal_phys, K, fctid_loc_plus);

      // Set DOFs for cell
      // FIXME - Add jump contribution
      T jump = 0;
      coeffs_flux[id_a][0] = jump - coeffs_flux[id_a - 1][1];
      coeffs_flux[id_a][1] = detJ / 6 - coeffs_flux[id_a][0];
    }

    /* Solution step 2: Minimisation */
  }
}
} // namespace dolfinx_adaptivity::equilibration