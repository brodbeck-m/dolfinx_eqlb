#pragma once

#include "KernelData.hpp"
#include "PatchFluxCstm.hpp"
#include "ProblemDataFluxCstm.hpp"
#include "eigen3/Eigen/Dense"
#include "utils.hpp"

#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/graph/AdjacencyList.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <iterator>
#include <span>
#include <tuple>
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

  // +/- cell on facet Eam1
  std::int32_t cell_plus_eam1, cell_minus_eam1;

  // Physical normal
  std::array<double, 2> normal_phys;

  // Jump within projected flux
  std::array<double, 2> jump_proj_flux;

  // DOFs flux (cell-wise H(div))
  T c_ta_ea, c_ta_eam1, c_tam1_eam1;

  // Storage cell geometries/ normal orientation
  const int cstride_geom = 3 * nnodes_cell;
  std::vector<double> coordinate_dofs(ncells * cstride_geom, 0);

  std::int8_t fctloc_ea, fctloc_eam1;
  bool noutward_ea, noutward_eam1;
  std::vector<double> dprefactor_dof(ncells * 2, 1.0);
  dolfinx_adaptivity::mdspan2_t prefactor_dof(dprefactor_dof.data(), ncells, 2);

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

    // Get local fact ids on cell_i
    std::tie(fctloc_ea, fctloc_eam1) = patch.fctid_local(index + 1);

    // Get indicators if reference normals are pointing outward
    std::tie(noutward_eam1, noutward_ea)
        = kernel_data.fct_normal_is_outward(fctloc_ea, fctloc_eam1);

    // Set prefactor
    dprefactor_dof[2 * index] = (noutward_ea) ? 1.0 : -1.0;
    dprefactor_dof[2 * index + 1] = (noutward_eam1) ? 1.0 : -1.0;
  }

  /* Solve equilibration */
  for (std::size_t i_rhs = 0; i_rhs < problem_data.nlhs(); ++i_rhs)
  {
    // Patch type
    int type_patch = patch.type(i_rhs);

    // Solution vector (flux, picewise-H(div))
    std::span<T> x_flux_dhdiv = problem_data.flux(i_rhs).x()->mutable_array();

    // Projected primal flux
    std::span<T> x_flux_proj
        = problem_data.projected_flux(i_rhs).x()->mutable_array();

    // Projected RHS
    std::span<T> x_rhs_proj
        = problem_data.projected_rhs(i_rhs).x()->mutable_array();

    /* Solution step 1: Jump and divergence condition */
    int loop_end;
    if (type_patch == 0 | type_patch == 2)
    {
      loop_end = ncells + 1;

      // Physical coordinates of cell
      std::span<double> coordinate_dofs_e(coordinate_dofs.data(), cstride_geom);
      dolfinx_adaptivity::cmdspan2_t coords(coordinate_dofs_e.data(),
                                            nnodes_cell, 3);

      // Isoparametric mappring for cell
      const double detJ
          = kernel_data.compute_jacobian(J, K, detJ_scratch, coords);

      // Extract RHS value on current cell
      // FIXME - Acess onto c seems to be wrong --> Check DOFmap
      T f_i = x_rhs_proj[cells[0]];

      // Set DOFs for cell 1
      c_ta_ea = prefactor_dof(0, 1) * f_i * detJ / 6;

      // Store coefficients and set history values
      std::span<const std::int32_t> gdofs_flux = patch.dofs_flux_fct_global(1);

      x_flux_dhdiv[gdofs_flux[0]] = 0;
      x_flux_dhdiv[gdofs_flux[1]] = c_ta_ea;

      c_tam1_eam1 = c_ta_ea;
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

      // Global cell id
      std::int32_t c = cells[id_a];

      // Physical coordinates of cell
      std::span<double> coordinate_dofs_e(
          coordinate_dofs.data() + id_a * cstride_geom, cstride_geom);

      dolfinx_adaptivity::cmdspan2_t coords(coordinate_dofs_e.data(),
                                            nnodes_cell, 3);

      // Isoparametric mappring cell
      const double detJ
          = kernel_data.compute_jacobian(J, K, detJ_scratch, coords);

      // Compute physical normal
      std::tie(cell_plus_eam1, cell_minus_eam1) = patch.cellpm(a - 1);
      std::int8_t fctid_loc_plus = patch.fctid_local(a - 1, cell_plus_eam1);

      kernel_data.physical_fct_normal(normal_phys, K, fctid_loc_plus);

      // Extract RHS value
      T f_i = x_rhs_proj[c];

      // Extract gadients (+/- side) ond facet E_am1
      std::span<const std::int32_t> dofs_projflux_fct
          = patch.dofs_projflux_fct(a - 1);

      jump_proj_flux[0] = x_flux_proj[dofs_projflux_fct[0]]
                          - x_flux_proj[dofs_projflux_fct[2]];
      jump_proj_flux[1] = x_flux_proj[dofs_projflux_fct[1]]
                          - x_flux_proj[dofs_projflux_fct[3]];

      double jump_i = jump_proj_flux[0] * normal_phys[0]
                      + jump_proj_flux[1] * normal_phys[1];

      // Set DOFs for cell
      // FIXME - Add length of facet --> integral incomplete
      c_ta_eam1 = prefactor_dof(id_a, 0) * (jump_i - c_tam1_eam1);
      c_ta_ea = prefactor_dof(id_a, 1) * (f_i * detJ / 6 - c_ta_eam1);

      // Store coefficients and set history values
      std::span<const std::int32_t> gdofs_flux = patch.dofs_flux_fct_global(a);

      x_flux_dhdiv[gdofs_flux[0]] = c_ta_eam1;
      x_flux_dhdiv[gdofs_flux[1]] = c_ta_ea;

      c_tam1_eam1 = c_ta_ea;
    }

    /* Solution step 2: Minimisation */
  }
}
} // namespace dolfinx_adaptivity::equilibration