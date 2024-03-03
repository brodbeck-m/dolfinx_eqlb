#pragma once

#include "eigen3/Eigen/Dense"

#include "KernelData.hpp"
#include "Patch.hpp"
#include "PatchCstm.hpp"
#include "ProblemDataStress.hpp"
#include "minimise_flux.hpp"
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
#include <ios>
#include <iostream>
#include <iterator>
#include <span>
#include <tuple>
#include <vector>

using namespace dolfinx;

namespace dolfinx_eqlb
{

namespace stdex = std::experimental;

template <typename T, int id_flux_order = 3>
void impose_weak_symmetry(const mesh::Geometry& geometry,
                          PatchCstm<T, id_flux_order, true>& patch,
                          ProblemDataStress<T>& problem_data,
                          KernelDataEqlb<T>& kernel_data,
                          kernel_fn<T, true>& minkernel)
{
  /* Extract data */
  // The spatial dimension
  const int gdim = 2;

  // The geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();
  std::span<const double> x = geometry.x();

  // The patch
  std::span<const std::int32_t> cells = patch.cells();

  const int ncells = patch.ncells();
  const int nfcts = patch.nfcts();
  const int nnodes = nfcts + 1;

  // Nodes constructing one element
  const int nnodes_cell = kernel_data.nnodes_cell();

  // The flux space
  const graph::AdjacencyList<std::int32_t>& flux_dofmap
      = problem_data.fspace_flux_hdiv()->dofmap()->list();

  const int degree_flux_rt = patch.degree_raviart_thomas();

  const int ndofs_flux = patch.ndofs_flux();
  const int ndofs_flux_fct = patch.ndofs_flux_fct();
  const int ndofs_flux_cell_add = patch.ndofs_flux_cell_add();

  const int ndofs_flux_hdivz
      = 1 + degree_flux_rt * nfcts
        + 0.5 * degree_flux_rt * (degree_flux_rt - 1) * ncells;

  const int ndofs_per_cell
      = (gdim == 2) ? 2 * (2 * (ndofs_flux_fct - 1) + 1 + ndofs_flux_cell_add)
                          + nnodes_cell
                    : 3 * (3 * (ndofs_flux_fct - 1) + 1 + ndofs_flux_cell_add)
                          + 3 * nnodes_cell;

  // Intermediate storage of the stress corrector
  std::span<T> storage_corrector = problem_data.stress_corrector();

  /* Initialisations */
  // Isoparametric mapping
  std::array<double, 9> Jb;
  mdspan_t<double, 2> J(Jb.data(), 2, 2);
  std::array<double, 18> detJ_scratch;

  std::vector<double> storage_K;
  std::vector<double> storage_detJ(ncells, 0), storage_J(ncells * 4, 0);

  // Cell geometry
  const int cstride_geom = 3 * nnodes_cell;
  std::vector<double> coordinate_dofs_e(cstride_geom, 0);

  // DOFmap
  std::array<std::size_t, 3> shape_asmblinfo;
  std::vector<std::int32_t> dasmblinfo;

  // Equation system
  const std::size_t dim_minspace = dimension_minimisation_space(
      Kernel::StressMin, gdim, nnodes, ndofs_flux_hdivz);

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A_patch;
  Eigen::Matrix<T, Eigen::Dynamic, 1> L_patch, u_patch;

  A_patch.resize(dim_minspace, dim_minspace);
  L_patch.resize(dim_minspace);
  u_patch.resize(dim_minspace);

  A_patch.setZero();
  L_patch.setZero();

  // Boundary markers
  std::array<PatchType, 3> patch_types;
  std::array<bool, 3> patch_reversions;

  std::vector<std::int8_t> boundary_markers = initialise_boundary_markers(
      Kernel::StressMin, gdim, nnodes, ndofs_flux_hdivz);

  // Coefficients (solution without symmetry)
  std::vector<T> dcoefficients(gdim * ncells * ndofs_flux, 0);
  mdspan_t<T, 2> coefficients(dcoefficients.data(), ncells, ndofs_flux);

  // Local solver (Cholesky decomposition)
  Eigen::PartialPivLU<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> solver;

  /* Move data into flattened storage */
  for (std::size_t i_row = 0; i_row < gdim; ++i_row)
  {
    // Set patch type
    patch_types[i_row] = patch.type(i_row);

    // Check if patch reversion required
    patch_reversions[i_row] = patch.reversion_required(i_row);

    // The global DOF vector
    std::span<const T> x_flux = problem_data.flux(i_row).x()->array();

    // Initialise offset
    int offs_base = i_row * ndofs_flux;

    // Loop over cells
    for (std::size_t a = 1; a < ncells + 1; ++a)
    {
      // Current cell
      std::size_t id_a = a - 1;
      std::int32_t c = cells[a];

      // DOFs on current cell
      std::span<const std::int32_t> dofs_cell = flux_dofmap.links(c);

      // Copy DOFs into flattened storage
      for (std::size_t i = 0; i < ndofs_flux; ++i)
      {
        coefficients(id_a, offs_base + i) = x_flux[dofs_cell[i]];
      }

      // Piola transformation
      if (i_row == 0)
      {
        // Copy points of current cell
        std::span<const std::int32_t> x_dofs = x_dofmap.links(c);
        copy_cell_data<double, 3>(x, x_dofs, coordinate_dofs_e, 3);

        /* Piola mapping */
        // Reshape geometry infos
        mdspan_t<const double, 2> coords(coordinate_dofs_e.data(), nnodes_cell,
                                         3);

        // Calculate Jacobi, inverse, and determinant
        storage_detJ[id_a]
            = kernel_data.compute_jacobian(J, detJ_scratch, coords);

        // Storage of (inverse) Jacobian
        store_mapping_data(id_a, storage_J, J);
      }
    }
  }

  /* Solve minimisation problem */
  // Set DOFmap on patch
  patch.set_assembly_informations(kernel_data.fct_normal_is_outward(),
                                  storage_detJ);

  std::tie(shape_asmblinfo, dasmblinfo)
      = set_flux_dofmap(patch, ndofs_flux_hdivz);
  mdspan_t<const std::int32_t, 3> asmbl_info(dasmblinfo.data(),
                                             shape_asmblinfo);

  // Set boundary markers
  set_boundary_markers(
      boundary_markers, Kernel::StressMin, {patch.type(0), patch.type(1)}, gdim,
      ncells, ndofs_flux_fct,
      {patch.reversion_required(0), patch.reversion_required(1)});

  assemble_fluxminimiser<T, id_flux_order, true>(
      minkernel, A_patch, L_patch, boundary_markers, asmbl_info, ndofs_per_cell,
      dcoefficients, gdim * ndofs_flux, storage_detJ, storage_J, storage_K,
      patch.requires_flux_bcs());

  // Solve equation system
  solver.compute(A_patch);
  u_patch = solver.solve(L_patch);

  /* Store local solution into global storage */
  const int ndofs_flux_per_cell
      = gdim * (ndofs_flux_fct - 1) + 1 + ndofs_flux_cell_add;

  for (std::int32_t a = 1; a < ncells + 1; ++a)
  {
    // Set id for accessing storage
    std::size_t id_a = a - 1;

    // Map solution from H(div=0) to H(div) space
    for (std::size_t i = 0; i < ndofs_flux_per_cell; ++i)
    {
      // Apply correction
      for (std::size_t j = 0; j < gdim; ++j)
      {
        // Offsets
        int offs_i = gdim * i + j;
        int pos_strg = gdim * asmbl_info(1, a, offs_i) + j;

        storage_corrector[pos_strg]
            += asmbl_info(3, a, offs_i) * u_patch(asmbl_info(2, a, offs_i));
      }
    }
  }
}
} // namespace dolfinx_eqlb