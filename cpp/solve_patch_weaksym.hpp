#pragma once

#include "eigen3/Eigen/Dense"

#include "KernelData.hpp"
#include "Patch.hpp"
#include "PatchCstm.hpp"
#include "PatchData.hpp"
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

template <typename T, int id_flux_order, bool modified_patch>
void impose_weak_symmetry(const mesh::Geometry& geometry,
                          PatchFluxCstm<T, id_flux_order>& patch,
                          PatchDataCstm<T, id_flux_order>& patch_data,
                          ProblemDataFluxCstm<T>& problem_data,
                          KernelDataEqlb<T>& kernel_data,
                          kernel_fn_schursolver<T>& kernel)
{
  /* Extract data */
  // The spatial dimension
  const int gdim = patch.dim();

  // The patch
  const int ncells = patch.ncells();

  // The flux space
  const int ndofs_flux_fct = patch.ndofs_flux_fct();

  /* Initialisations */
  // Patch type and reversion information
  bool requires_bcs = false;
  std::vector<PatchType> patch_types(gdim, PatchType::internal);
  std::vector<bool> patch_reversions(gdim, false);

  if (patch.is_on_boundary())
  {
    for (std::size_t i = 0; i < gdim; ++i)
    {
      patch_types[i] = patch.type(i);
      patch_reversions[i] = patch.reversion_required(i);

      if (patch_types[i] == PatchType::bound_essnt_dual
          || patch_types[i] == PatchType::bound_mixed)
      {
        requires_bcs = true;
      }
    }
  }

  // Data for Piola mapping
  if constexpr (modified_patch)
  {
    /* Extract data */
    // The geometry
    const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();
    std::span<const double> x = geometry.x();

    // The patch
    std::span<const std::int32_t> cells = patch.cells();

    /* Initialisation */
    // Isoparametric mapping
    std::array<double, 9> Jb;
    std::array<double, 18> detJ_scratch;
    mdspan_t<double, 2> J(Jb.data(), 2, 2);

    // Storage cell geometry
    std::array<double, 12> coordinate_dofs_e;
    mdspan_t<const double, 2> coords(coordinate_dofs_e.data(),
                                     kernel_data.nnodes_cell(), 3);

    /* Evaluate Piola mapping */
    for (std::size_t a = 1; a < ncells + 1; ++a)
    {
      // Current cell
      std::size_t id_a = a - 1;
      std::int32_t c = cells[a];

      // Copy points of current cell
      std::span<const std::int32_t> x_dofs = x_dofmap.links(c);
      copy_cell_data<double, 3>(x, x_dofs, coordinate_dofs_e, 3);

      // Calculate Jacobi matrix and determinant
      double detJ = kernel_data.compute_jacobian(J, detJ_scratch, coords);
      patch_data.store_piola_mapping(id_a, detJ, J);
    }

    /* Initialise assembly informations */
    patch.set_assembly_informations(kernel_data.fct_normal_is_outward(),
                                    patch_data.jacobi_determinant());
  }

  // Coefficients of the stress tensor
  if (!(modified_patch))
  {
    // Dimension of the flux space
    const int ndofs_flux = patch.ndofs_flux();

    // Flattened storage of stress coefficients
    std::span<T> stress_coefficients = patch_data.coefficients_stress();

    // Copy the coefficients
    for (std::size_t i_row = 0; i_row < gdim; ++i_row)
    {
      for (std::size_t a = 1; a < ncells + 1; ++a)
      {
        // Set offset
        std::size_t offset = (a - 1) * ndofs_flux * gdim + i_row * ndofs_flux;

        // Coefficients of flux i on cell a
        std::span<const T> coeffs_rowi_cella
            = patch_data.coefficients_flux(i_row, a);

        // Move coefficients to flattened storage
        std::copy_n(coeffs_rowi_cella.begin(), ndofs_flux,
                    stress_coefficients.begin() + offset);
      }
    }
  }

  /* Solve minimisation problem */
  // The assembly information
  mdspan_t<const std::int32_t, 3> asmbl_info
      = patch.assembly_info_minimisation();

  // Assemble equation system
  if constexpr (modified_patch)
  {
    // Create DOFmap on modified patch

    // Set boundary markers

    // Assemble equation system
  }
  else
  {
    // Set boundary markers
    set_boundary_markers(patch_data.boundary_markers(true), Kernel::StressMin,
                         patch_types, gdim, ncells, ndofs_flux_fct,
                         patch_reversions);

    // Assemble equation system
    assemble_stressminimiser<T, id_flux_order, modified_patch>(
        kernel, patch_data, asmbl_info, requires_bcs);
  }

  // Solve equation system
  patch_data.solve_constrained_minimisation(requires_bcs);

  /* Store local solution into global storage */
  const int dim_hdivz = patch.ndofs_minspace_flux(false);
  const int ndofs_flux_per_cell
      = gdim * ndofs_flux_fct + patch.ndofs_flux_cell_add();

  Eigen::Matrix<T, Eigen::Dynamic, 1>& u_patch = patch_data.vector_u_sigma();

  for (std::size_t i_row = 0; i_row < gdim; ++i_row)
  {
    // Global storage of the solution
    std::span<T> x_flux_dhdiv = problem_data.flux(i_row).x()->mutable_array();

    // Initialise offset
    int offset = i_row * dim_hdivz;

    // Loop over cells
    for (std::int32_t a = 1; a < ncells + 1; ++a)
    {
      // Map solution from H(div=0) to H(div) space
      for (std::size_t i = 0; i < ndofs_flux_per_cell; ++i)
      {
        // Local to global storage
        x_flux_dhdiv[asmbl_info(1, a, i)]
            += asmbl_info(3, a, i) * u_patch(offset + asmbl_info(2, a, i));
      }
    }
  }
}
} // namespace dolfinx_eqlb