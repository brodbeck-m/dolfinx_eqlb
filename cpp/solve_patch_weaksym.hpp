#pragma once

#include "eigen3/Eigen/Dense"

#include "KernelData.hpp"
#include "Patch.hpp"
#include "PatchCstm.hpp"
#include "PatchData.hpp"
#include "assemble_patch_semiexplt.hpp"
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
  const int ndofs_flux = patch.ndofs_flux();
  const int ndofs_flux_fct = patch.ndofs_flux_fct();
  const int ndofs_hdivz = patch.ndofs_flux_hdiz_zero();

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

  // Coefficients of the stress tensor
  if constexpr (modified_patch)
  {
    // Cells on current patch
    std::span<const std::int32_t> cells = patch.cells();

    // (Global) DOFmap of the flux space
    const graph::AdjacencyList<std::int32_t>& flux_dofmap
        = problem_data.fspace_flux_hdiv()->dofmap()->list();

    for (std::size_t i = 0; i < gdim; ++i)
    {
      // Global storage of the stress (row i)
      std::span<const T> x_flux = problem_data.flux(i).x()->array();

      // Loop over cells
      for (std::int32_t a = 1; a < ncells + 1; ++a)
      {
        // Global DOFs
        std::span<const std::int32_t> gdofs = flux_dofmap.links(cells[a]);

        // Flattened storage of stress
        std::span<T> stress_coefficients = patch_data.coefficients_stress(i, a);

        // Loop over DOFs an cell
        for (std::size_t i = 0; i < ndofs_flux; ++i)
        {
          // Set zero-order DOFs on facets
          stress_coefficients[i] = x_flux[gdofs[i]];
        }
      }
    }
  }
  else
  {
    for (std::size_t i_row = 0; i_row < gdim; ++i_row)
    {
      for (std::size_t a = 1; a < ncells + 1; ++a)
      {
        // Move coefficients to flattened storage
        std::copy_n(patch_data.coefficients_flux(i_row, a).begin(), ndofs_flux,
                    patch_data.coefficients_stress(i_row, a).begin());
      }
    }
  }

  /* Solve minimisation problem */
  // The assembly information
  mdspan_t<const std::int32_t, 3> asmbl_info
      = patch.assembly_info_minimisation();

  // Set boundary markers
  if (patch.is_on_boundary())
  {
    set_boundary_markers(patch_data.boundary_markers(true), patch_types,
                         patch_reversions, ncells, ndofs_hdivz, ndofs_flux_fct);
  }

  // Assemble equation system
  if (requires_bcs)
  {
    assemble_stressminimiser<T, id_flux_order, true>(kernel, patch_data,
                                                     asmbl_info);
  }
  else
  {
    assemble_stressminimiser<T, id_flux_order, false>(kernel, patch_data,
                                                      asmbl_info);
  }

  // Solve equation system
  patch_data.solve_constrained_minimisation(requires_bcs);

  /* Store local solution into global storage */
  // The flux space
  const int ndofs_hdivz_per_cell
      = gdim * ndofs_flux_fct + patch.ndofs_flux_cell_add();

  // The patch solution
  Eigen::Matrix<T, Eigen::Dynamic, 1>& u_patch = patch_data.vector_u_sigma();

  // Move solution from patch-wise into global storage
  for (std::size_t i = 0; i < gdim; ++i)
  {
    // Global storage of the solution
    std::span<T> x_stress = problem_data.flux(i).x()->mutable_array();

    // Initialise offset
    int offset_u = i * ndofs_hdivz;

    // Loop over cells
    for (std::int32_t a = 1; a < ncells + 1; ++a)
    {
      // Map solution from H(div=0) to H(div) space
      for (std::size_t j = 0; j < ndofs_hdivz_per_cell; ++j)
      {
        // Local to global storage
        x_stress[asmbl_info(1, a, j)]
            += asmbl_info(3, a, j) * u_patch(offset_u + asmbl_info(2, a, j));
      }
    }
  }
}
} // namespace dolfinx_eqlb