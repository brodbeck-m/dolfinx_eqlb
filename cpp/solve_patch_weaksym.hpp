// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

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

/// Impose weak symmetry on a reconstructed stress tensor
///
/// Based on o row-wise equilibrated stress tensors, weak symmetry
/// - tested against a mesh-wide P1 space - is enforced based on a
/// constrained minimisation problem, following [1].
///
/// [1] Bertrand, F. et al.: https://doi.org/10.1002/num.22741, 2021
///
/// @tparam T              The scalar type
/// @tparam id_flux_order  Parameter for flux order (1->RT1, 2->RT2, 3->general)
/// @tparam modified_patch Identifier fou grouped patches
/// @param mesh         The mesh
/// @param patch        The patch
/// @param patch_data   The patch data
/// @param problem_data The problem data (Functions of flux, flux_dg, RHS_dg)
/// @param kernel_data  The kernel data (Quadrature data, tabulated basis)
/// @param kernel       The kernel function
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

  // Marker for reversed facets
  mdspan_t<std::uint8_t, 2> reversed_fct
      = patch_data.reversed_facets_per_cell();

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
                                                     asmbl_info, reversed_fct);
  }
  else
  {
    assemble_stressminimiser<T, id_flux_order, false>(kernel, patch_data,
                                                      asmbl_info, reversed_fct);
  }

  // Solve equation system
  patch_data.solve_constrained_minimisation(requires_bcs);

  /* Store local solution into global storage */
  // The flux space
  const int ndofs_hdivz_per_cell
      = gdim * ndofs_flux_fct + patch.ndofs_flux_cell_add();

  // The patch solution
  Eigen::Matrix<T, Eigen::Dynamic, 1>& u_patch = patch_data.vector_u_sigma();

  // DOF transformation data
  mdspan_t<const double, 2> doftrafo
      = kernel_data.entity_transformations_flux();

  // Move solution from patch-wise into global storage
  for (std::size_t i = 0; i < gdim; ++i)
  {
    // Global storage of the solution
    std::span<T> x_stress = problem_data.flux(i).x()->mutable_array();
    mdspan_t<T, 2> coefficients_flux = patch_data.coefficients_flux(i);

    // Initialise offset
    int offset_u = i * ndofs_hdivz;

    for (std::int32_t a = 1; a < ncells + 1; ++a)
    {
      // Start of general loop over cell DOFs
      std::size_t start_j = 0;

      // Loop over DOFs on reversed facet
      if (reversed_fct(a - 1, 0))
      {
        // DOFs on reversed facet
        for (std::size_t j = 0; j < ndofs_flux_fct; ++j)
        {
          // Transform global- into cell-local DOFs
          T local_value = 0.0;

          for (std::size_t k = 0; k < ndofs_flux_fct; ++k)
          {
            T pf_k = doftrafo(k, j) * asmbl_info(3, a, k);
            local_value += pf_k * u_patch(offset_u + asmbl_info(2, a, k));
          }

          x_stress[asmbl_info(1, a, j)] += local_value;
        }

        // Modify start index of general loop over DOFs
        start_j = ndofs_flux_fct;
      }

      // General loop over cell DOFs
      for (std::size_t j = start_j; j < ndofs_hdivz_per_cell; ++j)
      {
        x_stress[asmbl_info(1, a, j)]
            += asmbl_info(3, a, j) * u_patch(offset_u + asmbl_info(2, a, j));
      }
    }
  }
}
} // namespace dolfinx_eqlb