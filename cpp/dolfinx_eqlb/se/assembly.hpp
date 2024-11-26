// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "eigen3/Eigen/Dense"

#include "KernelData.hpp"
#include "PatchData.hpp"
#include "utils.hpp"

#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx_eqlb/base/Patch.hpp>
#include <dolfinx_eqlb/base/mdspan.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <iterator>
#include <span>
#include <vector>

using namespace dolfinx;

namespace dolfinx_eqlb::se
{

/* Assemblers for patch-wise minimisation problems */

/// Set boundary markers for patch-wise H(div=0) space
/// @param boundary_markers    The boundary markers
/// @param types_patch         The patch types
/// @param reversions_required Identifier if patch requires reversion
/// @param ncells              Number of cells on patch
/// @param ndofs_flux_hivz     nDOFs patch-wise H(div=0) flux-space
/// @param ndofs_flux_fct      nDOFs flux-space space per facet
void set_boundary_markers(std::span<std::int8_t> boundary_markers,
                          std::vector<base::PatchType> types_patch,
                          std::vector<bool> reversions_required,
                          const int ncells, const int ndofs_flux_hivz,
                          const int ndofs_flux_fct)
{
  // Reinitialise markers
  std::fill(boundary_markers.begin(), boundary_markers.end(), false);

  // Auxiliaries
  const int offset_En = ncells * (ndofs_flux_fct - 1);

  // Set boundary markers
  for (std::size_t i = 0; i < types_patch.size(); ++i)
  {
    if (types_patch[i] != base::PatchType::bound_essnt_primal)
    {
      // Basic offset
      int offset_i = i * ndofs_flux_hivz;

      // Set boundary markers for d0
      boundary_markers[offset_i] = true;

      for (std::size_t j = 1; j < ndofs_flux_fct; ++j)
      {
        if (types_patch[i] == base::PatchType::bound_essnt_dual)
        {
          // Mark DOFs on facet E0
          int offset_E0 = offset_i + j;
          boundary_markers[offset_E0] = true;

          // Mark DOFs on facet En
          boundary_markers[offset_E0 + offset_En] = true;
        }
        else
        {
          if (reversions_required[i])
          {
            // Mark DOFs in facet En
            // (Mixed patch with reversed order)
            boundary_markers[offset_i + offset_En + j] = true;
          }
          else
          {
            // Mark DOFs in facet E0
            // (Mixed patch with original order)
            boundary_markers[offset_i + j] = true;
          }
        }
      }
    }
  }
}

/// Assemble EQS for flux minimisation
///
/// Assembles system-matrix and load vector for unconstrained flux
/// minimisation on patch-wise divergence free H(div) space. Explicit ansatz
/// for such a space see [1, Lemma 12].
///
/// [1] Bertrand, F. et al.: https://doi.org/10.1007/s00211-023-01366-8, 2023
///
/// @tparam T                       The scalar type
/// @tparam id_flux_order           The flux order (1->RT1, 2->RT2, 3->general)
/// @tparam asmbl_systmtrx          Flag if entire tangent or only load
///                                 vector is assembled
/// @param minimisation_kernel      The kernel for minimisation
/// @param patch_data               The temporary storage for the patch
/// @param asmbl_info               Informations patch-wise H(div=0) space
/// @param fct_reversion            Marker for reversed facets
/// @param i_rhs                    Index of the right-hand side
/// @param requires_flux_bc         Marker if flux BCs are required
template <typename T, int id_flux_order, bool asmbl_systmtrx>
void assemble_fluxminimiser(kernel_fn<T, asmbl_systmtrx>& minimisation_kernel,
                            PatchData<T, id_flux_order>& patch_data,
                            base::mdspan_t<const std::int32_t, 3> asmbl_info,
                            base::mdspan_t<const std::uint8_t, 2> fct_reversion,
                            const int i_rhs, const bool requires_flux_bc)
{
  assert(id_flux_order < 0);

  /* Extract data */
  // The spatial dimension
  const int gdim = patch_data.gdim();

  // Number of elements/facets on patch
  const int ncells = patch_data.ncells();

  // Tangent storage
  base::mdspan_t<T, 2> Te = patch_data.Te();

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& A = patch_data.matrix_A();
  Eigen::Matrix<T, Eigen::Dynamic, 1>& L = patch_data.vector_L();
  std::span<const std::int8_t> boundary_markers
      = patch_data.boundary_markers(false);

  /* Initialisation */
  if constexpr (asmbl_systmtrx)
  {
    A.setZero();
    L.setZero();
  }
  else
  {
    L.setZero();
  }

  /* Calculation and assembly */
  const int ndofs_per_cell = patch_data.ndofs_flux_hdivz_per_cell();
  const int ndofs_constr_per_cell = gdim + 1;

  const int index_load = ndofs_per_cell;

  for (std::size_t a = 1; a < ncells + 1; ++a)
  {
    int id_a = a - 1;

    // Isoparametric mapping
    const double detJ = patch_data.jacobi_determinant(id_a);
    base::mdspan_t<const double, 2> J = patch_data.jacobian(id_a);

    // DOFmap on cell
    base::smdspan_t<const std::int32_t, 2> asmbl_info_cell
        = std::experimental::submdspan(asmbl_info,
                                       std::experimental::full_extent, a,
                                       std::experimental::full_extent);

    // DOFs on cell
    std::span<const T> coefficients = patch_data.coefficients_flux(i_rhs, a);

    // Evaluate linear- and bilinear form
    patch_data.reinitialise_Te();
    minimisation_kernel(Te, coefficients, asmbl_info_cell,
                        fct_reversion(id_a, 0), detJ, J);

    // Assemble linear- and bilinear form
    if constexpr (id_flux_order == 1)
    {
      if (requires_flux_bc)
      {
        // Assemble linar form
        L(0) = 0;

        if constexpr (asmbl_systmtrx)
        {
          // Assemble bilinear form
          A(0, 0) = 1;
        }
      }
      else
      {
        // Assemble linar form
        L(0) += Te(1, 0);

        if constexpr (asmbl_systmtrx)
        {
          // Assemble bilinear form
          A(0, 0) += Te(0, 0);
        }
      }
    }
    else
    {
      if (requires_flux_bc)
      {
        for (std::size_t i = 0; i < ndofs_per_cell; ++i)
        {
          std::int32_t dof_i = asmbl_info_cell(2, i + 1);
          std::int8_t bmarker_i = boundary_markers[dof_i];

          // Assemble load vector
          if (bmarker_i)
          {
            L(dof_i) = 0;
          }
          else
          {
            L(dof_i) += Te(index_load, i);
          }

          // Assemble bilinear form
          if constexpr (asmbl_systmtrx)
          {
            if (bmarker_i)
            {
              A(dof_i, dof_i) = 1;
            }
            else
            {
              for (std::size_t j = 0; j < ndofs_per_cell; ++j)
              {
                std::int32_t dof_j = asmbl_info_cell(2, j + 1);
                std::int8_t bmarker_j = boundary_markers[dof_j];

                if (bmarker_j)
                {
                  A(dof_i, dof_j) = 0;
                }
                else
                {
                  A(dof_i, dof_j) += Te(i, j);
                }
              }
            }
          }
        }
      }
      else
      {
        for (std::size_t i = 0; i < ndofs_per_cell; ++i)
        {
          std::int32_t dof_i = asmbl_info_cell(2, i + 1);

          // Assemble load vector
          L(dof_i) += Te(index_load, i);

          // Assemble bilinear form
          if constexpr (asmbl_systmtrx)
          {
            for (std::size_t j = 0; j < ndofs_per_cell; ++j)
            {
              A(dof_i, asmbl_info_cell(2, j + 1)) += Te(i, j);
            }
          }
        }
      }
    }
  }
}

/// Assemble EQS for constrained stress minimisation
///
/// Assembles system-matrix and load vector for constrained flux
/// minimisation on patch-wise divergence free H(div) space. Explicit ansatz
/// for such a space see [1, Lemma 12].
///
/// [1] Bertrand, F. et al.: https://doi.org/10.1007/s00211-023-01366-8, 2023
///
/// @tparam T                       The scalar type
/// @tparam id_flux_order           The flux order (1->RT1, 2->RT2, 3->general)
/// @tparam requires_bcs            Flag if BCs have to be considered
/// @param minimisation_kernel      The kernel for minimisation
/// @param patch_data               The temporary storage for the patch
/// @param asmbl_info               Informations patch-wise H(div=0) space
/// @param fct_reversion            Marker for reversed facets
template <typename T, int id_flux_order, bool bcs_required>
void assemble_stressminimiser(
    kernel_fn_schursolver<T>& minimisation_kernel,
    PatchData<T, id_flux_order>& patch_data,
    base::mdspan_t<const std::int32_t, 3> asmbl_info,
    base::mdspan_t<const std::uint8_t, 2> fct_reversion)
{
  assert(id_flux_order < 0);

  /* Extract data */
  // The spatial dimension
  const int gdim = patch_data.gdim();

  // Number of elements/facets on patch
  const int ncells = patch_data.ncells();

  // Check if Lagrange multiplier is required
  const bool requires_lagrmp = patch_data.meanvalue_zero_condition_required();

  // Tangent storage
  base::mdspan_t<T, 2> Ae = patch_data.Ae();
  base::mdspan_t<T, 2> Be = patch_data.Be();
  std::span<T> Ce = patch_data.Ce();
  std::span<T> Le = patch_data.Le();

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& A
      = patch_data.matrix_A_without_bc();
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& B = patch_data.matrix_B();
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& C = patch_data.matrix_C();
  Eigen::Matrix<T, Eigen::Dynamic, 1>& L = patch_data.vector_L();

  // The boundary markers
  std::span<const std::int8_t> boundary_markers
      = patch_data.boundary_markers(true);

  // DOF counters
  const int ndofs_flux_hdivz = patch_data.ndofs_flux_hdivz();
  const int ndofs_constr = patch_data.ndofs_constraint();

  const int ndofs_flux_per_cell = patch_data.ndofs_flux_hdivz_per_cell();
  ;
  const int ndofs_constr_per_cell = gdim + 1;

  /* Initialisation */
  if constexpr (bcs_required)
  {
    A.setZero();
    B.setZero();
    C.setZero();
    L.setZero();
  }
  else
  {
    B.setZero();
    C.setZero();
    L.setZero();
  }

  /* Assembly of the system matrices */
  // Offsets
  const int offset_dofmap_c = ndofs_flux_per_cell + 1;
  const int offset_Lc = gdim * ndofs_flux_hdivz;
  const int offset_Lec = gdim * ndofs_flux_per_cell;

  for (std::size_t a = 1; a < ncells + 1; ++a)
  {
    int id_a = a - 1;

    // Isoparametric mapping
    const double detJ = patch_data.jacobi_determinant(id_a);
    base::mdspan_t<const double, 2> J = patch_data.jacobian(id_a);

    // DOFmap on cell
    base::smdspan_t<const std::int32_t, 2> asmbl_info_cell
        = std::experimental::submdspan(asmbl_info,
                                       std::experimental::full_extent, a,
                                       std::experimental::full_extent);

    // DOFs on cell
    std::span<const T> coefficients = patch_data.coefficients_stress(a);

    // Evaluate linear- and bilinear form
    if constexpr (bcs_required)
    {
      // Initilaisation stoarge of element contributions
      patch_data.reinitialise_Ae();
      patch_data.reinitialise_Be();
      patch_data.reinitialise_Ce();
      patch_data.reinitialise_Le();

      // Evaluate kernel
      minimisation_kernel(Ae, Be, Ce, Le, coefficients, asmbl_info_cell,
                          fct_reversion(id_a, 0), detJ, J, true);
    }
    else
    {
      // Initilaisation stoarge of element contributions
      patch_data.reinitialise_Be();
      patch_data.reinitialise_Ce();
      patch_data.reinitialise_Le();

      // Evaluate kernel
      minimisation_kernel(Ae, Be, Ce, Le, coefficients, asmbl_info_cell,
                          fct_reversion(id_a, 0), detJ, J, false);
    }

    for (std::size_t k = 0; k < gdim; ++k)
    {
      // Offsets
      int offset_uk = k * ndofs_flux_hdivz;
      int offset_uek = k * ndofs_flux_per_cell;
      int offset_bk = k * ndofs_constr;
      int offset_bek = k * ndofs_constr_per_cell;

      // Assemble A, B_k and L_uk
      for (std::size_t i = 0; i < ndofs_flux_per_cell; ++i)
      {
        std::int32_t dof_i = asmbl_info_cell(2, i + 1);

        if constexpr (bcs_required)
        {
          // Linearforms L_uk
          L(offset_uk + dof_i) += Le[offset_uek + i];

          // Sub-matrix A
          if (k == 0)
          {
            for (std::size_t j = 0; j < ndofs_flux_per_cell; ++j)
            {
              A(dof_i, asmbl_info_cell(2, j + 1)) += Ae(i, j);
            }
          }
        }

        // Sub-matrices B_k
        for (std::size_t j = 0; j < ndofs_constr_per_cell; ++j)
        {
          std::int32_t dof_j = asmbl_info_cell(2, offset_dofmap_c + j);

          if constexpr (bcs_required)
          {
            if (!(boundary_markers[offset_uk + dof_i]))
            {
              B(dof_i, offset_bk + dof_j) += Be(i, offset_bek + j);
            }
          }
          else
          {
            B(dof_i, offset_bk + dof_j) += Be(i, offset_bek + j);
          }
        }
      }

      // Assemble C and L_c
      if (k == 0)
      {
        if (requires_lagrmp)
        {
          for (std::size_t i = 0; i < ndofs_constr_per_cell; ++i)
          {
            std::int32_t dof_i = asmbl_info_cell(2, offset_dofmap_c + i);

            // Sub-Matrix C
            C(dof_i, ndofs_constr) += Ce[i];
            C(ndofs_constr, dof_i) += Ce[i];

            // Linear form Lc
            L(offset_Lc + dof_i) += Le[offset_Lec + i];
          }
        }
        else
        {
          for (std::size_t i = 0; i < ndofs_constr_per_cell; ++i)
          {
            std::int32_t dof_i = asmbl_info_cell(2, offset_dofmap_c + i);
            L(offset_Lc + dof_i) += Le[offset_Lec + i];
          }
        }
      }
    }
  }
}

} // namespace dolfinx_eqlb::se
