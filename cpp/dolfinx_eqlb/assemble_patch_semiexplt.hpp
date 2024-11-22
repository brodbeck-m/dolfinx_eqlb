// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "eigen3/Eigen/Dense"

#include "KernelData.hpp"
#include "PatchCstm.hpp"
#include "PatchData.hpp"
#include "utils.hpp"

#include <dolfinx_eqlb/base/Patch.hpp>

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
#include <vector>

using namespace dolfinx;

namespace dolfinx_eqlb
{

namespace stdex = std::experimental;

// ------------------------------------------------------------------------------

/* The minimisation kernels */

enum class Kernel
{
  FluxMin,
  StressMin,
  StressMinNL,
};

/// Generate kernel for the assembly flux minimisation
/// @tparam T              The scalar type
/// @tparam asmbl_systmtrx Id if entire tangent or only load vector is assembled
/// @param kernel_data     The kernel data
/// @param gdim            The spatial dimension
/// @param degree_rt       The degree of the Raviart-Thomas space
/// @return                The kernel function
template <typename T, bool asmbl_systmtrx = true>
kernel_fn<T, asmbl_systmtrx>
generate_flux_minimisation_kernel(KernelDataEqlb<T>& kernel_data,
                                  const int gdim, const int degree_rt)
{
  /* DOF counters */
  const int ndofs_flux_fct = degree_rt + 1;
  const int ndofs_flux_cell_add = 0.5 * degree_rt * (degree_rt - 1);

  const int ndofs_hdivzero_per_cell
      = 2 * ndofs_flux_fct + ndofs_flux_cell_add - 1;

  /* The kernel */

  /// Kernel for unconstrained flux minimisation
  ///
  /// Calculates system-matrix and load vector for unconstrained flux
  /// minimisation on patch-wise divergence free H(div) space.
  ///
  /// @tparam T               The scalar type
  /// @tparam asmbl_systmtrx  Flag if entire tangent or only load vector is
  ///                         assembled
  /// @param Te           Storage for tangent arrays
  /// @param coefficients The Coefficients
  /// @param asmbl_info   Information to create the patch-wise H(div=0) space
  /// @param detJ         The Jacobi determinant
  /// @param J            The Jacobi matrix
  kernel_fn<T, asmbl_systmtrx> unconstrained_flux_minimisation
      = [kernel_data, ndofs_hdivzero_per_cell,
         ndofs_flux_fct](mdspan_t<T, 2> Te, std::span<const T> coefficients,
                         smdspan_t<const std::int32_t, 2> asmbl_info,
                         const std::uint8_t fct_eam1_reversed,
                         const double detJ, mdspan_t<const double, 2> J) mutable
  {
    const int index_load = ndofs_hdivzero_per_cell;

    /* Extract shape functions and quadrature data */
    smdspan_t<double, 3> phi = kernel_data.shapefunctions_flux(J, detJ);

    mdspan_t<const double, 2> shapetrafo
        = kernel_data.entity_transformations_flux();

    std::span<const double> quadrature_weights
        = kernel_data.quadrature_weights(0);

    /* Initialisation */
    // Interpolated solution from step 1
    std::array<T, 2> sigtilde_q;

    // Intermediate storage for higher-order facet moments on reversed facet
    std::vector<double> data_gphi_Eam1(2 * ndofs_flux_fct, 0);
    mdspan_t<double, 2> gphi_Eam1(data_gphi_Eam1.data(), ndofs_flux_fct, 2);

    // Data manipulation of shape function for d0
    std::int32_t ld0_Eam1 = asmbl_info(0, 0),
                 ld0_Ea = asmbl_info(0, ndofs_flux_fct);
    std::int32_t p_Eam1 = asmbl_info(3, 0),
                 p_Ea = asmbl_info(3, ndofs_flux_fct);

    /* Assemble tangents */
    for (std::size_t iq = 0; iq < quadrature_weights.size(); ++iq)
    {
      // Interpolate flux_tilde
      sigtilde_q[0] = 0;
      sigtilde_q[1] = 0;

      for (std::size_t i = 0; i < phi.extent(1); ++i)
      {
        sigtilde_q[0] += coefficients[i] * phi(iq, i, 0);
        sigtilde_q[1] += coefficients[i] * phi(iq, i, 1);
      }

      // Transform shape functions in case of facet reversion
      if (fct_eam1_reversed)
      {
        // Transform higher order shape functions on facet Ea
        for (std::size_t i = 0; i < ndofs_flux_fct; ++i)
        {
          for (std::size_t j = 0; j < ndofs_flux_fct; ++j)
          {
            int ldj_Eam1 = asmbl_info(0, j);

            gphi_Eam1(i, 0) += shapetrafo(i, j) * phi(iq, ldj_Eam1, 0);
            gphi_Eam1(i, 1) += shapetrafo(i, j) * phi(iq, ldj_Eam1, 1);
          }
        }

        // Write data into storage
        for (std::size_t i = 0; i < ndofs_flux_fct; ++i)
        {
          int ldi_Eam1 = asmbl_info(0, i);

          phi(iq, ldi_Eam1, 0) = gphi_Eam1(i, 0);
          phi(iq, ldi_Eam1, 1) = gphi_Eam1(i, 1);
        }

        // Set intermediate storage to zero
        std::fill(data_gphi_Eam1.begin(), data_gphi_Eam1.end(), 0);
      }

      // Manipulate shape function for coefficient d_0
      phi(iq, ld0_Ea, 0)
          = p_Ea * (p_Eam1 * phi(iq, ld0_Eam1, 0) + p_Ea * phi(iq, ld0_Ea, 0));
      phi(iq, ld0_Ea, 1)
          = p_Ea * (p_Eam1 * phi(iq, ld0_Eam1, 1) + p_Ea * phi(iq, ld0_Ea, 1));

      // Volume integrator
      double dvol = quadrature_weights[iq] * std::fabs(detJ);

      // Assemble linear- and bilinear form
      for (std::size_t i = 0; i < ndofs_hdivzero_per_cell; ++i)
      {
        // Offset
        std::size_t ip1 = i + 1;

        // Manipulate shape functions
        double alpha = asmbl_info(3, ip1) * dvol;

        double phi_i0 = phi(iq, asmbl_info(0, ip1), 0) * alpha;
        double phi_i1 = phi(iq, asmbl_info(0, ip1), 1) * alpha;

        // Linear form
        Te(index_load, i) -= phi_i0 * sigtilde_q[0] + phi_i1 * sigtilde_q[1];

        if constexpr (asmbl_systmtrx)
        {
          for (std::size_t j = i; j < ndofs_hdivzero_per_cell; ++j)
          {
            // Offset
            std::size_t jp1 = j + 1;

            // Manipulate shape functions
            double phi_j0 = phi(iq, asmbl_info(0, jp1), 0) * asmbl_info(3, jp1);
            double phi_j1 = phi(iq, asmbl_info(0, jp1), 1) * asmbl_info(3, jp1);

            // Bilinear form
            Te(i, j) += phi_i0 * phi_j0 + phi_i1 * phi_j1;
          }
        }
      }
    }

    // Set symmetric contributions of element mass-matrix
    if constexpr (asmbl_systmtrx)
    {
      if (ndofs_hdivzero_per_cell > 1)
      {
        for (std::size_t i = 1; i < ndofs_hdivzero_per_cell; ++i)
        {
          for (std::size_t j = 0; j < i; ++j)
          {
            Te(i, j) = Te(j, i);
          }
        }
      }
    }
  };

  if (gdim == 2)
  {
    return unconstrained_flux_minimisation;
  }
  else
  {
    throw std::runtime_error(
        "Kernel for flux minimisation (3D) not implemented");
  }
}

/// Generate kernel for the assembly stress minimisation user consideration of a
/// weak symmetry condition
/// @tparam T              The scalar type
/// @param type            The kernel type
/// @param kernel_data     The kernel data
/// @param gdim            The spatial dimension
/// @param facets_per_cell The number of facets per cell
/// @param degree_rt       The degree of the Raviart-Thomas space
/// @return                The kernel function
template <typename T>
kernel_fn_schursolver<T>
generate_stress_minimisation_kernel(Kernel type, KernelDataEqlb<T>& kernel_data,
                                    const int gdim, const int facets_per_cell,
                                    const int degree_rt)
{
  /* DOF counters */
  const int ndofs_flux_fct = degree_rt + 1;
  const int ndofs_flux_cell_add = 0.5 * degree_rt * (degree_rt - 1);

  const int ndofs_hdivzero_per_cell
      = 2 * ndofs_flux_fct + ndofs_flux_cell_add - 1;
  const int ndofs_constr_per_cell
      = (gdim == 2) ? facets_per_cell : gdim * facets_per_cell;

  /* The kernel */

  /// Kernel for the (constrained) stress correction (2D)
  ///
  /// Calculates system-matrix and load vector for constrained stress
  /// minimisation on patch-wise divergence free H(div=0) space. Weak symmetry
  /// is zero-tested against a patch-wise continuous P1 space.
  ///
  /// @tparam T             The scalar type
  /// @param Ae             Storage for sub-matrix Ae
  /// @param Be             Storage for sub-matrices Be_k
  /// @param Ce             Storage for sub-matrix Ce
  ///                       (contribution of the Lagrangian multiplier)
  /// @param Le             Storage for load vector Le
  /// @param coefficients   The coefficients
  /// @param asmbl_info     Information to create the patch-wise H(div=0) space
  /// @param detJ           The Jacobi determinant
  /// @param J              The Jacobi matrix
  /// @param assemble_A     Flag if sub-matrix A has to be assembled
  kernel_fn_schursolver<T> stress_minimisation_2D
      = [kernel_data, ndofs_hdivzero_per_cell, ndofs_constr_per_cell,
         ndofs_flux_fct](mdspan_t<T, 2> Ae, mdspan_t<T, 2> Be, std::span<T> Ce,
                         std::span<T> Le, std::span<const T> coefficients,
                         smdspan_t<const std::int32_t, 2> asmbl_info,
                         const std::uint8_t fct_eam1_reversed,
                         const double detJ, mdspan_t<const double, 2> J,
                         const bool assemble_A) mutable
  {
    /* Extract shape functions and quadrature data */
    smdspan_t<double, 3> phi_f = kernel_data.shapefunctions_flux(J, detJ);
    smdspan_t<const double, 2> phi_c = kernel_data.shapefunctions_cell_hat();

    mdspan_t<const double, 2> shapetrafo
        = kernel_data.entity_transformations_flux();

    std::span<const double> quadrature_weights
        = kernel_data.quadrature_weights(0);

    /* Initialisation */
    const int ndofs_flux = phi_f.extent(1);

    // Interpolated solution from step 1
    std::array<T, 2> sig_r0, sig_r1;

    // Intermediate storage for modified shape-functions on reversed facet
    std::vector<double> data_gphif_Eam1(2 * ndofs_flux_fct, 0);
    mdspan_t<double, 2> gphif_Eam1(data_gphif_Eam1.data(), ndofs_flux_fct, 2);

    // Data manipulation of shape function for d0
    std::int32_t ld0_Eam1 = asmbl_info(0, 0),
                 ld0_Ea = asmbl_info(0, ndofs_flux_fct);
    std::int32_t p_Eam1 = asmbl_info(3, 0),
                 p_Ea = asmbl_info(3, ndofs_flux_fct);

    // Offset constrained space
    const int offs_c = ndofs_hdivzero_per_cell + 1;
    const int offs_Lc = 2 * ndofs_hdivzero_per_cell;

    /* Assemble tangents */
    for (std::size_t iq = 0; iq < quadrature_weights.size(); ++iq)
    {
      // Interpolate stress
      sig_r0[0] = 0, sig_r0[1] = 0;
      sig_r1[0] = 0, sig_r1[1] = 0;

      for (std::size_t i = 0; i < ndofs_flux; ++i)
      {
        int offs = ndofs_flux + i;

        sig_r0[0] += coefficients[i] * phi_f(iq, i, 0);
        sig_r0[1] += coefficients[i] * phi_f(iq, i, 1);

        sig_r1[0] += coefficients[offs] * phi_f(iq, i, 0);
        sig_r1[1] += coefficients[offs] * phi_f(iq, i, 1);
      }

      // Transform shape functions in case of facet reversion
      if (fct_eam1_reversed)
      {
        // Transform higher order shape functions on facet Ea
        for (std::size_t i = 0; i < ndofs_flux_fct; ++i)
        {
          for (std::size_t j = 0; j < ndofs_flux_fct; ++j)
          {
            int ldj_Eam1 = asmbl_info(0, j);

            gphif_Eam1(i, 0) += shapetrafo(i, j) * phi_f(iq, ldj_Eam1, 0);
            gphif_Eam1(i, 1) += shapetrafo(i, j) * phi_f(iq, ldj_Eam1, 1);
          }
        }

        // Write data into storage
        for (std::size_t i = 0; i < ndofs_flux_fct; ++i)
        {
          int ldi_Eam1 = asmbl_info(0, i);

          phi_f(iq, ldi_Eam1, 0) = gphif_Eam1(i, 0);
          phi_f(iq, ldi_Eam1, 1) = gphif_Eam1(i, 1);
        }

        // Set intermediate storage to zero
        std::fill(data_gphif_Eam1.begin(), data_gphif_Eam1.end(), 0);
      }

      // Manipulate shape function for coefficient d_0
      phi_f(iq, ld0_Ea, 0)
          = p_Ea
            * (p_Eam1 * phi_f(iq, ld0_Eam1, 0) + p_Ea * phi_f(iq, ld0_Ea, 0));
      phi_f(iq, ld0_Ea, 1)
          = p_Ea
            * (p_Eam1 * phi_f(iq, ld0_Eam1, 1) + p_Ea * phi_f(iq, ld0_Ea, 1));

      // Volume integrator
      double dvol = quadrature_weights[iq] * std::fabs(detJ);

      for (std::size_t i = 0; i < ndofs_hdivzero_per_cell; ++i)
      {
        // Offset
        std::size_t ip1 = i + 1;

        // Cell-local DOF (index i)
        std::int32_t dl_i = asmbl_info(0, ip1);

        // Manipulate shape functions
        double alpha = asmbl_info(3, ip1) * dvol;

        double phi_i0 = phi_f(iq, dl_i, 0) * alpha;
        double phi_i1 = phi_f(iq, dl_i, 1) * alpha;

        if (assemble_A)
        {
          // Load vector L_sigma
          Le[i] -= sig_r0[0] * phi_i0 + sig_r0[1] * phi_i1;
          Le[ndofs_hdivzero_per_cell + i]
              -= sig_r1[0] * phi_i0 + sig_r1[1] * phi_i1;

          // Sub-matrix A
          for (std::size_t j = i; j < ndofs_hdivzero_per_cell; ++j)
          {
            // Offset
            std::size_t jp1 = j + 1;

            // Cell-local DOF (index j)
            std::int32_t dl_j = asmbl_info(0, jp1);

            // Manipulate shape functions
            double phi_j0 = phi_f(iq, dl_j, 0) * asmbl_info(3, jp1);
            double phi_j1 = phi_f(iq, dl_j, 1) * asmbl_info(3, jp1);

            // System matrix
            Ae(i, j) += phi_i0 * phi_j0 + phi_i1 * phi_j1;
          }
        }

        // Sub-matrices B1 and B2
        for (std::size_t j = 0; j < ndofs_constr_per_cell; ++j)
        {
          // Shape function (index j)
          double phi_j = phi_c(iq, asmbl_info(0, offs_c + j));

          // System matrix
          Be(i, j) += phi_i1 * phi_j;
          Be(i, ndofs_constr_per_cell + j) -= phi_i0 * phi_j;
        }
      }

      for (std::size_t i = 0; i < ndofs_constr_per_cell; ++i)
      {
        // Manipulate shape function (index i)
        double phi_i = phi_c(iq, asmbl_info(0, offs_c + i)) * dvol;

        // Submatrix Ce
        Ce[i] += phi_i;

        // Load vector L_c
        Le[offs_Lc + i] -= phi_i * (sig_r0[1] - sig_r1[0]);
      }
    }

    if (assemble_A)
    {
      for (std::size_t i = 1; i < ndofs_hdivzero_per_cell; ++i)
      {
        for (std::size_t j = 0; j < i; ++j)
        {
          Ae(i, j) = Ae(j, i);
        }
      }
    }
  };

  switch (type)
  {
  case Kernel::StressMin:
    if (gdim == 2)
    {
      return stress_minimisation_2D;
    }
    else
    {
      throw std::runtime_error(
          "Kernel for weakly symmetric stresses (3D) not implemented");
    }
  case Kernel::StressMinNL:
    if (gdim == 2)
    {
      throw std::runtime_error(
          "Kernel for weakly symmetric 1.PK stress (2D) not implemented");
    }
    else
    {
      throw std::runtime_error(
          "Kernel for weakly symmetric 1.PK stress (3D) not implemented");
    }
  default:
    throw std::invalid_argument("Unrecognised kernel");
  }
}

// ------------------------------------------------------------------------------

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
                            PatchDataCstm<T, id_flux_order>& patch_data,
                            mdspan_t<const std::int32_t, 3> asmbl_info,
                            mdspan_t<const std::uint8_t, 2> fct_reversion,
                            const int i_rhs, const bool requires_flux_bc)
{
  assert(id_flux_order < 0);

  /* Extract data */
  // The spatial dimension
  const int gdim = patch_data.gdim();

  // Number of elements/facets on patch
  const int ncells = patch_data.ncells();

  // Tangent storage
  mdspan_t<T, 2> Te = patch_data.Te();

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
    mdspan_t<const double, 2> J = patch_data.jacobian(id_a);

    // DOFmap on cell
    smdspan_t<const std::int32_t, 2> asmbl_info_cell = stdex::submdspan(
        asmbl_info, stdex::full_extent, a, stdex::full_extent);

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
void assemble_stressminimiser(kernel_fn_schursolver<T>& minimisation_kernel,
                              PatchDataCstm<T, id_flux_order>& patch_data,
                              mdspan_t<const std::int32_t, 3> asmbl_info,
                              mdspan_t<const std::uint8_t, 2> fct_reversion)
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
  mdspan_t<T, 2> Ae = patch_data.Ae();
  mdspan_t<T, 2> Be = patch_data.Be();
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
    mdspan_t<const double, 2> J = patch_data.jacobian(id_a);

    // DOFmap on cell
    smdspan_t<const std::int32_t, 2> asmbl_info_cell = stdex::submdspan(
        asmbl_info, stdex::full_extent, a, stdex::full_extent);

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
} // namespace dolfinx_eqlb
