// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "KernelData.hpp"
#include "utils.hpp"

#include <dolfinx_eqlb/base/mdspan.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

using namespace dolfinx;

namespace dolfinx_eqlb::se
{

enum class Kernel
{
  StressMin,
  StressMinNL,
};

/// Generate kernel for the assembly stress minimisation under consideration of
/// a weak symmetry condition
/// @tparam T              The scalar type
/// @param type            The kernel type
/// @param kernel_data     The kernel data
/// @param gdim            The spatial dimension
/// @param facets_per_cell The number of facets per cell
/// @param degree_rt       The degree of the Raviart-Thomas space
/// @return                The kernel function
template <typename T>
kernel_fn_schursolver<T>
generate_stress_minimisation_kernel(Kernel type, KernelData<T>& kernel_data,
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
         ndofs_flux_fct](
            base::mdspan_t<T, 2> Ae, base::mdspan_t<T, 2> Be, std::span<T> Ce,
            std::span<T> Le, std::span<const T> coefficients,
            base::smdspan_t<const std::int32_t, 2> asmbl_info,
            const std::uint8_t fct_eam1_reversed, const double detJ,
            base::mdspan_t<const double, 2> J, const bool assemble_A) mutable
  {
    /* Extract shape functions and quadrature data */
    base::smdspan_t<double, 3> phi_f = kernel_data.shapefunctions_flux(J, detJ);
    base::smdspan_t<const double, 2> phi_c
        = kernel_data.shapefunctions_cell_hat();

    base::mdspan_t<const double, 2> shapetrafo
        = kernel_data.entity_transformations_flux();

    std::span<const double> quadrature_weights
        = kernel_data.quadrature_weights(0);

    /* Initialisation */
    const int ndofs_flux = phi_f.extent(1);

    // Interpolated solution from step 1
    std::array<T, 2> sig_r0, sig_r1;

    // Intermediate storage for modified shape-functions on reversed facet
    std::vector<double> data_gphif_Eam1(2 * ndofs_flux_fct, 0);
    base::mdspan_t<double, 2> gphif_Eam1(data_gphif_Eam1.data(), ndofs_flux_fct,
                                         2);

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

} // namespace dolfinx_eqlb::se