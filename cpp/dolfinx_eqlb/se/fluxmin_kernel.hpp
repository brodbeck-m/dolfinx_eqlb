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
/// Generate kernel for flux minimisation
/// @tparam T              The scalar type
/// @tparam asmbl_systmtrx Id if entire tangent or only load vector is assembled
/// @param kernel_data     The kernel data
/// @param gdim            The spatial dimension
/// @param degree_rt       The degree of the Raviart-Thomas space
/// @return                The kernel function
template <typename T, bool asmbl_systmtrx = true>
se::kernel_fn<T, asmbl_systmtrx>
generate_flux_minimisation_kernel(se::KernelData<T>& kernel_data,
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
      = [kernel_data, ndofs_hdivzero_per_cell, ndofs_flux_fct](
            base::mdspan_t<T, 2> Te, std::span<const T> coefficients,
            base::smdspan_t<const std::int32_t, 2> asmbl_info,
            const std::uint8_t fct_eam1_reversed, const double detJ,
            base::mdspan_t<const double, 2> J) mutable
  {
    const int index_load = ndofs_hdivzero_per_cell;

    /* Extract shape functions and quadrature data */
    base::smdspan_t<double, 3> phi = kernel_data.shapefunctions_flux(J, detJ);

    base::mdspan_t<const double, 2> shapetrafo
        = kernel_data.entity_transformations_flux();

    std::span<const double> quadrature_weights
        = kernel_data.quadrature_weights(0);

    /* Initialisation */
    // Interpolated solution from step 1
    std::array<T, 2> sigtilde_q;

    // Intermediate storage for higher-order facet moments on reversed facet
    std::vector<double> data_gphi_Eam1(2 * ndofs_flux_fct, 0);
    base::mdspan_t<double, 2> gphi_Eam1(data_gphi_Eam1.data(), ndofs_flux_fct,
                                        2);

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

} // namespace dolfinx_eqlb::se