// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "eigen3/Eigen/Dense"
#include "mdspan.hpp"

#include <cmath>
#include <span>

namespace dolfinx_eqlb::base
{

template <typename T>
void boundary_projection_kernel(
    std::span<const double> ntrace_flux_boundary,
    std::span<const double> facet_normal, mdspan_t<double, 3> phi,
    std::span<const double> weights, const double detJ,
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& A_e,
    Eigen::Matrix<T, Eigen::Dynamic, 1>& L_e)
{
  // The spacial dimension
  const int gdim = phi.extent(2);

  // The number of quadrature points
  const int num_qpoints = weights.size();

  // The number of shape functions per facet
  const int ndofs_per_fct = phi.extent(1);

  // Initialise tangent arrays
  A_e.setZero();
  L_e.setZero();

  // Initialise normal-trace of flux
  double ntrace_phi_i, ntrace_phi_j;

  // Quadrature loop
  for (std::size_t iq = 0; iq < num_qpoints; ++iq)
  {
    for (std::size_t i = 0; i < ndofs_per_fct; ++i)
    {
      // Normal trace of phi_i
      ntrace_phi_i = 0.0;

      for (std::size_t k = 0; k < gdim; ++k)
      {
        ntrace_phi_i += phi(iq, i, k) * facet_normal[k];
      }

      // Set RHS
      L_e(i) += ntrace_phi_i * ntrace_flux_boundary[iq] * weights[iq];

      for (std::size_t j = i; j < ndofs_per_fct; ++j)
      {
        // Normal trace of phi_i
        ntrace_phi_j = 0.0;

        for (std::size_t k = 0; k < gdim; ++k)
        {
          ntrace_phi_j += phi(iq, j, k) * facet_normal[k];
        }

        // Set entry mass matrix
        A_e(i, j) += ntrace_phi_i * ntrace_phi_j * weights[iq];
      }
    }
  }

  // Add symmetric entries of mass-matrix
  for (std::size_t i = 1; i < ndofs_per_fct; ++i)
  {
    for (std::size_t j = 0; j < i; ++j)
    {
      A_e(i, j) += A_e(j, i);
    }
  }
}

} // namespace dolfinx_eqlb::base