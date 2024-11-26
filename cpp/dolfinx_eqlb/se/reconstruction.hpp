// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "KernelData.hpp"
#include "Patch.hpp"
#include "PatchData.hpp"
#include "ProblemData.hpp"
#include "fluxmin_kernel.hpp"
#include "solve_patch_semiexplt.hpp"
#include "stressmin_kernel.hpp"
#include "utils.hpp"

#include <basix/e-lagrange.h>
#include <basix/finite-element.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx_eqlb/base/BoundaryData.hpp>
#include <dolfinx_eqlb/base/QuadratureRule.hpp>

#include <algorithm>
#include <array>
#include <exception>
#include <functional>
#include <iterator>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace dolfinx;

namespace base = dolfinx_eqlb::base;

namespace dolfinx_eqlb::se
{

/// Execute flux calculation based on H(div) conforming equilibration
///
/// Equilibration procedure is based on an explicitly calculated flux and an
/// unconstrained minimisation problem on a patch-wise divergence-free H(div)
/// space [1]. If stresses are considered the symmetry in considered in a weak
/// sense following [2]. The cells squared Korn constants are estimated based on
/// [3]. The grouping follows the idea in [4].
///
/// [1] Bertrand, F. et al.: https://doi.org/10.1007/s00211-023-01366-8, 2023
/// [2] Bertrand, F. et al.: https://doi.org/10.1002/num.22741, 2021
/// [3] Kim, K.-W.: https://doi.org/10.1137/110823031, 2011
/// [4] Moldenhauer, M.: http://d-nb.info/1221061712/34, 2020
///
/// @tparam T             The scalar type
/// @tparam id_flux_order The flux order (1->RT1, 2->RT2, 3->general)
/// @param problem_data       The problem data
/// @param symconstr_required Flag if weak symmetry condition is required
/// @param cells_kornconst    Upper bounds for cells Korn constant
template <typename T, int id_flux_order>
void reconstruction(ProblemData<T>& problem_data, const bool symconstr_required,
                    std::shared_ptr<fem::Function<T>> cells_kornconst)
{
  assert(id_flux_order < 0);

  // Determination of Korns constant
  const bool estimate_kornconst = (cells_kornconst != nullptr) ? true : false;

  std::span<T> x_kornconst;
  if (estimate_kornconst)
  {
    x_kornconst = cells_kornconst->x()->mutable_array();
  }

  /* Geometry */
  // Extract mesh
  std::shared_ptr<const mesh::Mesh> mesh = problem_data.mesh();
  const fem::CoordinateElement& cmap = mesh->geometry().cmap();

  // Permutations of facets
  const std::vector<std::uint8_t>& fct_perms
      = mesh->topology().get_facet_permutations();

  // Spacial dimension
  const int dim = mesh->geometry().dim();

  // Number of nodes on processor
  int n_nodes = mesh->topology().index_map(0)->size_local();

  // Number of elements on processor
  int n_cells = mesh->topology().index_map(dim)->size_local();

  /* Basix elements */
  // Basix element of pice-wise H(div) flux
  const basix::FiniteElement& basix_element_fluxhdiv
      = problem_data.fspace_flux_hdiv()->element()->basix_element();

  const int degree_flux_hdiv = basix_element_fluxhdiv.degree();
  const int degree_rt_flux_hdiv = degree_flux_hdiv - 1;

  // Basix element of projected flux/ RHS
  const basix::FiniteElement& basix_element_rhs
      = problem_data.fspace_flux_dg()->element()->basix_element();

  const int degree_rhs = basix_element_rhs.degree();

  bool is_dg = (degree_rhs == 0) ? true : false;

  basix::FiniteElement basix_element_rhscg = basix::element::create_lagrange(
      basix_element_rhs.cell_type(), degree_rhs,
      basix_element_rhs.lagrange_variant(), is_dg);

  // Basix element of hat-function
  basix::FiniteElement basix_element_hat = basix::element::create_lagrange(
      basix_element_rhs.cell_type(), 1, basix_element_rhs.lagrange_variant(),
      false);

  /* Equilibration */
  // Set quadrature rule
  const int quadrature_degree
      = (degree_flux_hdiv == 1) ? 2 : 2 * degree_flux_hdiv + 1;
  base::QuadratureRule quadrature_rule = base::QuadratureRule(
      mesh->topology().cell_type(), quadrature_degree, dim);

  // Initialize KernelData
  KernelData<T> kernel_data = KernelData<T>(
      mesh, std::make_shared<base::QuadratureRule>(quadrature_rule),
      basix_element_fluxhdiv, basix_element_rhs, basix_element_hat);

  // Generate minimisation kernels
  kernel_fn<T, true> kernel_fluxmin
      = generate_flux_minimisation_kernel<T, true>(kernel_data, dim,
                                                   degree_rt_flux_hdiv);

  kernel_fn<T, false> kernel_fluxmin_l
      = generate_flux_minimisation_kernel<T, false>(kernel_data, dim,
                                                    degree_rt_flux_hdiv);

  // Execute equilibration
  // FIXME - Currently only 2D meshes supported
  if (symconstr_required)
  {
    // Get list with node markers on stress boundary
    std::span<const std::int8_t> pnt_on_stress_boundary
        = problem_data.node_on_essnt_boundary_stress();

    // Initialise patch
    Patch<T, id_flux_order> patch = Patch<T, id_flux_order>(
        mesh, problem_data.facet_type(), pnt_on_stress_boundary,
        problem_data.fspace_flux_hdiv(), problem_data.fspace_flux_dg(),
        basix_element_rhscg, true, 1, 2);

    // Initialise storage for equilibration
    PatchData<T, id_flux_order> patch_data = PatchData<T, id_flux_order>(
        patch, kernel_data.nipoints_facet(), true);

    // Set kernel for weak symmetry condition
    kernel_fn_schursolver<T> kernel_weaksym
        = generate_stress_minimisation_kernel<T>(Kernel::StressMin, kernel_data,
                                                 dim, patch.fcts_per_cell(),
                                                 degree_rt_flux_hdiv);

    // Initialise list with equilibration markers
    std::vector<bool> perform_equilibration(n_nodes, true);

    // Loop over extended patches on essential boundary
    // TODO - Extend patch grouping on mixed patches
    if (degree_flux_hdiv == 2)
    {
      for (std::int32_t i_node = 0; i_node < n_nodes; ++i_node)
      {
        if (pnt_on_stress_boundary[i_node] && perform_equilibration[i_node])
        {
          // Group the patches
          std::vector<std::int32_t> grouped_patches
              = patch.group_boundary_patches(i_node, pnt_on_stress_boundary, 2);

          // Check if modification of patch is required
          if (grouped_patches.size() >= 2)
          {
            // Equilibration step 1: Explicit step and minimisation
            for (std::size_t i = grouped_patches.size(); i-- > 0;)
            {
              // Patch-central node
              const std::int32_t node_i = grouped_patches[i];

              // Check if patch has already been considered
              if (!perform_equilibration[node_i])
              {
                throw std::runtime_error("Incompatible mesh! To many patches "
                                         "with 2 cells on neumann boundary.");
              }
              else
              {
                perform_equilibration[node_i] = false;
              }

              // Create Sub-DOFmap
              patch.create_subdofmap(node_i);

              // Estimate the Korn constant
              if (estimate_kornconst)
              {
                // Estimate squared Korn constant
                double cks = patch.estimate_squared_korn_constant() * (dim + 1);

                // Store Korn's constant
                for (std::int32_t cell : patch.cells())
                {
                  x_kornconst[cell] += cks;
                }
              }

              // Re-initialise PatchData
              patch_data.reinitialisation(patch.type(), patch.ncells());

              // Perform equilibration
              equilibrate_flux_semiexplt<T, id_flux_order>(
                  mesh->geometry(), fct_perms, patch, patch_data, problem_data,
                  kernel_data, kernel_fluxmin, kernel_fluxmin_l);
            }

            // Equilibration step 2: Weak symmetry condition
            impose_weak_symmetry<T, id_flux_order, true>(
                mesh->geometry(), patch, patch_data, problem_data, kernel_data,
                kernel_weaksym);
          }
        }
      }
    }

    // Loop over all other patches
    for (std::size_t i_node = 0; i_node < n_nodes; ++i_node)
    {
      if (perform_equilibration[i_node])
      {
        // Set marker for patch
        perform_equilibration[i_node] = false;

        // Create Sub-DOFmap
        patch.create_subdofmap(i_node);

        // Estimate the Korn constant
        if (estimate_kornconst)
        {
          // Estimate squared Korn constant
          double cks = patch.estimate_squared_korn_constant() * (dim + 1);

          // Store Korn's constant
          for (std::int32_t cell : patch.cells())
          {
            x_kornconst[cell] += cks;
          }
        }

        // Reinitialise patch-data
        patch_data.reinitialisation(patch.type(), patch.ncells());

        // Calculate solution patch
        equilibrate_flux_semiexplt<T, id_flux_order>(
            mesh->geometry(), fct_perms, patch, patch_data, problem_data,
            kernel_data, kernel_fluxmin, kernel_fluxmin_l, kernel_weaksym);
      }
    }
  }
  else
  {
    // Initialise patch
    Patch<T, id_flux_order> patch = Patch<T, id_flux_order>(
        mesh, problem_data.facet_type(),
        problem_data.node_on_essnt_boundary_stress(),
        problem_data.fspace_flux_hdiv(), problem_data.fspace_flux_dg(),
        basix_element_rhscg);

    // Initialise storage for equilibration
    PatchData<T, id_flux_order> patch_data = PatchData<T, id_flux_order>(
        patch, kernel_data.nipoints_facet(), false);

    // Loop over all patches
    for (std::size_t i_node = 0; i_node < n_nodes; ++i_node)
    {
      // Create Sub-DOFmap
      patch.create_subdofmap(i_node);

      // Estimate the Korn constant
      if (estimate_kornconst)
      {
        // Estimate squared Korn constant
        double cks = patch.estimate_squared_korn_constant() * (dim + 1);

        // Store Korn's constant
        for (std::int32_t cell : patch.cells())
        {
          x_kornconst[cell] += cks;
        }
      }

      // Reinitialise patch-data
      patch_data.reinitialisation(patch.type(), patch.ncells());

      // Calculate solution patch
      equilibrate_flux_semiexplt<T, id_flux_order>(
          mesh->geometry(), fct_perms, patch, patch_data, problem_data,
          kernel_data, kernel_fluxmin, kernel_fluxmin_l);
    }
  }
}

/// Execute flux calculation based on H(div) conforming equilibration
///
/// Equilibration procedure is based on an explicitly calculated flux and an
/// unconstrained minimisation problem on a patch-wise divergence-free H(div)
/// space [1]. If stresses are considered the symmetry in considered in a weak
/// sense following [2]. The cells squared Korn constants are estimated based on
/// [3]. The grouping follows the idea in [4].
///
/// [1] Bertrand, F. et al.: https://doi.org/10.1007/s00211-023-01366-8, 2023
/// [2] Bertrand, F. et al.: https://doi.org/10.1002/num.22741, 2021
/// [3] Kim, K.-W.: https://doi.org/10.1137/110823031, 2011
/// [4] Moldenhauer, M.: http://d-nb.info/1221061712/34, 2020
///
/// @param flux_hdiv           Function that holds the reconstructed flux
/// @param flux_dg             Function that holds the projected primal flux
/// @param rhs_dg              Function that holds the projected rhs
/// @param fct_esntbound_prime Facets of essential BCs of primal problem
/// @param fct_esntbound_flux  Facets of essential BCs on flux field
/// @param bcs_flux            Essential boundary conditions for the flux
template <typename T>
void reconstruction(std::vector<std::shared_ptr<fem::Function<T>>>& flux_hdiv,
                    std::vector<std::shared_ptr<fem::Function<T>>>& flux_dg,
                    std::vector<std::shared_ptr<fem::Function<T>>>& rhs_dg,
                    std::shared_ptr<base::BoundaryData<T>> boundary_data,
                    const bool reconstruct_stress,
                    std::shared_ptr<dolfinx::fem::Function<T>> cells_kornconst)
{
  // Input size and polynomial degrees
  const int n_rhs = rhs_dg.size();
  const int n_flux_hdiv = flux_hdiv.size();
  const int n_flux_dg = flux_dg.size();
  const int n_bcs = boundary_data->num_rhs();

  const int order_flux
      = flux_hdiv[0]->function_space()->element()->basix_element().degree();
  const int degree_flux_dg
      = flux_dg[0]->function_space()->element()->basix_element().degree();
  const int degree_rhs
      = rhs_dg[0]->function_space()->element()->basix_element().degree();

  // Check input
  if (n_rhs != n_bcs || n_rhs != n_flux_hdiv || n_rhs != n_flux_dg)
  {
    throw std::runtime_error("Equilibration: Input sizes does not match");
  }

  if (degree_rhs > (order_flux - 1) || degree_flux_dg > degree_rhs)
  {
    throw std::runtime_error(
        "Equilibration: Wrong polynomial degree of the projected RHS");
  }

  if (degree_flux_dg != degree_rhs)
  {
    throw std::runtime_error(
        "Equilibration: Degrees of projected flux and RHS have to match");
  }

  // Additional checks for stress equilibration
  if (reconstruct_stress)
  {
    if (n_rhs < flux_hdiv[0]->function_space()->mesh()->geometry().dim())
    {
      throw std::runtime_error(
          "Stress equilibration: Specify all rows of stress tensor");
    }

    if (n_flux_hdiv < 2)
    {
      throw std::runtime_error("Stress equilibration: RT_k with k>1 required!");
    }
  }

  /* Set problem data */
  ProblemData<T> problem_data
      = ProblemData<T>(flux_hdiv, flux_dg, rhs_dg, boundary_data);

  /* Call equilibration */
  if (order_flux == 1)
  {
    reconstruction<T, 1>(problem_data, reconstruct_stress, cells_kornconst);
  }
  else if (order_flux == 2)
  {
    reconstruction<T, 2>(problem_data, reconstruct_stress, cells_kornconst);
  }
  else
  {
    reconstruction<T, 3>(problem_data, reconstruct_stress, cells_kornconst);
  }
}

} // namespace dolfinx_eqlb::se