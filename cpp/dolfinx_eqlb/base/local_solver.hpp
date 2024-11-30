// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "ProblemData.hpp"
#include "mdspan.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/graph/AdjacencyList.h>

#include <algorithm>
#include <functional>
#include <iostream>
#include <iterator>
#include <span>
#include <vector>

using namespace dolfinx;

namespace dolfinx_eqlb::base
{

/// Solve cell-local problems
///
/// Solve a set of cell local problems with identical bilinear- but multiple,
/// different linear forms. Each problem has no boundary conditions.
///
/// @param[in,out] solutions List of solution functions
/// @param[in] a             The Bilinear form
/// @param[in] ls            The linear forms
template <dolfinx::scalar T, std::floating_point U>
void local_solver(std::vector<std::shared_ptr<fem::Function<T, U>>>& solutions,
                  const fem::Form<T, U>& a,
                  const std::vector<std::shared_ptr<const fem::Form<T, U>>>& ls)
{
  /* Check input */
  if (ls.size() != solutions.size())
  {
    throw std::runtime_error("Local solver: Input sizes does not match");
  }

  /* Initialise data */
  // Cell geometry
  std::span<const dolfinx::scalar_value_type_t<T>> x = a.mesh()->geometry().x();
  mdspan_t<const std::int32_t, 2> x_dofmap = a.mesh()->geometry().dofmap();

  std::vector<scalar_value_type_t<T>> coordinate_dofs(3 * x_dofmap.extent(1));

  // Constants and coefficients (bilinear form)
  const std::vector<T> constants_a = fem::pack_constants(a);

  auto interm_coefficients_a = fem::allocate_coefficient_storage(a);
  fem::pack_coefficients(a, interm_coefficients_a);

  std::map<std::pair<fem::IntegralType, int>,
           std::pair<std::span<const T>, int>>
      coefficients_a = fem::make_coefficients_span(interm_coefficients_a);

  // Data for the linear form
  ProblemData<T, U> problem_data = ProblemData<T, U>(solutions, ls);

  // The DofMap
  std::shared_ptr<const fem::DofMap> dofmap
      = a.function_spaces().at(0)->dofmap();

  mdspan_t<const std::int32_t, 2> dofs = dofmap->map();
  const int bs = dofmap->bs();

  // The equation system
  const int num_dofs = dofs.extent(1);
  const int ndim = bs * num_dofs;

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A_e;
  Eigen::Matrix<T, Eigen::Dynamic, 1> L_e, u_e;

  A_e.resize(ndim, ndim);
  L_e.resize(ndim);
  u_e.resize(ndim);

  // The solver
  Eigen::LLT<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> solver;

  /* Solve element-wise equation systems */
  // Loop over all cell domains
  for (int i : a.integral_ids(fem::IntegralType::cell))
  {
    // Extract cells
    const std::span<const std::int32_t> cells
        = a.domain(fem::IntegralType::cell, i);

    // Prepare assembly bilinear form
    auto kernel_a = a.kernel(fem::IntegralType::cell, i);

    auto& [coeffs_a, cstride_a]
        = coefficients_a.at({fem::IntegralType::cell, i});

    // Initialize RHS for current integrator
    problem_data.initialize_kernel(fem::IntegralType::cell, i);

    // Loop over all cells
    if (!cells.empty())
    {
      for (std::size_t index = 0; index < cells.size(); ++index)
      {
        // Id of current cell
        std::int32_t c = cells[index];

        // Get cell coordinates
        auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
            x_dofmap, c, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);

        for (std::size_t i = 0; i < x_dofs.size(); ++i)
        {
          std::copy_n(std::next(x.begin(), 3 * x_dofs[i]), 3,
                      std::next(coordinate_dofs.begin(), 3 * i));
        }

        // Loop over all RHS
        for (std::size_t i_rhs = 0; i_rhs < problem_data.nrhs(); ++i_rhs)
        {
          /* Extract data for current RHS */
          // Integration kernel
          const auto& kernel_l = problem_data.kernel(i_rhs);

          // Constants and coefficients
          std::span<const T> constants_l = problem_data.constants(i_rhs);
          std::span<T> coefficients_l = problem_data.coefficients(i_rhs);

          // Infos about coefficients
          int cstride_l = problem_data.cstride(i_rhs);

          /* Solve cell-wise problem */
          if (i_rhs == 0)
          {
            // Evaluate bilinear form
            A_e.setZero();
            kernel_a(A_e.data(), coeffs_a.data() + index * cstride_a,
                     constants_a.data(), coordinate_dofs.data(), nullptr,
                     nullptr);

            // Prepare solver
            solver.compute(A_e);
          }

          // Evaluate linear form
          L_e.setZero();
          kernel_l(L_e.data(), coefficients_l.data() + index * cstride_l,
                   constants_l.data(), coordinate_dofs.data(), nullptr,
                   nullptr);

          // Solve equation system
          u_e = solver.solve(L_e);

          // Global dofs of currect element
          smdspan_t<const std::int32_t, 1> sol_dof
              = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                  dofs, c, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);

          // Map solution into global function space
          std::span<T> sol_i
              = problem_data.solution_function(i_rhs).x()->mutable_array();
          if (bs == 1)
          {
            for (std::size_t k = 0; k < num_dofs; ++k)
            {
              sol_i[sol_dof(k)] = u_e[k];
            }
          }
          else
          {
            for (std::size_t k = 0; k < num_dofs; ++k)
            {
              for (std::size_t m = 0; m < bs; ++m)
              {
                sol_i[bs * sol_dof(k) + m] = u_e[bs * k + m];
              }
            }
          }
        }
      }
    }
  }
}

} // namespace dolfinx_eqlb::base