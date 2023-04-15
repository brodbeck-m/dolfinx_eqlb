#pragma once

#include "ProblemData.hpp"
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Sparse"
#include <algorithm>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <functional>
#include <iostream>
#include <iterator>
#include <span>
#include <vector>

using namespace dolfinx;

namespace dolfinx_adaptivity
{
/// Execute solution of problem in element-wise manner
///
/// @param vec_sol List of solution vectors
/// @param a       Bilinear form of all problems
/// @param vec_l   List of multiple linear forms
/// @param solver  Solver for linear equation system
template <typename T, typename S>
void local_solver(std::vector<std::shared_ptr<fem::Function<T>>>& vec_sol,
                  const fem::Form<T>& a,
                  const std::vector<std::shared_ptr<const fem::Form<T>>>& vec_l,
                  S& solver)
{
  /* Handle multiple LHS */
  // Check input data
  if (vec_l.size() != vec_sol.size())
  {
    throw std::runtime_error("Local solver: Input sizes does not match");
  }

  // Initilize data of LHS
  const std::vector<std::vector<std::shared_ptr<const fem::DirichletBC<T>>>>
      bcs;
  equilibration::ProblemData<T> problem_data
      = equilibration::ProblemData<T>(vec_sol);

  problem_data.set_rhs(vec_l, bcs);

  // Prepare cell geometry
  const mesh::Geometry& geometry = a.mesh()->geometry();

  const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();
  const std::size_t num_dofs_g = geometry.cmap().dim();
  std::span<const double> x = geometry.x();
  std::vector<double> coordinate_dofs(3 * num_dofs_g);

  // Prepare constants and coefficients
  const std::vector<T> constants_a = fem::pack_constants(a);

  auto coefficients_a_stdvec = fem::allocate_coefficient_storage(a);
  fem::pack_coefficients(a, coefficients_a_stdvec);

  std::map<std::pair<fem::IntegralType, int>,
           std::pair<std::span<const T>, int>>
      coefficients_a = fem::make_coefficients_span(coefficients_a_stdvec);

  // Extract dofmap from functionspace
  std::shared_ptr<const fem::DofMap> dofmap0
      = a.function_spaces().at(0)->dofmap();
  assert(dofmap0);

  const graph::AdjacencyList<std::int32_t>& dofs0 = dofmap0->list();
  const int bs0 = dofmap0->bs();

  // Initialize element arrays
  const int num_dofs0 = dofs0.links(0).size();
  const int ndim0 = bs0 * num_dofs0;

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A_e;
  Eigen::Matrix<T, Eigen::Dynamic, 1> L_e, u_e;

  A_e.resize(ndim0, ndim0);
  L_e.resize(ndim0);
  u_e.resize(ndim0);

  /* Solve element-wise equation systems */
  // Loop over all cell domains
  for (int i : a.integral_ids(fem::IntegralType::cell))
  {
    // Extract cells
    const std::vector<std::int32_t>& cells = a.cell_domains(i);

    // Prepare assembly bilinear form
    const auto& kernel_a = a.kernel(fem::IntegralType::cell, i);

    const auto& [coeffs_a, cstride_a]
        = coefficients_a.at({fem::IntegralType::cell, i});

    // Initialize LHS for current integrator
    problem_data.initialize_kernels(fem::IntegralType::cell, i);

    // Loop over all cells
    if (!cells.empty())
    {
      for (std::size_t index = 0; index < cells.size(); ++index)
      {
        // Id of current cell
        std::int32_t c = cells[index];

        // Get cell coordinates
        auto x_dofs = x_dofmap.links(c);
        for (std::size_t j = 0; j < x_dofs.size(); ++j)
        {
          std::copy_n(std::next(x.begin(), 3 * x_dofs[j]), 3,
                      std::next(coordinate_dofs.begin(), 3 * j));
        }

        // Loop over all LHS
        for (std::size_t i_lhs = 0; i_lhs < problem_data.nlhs(); ++i_lhs)
        {
          /* Extract data for current LHS */
          // Integration kernel
          const auto& kernel_l = problem_data.kernel(i_lhs);

          // Constants and coefficients
          std::span<const T> constants_l = problem_data.constants(i_lhs);
          std::span<T> coefficients_l = problem_data.coefficients(i_lhs);

          // Infos about coefficients
          int cstride_l = problem_data.cstride(i_lhs);

          // Solution vector
          std::span<T> vec_sol_elmt
              = problem_data.solution_function(i_lhs).x()->mutable_array();

          /* Solve cell-wise problem */
          if (i_lhs == 0)
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
          std::span<const int32_t> sol_dof = dofs0.links(c);

          // Map solution into global function space
          if (bs0 == 1)
          {
            for (std::size_t k = 0; k < num_dofs0; ++k)
            {
              vec_sol_elmt[sol_dof[k]] = u_e[k];
            }
          }
          else
          {
            for (std::size_t k = 0; k < num_dofs0; ++k)
            {
              for (std::size_t cb = 0; cb < bs0; ++cb)
              {
                vec_sol_elmt[bs0 * sol_dof[k] + cb] = u_e[bs0 * k + cb];
              }
            }
          }
        }
      }
    }
  }
}

/// Local solver (using LU decomposition)
///
/// @param vec_sol List of solution vectors
/// @param a       Bilinear form of all problems
/// @param vec_l   List of multiple linear forms
template <typename T>
void local_solver_lu(
    std::vector<std::shared_ptr<fem::Function<T>>>& vec_sol,
    const fem::Form<T>& a,
    const std::vector<std::shared_ptr<const fem::Form<T>>>& vec_l)
{
  // Initialize solver
  Eigen::PartialPivLU<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
      lu_solver;

  // Solve problems element-wise
  local_solver(vec_sol, a, vec_l, lu_solver);
}

/// Local solver (using Cholesky decomposition)
///
/// @param vec_sol List of solution vectors
/// @param a       Bilinear form of all problems
/// @param vec_l   List of multiple linear forms
template <typename T>
void local_solver_cholesky(
    std::vector<std::shared_ptr<fem::Function<T>>>& vec_sol,
    const fem::Form<T>& a,
    const std::vector<std::shared_ptr<const fem::Form<T>>>& vec_l)
{
  // Initialize solver
  Eigen::LLT<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> llt_solver;

  // Solve problems element-wise
  local_solver(vec_sol, a, vec_l, llt_solver);
}

/// Local solver (using the CG solver)
///
/// @param vec_sol List of solution vectors
/// @param a       Bilinear form of all problems
/// @param vec_l   List of multiple linear forms
template <typename T>
void local_solver_cg(
    std::vector<std::shared_ptr<fem::Function<T>>>& vec_sol,
    const fem::Form<T>& a,
    const std::vector<std::shared_ptr<const fem::Form<T>>>& vec_l)
{
  // Initialize solver
  Eigen::ConjugateGradient<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>,
                           Eigen::Lower | Eigen::Upper>
      cg_solver;

  // Solve problems element-wise
  local_solver(vec_sol, a, vec_l, cg_solver);
}
} // namespace dolfinx_adaptivity