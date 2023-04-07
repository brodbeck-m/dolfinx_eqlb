#pragma once

#include "eigen3/Eigen/Dense"
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

void test_pybind()
{
  std::cout << "Hello World221\n";
  std::cout << "Ich bin ein kleines Testprogramm\n";
}

void test_eigen()
{
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> A_e;
  Eigen::Matrix<double, Eigen::Dynamic, 1> _A_e, L_e, u_e;
  Eigen::FullPivLU<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
      solver;

  _A_e.resize(9);
  L_e.resize(3);

  std::vector<double> data_vector = {1, 0, 5, 2, 0.5, 0, 0, 7, 2};

  std::fill(_A_e.begin(), _A_e.end(), 0);
  std::fill(L_e.begin(), L_e.end(), 10);

  std::cout << "Solution:\n" << _A_e << std::endl;
  std::cout << "Solution:\n" << L_e << std::endl;

  _A_e(0, 0) = 1;
  _A_e(2, 0) = 5;
  _A_e(3, 0) = 2;
  _A_e(4, 0) = 0.5;
  _A_e(7, 0) = 7;
  _A_e(8, 0) = 2;

  L_e(0) = 2;
  L_e(1) = 0;
  L_e(2) = 1;

  A_e.resize(3, 3);
  A_e = _A_e.reshaped(3, 3);

  // std::fill(A_e.reshaped().begin(), A_e.reshaped().end(), 50);

  std::cout << "Solution:\n" << A_e.transpose() << std::endl;
  std::cout << "Solution:\n" << L_e << std::endl;
  // std::fill(A_e.reshaped().begin(), A_e.reshaped().end(), 0);
  // std::cout << "Solution:\n" << A_e << std::endl;

  solver.compute(A_e.transpose());
  u_e = solver.solve(L_e);
  std::cout << "Solution:\n" << u_e << std::endl;
}

template <typename T>
void local_solver(fem::Function<T>& sol_elmt, const fem::Form<T>& a,
                  const fem::Form<T>& l)
{
  // Prepare cell geometry
  const mesh::Geometry& geometry = a.mesh()->geometry();

  const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();
  const std::size_t num_dofs_g = geometry.cmap().dim();
  std::span<const double> x = geometry.x();
  std::vector<double> coordinate_dofs(3 * num_dofs_g);

  // Prepare constants and coefficients
  const std::vector<T> constants_a = fem::pack_constants(a);
  const std::vector<T> constants_l = fem::pack_constants(l);

  auto coefficients_a_stdvec = fem::allocate_coefficient_storage(a);
  fem::pack_coefficients(a, coefficients_a_stdvec);
  auto coefficients_l_stdvec = fem::allocate_coefficient_storage(l);
  fem::pack_coefficients(l, coefficients_l_stdvec);

  std::map<std::pair<fem::IntegralType, int>,
           std::pair<std::span<const T>, int>>
      coefficients_a = fem::make_coefficients_span(coefficients_a_stdvec);
  std::map<std::pair<fem::IntegralType, int>,
           std::pair<std::span<const T>, int>>
      coefficients_l = fem::make_coefficients_span(coefficients_l_stdvec);

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

  Eigen::LLT<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> solver;

  A_e.resize(ndim0, ndim0);
  L_e.resize(ndim0);
  u_e.resize(ndim0);

  // Pointer onto global solution sol_elmt
  std::span<T> vec_sol_elmt = sol_elmt.x()->mutable_array();

  // Solve equation system on all cells
  for (int i : a.integral_ids(fem::IntegralType::cell))
  {
    // Extract kernels
    const auto& kernel_a = a.kernel(fem::IntegralType::cell, i);
    const auto& kernel_l = l.kernel(fem::IntegralType::cell, i);

    // Extract cells
    const std::vector<std::int32_t>& cells = a.cell_domains(i);

    // Extract coefficients
    const auto& [coeffs_a, cstride_a]
        = coefficients_a.at({fem::IntegralType::cell, i});
    const auto& [coeffs_l, cstride_l]
        = coefficients_l.at({fem::IntegralType::cell, i});

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

        // Initialize element array with zeros
        A_e.setZero();
        L_e.setZero();
        u_e.setZero();

        // Tabluate system
        kernel_a(A_e.data(), coeffs_a.data() + index * cstride_a,
                 constants_a.data(), coordinate_dofs.data(), nullptr, nullptr);
        kernel_l(L_e.data(), coeffs_l.data() + index * cstride_l,
                 constants_l.data(), coordinate_dofs.data(), nullptr, nullptr);

        // Solve equation system
        solver.compute(A_e);
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

} // namespace dolfinx_adaptivity