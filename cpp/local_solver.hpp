#pragma once

#include "eigen3/Eigen/Dense"
#include <algorithm>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <functional>
#include <iostream>
#include <iterator>
#include <span>
#include <vector>

namespace dolfinx_eqlb
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
void local_projector(fem::Function<T>& sol_elmt, const fem::Form<T>& a,
                     const fem::Form<T>& l)
{
  // Prepare cell geometry
  const mesh::Geometry& geometry = a.mesh()->geometry();

  const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();
  const std::size_t num_dofs_g = geometry.cmap().dim();
  std::span<const double> x = geometry.x();
  std::vector<double> coordinate_dofs(3 * num_dofs_g);

  // Extract dofmap from functionspace
  std::shared_ptr<const fem::DofMap> dofmap0
      = a.function_spaces().at(0)->dofmap();
  assert(dofmap0);

  const graph::AdjacencyList<std::int32_t>& dofs0 = dofmap0->list();
  const int bs0 = dofmap0->bs();

  // Extract DOF transformations
  // const fem::FiniteElement& element0 = a.function_spaces().at(0)->element();
  // const std::function<void(const std::span<T>&,
  //                          const std::span<const std::uint32_t>&,
  //                          std::int32_t, int)>& dof_transform
  //     = element0.get_dof_transformation_function<T>();
  // const std::function<void(const std::span<T>&,
  //                          const std::span<const std::uint32_t>&,
  //                          std::int32_t, int)>& dof_transform_to_transpose
  //     = element0.get_dof_transformation_to_transpose_function<T>();

  // const bool needs_transformation_data
  //     = element0.needs_dof_transformations() or a.needs_facet_permutations();
  // std::span<const std::uint32_t> cell_info;

  // if (needs_transformation_data)
  // {
  //   // Get cell_date
  //   a.mesh().topology_mutable().create_entity_permutations();

  //   // Extract transformation functions
  //   cell_info = std::span(a.mesh().topology().get_cell_permutation_info());
  // }

  // Initialize element arrays
  const int num_dofs0 = dofs0.links(0).size();
  const int ndim0 = bs0 * num_dofs0;

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A_e;
  Eigen::Matrix<double, Eigen::Dynamic, 1> L_e, u_e;
  Eigen::FullPivLU<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      solver;

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
        std::fill(A_e.reshaped().begin(), A_e.reshaped().end(), 0);
        std::fill(L_e.begin(), L_e.end(), 0);
        std::fill(u_e.begin(), u_e.end(), 5);

        // Tabluate system
        kernel_a(A_e.data(), nullptr, nullptr, coordinate_dofs.data(), nullptr,
                 nullptr);
        kernel_l(L_e.data(), nullptr, nullptr, coordinate_dofs.data(), nullptr,
                 nullptr);

        // Apply dof permutations

        // Solve equation system
        solver.compute(A_e);
        u_e = solver.solve(L_e);

        // Global dofs of currect element
        std::span<const int32_t> sol_dof = dofs0.links(c);

        // Map solution into global function space
        for (std::size_t k = 0; k < ndim0; ++k)
        {
          vec_sol_elmt[sol_dof[k]] = u_e[k];
        }
      }
    }
  }
}

} // namespace dolfinx_eqlb