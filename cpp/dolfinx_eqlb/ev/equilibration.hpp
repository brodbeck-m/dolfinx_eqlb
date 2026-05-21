// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "subdofmap.hpp"
#include <dolfinx_eqlb/base/ProblemData.hpp>
#include <dolfinx_eqlb/base/equilibration.hpp>
#include <dolfinx_eqlb/base/mdspan.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <dolfinx/common/types.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Mesh.h>

#include <algorithm>
#include <functional>
#include <iostream>
#include <iterator>
#include <span>
#include <vector>

using namespace dolfinx;

namespace dolfinx_eqlb::ev
{

/// Solve the equilibration problem
///
/// @param[in,out] fluxes The equilibrated fluxes
/// @param[in] a          The bilinear forms
///                       (mass-matrix, constraints, Lagrange multiplier)
/// @param[in] ls         The linear forms
template <dolfinx::scalar T, std::floating_point U>
void equilibrate(std::vector<std::shared_ptr<fem::Function<T, U>>>& fluxes,
                 const std::vector<std::shared_ptr<const fem::Form<T, U>>>& as,
                 const std::vector<std::shared_ptr<const fem::Form<T, U>>>& ls)
{
  if (as.size() != 3)
  {
    throw std::runtime_error(
        "Equilibration: System matrix requires form of mass matrix, constraint "
        "contribution and Lagrange multiplyer.");
  }

  if (ls.size() != fluxes.size())
  {
    throw std::runtime_error("Equilibration: Input sizes does not match.");
  }

  if ((as[0]->mesh()->topology()->cell_types().size()) > 1)
  {
    throw std::runtime_error("Equilibration: Meshes .");
  }

  /* Initialise data */
  // The number of equilibrated fluxes
  const int n_rhs = fluxes.size();

  // The mesh
  std::shared_ptr<const mesh::Mesh<U>> msh = as[0]->mesh();

  const int gdim = msh->geometry().dim();
  const int fdim = gdim - 1;

  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> node_to_cell
      = msh->topology()->connectivity(0, gdim);
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> node_to_fct
      = msh->topology()->connectivity(0, fdim);
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> fct_to_node
      = msh->topology()->connectivity(fdim, 0);
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> fct_to_cell
      = msh->topology()->connectivity(fdim, gdim);
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> cell_to_fct
      = msh->topology()->connectivity(gdim, fdim);
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> cell_to_node
      = msh->topology()->connectivity(gdim, 0);

  std::span<const dolfinx::scalar_value_t<T>> x = msh->geometry().x();
  base::mdspan_t<const std::int32_t, 2> x_dofmap = msh->geometry().dofmap();
  std::vector<dolfinx::scalar_value_t<T>> coordinate_dofs(3
                                                          * x_dofmap.extent(1));

  //  Initialise the patch
  const std::int32_t max_size_patch = max_patch_size(
      msh->topology()->index_map(0)->size_local(), node_to_cell);

  const std::int32_t max_cells_per_patch
      = node_to_cell->links(max_size_patch).size();
  const std::int32_t max_fcts_per_patch
      = node_to_fct->links(max_size_patch).size();

  // The (global) DofMap of the flux space
  std::shared_ptr<const fem::FunctionSpace<U>> fspace_v
      = as[0]->function_spaces().at(0);
  std::shared_ptr<const fem::DofMap> dofmap_v = fspace_v->dofmap();

  const int fluxdofs_per_cell = fspace_v->element()->space_dimension();
  const std::vector<int> fluxdofs_per_entity = ndofs_per_entity(fspace_v);

  // The (global) DofMap of the constrained space
  const int id_q = (fspace_v == as[1]->function_spaces().at(0)) ? 1 : 0;
  std::shared_ptr<const fem::FunctionSpace<U>> fspace_q
      = as[1]->function_spaces().at(id_q);
  std::shared_ptr<const fem::DofMap> dofmap_q = fspace_q->dofmap();

  // The linear solvers
  ndofs_flux_max = max_fcts_per_patch * fluxdofs_per_entity[fdim]
                   + max_cells_per_patch * fluxdofs_per_entity[gdim];
  ndofs_cnstrs_max
      = max_cells_per_patch * fspace_q->element()->space_dimension() + 1;

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A, B;
  Eigen::Matrix<T, Eigen::Dynamic, 1> Lu, Lc, u, c;

  A.resize(ndofs_flux_max, ndofs_flux_max);
  B.resize(ndofs_flux_max, ndofs_cnstrs_max);
  Lu.resize(ndofs_flux_max);
  Lc.resize(ndofs_cnstrs_max);
  u.resize(ndofs_flux_max);
  c.resize(ndofs_cnstrs_max);

  /* Solve patch-wise equation systems */
  // Data for mass matrix
  const std::vector<T> constants_a = fem::pack_constants(*as[0]);

  auto interm_coefficients_a = fem::allocate_coefficient_storage(*as[0]);
  fem::pack_coefficients(*as[0], interm_coefficients_a);

  std::map<std::pair<fem::IntegralType, int>,
           std::pair<std::span<const T>, int>>
      coefficients_a = fem::make_coefficients_span(interm_coefficients_a);

  // Data for the linear forms
  base::ProblemData<T, U> problem_data = base::ProblemData<T, U>(fluxes, ls);

  // Loop over all cell types and cell domains
  // const int num_cell_types
  //     = static_cast<int>(a.mesh()->topology()->cell_types().size());
  // for (int cell_type_idx = 0; cell_type_idx < num_cell_types;
  // ++cell_type_idx)
  // {
  //   for (int i = 0; i < a.num_integrals(fem::IntegralType::cell,
  //   cell_type_idx);
  //        ++i)
  //   {
  //     // Extract cells
  //     const std::span<const std::int32_t> cells
  //         = a.domain(fem::IntegralType::cell, i, cell_type_idx);
  //   }
  // }

  std::cout << "n_rhs: " << n_rhs << std::endl;
  std::cout << "max_cells_per_patch: " << max_cells_per_patch << std::endl;
  std::cout << "max_facets_per_patch: " << max_fcts_per_patch << std::endl;

  std::cout << "DOFs per entity: ";
  for (auto v : fluxdofs_per_entity)
  {
    std::cout << v << " ";
  }
  std::cout << std::endl;
}

} // namespace dolfinx_eqlb::ev