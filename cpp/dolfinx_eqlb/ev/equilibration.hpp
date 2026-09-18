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
    throw std::runtime_error("Equilibration: System matrix requires form of mass matrix, constraint "
                             "contribution and Lagrange multiplier.");
  }

  if (ls.size() != fluxes.size())
  {
    throw std::runtime_error("Equilibration: Input sizes does not match.");
  }

  if ((as[0]->mesh()->topology()->cell_types().size()) > 1)
  {
    throw std::runtime_error("Equilibration: Mixed meshes not supported.");
  }

  /* Initialise data */
  // The number of equilibrated fluxes
  const int n_rhs = fluxes.size();

  // The mesh
  std::shared_ptr<const mesh::Mesh<U>> msh = as[0]->mesh();
  const int nvert = msh->topology()->index_map(0)->size_local();

  const int gdim = msh->geometry().dim();
  const int fdim = gdim - 1;

  // Counter for final normalization
  std::vector<std::vector<int>> occurance_count(fluxes.size(), std::vector<int>(nvert));

  // connectivity of the mesh
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> node_to_cell = msh->topology()->connectivity(0, gdim);
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> node_to_fct = msh->topology()->connectivity(0, fdim);
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> fct_to_node = msh->topology()->connectivity(fdim, 0);
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> fct_to_cell = msh->topology()->connectivity(fdim, gdim);
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> cell_to_fct = msh->topology()->connectivity(gdim, fdim);
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> cell_to_node = msh->topology()->connectivity(gdim, 0);

  std::span<const dolfinx::scalar_value_t<T>> x = msh->geometry().x();
  base::mdspan_t<const std::int32_t, 2> x_dofmap = msh->geometry().dofmap();
  std::vector<dolfinx::scalar_value_t<T>> coordinate_dofs(
      x_dofmap.extent(0) * x_dofmap.extent(1)); // TODO Why different from local solver?

  //  Initialise the patch
  const std::int32_t max_size_patch_id = max_patch_size(msh->topology()->index_map(0)->size_local(), node_to_cell);

  const std::int32_t max_cells_per_patch = node_to_cell->links(max_size_patch_id).size();
  const std::int32_t max_fcts_per_patch = node_to_fct->links(max_size_patch_id).size();
  // The (global) DofMap of the flux space
  std::shared_ptr<const fem::FunctionSpace<U>> fspace_v = as[0]->function_spaces().at(0);

  // Different DOFMAP call and return value from above!!!
  std::shared_ptr<const fem::DofMap> dofmap_v = fspace_v->dofmap();
  const int fluxdofs_per_cell = fspace_v->element()->space_dimension();
  const std::vector<int> fluxdofs_per_entity = ndofs_per_entity(fspace_v);

  // The (local) DofMap of the flux space
  const std::int32_t ndofs_flux_max
      = (max_fcts_per_patch * fluxdofs_per_entity[fdim]) + (max_cells_per_patch * fluxdofs_per_entity[gdim]);
  std::vector<std::int32_t> subdofmap_flux;

  // The (global) DofMap of the constrained space
  const int id_q = (fspace_v == as[1]->function_spaces().at(0)) ? 1 : 0;
  std::shared_ptr<const fem::FunctionSpace<U>> fspace_q = as[1]->function_spaces().at(id_q);
  std::shared_ptr<const fem::DofMap> dofmap_q = fspace_q->dofmap();
  const int constrdofs_per_cell = fspace_q->element()->space_dimension();

  // The (local) DofMap of the constrained space
  const std::int32_t ndofs_cnstrs_max = max_cells_per_patch * fspace_q->element()->space_dimension() + 1;

  // The linear solvers
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

  std::map<std::pair<fem::IntegralType, int>, std::pair<std::span<const T>, int>> coefficients_a
      = fem::make_coefficients_span(interm_coefficients_a);

  // Data for the linear forms
  base::ProblemData<T, U> problem_data = base::ProblemData<T, U>(fluxes, ls);

  // Reusable solver
  Eigen::PartialPivLU<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> solver;

  // Initialize geometry storage/ hat-function
  // const int cstride_geom = 3 * geometry.cmap().dim();

  // Initialize the global fluxes to zero
  for (int rhs_idx = 0; rhs_idx < n_rhs; ++rhs_idx)
  {
    auto& global_flux_func = *fluxes[rhs_idx];
    std::span<T> global_flux_data = global_flux_func.x()->mutable_array();
    std::fill(global_flux_data.begin(), global_flux_data.end(), 0.0);
  }

  // Loop over all vertices -> patches
  for (std::int32_t v = 0; v < nvert; ++v)
  {
    // Get the cells associated with the current vertex
    std::span<const std::int32_t> patch_cells = node_to_cell->links(v);
    if (patch_cells.empty())
    {
      std::cout << "Warning: Vertex " << v << " has no associated cells." << std::endl;
      continue;
    }

    const int ncells = patch_cells.size();

    // Get global patch dofs
    std::vector<int> raw_patch_flux_dofs;
    std::vector<int> raw_patch_constr_dofs;

    for (const std::int32_t cell : patch_cells)
    {
      auto cell_flux = dofmap_v->cell_dofs(cell);
      raw_patch_flux_dofs.insert(raw_patch_flux_dofs.end(), cell_flux.begin(), cell_flux.end());

      auto cell_constr = dofmap_q->cell_dofs(cell);
      raw_patch_constr_dofs.insert(raw_patch_constr_dofs.end(), cell_constr.begin(), cell_constr.end());
    }

    // remap the patch dofs to a compacted local numbering
    std::vector<int> flux_remap = compact_dof_map(raw_patch_flux_dofs); // TODO raw_patch_flux_dofs has duplicate values
    std::vector<int> constr_remap = compact_dof_map(raw_patch_constr_dofs);

    // Get local patch dofs
    std::vector<std::int32_t> local_patch_flux_dofs;
    for (int idx : flux_remap)
    {
      std::int32_t val = raw_patch_flux_dofs[idx];
      if (local_patch_flux_dofs.empty() || local_patch_flux_dofs.back() != val)
        local_patch_flux_dofs.push_back(val);
    }

    std::vector<std::int32_t> local_patch_constr_dofs;
    for (int idx : constr_remap)
    {
      std::int32_t val = raw_patch_constr_dofs[idx];
      if (local_patch_constr_dofs.empty() || local_patch_constr_dofs.back() != val)
        local_patch_constr_dofs.push_back(val);
    }

    // Dimensions of the problem
    const std::size_t n_flux = local_patch_flux_dofs.size();
    const std::size_t n_constr = local_patch_constr_dofs.size() + 1; // +1 for lagrange multiplier

    // Loop over all RHS and solve the patch-wise system
    for (int rhs_idx = 0; rhs_idx < n_rhs; ++rhs_idx)
    {
      // Zero the matrices and vectors for the current patch
      A.setZero();
      B.setZero();
      Lu.setZero();
      Lc.setZero();

      const std::vector<T> constants_l = fem::pack_constants(*ls[rhs_idx]);
      auto interm_coefficients_l = fem::allocate_coefficient_storage(*ls[rhs_idx]);
      fem::pack_coefficients(*ls[rhs_idx], interm_coefficients_l);
      auto coeffs_l_map = fem::make_coefficients_span(interm_coefficients_l);
      auto coeffs_l_span = coeffs_l_map.at({fem::IntegralType::cell, 0}).first;

      // Loop over the cells of the patch for cell-wise assembly
      for (const std::int32_t cell : patch_cells)
      {
        auto cell_flux_dofs = dofmap_v->cell_dofs(cell);
        auto cell_constr_dofs = dofmap_q->cell_dofs(cell);

        for (std::size_t i = 0; i < x_dofmap.extent(1); ++i)
        {
          const auto node_idx = x_dofmap(cell, i);

          coordinate_dofs[3 * i + 0] = x[3 * node_idx + 0];
          coordinate_dofs[3 * i + 1] = x[3 * node_idx + 1];
          coordinate_dofs[3 * i + 2] = x[3 * node_idx + 2];
        }

        const int n_cell_flux = cell_flux_dofs.size();
        const int n_cell_constr = cell_constr_dofs.size();

        std::vector<T> A_cell(n_cell_flux * n_cell_flux, 0.0);
        std::vector<T> B_cell(n_cell_flux * n_cell_constr, 0.0);
        std::vector<T> Lu_cell(n_cell_flux, 0.0);
        std::vector<T> Lc_cell(n_cell_constr, 0.0);

        auto kernel_A = as[0]->kernel(fem::IntegralType::cell, 0, 0);
        auto kernel_B = as[1]->kernel(fem::IntegralType::cell, 0, 0);
        auto kernel_Lu = ls[rhs_idx]->kernel(fem::IntegralType::cell, 0, 0);

        auto coeffs_a_span = coefficients_a.at({fem::IntegralType::cell, 0}).first;

        kernel_A(A_cell.data(), coeffs_a_span.data(), constants_a.data(), coordinate_dofs.data(), nullptr, nullptr,
                 nullptr);
        kernel_B(B_cell.data(), coeffs_a_span.data(), constants_a.data(), coordinate_dofs.data(), nullptr, nullptr,
                 nullptr);

        kernel_Lu(Lu_cell.data(), coeffs_l_span.data(), constants_l.data(), coordinate_dofs.data(), nullptr, nullptr,
                  nullptr);

        std::vector<int> local_flux_indices(n_cell_flux);
        for (int i = 0; i < n_cell_flux; ++i)
        {
          auto it = std::lower_bound(local_patch_flux_dofs.begin(), local_patch_flux_dofs.end(), cell_flux_dofs[i]);
          local_flux_indices[i] = std::distance(local_patch_flux_dofs.begin(), it);
        }

        std::vector<int> local_constr_indices(n_cell_constr);
        for (int i = 0; i < n_cell_constr; ++i)
        {
          auto it
              = std::lower_bound(local_patch_constr_dofs.begin(), local_patch_constr_dofs.end(), cell_constr_dofs[i]);
          local_constr_indices[i] = std::distance(local_patch_constr_dofs.begin(), it);
        }

        for (int i = 0; i < n_cell_flux; ++i)
        {
          int patch_row = local_flux_indices[i];
          Lu(patch_row) += Lu_cell[i];

          for (int j = 0; j < n_cell_flux; ++j)
          {
            int patch_col = local_flux_indices[j];
            A(patch_row, patch_col) += A_cell[i * n_cell_flux + j];
          }
        }

        for (int i = 0; i < n_cell_flux; ++i)
        {
          int patch_row = local_flux_indices[i];
          for (int j = 0; j < n_cell_constr; ++j)
          {
            int patch_col = local_constr_indices[j];
            B(patch_row, patch_col) += B_cell[i * n_cell_constr + j];
          }
        }

        for (int i = 0; i < n_cell_constr; ++i)
        {
          int patch_row = local_constr_indices[i];
          Lc(patch_row) += Lc_cell[i];
        }
      }

      const std::size_t n_total = n_flux + n_constr;

      // Formulate the KKT system
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> KKT
          = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(n_total, n_total);
      Eigen::Matrix<T, Eigen::Dynamic, 1> RHS = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(n_total);

      KKT.block(0, 0, n_flux, n_flux) = A.topLeftCorner(n_flux, n_flux);
      KKT.block(0, n_flux, n_flux, n_constr) = B.topLeftCorner(n_flux, n_constr);
      KKT.block(n_flux, 0, n_constr, n_flux) = B.topLeftCorner(n_flux, n_constr).transpose();

      RHS.segment(0, n_flux) = Lu.head(n_flux);
      RHS.segment(n_flux, n_constr) = Lc.head(n_constr);

      // Solve using existing solver
      solver.compute(KKT);
      Eigen::Matrix<T, Eigen::Dynamic, 1> u_patch = solver.solve(RHS);

      // Assembly
      // get mut ref
      auto fluxes_array = fluxes[rhs_idx]->x()->mutable_array();

      // TODO: what did I do here?
      // Loop over all verts
      for (int i = 0; i < n_flux; ++i)
      {
        // Get global dof id and add solution
        const std::int32_t global_dof = local_patch_flux_dofs[i];
        fluxes_array[global_dof] += u_patch(i);

        // Record occurance for normalization
        occurance_count[rhs_idx][global_dof] += 1;
      }

      // Normalize the fluxes by the number of times each DOF was updated
      for (std::size_t global_dof = 0; global_dof < occurance_count[rhs_idx].size(); ++global_dof)
      {
        const int count = occurance_count[rhs_idx][global_dof];
        if (count > 0)
        {
          fluxes_array[global_dof] /= static_cast<T>(count);
        }
      }
    }
  }

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