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

  if (ls.size() != 2 * fluxes.size())
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
  std::vector<dolfinx::scalar_value_t<T>> coordinate_dofs(3 * x_dofmap.extent(1)); // 3 for 3D coordinates of fenics

  //  Initialise the patch
  const std::int32_t max_size_patch_id = max_patch_size(msh->topology()->index_map(0)->size_local(), node_to_cell);

  const std::int32_t max_cells_per_patch = node_to_cell->links(max_size_patch_id).size();
  const std::int32_t max_fcts_per_patch = node_to_fct->links(max_size_patch_id).size();

  // The mixed FunctionSpace W
  std::shared_ptr<const fem::FunctionSpace<U>> fspace_w = as[0]->function_spaces()[0];

  // Extract subspaces: 0 -> flux space (RT), 1 -> constrained space (DG)
  auto fspace_v = std::make_shared<const fem::FunctionSpace<U>>(fspace_w->sub({0}));
  auto fspace_q = std::make_shared<const fem::FunctionSpace<U>>(fspace_w->sub({1}));

  // Sub-dofmaps
  std::shared_ptr<const fem::DofMap> dofmap_v = fspace_v->dofmap();
  const int fluxdofs_per_cell = fspace_v->element()->space_dimension();
  const std::vector<int> fluxdofs_per_entity = ndofs_per_entity(fspace_v);

  // The (local) DofMap of the flux space
  const std::int32_t ndofs_flux_max
      = (max_fcts_per_patch * fluxdofs_per_entity[fdim]) + (max_cells_per_patch * fluxdofs_per_entity[gdim]);
  std::vector<std::int32_t> subdofmap_flux;

  // The (global) DofMap of the constrained space
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
  base::ProblemData<T, U> problem_data(fluxes, ls);
  problem_data.initialize_kernel(fem::IntegralType::cell, 0);

  // Reusable solver
  Eigen::PartialPivLU<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> solver;

  // Initialize the global mixed functions to zero
  for (int rhs_idx = 0; rhs_idx < n_rhs; ++rhs_idx)
  {
    auto& global_flux_func = *fluxes[rhs_idx];
    std::span<T> global_flux_data = global_flux_func.x()->mutable_array();
    std::fill(global_flux_data.begin(), global_flux_data.end(), 0.0);
  }

  // Pre-fetch local entity DOF layouts for RT
  const auto& rt_entity_dofs = fspace_v->element()->entity_dofs();

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

    // set hat function
    std::span<const std::int32_t> nodeIndices = cell_to_node->links(patch_cells[0]);
    int localPatch0NodeId = -1;
    for (int i = 0; i < nodeIndices.size(); i++)
    {
      if (nodeIndices[i] == v)
      {
        localPatch0NodeId = i;
        break;
      }
    }

    // Activate hat function at vertex v across every cell in the patch
    std::vector<std::int8_t> patch_local_nodes(patch_cells.size(), -1);
    for (std::size_t c = 0; c < patch_cells.size(); ++c)
    {
      auto nodes = cell_to_node->links(patch_cells[c]);
      for (std::size_t i = 0; i < nodes.size(); ++i)
      {
        if (nodes[i] == v)
        {
          patch_local_nodes[c] = static_cast<std::int8_t>(i);
          problem_data.set_hat_function(patch_cells[c], patch_local_nodes[c], static_cast<T>(1.0));
          break;
        }
      }
    }

    const int ncells = patch_cells.size();

    // Get global patch dofs
    std::vector<int> raw_patch_flux_dofs;
    std::vector<int> raw_patch_constr_dofs;

    // https://docs.fenicsproject.org/dolfinx/main/cpp/doxygen/df/d27/classdolfinx_1_1fem_1_1FiniteElement.html#ab65e0ed248053420d4fe672743526e29
    // https://docs.fenicsproject.org/basix/v0.10.0.post0/cpp/classbasix_1_1FiniteElement.html#a12145f082817587bd17ede85cb251706
    for (const std::int32_t cell : patch_cells)
    {
      auto cell_flux = dofmap_v->cell_dofs(cell);
      auto cell_fcts = cell_to_fct->links(cell);

      // Add RT DOFs on facets touching the current vertex
      for (std::size_t local_f = 0; local_f < cell_fcts.size(); ++local_f)
      {
        const std::int32_t fct = cell_fcts[local_f];
        auto fct_nodes = fct_to_node->links(fct);

        // Check if the facet touches the current vertex
        bool touches_v = (std::find(fct_nodes.begin(), fct_nodes.end(), v) != fct_nodes.end());
        if (touches_v)
        {
          for (int local_dof : rt_entity_dofs[fdim][local_f])
            raw_patch_flux_dofs.push_back(cell_flux[local_dof]);
        }
      }

      // Add cell-interior RT DOFs (for RT degree >= 2)
      for (int local_dof : rt_entity_dofs[gdim][0])
        raw_patch_flux_dofs.push_back(cell_flux[local_dof]);

      // Add all DG constraint DOFs on the patch cells
      auto cell_constr = dofmap_q->cell_dofs(cell);
      raw_patch_constr_dofs.insert(raw_patch_constr_dofs.end(), cell_constr.begin(), cell_constr.end());
    }

    // Remap the patch DOFs to a compacted local numbering
    std::vector<int> flux_remap = compact_dof_map(raw_patch_flux_dofs);
    std::vector<int> constr_remap = compact_dof_map(raw_patch_constr_dofs);

    const std::size_t n_flux
        = flux_remap.empty() ? 0
                             : static_cast<std::size_t>(*std::max_element(flux_remap.begin(), flux_remap.end()) + 1);
    const std::size_t n_constr_dofs
        = constr_remap.empty()
              ? 0
              : static_cast<std::size_t>(*std::max_element(constr_remap.begin(), constr_remap.end()) + 1);

    // Map each compact local index back to its global DOF in Function(W)
    std::vector<std::int32_t> local_patch_flux_dofs(n_flux, -1);
    for (std::size_t i = 0; i < raw_patch_flux_dofs.size(); ++i)
      local_patch_flux_dofs[flux_remap[i]] = raw_patch_flux_dofs[i];

    std::vector<std::int32_t> local_patch_constr_dofs(n_constr_dofs, -1);
    for (std::size_t i = 0; i < raw_patch_constr_dofs.size(); ++i)
      local_patch_constr_dofs[constr_remap[i]] = raw_patch_constr_dofs[i];

    // Build fast global-to-local patch lookup maps (since compact_dof_map is not sorted)
    std::unordered_map<std::int32_t, int> global_to_local_flux;
    global_to_local_flux.reserve(n_flux);
    for (std::size_t i = 0; i < n_flux; ++i)
      global_to_local_flux[local_patch_flux_dofs[i]] = static_cast<int>(i);

    std::unordered_map<std::int32_t, int> global_to_local_constr;
    global_to_local_constr.reserve(n_constr_dofs);
    for (std::size_t i = 0; i < n_constr_dofs; ++i)
      global_to_local_constr[local_patch_constr_dofs[i]] = static_cast<int>(i);

    // Pre-fetch DOF transformation functions and cell permutation info for mixed space W
    const bool needs_transform = fspace_w->element()->needs_dof_transformations();
    auto apply_dof_trans_left = fspace_w->element()->template dof_transformation_fn<T>(fem::doftransform::transpose);
    auto apply_dof_trans_right
        = fspace_w->element()->template dof_transformation_right_fn<T>(fem::doftransform::transpose);
    std::span<const std::uint32_t> cell_info;
    if (needs_transform)
    {
      msh->topology_mutable()->create_entity_permutations();
      cell_info = msh->topology()->get_cell_permutation_info();
    }

    // Pack Lagrange multiplier form as[2] = q * dx (constant across RHS)
    const std::vector<T> constants_lagr = fem::pack_constants(*as[2]);
    auto interm_coeffs_lagr = fem::allocate_coefficient_storage(*as[2]);
    fem::pack_coefficients(*as[2], interm_coeffs_lagr);
    auto coeffs_lagr_map = fem::make_coefficients_span(interm_coeffs_lagr);
    const T* coeffs_lagr_ptr = coeffs_lagr_map.empty() ? nullptr : coeffs_lagr_map.begin()->second.first.data();

    // Dimensions of the saddle-point system
    const std::size_t n_constr = n_constr_dofs + 1; // +1 for scalar Lagrange multiplier

    // Loop over all RHS and solve the patch-wise system
    for (int rhs_idx = 0; rhs_idx < n_rhs; ++rhs_idx)
    {
      // Zero the matrices and vectors for the current patch
      A.topLeftCorner(n_flux, n_flux).setZero();
      B.topLeftCorner(n_flux, n_constr).setZero();
      Lu.head(n_flux).setZero();
      Lc.head(n_constr).setZero();

      auto kernel_A = as[0]->kernel(fem::IntegralType::cell, 0, 0);
      auto kernel_B = as[1]->kernel(fem::IntegralType::cell, 0, 0);
      auto kernel_lagr = as[2]->kernel(fem::IntegralType::cell, 0, 0);

      auto kernel_Lu = problem_data.kernel(2 * rhs_idx);
      auto kernel_Lc = problem_data.kernel(2 * rhs_idx + 1);
      auto coeffs_lu_span = problem_data.coefficients(2 * rhs_idx);
      auto coeffs_lc_span = problem_data.coefficients(2 * rhs_idx + 1);
      auto constants_lu = problem_data.constants(2 * rhs_idx);
      auto constants_lc = problem_data.constants(2 * rhs_idx + 1);

      auto coeffs_a_map = coefficients_a;
      const T* coeffs_a_ptr = coeffs_a_map.empty() ? nullptr : coeffs_a_map.begin()->second.first.data();

      const int cstride_lu = problem_data.cstride(2 * rhs_idx);
      const int cstride_lc = problem_data.cstride(2 * rhs_idx + 1);

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
        const int n_cell_w = n_cell_flux + n_cell_constr;

        // Allocate full mixed-element local buffers
        std::vector<T> A_cell(n_cell_w * n_cell_w, 0.0);
        std::vector<T> B_cell(n_cell_w * n_cell_w, 0.0);
        std::vector<T> Lu_cell(n_cell_w, 0.0);
        std::vector<T> Lc_cell(n_cell_w, 0.0);
        std::vector<T> Lag_cell(n_cell_w, 0.0);

        kernel_A(A_cell.data(), coeffs_a_ptr, constants_a.data(), coordinate_dofs.data(), nullptr, nullptr, nullptr);
        kernel_B(B_cell.data(), coeffs_a_ptr, constants_a.data(), coordinate_dofs.data(), nullptr, nullptr, nullptr);
        kernel_lagr(Lag_cell.data(), coeffs_lagr_ptr, constants_lagr.data(), coordinate_dofs.data(), nullptr, nullptr,
                    nullptr);

        const T* cell_coeffs_lu = coeffs_lu_span.empty() ? nullptr : coeffs_lu_span.data() + cell * cstride_lu;
        const T* cell_coeffs_lc = coeffs_lc_span.empty() ? nullptr : coeffs_lc_span.data() + cell * cstride_lc;

        kernel_Lu(Lu_cell.data(), cell_coeffs_lu, constants_lu.data(), coordinate_dofs.data(), nullptr, nullptr,
                  nullptr);
        kernel_Lc(Lc_cell.data(), cell_coeffs_lc, constants_lc.data(), coordinate_dofs.data(), nullptr, nullptr,
                  nullptr);

        if (needs_transform)
        {
          apply_dof_trans_left(A_cell, cell_info, cell, n_cell_w);
          apply_dof_trans_right(A_cell, cell_info, cell, n_cell_w);
          apply_dof_trans_left(B_cell, cell_info, cell, n_cell_w);
          apply_dof_trans_right(B_cell, cell_info, cell, n_cell_w);
          apply_dof_trans_left(Lu_cell, cell_info, cell, 1);
          apply_dof_trans_left(Lc_cell, cell_info, cell, 1);
          apply_dof_trans_left(Lag_cell, cell_info, cell, 1);
        }

        // Scatter A (top-left RT x RT block) and Lu (top RT segment)
        for (int i = 0; i < n_cell_flux; ++i)
        {
          auto it_row = global_to_local_flux.find(cell_flux_dofs[i]);
          if (it_row == global_to_local_flux.end())
            continue;
          const int patch_row = it_row->second;

          Lu(patch_row) += Lu_cell[i];

          for (int j = 0; j < n_cell_flux; ++j)
          {
            auto it_col = global_to_local_flux.find(cell_flux_dofs[j]);
            if (it_col == global_to_local_flux.end())
              continue;
            const int patch_col = it_col->second;
            A(patch_row, patch_col) += A_cell[i * n_cell_w + j];
          }

          // Scatter B (top-right RT x DG block: columns n_cell_flux .. n_cell_w - 1)
          for (int j = 0; j < n_cell_constr; ++j)
          {
            const int patch_col = global_to_local_constr.at(cell_constr_dofs[j]);
            B(patch_row, patch_col) += B_cell[i * n_cell_w + (n_cell_flux + j)];
          }
        }

        // Scatter Lc (bottom DG segment) and Lagrange multiplier into last column of B
        for (int j = 0; j < n_cell_constr; ++j)
        {
          const int patch_idx = global_to_local_constr.at(cell_constr_dofs[j]);
          Lc(patch_idx) += Lc_cell[n_cell_flux + j];
          // Note: Lagrange multiplier couples with p (constraint space), not u!
        }
      }

      // -------------------------------------------------------------------------
      // Solve the saddle-point system:
      //   [ A    B   0 ] [ u   ]   [ Lu ]
      //   [ B^T  0   g ] [ p   ] = [ Lc ]
      //   [ 0    g^T 0 ] [ lam ]   [ 0  ]
      // where g = \int q dx enforces zero mean on p (making B^T full row rank).
      // -------------------------------------------------------------------------
      const std::size_t n_total = n_flux + n_constr;
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> KKT
          = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(n_total, n_total);
      Eigen::Matrix<T, Eigen::Dynamic, 1> RHS = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(n_total);

      // Flux-flux mass block A
      KKT.block(0, 0, n_flux, n_flux) = A.topLeftCorner(n_flux, n_flux);
      // Flux-DG divergence block B_div (size n_flux x n_constr_dofs)
      KKT.block(0, n_flux, n_flux, n_constr_dofs) = B.topLeftCorner(n_flux, n_constr_dofs);
      KKT.block(n_flux, 0, n_constr_dofs, n_flux) = B.topLeftCorner(n_flux, n_constr_dofs).transpose();

      // Scalar Lagrange multiplier column g (couples with DG space p, rows n_flux .. n_flux + n_constr_dofs - 1)
      // Assemble g directly from as[2] over patch_cells:
      for (const std::int32_t cell : patch_cells)
      {
        auto cell_constr_dofs = dofmap_q->cell_dofs(cell);
        for (std::size_t i = 0; i < x_dofmap.extent(1); ++i)
        {
          const auto node_idx = x_dofmap(cell, i);
          coordinate_dofs[3 * i + 0] = x[3 * node_idx + 0];
          coordinate_dofs[3 * i + 1] = x[3 * node_idx + 1];
          coordinate_dofs[3 * i + 2] = x[3 * node_idx + 2];
        }
        const int n_cell_flux = dofmap_v->cell_dofs(cell).size();
        const int n_cell_constr = cell_constr_dofs.size();
        const int n_cell_w = n_cell_flux + n_cell_constr;
        std::vector<T> Lag_cell(n_cell_w, 0.0);
        kernel_lagr(Lag_cell.data(), coeffs_lagr_ptr, constants_lagr.data(), coordinate_dofs.data(), nullptr, nullptr,
                    nullptr);

        for (int j = 0; j < n_cell_constr; ++j)
        {
          const int patch_row = n_flux + global_to_local_constr.at(cell_constr_dofs[j]);
          KKT(patch_row, n_total - 1) += Lag_cell[n_cell_flux + j];
          KKT(n_total - 1, patch_row) += Lag_cell[n_cell_flux + j];
        }
      }

      RHS.segment(0, n_flux) = Lu.head(n_flux);
      RHS.segment(n_flux, n_constr_dofs) = Lc.head(n_constr_dofs);

      // Solve patch system (use FullPivLU or LDLT for symmetric indefinite saddle-point systems)
      Eigen::FullPivLU<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> kkt_solver(KKT);
      Eigen::Matrix<T, Eigen::Dynamic, 1> sol_patch = kkt_solver.solve(RHS);

      // Accumulate partition-of-unity patch flux contributions (sum, do NOT divide by count)
      auto& fluxes_array = fluxes[rhs_idx]->x()->mutable_array();
      for (std::size_t i = 0; i < n_flux; ++i)
      {
        const std::int32_t global_dof = local_patch_flux_dofs[i];
        fluxes_array[global_dof] += sol_patch(i);
      }
    }
    // Deactivate hat function after solving all RHS on this patch
    for (std::size_t c = 0; c < patch_cells.size(); ++c)
    {
      problem_data.set_hat_function(patch_cells[c], patch_local_nodes[c], static_cast<T>(0.0));
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