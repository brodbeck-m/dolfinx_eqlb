// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Patch.hpp"
#include "ProblemData.hpp"
#include "StorageStiffness.hpp"
#include "solve_patch.hpp"

#include <basix/e-lagrange.h>
#include <basix/finite-element.h>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx_eqlb/base/BoundaryData.hpp>

#include <algorithm>
#include <array>
#include <exception>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace dolfinx;

namespace base = dolfinx_eqlb::base;

namespace dolfinx_eqlb::ev
{
/// Calculation of patch contributions to flux
///
/// Equilibration procedure is based on the solution of patch-wise constrained
/// minimisation problems as described in [1]
///
/// [1] Ern, A. and Vohralík, M.: https://doi.org/10.1137/130950100, 2015
///
/// @tparam T The scalar type
/// @param a              The bilinear forms to assemble
/// @param l_pen          Penalization terms to assemble
/// @param problem_data   Linear forms and problem dependent input data
/// @param fct_type       Lookup-table for facet-types
/// @param flux_dg        Function that holds the projected fluxes
template <dolfinx::scalar T, std::floating_point U>
void reconstruction(const fem::Form<T, U>& a, const fem::Form<T, U>& l_pen,
                    ProblemData<T, U>& problem_data)
{
  /* Geometry */
  const mesh::Geometry<U>& geometry = a.mesh()->geometry();
  const int dim = geometry.dim();

  // Number of nodes on processor
  int n_nodes = a.mesh()->topology()->index_map(0)->size_local();

  // Number of elements on processor
  int n_cells = a.mesh()->topology()->index_map(dim)->size_local();

  /* Function space */
  // Get function space
  const std::shared_ptr<const fem::FunctionSpace<U>> function_space
      = a.function_spaces().at(0);

  // Get DOFmap
  std::shared_ptr<const fem::DofMap> dofmap0 = function_space->dofmap();
  assert(dofmap0);

  /* DOF transformation */
  std::shared_ptr<const fem::FiniteElement<U>> element0
      = function_space->element();

  fem::DofTransformKernel<T> auto P0
      = element0->template dof_transformation_fn<T>(
          fem::doftransform::standard);
  fem::DofTransformKernel<T> auto P0T
      = element0->template dof_transformation_fn<T>(
          fem::doftransform::transpose);

  const bool needs_transformation_data
      = element0->needs_dof_transformations() or a.needs_facet_permutations();
  std::span<const std::uint32_t> cell_info;
  if (needs_transformation_data)
  {
    std::shared_ptr<const mesh::Mesh<U>> mesh = a.mesh();
    assert(mesh);
    mesh->topology_mutable()->create_entity_permutations();
    cell_info = std::span(mesh->topology()->get_cell_permutation_info());
  }

  /* Initialize Patch */
  // BasiX elements of flux subspaces
  std::vector<int> sub0(1, 0);
  const basix::FiniteElement<U>& basix_element_flux
      = function_space->sub(sub0).element()->basix_element();

  Patch<U> patch = Patch<U>(
      n_nodes, a.mesh(), problem_data.facet_type(), a.function_spaces().at(0),
      problem_data.flux(0).function_space(), basix_element_flux);

  /* Prepare Assembly */
  // Set kernels
  auto kernel_a = a.kernel(fem::IntegralType::cell, -1);
  auto kernel_lpen = l_pen.kernel(fem::IntegralType::cell, -1);
  problem_data.initialize_kernels(fem::IntegralType::cell, -1);

  // Initialize storage of tangents on each cell
  StorageStiffness<T> storage_stiffness
      = StorageStiffness<T>(n_cells, patch.ndofs_elmt(), patch.ndofs_cons());

  /* Solve flux reconstruction on each patch */
  // Loop over all nodes and solve patch problem
  for (std::size_t i_node = 0; i_node < n_nodes; ++i_node)
  {
    // Create Sub-DOFmap
    patch.create_subdofmap(i_node);

    // Solve patch problem
    equilibrate_flux_constrmin<T, U>(geometry, patch, P0, P0T, cell_info,
                                     kernel_a, kernel_lpen, problem_data,
                                     storage_stiffness);
  }
}

/// Execute flux calculation based on H(div) conforming equilibration
///
/// Equilibration procedure is based on the solution of patch-wise
/// constrained minimisation problems as described in [1].
///
/// [1] Ern, A. and Vohralík, M.: https://doi.org/10.1137/130950100, 2015
///
/// @param a                   The bilinear form to assemble
/// @param l                   The linear form to assemble
/// @param fct_esntbound_prime Facets of essential BCs of primal problem
/// @param fct_esntbound_flux  Facets of essential BCs on flux field
/// @param bcs_flux            Essential boundary conditions for the flux
/// @param flux                Function that holds the reconstructed flux
template <dolfinx::scalar T, std::floating_point U>
void reconstruction(
    const fem::Form<T, U>& a, const fem::Form<T, U>& l_pen,
    const std::vector<std::shared_ptr<const fem::Form<T, U>>>& l,
    std::vector<std::shared_ptr<fem::Function<T, U>>>& flux_hdiv,
    std::shared_ptr<base::BoundaryData<T, U>> boundary_data)
{
  // Check input
  int n_rhs = l.size();
  int n_bcs = boundary_data->num_rhs();
  int n_flux = flux_hdiv.size();

  if (n_rhs != n_bcs || n_rhs != n_flux)
  {
    throw std::runtime_error("Equilibration: Input sizes does not match");
  }

  /* Initialize problem data */
  ProblemData<T, U> problem_data
      = ProblemData<T, U>(flux_hdiv, l, boundary_data);

  /* Call equilibration */
  reconstruction<T, U>(a, l_pen, problem_data);
}

} // namespace dolfinx_eqlb::ev