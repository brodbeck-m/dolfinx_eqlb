#pragma once

#include "PatchFluxEV.hpp"
#include "ProblemDataFlux.hpp"
#include "StorageStiffness.hpp"
#include "solve_patch_constrmin.hpp"
#include <algorithm>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/DofMap.h>
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
#include <exception>
#include <functional>
#include <iostream>
#include <iterator>
#include <span>
#include <vector>

using namespace dolfinx;

namespace dolfinx_adaptivity::equilibration
{
/// Execute calculation of patch constributions to flux
///
/// @param a              The bilinear forms to assemble
/// @param l_pen          Penalisation terms to assemble
/// @param problem_data   Linear forms and problem dependent input data
/// @param fct_type       Lookup-table for facet-types
/// @param flux_dg        Function that holds the projected fluxes
template <typename T>
void reconstruct_fluxes_patch(const fem::Form<T>& a, const fem::Form<T>& l_pen,
                              ProblemDataFlux<T>& problem_data,
                              graph::AdjacencyList<std::int8_t>& fct_type)
{
  /* Geometry */
  const mesh::Geometry& geometry = a.mesh()->geometry();
  const int dim = geometry.dim();

  // Number of nodes on processor
  int n_nodes = a.mesh()->topology().index_map(0)->size_local();

  // Number of elements on processor
  int n_cells = a.mesh()->topology().index_map(dim)->size_local();

  /* Function space */
  // Get function space
  const std::shared_ptr<const fem::FunctionSpace>& function_space
      = a.function_spaces().at(0);

  // Get DOFmap
  std::shared_ptr<const fem::DofMap> dofmap0 = function_space->dofmap();
  assert(dofmap0);

  /* DOF transformation */
  std::shared_ptr<const fem::FiniteElement> element0
      = function_space->element();
  const std::function<void(const std::span<T>&,
                           const std::span<const std::uint32_t>&, std::int32_t,
                           int)>& dof_transform
      = element0->get_dof_transformation_function<T>();
  const std::function<void(const std::span<T>&,
                           const std::span<const std::uint32_t>&, std::int32_t,
                           int)>& dof_transform_to_transpose
      = element0->get_dof_transformation_to_transpose_function<T>();

  const bool needs_transformation_data
      = element0->needs_dof_transformations() or a.needs_facet_permutations();
  std::span<const std::uint32_t> cell_info;
  if (needs_transformation_data)
  {
    std::shared_ptr<const mesh::Mesh> mesh = a.mesh();
    assert(mesh);
    mesh->topology_mutable().create_entity_permutations();
    cell_info = std::span(mesh->topology().get_cell_permutation_info());
  }

  /* Initialize Patch */
  // BasiX elements of flux subspaces
  std::vector<int> sub0(1, 0);
  const basix::FiniteElement& basix_element_flux
      = function_space->sub(sub0)->element()->basix_element();

  PatchFluxEV patch
      = PatchFluxEV(n_nodes, a.mesh(), fct_type, a.function_spaces().at(0),
                    problem_data.flux(0).function_space(), basix_element_flux);

  /* Prepare Assembly */
  // Set kernels
  const auto& kernel_a = a.kernel(fem::IntegralType::cell, -1);
  const auto& kernel_lpen = l_pen.kernel(fem::IntegralType::cell, -1);
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
    equilibrate_flux_constrmin(geometry, patch, dofmap0->list(), dof_transform,
                               dof_transform_to_transpose, cell_info, kernel_a,
                               kernel_lpen, problem_data, storage_stiffness);
  }
}

/// Execute flux calculation based on H(div) conforming equilibration
///
/// @param a                   The bilinears form to assemble
/// @param l                   The linar form to assemble
/// @param fct_esntbound_prime Facets of essential BCs of primal problem
/// @param fct_esntbound_flux  Facets of essential BCs on flux field
/// @param flux                Function that holds the reconstructed flux
template <typename T>
void reconstruct_fluxes(
    const fem::Form<T>& a, const fem::Form<T>& l_pen,
    const std::vector<std::shared_ptr<const fem::Form<T>>>& l,
    const std::vector<std::vector<std::int32_t>>& fct_esntbound_prime,
    const std::vector<std::vector<std::int32_t>>& fct_esntbound_flux,
    const std::vector<std::vector<std::shared_ptr<const fem::DirichletBC<T>>>>&
        bcs_flux,
    std::vector<std::shared_ptr<fem::Function<T>>>& flux_hdiv)
{
  // Check input
  int n_lhs = l.size();
  int n_fbp = fct_esntbound_prime.size();
  int n_fbf = fct_esntbound_flux.size();
  int n_bcs = bcs_flux.size();
  int n_flux = flux_hdiv.size();

  if (n_lhs != n_fbp || n_lhs != n_fbf || n_lhs != n_bcs || n_lhs != n_flux)
  {
    throw std::runtime_error("Equilibration: Input sizes does not match");
  }
  /* Geometry data */
  // Get topology
  const mesh::Topology& topology = a.mesh()->topology();

  // Facet dimansion
  const int dim_fct = topology.dim() - 1;

  /* Mark facest (0->internal, 1->esnt_prim, 2->esnt_flux) */
  // Initialize data storage
  std::int32_t nnodes = topology.index_map(dim_fct)->size_local();
  std::vector<std::int8_t> data_fct_type(nnodes * n_lhs, 0);
  std::vector<std::int32_t> offsets_fct_type(n_lhs + 1, 0);

  // Create adjacency list
  std::generate(offsets_fct_type.begin(), offsets_fct_type.end(),
                [n = 0, nnodes]() mutable { return nnodes * (n++); });

  graph::AdjacencyList<std::int8_t> fct_type
      = graph::AdjacencyList<std::int8_t>(std::move(data_fct_type),
                                          std::move(offsets_fct_type));

  // Set data
  for (std::size_t i = 0; i < n_lhs; ++i)
  {
    // Get lookup-table for l_i
    std::span<std::int8_t> fct_type_i = fct_type.links(i);

    // Mark facets with essential bcs for primal solution
    // FIXME - Parallel computation (input only local facets?)
    if (!fct_esntbound_prime[i].empty())
    {
      for (const std::int32_t fct : fct_esntbound_prime[0])
      {
        // Set marker facet
        fct_type_i[fct] = 1;
      }
    }

    // Mark facets with essential bcs for flux
    // FIXME - Parallel computation (input only local facets?)
    if (!fct_esntbound_flux[i].empty())
    {
      // Check if boundary conditions are set
      if (bcs_flux[i].empty())
      {
        throw std::runtime_error(
            "Equilibration: Essential BC for flux required");
      }

      // Set markers
      for (const std::int32_t fct : fct_esntbound_flux[0])
      {
        // Set marker facet
        fct_type_i[fct] = 2;
      }
    }
    else
    {
      // Check if boundary conditions are set
      if (!bcs_flux[i].empty())
      {
        throw std::runtime_error(
            "Equilibration: No essential BC for flux required");
      }
    }
  }

  /* Initialize essential boundary conditions for reconstructed flux */
  ProblemDataFlux<T> problem_data = ProblemDataFlux<T>(l, bcs_flux, flux_hdiv);

  /* Call equilibration */
  reconstruct_fluxes_patch(a, l_pen, problem_data, fct_type);
}

} // namespace dolfinx_adaptivity::equilibration