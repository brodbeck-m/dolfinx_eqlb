#pragma once

#include "solve_patch_constrmin.hpp"
#include <algorithm>
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
#include <exception>
#include <functional>
#include <iostream>
#include <iterator>
#include <span>
#include <vector>

namespace dolfinx_adaptivity::equilibration
{
/// Construction of a sub-DOFmap on each patch
///
/// Determines type of patch (0-> internal, 1->bc_neumann, 2->bc_dirichlet
/// 3->bc_mixed) and creats sorted DOFmap. Sorting of facets/elements/DOFs
/// follows [1,2].
///
/// [1] Moldenhauer, M.: Stress reconstructionand a-posteriori error
///     estimationfor elasticity (PhdThesis)
/// [2] Bertrand, F.; Carstensen, C.; Gräßle, B. & Tran, N. T.:
///     Stabilization-free HHO a posteriori error control, 2022
///
/// @param i_node         ID (local on partition!) of current node
/// @param function_space FunctionSpace of mixed problem
/// @param entity_dofs0   BasiX/FiniteElement.entity_dofs flux subspace
///                       subspace
/// @param fct_type       Vector (len: number of fcts on partition)
///                       determine the facet type
/// @return type_patch    Type of patch (0, 1, 2 or 3)
/// @return ndof_patch    Number of DOFs on patch
/// @return cells_patch   List of cells (consisnten to local DOFmap)
///                       on patch
/// @return dofs_local    Non-Zero DOFs on patch (element-local ID)
/// @return dofs_patch    Non-Zero DOFs on patch (patch-local ID)

std::tuple<int, const int, std::vector<std::int32_t>,
           graph::AdjacencyList<std::int32_t>,
           graph::AdjacencyList<std::int32_t>>
submap_equilibration_patch(
    int i_node, std::shared_ptr<const fem::FunctionSpace> function_space,
    const std::vector<std::vector<std::vector<int>>>& entity_dofs0,
    const std::vector<std::int8_t>& fct_type);

/// Execute calculation of patch constributions to flux
///
/// @param a              The bilinear form to assemble
/// @param l              The linar form to assemble
/// @param constants_l    Constants that appear in 'l'
/// @param coefficients_l Coefficients that appear in 'l'
/// @param fct_type       Vector (len: number of fcts on partition)
///                       determine the facet type
/// @param flux           Function that holds the reconstructed flux
template <typename T>
void reconstruct_fluxes_patch(
    const fem::Form<T>& a, const fem::Form<T>& l,
    std::span<const T> constants_l,
    const std::map<std::pair<fem::IntegralType, int>,
                   std::pair<std::span<const T>, int>>& coefficients_l,
    std::vector<std::int8_t>& fct_type, fem::Function<T>& flux,
    fem::Function<T>& flux_dg)
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

  // BasiX elements of subspaces
  std::vector<int> sub0(1, 0);
  const basix::FiniteElement& basix_element0
      = function_space->sub(sub0)->element()->basix_element();

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

  /* Prepare Assembly */
  // Integration kernels
  const auto& kernel_a = a.kernel(fem::IntegralType::cell, -1);
  const auto& kernel_l = l.kernel(fem::IntegralType::cell, -1);

  // Coefficients
  const auto& [coeffs_l, cstride_l]
      = coefficients_l.at({fem::IntegralType::cell, -1});

  // Local part of the solution vector (only flux!)
  std::span<T> x_flux = flux.x()->mutable_array();
  std::span<T> x_flux_dg = flux_dg.x()->mutable_array();

  /* Initialize storage of  stiffnes matrix on each cell*/
  // Get size for storage array
  const int dim_fe_space = element0->space_dimension();
  const int dim_stiffness = dim_fe_space * dim_fe_space;

  // Initialize id, if cell is already initilaized
  std::vector<std::int8_t> cell_is_evaluated(n_cells, 0);

  // Initialize adjaceny list
  std::vector<std::int32_t> offset_storage_stiffness(n_cells + 1);
  std::vector<T> data_storage_stiffness(n_cells * dim_stiffness, 0);
  std::generate(
      offset_storage_stiffness.begin(), offset_storage_stiffness.end(),
      [n = 0, dim_stiffness]() mutable { return dim_stiffness * (n++); });

  graph::AdjacencyList<T> storage_stiffness_cells = graph::AdjacencyList<T>(
      std::move(data_storage_stiffness), std::move(offset_storage_stiffness));

  /* Solve flux reconstruction on each patch */
  // Loop over all nodes and solve patch problem
  for (std::size_t i_node = 0; i_node < n_nodes; ++i_node)
  {
    // Create Sub-DOFmap
    auto [type_patch, ndof_patch, cells_patch, dofs_local, dofs_patch]
        = submap_equilibration_patch(i_node, function_space,
                                     basix_element0.entity_dofs(), fct_type);

    // Solve patch problem
    equilibrate_flux_constrmin(
        geometry, type_patch, ndof_patch, cells_patch, dofmap0->list(),
        dofs_local, dofs_patch, dof_transform, dof_transform_to_transpose,
        kernel_a, kernel_l, coeffs_l, cstride_l, constants_l, cell_info,
        cell_is_evaluated, storage_stiffness_cells, x_flux, x_flux_dg);
  }
}

/// Execute flux calculation based on H(div) conforming equilibration
///
/// @param a                   The bilinear form to assemble
/// @param l                   The linar form to assemble
/// @param fct_esntbound_prime Facets of essential BCs of primal problem
/// @param fct_esntbound_flux  Facets of essential BCs on flux field
/// @param flux                Function that holds the reconstructed flux
template <typename T>
void reconstruct_fluxes(const fem::Form<T>& a, const fem::Form<T>& l,
                        std::vector<std::int32_t>& fct_esntbound_prime,
                        std::vector<std::int32_t>& fct_esntbound_flux,
                        fem::Function<T>& flux, fem::Function<T>& flux_dg)
{
  /* Geometry data */
  // Get topology
  const mesh::Topology& topology = a.mesh()->topology();

  // Facet dimansion
  const int dim_fct = topology.dim() - 1;

  /* Constants and coefficients */
  const std::vector<T> constants_l = pack_constants(l);

  auto coefficients_l = fem::allocate_coefficient_storage(l);
  fem::pack_coefficients(l, coefficients_l);

  /* Mark facest (0->internal, 1->esnt_prim, 2->esnt_flux) */
  // Create look-up table for facets
  std::vector<std::int8_t> fct_type(topology.index_map(dim_fct)->size_local(),
                                    0);

  // Mark facets with essential bcs for primal solution
  // FIXME - Allow for empty arrays
  // FIXME - Parallel computation (input only local facets?)
  if (!fct_esntbound_prime.empty())
  {
    for (const std::int32_t fct : fct_esntbound_prime)
    {
      // Set marker facet
      fct_type[fct] = 1;
    }
  }

  // Mark facets with essential bcs for flux
  // FIXME - Allow for empty arrays
  // FIXME - Parallel computation (input only local facets?)
  if (!fct_esntbound_flux.empty())
  {
    for (const std::int32_t fct : fct_esntbound_flux)
    {
      // Set marker facet
      fct_type[fct] = 2;
    }
  }

  /* Initialize essential boundary conditions for reconstructed flux */
  // TODO - Implement preparation of boundary conditions
  reconstruct_fluxes_patch(a, l, std::span(constants_l),
                           fem::make_coefficients_span(coefficients_l),
                           fct_type, flux, flux_dg);
}

} // namespace dolfinx_adaptivity::equilibration