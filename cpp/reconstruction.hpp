#pragma once

#include "PatchFluxEV.hpp"
#include "StorageStiffness.hpp"
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
/// Execute calculation of patch constributions to flux
///
/// @param a              The bilinear forms to assemble
/// @param l_pen          Penalisation terms to assemble
/// @param l              The linar form to assemble
/// @param consts_l       Constants that appear in 'l'
/// @param coeffs_l       Coefficients that appear in 'l'
/// @param info_coeffs_l  Information about coefficient storage for 'l'
/// @param fct_type       Vector (len: number of fcts on partition)
///                       determine the facet type
/// @param flux           Function that holds the reconstructed flux
template <typename T>
void reconstruct_fluxes_patch(const fem::Form<T>& a, const fem::Form<T>& l_pen,
                              const fem::Form<T>& l,
                              std::span<const T> consts_l,
                              std::span<T> coeffs_l,
                              const std::vector<int>& info_coeffs_l,
                              std::span<std::int8_t> fct_type,
                              fem::Function<T>& flux, fem::Function<T>& flux_dg)
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
                    basix_element_flux);
  std::cout << "Test passed" << std::endl;

  /* Prepare Assembly */
  const auto& kernel_a = a.kernel(fem::IntegralType::cell, -1);
  const auto& kernel_lpen = l_pen.kernel(fem::IntegralType::cell, -1);
  const auto& kernel_l = l.kernel(fem::IntegralType::cell, -1);

  // Local part of the solution vector (only flux!)
  std::span<T> x_flux = flux.x()->mutable_array();
  std::span<T> x_flux_dg = flux_dg.x()->mutable_array();

  // Initialize storage of tangents on each cell
  // StorageStiffness<T> storage_stiffness
  //     = StorageStiffness<T>(n_cells, patch.ndofs_elmt(), patch.ndofs_cons());

  /* Solve flux reconstruction on each patch */
  // Loop over all nodes and solve patch problem
  for (std::size_t i_node = 0; i_node < n_nodes; ++i_node)
  {
    // Create Sub-DOFmap
    patch.create_subdofmap(i_node);

    // Solve patch problem
    equilibrate_flux_constrmin(geometry, patch, dofmap0->list(), dof_transform,
                               dof_transform_to_transpose, kernel_a,
                               kernel_lpen, kernel_l, consts_l, coeffs_l,
                               info_coeffs_l, cell_info, x_flux, x_flux_dg);
    std::cout << "Test passed" << std::endl;
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
void reconstruct_fluxes(const fem::Form<T>& a, const fem::Form<T>& l_pen,
                        const fem::Form<T>& l,
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
  // Allocate storage of coefficients/constants of linear form
  const std::vector<T> constants_l = pack_constants(l);

  auto coefficients_l = fem::allocate_coefficient_storage(l);
  fem::pack_coefficients(l, coefficients_l);

  auto& [coeffs_l, cstride_l]
      = coefficients_l.at({fem::IntegralType::cell, -1});

  // Pack relevant informations (0->cstride, 1->Begin _hat, 2->Begin flux_dg)
  std::vector<int> info_coeffs_l(3, 0);

  if (cstride_l - l.coefficient_offsets()[1] > l.coefficient_offsets()[1])
  {
    // Set cstride_l
    info_coeffs_l[0] = cstride_l;

    // Set beginn of _hat-data
    info_coeffs_l[1] = l.coefficient_offsets()[1];

    // Set beginn of flux_dg-data
    info_coeffs_l[1] = 0;
  }
  else
  {
    // Set cstride_l
    info_coeffs_l[0] = cstride_l;

    // Set beginn of _hat-data
    info_coeffs_l[1] = 0;

    // Set beginn of flux_dg-data
    info_coeffs_l[1] = l.coefficient_offsets()[1];
  }

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
  reconstruct_fluxes_patch(a, l_pen, l, std::span(constants_l),
                           std::span(coeffs_l), info_coeffs_l,
                           std::span(fct_type), flux, flux_dg);
}

} // namespace dolfinx_adaptivity::equilibration