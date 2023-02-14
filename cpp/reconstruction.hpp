#pragma once

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
#include <functional>
#include <iostream>
#include <iterator>
#include <span>
#include <vector>

namespace dolfinx_eqlb
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
/// @param entity_dofs1   BasiX/FiniteElement.entity_dofs constraint
///                       subspace
/// @param fct_type       Vector (len: number of fcts on partition)
///                       determine the facet type
/// @return type_facet    Type of facet (0, 1, 2 or 3)
/// @return cells_patch   List of cells (consisnten to local DOFmap)
///                       on patch
/// @return dofs_local    Non-Zero DOFs on patch (element-local ID)
/// @return dofs_patch    Non-Zero DOFs on patch (patch-local ID)

std::tuple<int, std::vector<std::int32_t>, graph::AdjacencyList<std::int32_t>,
           graph::AdjacencyList<std::int32_t>>
submap_equilibration_patch(
    int i_node, std::shared_ptr<const fem::FunctionSpace> function_space,
    const std::vector<std::vector<std::vector<int>>>& entity_dofs0,
    const std::vector<std::vector<std::vector<int>>>& entity_dofs1,
    const std::vector<std::int8_t>& fct_type);

/// Execute calculation of patch constributions to flux
///
/// @param a              The bilinear form to assemble
/// @param l              The linar form to assemble
/// @param constants_a    Constants that appear in 'a'
/// @param constants_l    Constants that appear in 'l'
/// @param coefficients_a Coefficients that appear in 'a'
/// @param coefficients_l Coefficients that appear in 'l'
/// @param fct_type       Vector (len: number of fcts on partition)
///                       determine the facet type
/// @param flux           Function that holds the reconstructed flux
template <typename T>
void reconstruct_fluxes_patch(
    const fem::Form<T>& a, const fem::Form<T>& l,
    std::span<const T> constants_a, std::span<const T> constants_l,
    const std::map<std::pair<fem::IntegralType, int>,
                   std::pair<std::span<const T>, int>>& coefficients_a,
    const std::map<std::pair<fem::IntegralType, int>,
                   std::pair<std::span<const T>, int>>& coefficients_l,
    std::vector<std::int8_t>& fct_type, fem::Function<T>& flux)
{
  /* Geometry */
  int n_nodes = a.mesh()->topology().index_map(0)->size_local();

  /* Function space */
  // Get function space
  const std::shared_ptr<const fem::FunctionSpace>& function_space
      = a.function_spaces().at(0);

  // Get DOFmap
  std::shared_ptr<const fem::DofMap> dofmap0 = function_space->dofmap();
  assert(dofmap0);
  const int bs0 = dofmap0->bs();

  // BasiX elements of subspaces
  std::vector<int> sub0(1, 0);
  std::vector<int> sub1(1, 1);
  const basix::FiniteElement& basix_element0
      = function_space->sub(sub0)->element()->basix_element();
  const basix::FiniteElement& basix_element1
      = function_space->sub(sub1)->element()->basix_element();

  //   /* DOF transformation */
  //   std::shared_ptr<const fem::FiniteElement> element0
  //       = function_space->element();
  //   const std::function<void(const std::span<T>&,
  //                            const std::span<const std::uint32_t>&,
  //                            std::int32_t, int)>& dof_transform
  //       = element0->get_dof_transformation_function<T>();
  //   const std::function<void(const std::span<T>&,
  //                            const std::span<const std::uint32_t>&,
  //                            std::int32_t, int)>& dof_transform_to_transpose
  //       = element0->get_dof_transformation_to_transpose_function<T>();

  //   const bool needs_transformation_data
  //       = element0->needs_dof_transformations() or
  //       a.needs_facet_permutations();
  //   std::span<const std::uint32_t> cell_info;
  //   if (needs_transformation_data)
  //   {
  //     std::shared_ptr<const mesh::Mesh> mesh = a.mesh();
  //     assert(mesh);
  //     mesh->topology_mutable().create_entity_permutations();
  //     cell_info = std::span(mesh->topology().get_cell_permutation_info());
  //   }

  /* Solve flux reconstruction on each patch */
  // Loop over all nodes and solve patch problem
  for (std::size_t i_node = 0; i_node < n_nodes; ++i_node)
  {
    // Create Sub-DOFmap
    auto [type_patch, cells_patch, dofs_local, dofs_patch]
        = submap_equilibration_patch(i_node, function_space,
                                     basix_element0.entity_dofs(),
                                     basix_element1.entity_dofs(), fct_type);
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
                        fem::Function<T>& flux)
{
  /* Geometry data */
  // Get topology
  const mesh::Topology& topology = a.mesh()->topology();

  // Facet dimansion
  const int dim_fct = topology.dim() - 1;

  /* Constants and coefficients */
  const std::vector<T> constants_a = pack_constants(a);
  const std::vector<T> constants_l = pack_constants(l);

  auto coefficients_a = fem::allocate_coefficient_storage(a);
  fem::pack_coefficients(a, coefficients_a);
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

  reconstruct_fluxes_patch(a, l, std::span(constants_a), std::span(constants_l),
                           fem::make_coefficients_span(coefficients_a),
                           fem::make_coefficients_span(coefficients_l),
                           fct_type, flux);
}

} // namespace dolfinx_eqlb