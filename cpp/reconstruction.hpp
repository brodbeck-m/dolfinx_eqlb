#pragma once

#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Sparse"
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

std::tuple<int, std::vector<std::int32_t>> submap_equilibration_patch(
    int i_node, std::shared_ptr<const fem::FunctionSpace> function_space,
    const std::vector<std::vector<std::vector<int>>>& entity_dofs0,
    const std::vector<std::vector<std::vector<int>>>& entity_dofs1,
    const std::vector<std::int8_t>& fct_type);

template <typename T>
void reconstruct_flux_patch(std::vector<std::int32_t>& fct_esntbound_prime,
                            std::vector<std::int32_t>& fct_esntbound_flux,
                            fem::Function<T>& sol_elmt, const fem::Form<T>& a,
                            const fem::Form<T>& l)
{
  /* Extract geometry data */
  // Get geometry and topology
  const mesh::Geometry& geometry = a.mesh()->geometry();
  const mesh::Topology& topology = a.mesh()->topology();

  // Get spacial dimension
  int dim = geometry.dim();
  int dim_fct = dim - 1;

  // Nodes: DOFmap and positions
  const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();
  const std::size_t num_nodes_g = geometry.cmap().dim();
  std::span<const double> x = geometry.x();
  std::vector<double> coordinate_dofs(3 * num_nodes_g);

  /* Constants and coefficients */
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

  /* DOFmap and function spaces */
  // Function space
  std::shared_ptr<const fem::FunctionSpace> function_space
      = a.function_spaces().at(0);

  // DOFmap
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

  // // Local DOFmap facets (sorted by entity)
  // const std::vector<std::vector<std::vector<int>>>& entity_dofs0
  //     = basix_element0.entity_dofs();
  // const std::vector<std::vector<std::vector<int>>>& entity_dofs1
  //     = basix_element1.entity_dofs();

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

  /* Initialize tangent arrays*/
  // DOFs per element
  const int num_dof_elmt = dofmap0->list().links(0).size();
  const int ndim0 = bs0 * num_dof_elmt;

  // Element
  std::vector<T> Ae(ndim0 * ndim0);
  std::span<T> _Ae(Ae);
  std::vector<T> Le(ndim0);
  std::span<T> _Le(Le);

  // Patch (dynamic, resized for every patch)
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A_patch;
  Eigen::Matrix<T, Eigen::Dynamic, 1> L_patch, u_patch;

  Eigen::ConjugateGradient<
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
      Eigen::Lower | Eigen::Upper>
      solver;

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

  /* Solve flux reconstruction on each patch */
  // Loop over all nodes and solve patch problem
  for (std::size_t i_node = 0; i_node < topology.index_map(0)->size_local();
       ++i_node)
  {
    // Create Sub-DOFmap
    auto [type_patch, cells_patch] = submap_equilibration_patch(
        i_node, function_space, basix_element0.entity_dofs(),
        basix_element1.entity_dofs(), fct_type);
  }
}

} // namespace dolfinx_eqlb