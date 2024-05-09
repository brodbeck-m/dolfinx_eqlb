#pragma once

#include "BoundaryData.hpp"
#include "KernelData.hpp"
#include "PatchCstm.hpp"
#include "PatchData.hpp"
#include "PatchFluxEV.hpp"
#include "ProblemDataFluxCstm.hpp"
#include "ProblemDataFluxEV.hpp"
#include "StorageStiffness.hpp"
#include "minimise_flux.hpp"
#include "solve_patch_constrmin.hpp"
#include "solve_patch_semiexplt.hpp"
#include "utils.hpp"

#include <basix/e-lagrange.h>
#include <basix/finite-element.h>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/CoordinateElement.h>
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

#include <algorithm>
#include <array>
#include <chrono>
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

namespace dolfinx_eqlb
{
/// Calculation of patch contributions to flux
///
/// Equilibration procedure is based on the solution of patch-wise constrained
/// minimisation problems as described in [1]
///
/// [1] Ern, A. & Vohralík, M.: Polynomial-Degree-Robust A Posteriori
///     Estimates in a Unified Setting for Conforming, Nonconforming,
///     Discontinuous Galerkin, and Mixed Discretizations, 2015
///
/// @tparam T The scalar type
/// @param a              The bilinear forms to assemble
/// @param l_pen          Penalization terms to assemble
/// @param problem_data   Linear forms and problem dependent input data
/// @param fct_type       Lookup-table for facet-types
/// @param flux_dg        Function that holds the projected fluxes
template <typename T>
void reconstruct_fluxes_patch(const fem::Form<T>& a, const fem::Form<T>& l_pen,
                              ProblemDataFluxEV<T>& problem_data)
{
  const std::int32_t n_repetition = 250000;

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

  PatchFluxEV patch = PatchFluxEV(
      n_nodes, a.mesh(), problem_data.facet_type(), a.function_spaces().at(0),
      problem_data.flux(0).function_space(), basix_element_flux);

  /* Prepare Assembly */
  // Set kernels
  const auto& kernel_a = a.kernel(fem::IntegralType::cell, -1);
  const auto& kernel_lpen = l_pen.kernel(fem::IntegralType::cell, -1);
  problem_data.initialize_kernels(fem::IntegralType::cell, -1);

  // Initialize storage of tangents on each cell
  StorageStiffness<T> storage_stiffness
      = StorageStiffness<T>(n_cells, patch.ndofs_elmt(), patch.ndofs_cons());

  /* Timing */
  // Initialise timing
  std::chrono::time_point<std::chrono::system_clock> begin, end;
  std::vector<std::int32_t> nodes{2, 4};

  for (std::int32_t i_node : nodes)
  {
    // --- Time patch creation
    begin = std::chrono::system_clock::now();
    for (std::size_t i = 0; i < n_repetition; ++i)
    {
      // Create patch
      patch.create_subdofmap(i_node);
    }
    end = std::chrono::system_clock::now();

    std::chrono::duration<double> total_create_patch = end - begin;

    // Time equilibration: Assemble equation system
    begin = std::chrono::system_clock::now();
    for (std::size_t i = 0; i < n_repetition; ++i)
    {
      // Create Sub-DOFmap
      patch.create_subdofmap(i_node);

      // Solve patch problem
      equilibrate_flux_constrmin<T, 1>(
          geometry, patch, dofmap0->list(), dof_transform,
          dof_transform_to_transpose, cell_info, kernel_a, kernel_lpen,
          problem_data, storage_stiffness, n_repetition);
    }
    end = std::chrono::system_clock::now();

    std::chrono::duration<double> total_eqlb_assembly = end - begin;

    // Time equilibration: Assemble equation system
    begin = std::chrono::system_clock::now();
    for (std::size_t i = 0; i < n_repetition; ++i)
    {
      // Create Sub-DOFmap
      patch.create_subdofmap(i_node);

      // Solve patch problem
      equilibrate_flux_constrmin<T, 2>(
          geometry, patch, dofmap0->list(), dof_transform,
          dof_transform_to_transpose, cell_info, kernel_a, kernel_lpen,
          problem_data, storage_stiffness, n_repetition);
    }
    end = std::chrono::system_clock::now();

    std::chrono::duration<double> total_eqlb_solve = end - begin;

    // --- Output timings
    double timing_eqlb_assembly
        = total_eqlb_assembly.count() - total_create_patch.count();
    double timing_eqlb_solve
        = total_eqlb_solve.count() - total_eqlb_assembly.count();

    std::cout << "Timings for patch-size " << patch.ncells() << std::endl;
    std::cout << "Patch creation: " << total_create_patch.count() << std::endl;
    std::cout << "Eqlb. - assembly flux minimisation: " << timing_eqlb_assembly
              << std::endl;
    std::cout << "Eqlb. - solve flux minimisation: " << timing_eqlb_solve
              << std::endl;
    std::cout << "Eqlb. - total: " << total_eqlb_solve.count() << std::endl;
  }
}

/// Calculation of patch contributions to flux
///
/// Equilibration procedure is based on an explicitly calculated flux and an
/// unconstrained minimisation problem on a patch-wise divergence-free H(div)
/// space (see [1]).
///
/// [1] Bertrand, F.; Carstensen, C.; Gräßle, B. & Tran, N. T.:
///     Stabilization-free HHO a posteriori error control, 2022
///
/// @tparam T             The scalar type
/// @tparam id_flux_order The flux order (1->RT1, 2->RT2, 3->general)
/// @param problem_data   The problem data
/// @param fct_type       Lookup-table for facet-types
template <typename T, int id_flux_order, bool symconstr_required>
void reconstruct_fluxes_patch(ProblemDataFluxCstm<T>& problem_data)
{
  assert(id_flux_order < 0);

  const std::int32_t n_repetition = 250000;

  /* Geometry */
  // Extract mesh
  std::shared_ptr<const mesh::Mesh> mesh = problem_data.mesh();
  const fem::CoordinateElement& cmap = mesh->geometry().cmap();

  // Spacial dimension
  const int dim = mesh->geometry().dim();

  // Number of nodes on processor
  int n_nodes = mesh->topology().index_map(0)->size_local();

  // Number of elements on processor
  int n_cells = mesh->topology().index_map(dim)->size_local();

  /* Basix elements */
  // Basix element of pice-wise H(div) flux
  const basix::FiniteElement& basix_element_fluxhdiv
      = problem_data.fspace_flux_hdiv()->element()->basix_element();

  const int degree_flux_hdiv = basix_element_fluxhdiv.degree();
  const int degree_rt_flux_hdiv = degree_flux_hdiv - 1;

  // Basix element of projected flux/ RHS
  const basix::FiniteElement& basix_element_rhs
      = problem_data.fspace_flux_dg()->element()->basix_element();

  const int degree_rhs = basix_element_rhs.degree();

  bool is_dg = (degree_rhs == 0) ? true : false;

  basix::FiniteElement basix_element_rhscg = basix::element::create_lagrange(
      basix_element_rhs.cell_type(), degree_rhs,
      basix_element_rhs.lagrange_variant(), is_dg);

  // Basix element of hat-function
  basix::FiniteElement basix_element_hat = basix::element::create_lagrange(
      basix_element_rhs.cell_type(), 1, basix_element_rhs.lagrange_variant(),
      false);

  /* Equilibration */
  // Set quadrature rule
  const int quadrature_degree
      = (degree_flux_hdiv == 1) ? 2 : 2 * degree_flux_hdiv + 1;
  QuadratureRule quadrature_rule
      = QuadratureRule(mesh->topology().cell_type(), quadrature_degree, dim);

  // Initialize KernelData
  KernelDataEqlb<T> kernel_data = KernelDataEqlb<T>(
      mesh, std::make_shared<QuadratureRule>(quadrature_rule),
      basix_element_fluxhdiv, basix_element_rhs, basix_element_hat);

  // Generate minimisation kernels
  kernel_fn<T, true> kernel_fluxmin
      = generate_flux_minimisation_kernel<T, true>(kernel_data, dim,
                                                   degree_rt_flux_hdiv);

  kernel_fn<T, false> kernel_fluxmin_l
      = generate_flux_minimisation_kernel<T, false>(kernel_data, dim,
                                                    degree_rt_flux_hdiv);

  // Initialise timing
  std::chrono::time_point<std::chrono::system_clock> begin, end;

  // Execute equilibration
  // FIXME - Currently only 2D meshes supported
  if constexpr (symconstr_required)
  {
    // Get list with node markers on stress boundary
    std::span<const std::int8_t> pnt_on_stress_boundary
        = problem_data.node_on_essnt_boundary_stress();

    // Initialise patch
    PatchFluxCstm<T, id_flux_order> patch = PatchFluxCstm<T, id_flux_order>(
        mesh, problem_data.facet_type(), 2, pnt_on_stress_boundary,
        problem_data.fspace_flux_hdiv(), problem_data.fspace_flux_dg(),
        basix_element_rhscg, true);

    // Initialise storage for equilibration
    PatchDataCstm<T, id_flux_order> patch_data
        = PatchDataCstm<T, id_flux_order>(patch, kernel_data.nipoints_facet(),
                                          true);

    // Set kernel for weak symmetry condition
    kernel_fn_schursolver<T> kernel_weaksym
        = generate_stress_minimisation_kernel<T>(Kernel::StressMin, kernel_data,
                                                 dim, patch.fcts_per_cell(),
                                                 degree_rt_flux_hdiv);

    std::vector<std::int32_t> nodes{2, 4};

    for (std::int32_t i_node : nodes)
    {
      // --- Time patch creation
      begin = std::chrono::system_clock::now();
      for (std::size_t i = 0; i < n_repetition; ++i)
      {
        // Create patch
        patch.create_subdofmap(i_node);

        // Reinitialise patch-data
        patch_data.reinitialisation(patch.type(), patch.ncells());
      }
      end = std::chrono::system_clock::now();

      std::chrono::duration<double> total_create_patch = end - begin;

      //  --- Time equilibration: Explicit setp
      begin = std::chrono::system_clock::now();
      for (std::size_t i = 0; i < n_repetition; ++i)
      {
        // Create patch
        patch.create_subdofmap(i_node);

        // Reinitialise patch-data
        patch_data.reinitialisation(patch.type(), patch.ncells());

        // Explicit setp equilibration
        equilibrate_flux_semiexplt<T, id_flux_order, 0>(
            mesh->geometry(), patch, patch_data, problem_data, kernel_data,
            kernel_fluxmin, kernel_fluxmin_l, kernel_weaksym, n_repetition);
      }
      end = std::chrono::system_clock::now();

      std::chrono::duration<double> total_eqlb_explicit = end - begin;

      //  --- Time equilibration: Assembly flux minimisation
      begin = std::chrono::system_clock::now();
      for (std::size_t i = 0; i < n_repetition; ++i)
      {
        // Create patch
        patch.create_subdofmap(i_node);

        // Reinitialise patch-data
        patch_data.reinitialisation(patch.type(), patch.ncells());

        // Explicit setp equilibration
        equilibrate_flux_semiexplt<T, id_flux_order, 1>(
            mesh->geometry(), patch, patch_data, problem_data, kernel_data,
            kernel_fluxmin, kernel_fluxmin_l, kernel_weaksym, n_repetition);
      }
      end = std::chrono::system_clock::now();

      std::chrono::duration<double> total_eqlb_fluxmin_assembly = end - begin;

      //  --- Time equilibration: Assembly flux minimisation
      begin = std::chrono::system_clock::now();
      for (std::size_t i = 0; i < n_repetition; ++i)
      {
        // Create patch
        patch.create_subdofmap(i_node);

        // Reinitialise patch-data
        patch_data.reinitialisation(patch.type(), patch.ncells());

        // Explicit setp equilibration
        equilibrate_flux_semiexplt<T, id_flux_order, 2>(
            mesh->geometry(), patch, patch_data, problem_data, kernel_data,
            kernel_fluxmin, kernel_fluxmin_l, kernel_weaksym, n_repetition);
      }
      end = std::chrono::system_clock::now();

      std::chrono::duration<double> total_eqlb_fluxmin_solve = end - begin;

      //  --- Time equilibration: Assembly imposition weak symmetry
      begin = std::chrono::system_clock::now();
      for (std::size_t i = 0; i < n_repetition; ++i)
      {
        // Create patch
        patch.create_subdofmap(i_node);

        // Reinitialise patch-data
        patch_data.reinitialisation(patch.type(), patch.ncells());

        // Explicit setp equilibration
        equilibrate_flux_semiexplt<T, id_flux_order, 3>(
            mesh->geometry(), patch, patch_data, problem_data, kernel_data,
            kernel_fluxmin, kernel_fluxmin_l, kernel_weaksym, n_repetition);
      }
      end = std::chrono::system_clock::now();

      std::chrono::duration<double> total_eqlb_weaksym_assembly = end - begin;

      //  --- Time equilibration: Solve imposition weak symmetry
      begin = std::chrono::system_clock::now();
      for (std::size_t i = 0; i < n_repetition; ++i)
      {
        // Create patch
        patch.create_subdofmap(i_node);

        // Reinitialise patch-data
        patch_data.reinitialisation(patch.type(), patch.ncells());

        // Explicit setp equilibration
        equilibrate_flux_semiexplt<T, id_flux_order, 4>(
            mesh->geometry(), patch, patch_data, problem_data, kernel_data,
            kernel_fluxmin, kernel_fluxmin_l, kernel_weaksym, n_repetition);
      }
      end = std::chrono::system_clock::now();

      std::chrono::duration<double> total_eqlb_weaksym_solve = end - begin;

      // --- Output timings
      double timing_eqlb_explicit
          = total_eqlb_explicit.count() - total_create_patch.count();
      double timing_eqlb_fluxmin_assembly
          = total_eqlb_fluxmin_assembly.count() - total_eqlb_explicit.count();
      double timing_eqlb_fluxmin_solve = total_eqlb_fluxmin_solve.count()
                                         - total_eqlb_fluxmin_assembly.count();
      double timing_weaksym_assembly = total_eqlb_weaksym_assembly.count()
                                       - total_eqlb_fluxmin_solve.count();
      double timing_weaksym_solve = total_eqlb_weaksym_solve.count()
                                    - total_eqlb_weaksym_assembly.count();

      std::cout << "Timings for patch-size " << patch.ncells() << std::endl;
      std::cout << "Patch creation: " << total_create_patch.count()
                << std::endl;
      std::cout << "Eqlb. - explicit setp: " << timing_eqlb_explicit
                << std::endl;
      std::cout << "Eqlb. - assembly flux minimisation: "
                << timing_eqlb_fluxmin_assembly << std::endl;
      std::cout << "Eqlb. - solve flux minimisation: "
                << timing_eqlb_fluxmin_solve << std::endl;
      std::cout << "Weak sym. - assembly: " << timing_weaksym_assembly
                << std::endl;
      std::cout << "Weak sym. - solve: " << timing_weaksym_solve << std::endl;
      std::cout << "Total: " << total_eqlb_weaksym_solve.count() << std::endl;
    }
  }
  else
  {
    // Initialise patch
    PatchFluxCstm<T, id_flux_order> patch = PatchFluxCstm<T, id_flux_order>(
        mesh, problem_data.facet_type(), 0,
        problem_data.node_on_essnt_boundary_stress(),
        problem_data.fspace_flux_hdiv(), problem_data.fspace_flux_dg(),
        basix_element_rhscg, false);

    // Initialise storage for equilibration
    PatchDataCstm<T, id_flux_order> patch_data
        = PatchDataCstm<T, id_flux_order>(patch, kernel_data.nipoints_facet(),
                                          false);

    // --- Time patch creation
    std::vector<std::int32_t> nodes{2, 4};

    for (std::int32_t i_node : nodes)
    {
      begin = std::chrono::system_clock::now();
      for (std::size_t i = 0; i < n_repetition; ++i)
      {
        // Create patch
        patch.create_subdofmap(i_node);

        // Reinitialise patch-data
        patch_data.reinitialisation(patch.type(), patch.ncells());
      }
      end = std::chrono::system_clock::now();

      std::chrono::duration<double> total_create_patch = end - begin;

      //  --- Time equilibration: Explicit setp
      begin = std::chrono::system_clock::now();
      for (std::size_t i = 0; i < n_repetition; ++i)
      {
        // Create patch
        patch.create_subdofmap(i_node);

        // Reinitialise patch-data
        patch_data.reinitialisation(patch.type(), patch.ncells());

        // Explicit setp equilibration
        equilibrate_flux_semiexplt<T, id_flux_order, 0>(
            mesh->geometry(), patch, patch_data, problem_data, kernel_data,
            kernel_fluxmin, kernel_fluxmin_l, n_repetition);
      }
      end = std::chrono::system_clock::now();

      std::chrono::duration<double> total_eqlb_explicit = end - begin;

      //  --- Time equilibration: Assembly flux minimisation
      begin = std::chrono::system_clock::now();
      for (std::size_t i = 0; i < n_repetition; ++i)
      {
        // Create patch
        patch.create_subdofmap(i_node);

        // Reinitialise patch-data
        patch_data.reinitialisation(patch.type(), patch.ncells());

        // Explicit setp equilibration
        equilibrate_flux_semiexplt<T, id_flux_order, 1>(
            mesh->geometry(), patch, patch_data, problem_data, kernel_data,
            kernel_fluxmin, kernel_fluxmin_l, n_repetition);
      }
      end = std::chrono::system_clock::now();

      std::chrono::duration<double> total_eqlb_fluxmin_assembly = end - begin;

      //  --- Time equilibration: Assembly flux minimisation
      begin = std::chrono::system_clock::now();
      for (std::size_t i = 0; i < n_repetition; ++i)
      {
        // Create patch
        patch.create_subdofmap(i_node);

        // Reinitialise patch-data
        patch_data.reinitialisation(patch.type(), patch.ncells());

        // Explicit setp equilibration
        equilibrate_flux_semiexplt<T, id_flux_order, 2>(
            mesh->geometry(), patch, patch_data, problem_data, kernel_data,
            kernel_fluxmin, kernel_fluxmin_l, n_repetition);
      }
      end = std::chrono::system_clock::now();

      std::chrono::duration<double> total_eqlb_fluxmin_solve = end - begin;

      // --- Output timings
      double timing_eqlb_explicit
          = total_eqlb_explicit.count() - total_create_patch.count();
      double timing_eqlb_fluxmin_assembly
          = total_eqlb_fluxmin_assembly.count() - total_eqlb_explicit.count();
      double timing_eqlb_fluxmin_solve = total_eqlb_fluxmin_solve.count()
                                         - total_eqlb_fluxmin_assembly.count();

      std::cout << "Timings for patch-size " << patch.ncells() << std::endl;
      std::cout << "Patch creation: " << total_create_patch.count()
                << std::endl;
      std::cout << "Eqlb. - explicit setp: " << timing_eqlb_explicit
                << std::endl;
      std::cout << "Eqlb. - assembly flux minimisation: "
                << timing_eqlb_fluxmin_assembly << std::endl;
      std::cout << "Eqlb. - solve flux minimisation: "
                << timing_eqlb_fluxmin_solve << std::endl;
      std::cout << "Eqlb. - total: " << total_eqlb_fluxmin_solve.count()
                << std::endl;
    }
  }
}

/// Execute flux calculation based on H(div) conforming equilibration
///
/// Equilibration based on local minimization problems. Weak forms
/// discretized using ufl.
///
/// @param a                   The bilinears form to assemble
/// @param l                   The linar form to assemble
/// @param fct_esntbound_prime Facets of essential BCs of primal problem
/// @param fct_esntbound_flux  Facets of essential BCs on flux field
/// @param bcs_flux            Essential boundary conditions for the flux
/// @param flux                Function that holds the reconstructed flux
template <typename T>
void reconstruct_fluxes_ev(
    const fem::Form<T>& a, const fem::Form<T>& l_pen,
    const std::vector<std::shared_ptr<const fem::Form<T>>>& l,
    std::vector<std::shared_ptr<fem::Function<T>>>& flux_hdiv,
    std::shared_ptr<BoundaryData<T>> boundary_data)
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
  ProblemDataFluxEV<T> problem_data
      = ProblemDataFluxEV<T>(flux_hdiv, l, boundary_data);

  /* Call equilibration */
  reconstruct_fluxes_patch<T>(a, l_pen, problem_data);
}

/// Execute flux calculation based on H(div) conforming equilibration
///
/// Equilibration based on semi-explicit formulas and small, unconstrained
/// minimisation problems.
///
/// @param flux_hdiv           Function that holds the reconstructed flux
/// @param flux_dg             Function that holds the projected primal flux
/// @param rhs_dg              Function that holds the projected rhs
/// @param fct_esntbound_prime Facets of essential BCs of primal problem
/// @param fct_esntbound_flux  Facets of essential BCs on flux field
/// @param bcs_flux            Essential boundary conditions for the flux
template <typename T>
void reconstruct_fluxes_cstm(
    std::vector<std::shared_ptr<fem::Function<T>>>& flux_hdiv,
    std::vector<std::shared_ptr<fem::Function<T>>>& flux_dg,
    std::vector<std::shared_ptr<fem::Function<T>>>& rhs_dg,
    std::shared_ptr<BoundaryData<T>> boundary_data,
    const bool reconstruct_stress)
{
  // Input size and polynomial degrees
  const int n_rhs = rhs_dg.size();
  const int n_flux_hdiv = flux_hdiv.size();
  const int n_flux_dg = flux_dg.size();
  const int n_bcs = boundary_data->num_rhs();

  const int order_flux
      = flux_hdiv[0]->function_space()->element()->basix_element().degree();
  const int degree_flux_dg
      = flux_dg[0]->function_space()->element()->basix_element().degree();
  const int degree_rhs
      = rhs_dg[0]->function_space()->element()->basix_element().degree();

  // Check input
  if (n_rhs != n_bcs || n_rhs != n_flux_hdiv || n_rhs != n_flux_dg)
  {
    throw std::runtime_error("Equilibration: Input sizes does not match");
  }

  if (degree_rhs > (order_flux - 1) || degree_flux_dg > degree_rhs)
  {
    throw std::runtime_error(
        "Equilibration: Wrong polynomial degree of the projected RHS");
  }

  if (degree_flux_dg != degree_rhs)
  {
    throw std::runtime_error(
        "Equilibration: Degrees of projected flux and RHS have to match");
  }

  // Additional checks for stress equilibration
  if (reconstruct_stress)
  {
    if (n_rhs < flux_hdiv[0]->function_space()->mesh()->geometry().dim())
    {
      throw std::runtime_error(
          "Stress equilibration: Specify all rows of stress tensor");
    }

    if (n_flux_hdiv < 2)
    {
      throw std::runtime_error("Stress equilibration: RT_k with k>1 required!");
    }
  }

  /* Set problem data */
  ProblemDataFluxCstm<T> problem_data
      = ProblemDataFluxCstm<T>(flux_hdiv, flux_dg, rhs_dg, boundary_data);

  /* Call equilibration */
  if (reconstruct_stress)
  {
    if (order_flux == 1)
    {
      reconstruct_fluxes_patch<T, 1, true>(problem_data);
    }
    else if (order_flux == 2)
    {
      reconstruct_fluxes_patch<T, 2, true>(problem_data);
    }
    else
    {
      reconstruct_fluxes_patch<T, 3, true>(problem_data);
    }
  }
  else
  {
    if (order_flux == 1)
    {
      reconstruct_fluxes_patch<T, 1, false>(problem_data);
    }
    else if (order_flux == 2)
    {
      reconstruct_fluxes_patch<T, 2, false>(problem_data);
    }
    else
    {
      reconstruct_fluxes_patch<T, 3, false>(problem_data);
    }
  }
}

} // namespace dolfinx_eqlb