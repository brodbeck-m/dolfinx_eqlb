// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "FluxBC.hpp"
#include "KernelDataBC.hpp"
#include "equilibration.hpp"
#include "mdspan.hpp"

#include <Eigen/Dense>
#include <basix/e-lagrange.h>
#include <basix/element-families.h>
#include <basix/finite-element.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/la/Vector.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/cell_types.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <memory>
#include <span>
#include <stdexcept>
#include <tuple>
#include <vector>

using namespace dolfinx;

namespace dolfinx_eqlb::base
{
// ------------------------------------------------------------------------------
/* Projection kernel */
// ------------------------------------------------------------------------------
/// Evaluate the projection kernel
///
/// For non-polynomial boundary expression or expression with to large degree
/// the boundary data hev to be projected onto the flux space. This routine
/// assembles mass-matrix as well as RHS required for the projection based on
/// the exact normal trace of the flux.
///
/// @param[in] ntrace_flux_boundary The exact flux normal-trace on the boundary
/// @param[in] facet_normal         The facet normal
/// @param[in] phi                  The (mapped) ansatz function of the flux
///                                 space
/// @param[in] weights              The quadrature weights
/// @param[in] detJ                 The determinant of the Jacobian of the
///                                 current element
/// @param[in, out] Ae              The mass matrix
/// @param[in, out] Le              The RHS of the projection
template <dolfinx::scalar T, std::floating_point U>
void boundary_projection_kernel(
    std::span<const U> ntrace_flux_boundary, std::span<const U> facet_normal,
    mdspan_t<U, 3> phi, std::span<const U> weights, const U detJ,
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& A_e,
    Eigen::Matrix<T, Eigen::Dynamic, 1>& L_e)
{
  // The spacial dimension
  const int gdim = phi.extent(2);

  // The number of quadrature points
  const int num_qpoints = weights.size();

  // The number of shape functions per facet
  const int ndofs_per_fct = phi.extent(1);

  // Initialise tangent arrays
  A_e.setZero();
  L_e.setZero();

  // Initialise normal-trace of flux
  double ntrace_phi_i, ntrace_phi_j;

  // Quadrature loop
  for (std::size_t iq = 0; iq < num_qpoints; ++iq)
  {
    for (std::size_t i = 0; i < ndofs_per_fct; ++i)
    {
      // Normal trace of phi_i
      ntrace_phi_i = 0.0;

      for (std::size_t k = 0; k < gdim; ++k)
      {
        ntrace_phi_i += phi(iq, i, k) * facet_normal[k];
      }

      // Set RHS
      L_e(i) += ntrace_phi_i * ntrace_flux_boundary[iq] * weights[iq];

      for (std::size_t j = i; j < ndofs_per_fct; ++j)
      {
        // Normal trace of phi_i
        ntrace_phi_j = 0.0;

        for (std::size_t k = 0; k < gdim; ++k)
        {
          ntrace_phi_j += phi(iq, j, k) * facet_normal[k];
        }

        // Set entry mass matrix
        A_e(i, j) += ntrace_phi_i * ntrace_phi_j * weights[iq];
      }
    }
  }

  // Add symmetric entries of mass-matrix
  for (std::size_t i = 1; i < ndofs_per_fct; ++i)
  {
    for (std::size_t j = 0; j < i; ++j)
    {
      A_e(i, j) += A_e(j, i);
    }
  }
}
// ------------------------------------------------------------------------------

// ------------------------------------------------------------------------------
/* BoundaryData */
// ------------------------------------------------------------------------------
template <dolfinx::scalar T, std::floating_point U>
class BoundaryData
{

  // TODO - Implement general casting between T and U in interpolation routines!

public:
  /// Storage and handling of boundary data
  ///
  /// Takes over a list of FluxBCs and evaluates therefrom the flux DOFs (within
  /// RT_k, k>=1) on the Neumann boundary of the primal problem. If the degree
  /// of the boundary expression is higher than k-1, a facet-local projection
  /// into the DG_(k-1) is performed.
  /// For the following equilibration the patch-wise boundary conditions are
  /// provided by evaluation the boundary DOFs from bflux x hat_function.
  ///
  /// @param list_bcs            Vector of Vectors with FluxBCs (one per RHS)
  /// @param boundary_flux       Vector with FE-Functions (one per RHS) within
  ///                            which the boundary conditions are stored
  /// @param fct_esntbound_prime List of facets, where essential BCs are applied
  ///                            on the primal problem
  /// @param problem_type        Type of the equilibration problem
  BoundaryData(
      std::vector<std::vector<std::shared_ptr<FluxBC<T, U>>>>& list_bcs,
      std::vector<std::shared_ptr<fem::Function<T, U>>>& boundary_fluxes,
      std::shared_ptr<const fem::FunctionSpace<U>> V,
      const std::vector<std::vector<std::int32_t>>& fct_esntbound_prime,
      KernelDataBC<T, U>& kernel_data, const ProblemType problem_type);

  /// Update the boundary values
  ///
  /// Update flux values on the boundary. The calculated values are then used
  /// for evaluating the BCs for each individual patch problem.
  ///
  /// @param time_functions List of time-dependent functions
  ///                       (on for each subspace)
  void
  update(std::vector<std::shared_ptr<const fem::Constant<T>>>& time_functions);

  /// Calculate the boundary DOFs for a patch problem
  ///
  /// Evaluates RT-DOFs on boundary_flux x hat_function. Before the evaluation
  /// of the boundary DOFs the required mappings are calculated. This function
  /// should be called once for alle equilibrated RHS.
  ///
  /// @param[in] bound_fcts      The boundary-facets of the (boundary-) patch
  /// @param[in] patchnode_local The element-local id of the patch-central node
  void calculate_patch_bc(std::span<const std::int32_t> bound_fcts,
                          std::span<const std::int8_t> patchnode_local);

  /// Calculate the boundary DOFs for a patch problem
  ///
  /// Evaluates RT-DOFs on boundary_flux x hat_function. The required mappings
  /// have to be provided. This function has to be called separately for each
  /// equilibrated RHS.
  ///
  /// @param rhs_i           The Id of the equilibrated RHS
  /// @param bound_fcts      The boundary-facets of the (boundary-) patch
  /// @param patchnode_local The element-local id of the patch-central node
  /// @param J               The Jacobian of the mapping function
  /// @param detJ            The determinant of the Jacobian
  /// @param K               The inverse of the Jacobian
  void calculate_patch_bc(const int rhs_i, const std::int32_t bound_fcts,
                          const std::int8_t patchnode_local,
                          mdspan_t<const U, 2> J, const U detJ,
                          mdspan_t<const U, 2> K);

  /* Getter methods: general */
  /// Get number of considered RHS
  /// @return Number of considered RHS
  int num_rhs() const { return _num_rhs; }

  /* Getter methods */
  /// Get list of facet types
  /// (marked with respect to FacetType)
  /// @param[in] rhs_i Index of RHS
  /// @return List of facet types (sorted by facet-ids)
  std::span<std::int8_t> facet_type(const int rhs_i)
  {
    return std::span<std::int8_t>(_facet_type.data() + _offset_fctdata[rhs_i],
                                  _offset_fctdata[rhs_i + 1]
                                      - _offset_fctdata[rhs_i]);
  }

  /// Get list of facet types
  /// (marked with respect to FacetType)
  /// @return List of all facet types (sorted by facet-ids)
  mdspan_t<const std::int8_t, 2> facet_type()
  {
    const std::int32_t num_fcts
        = _V->mesh()->topology()->index_map(_gdim - 1)->size_local();

    return mdspan_t<const std::int8_t, 2>(_facet_type.data(), _num_rhs,
                                          num_fcts);
  }

  /// Marker if mesh-node is on essential boundary of the stress field
  ///
  /// Essential boundary:
  ///    - Node resp. patch around node in on boundary of the domain
  ///    - All stress rows have pure neumann BCs on patch
  ///
  /// @return List of markers for all nodes
  std::span<const std::int8_t> node_on_essnt_boundary_stress() const
  {
    return std::span<const std::int8_t>(_pnt_on_esnt_boundary.data(),
                                        _pnt_on_esnt_boundary.size());
  }

  /// Get list of boundary markers
  /// @param[in] rhs_i Index of RHS
  /// @return List of boundary markers (sorted by DOF-ids)
  std::span<std::int8_t> boundary_markers(const int rhs_i)
  {
    return std::span<std::int8_t>(
        _boundary_markers.data() + _offset_dofdata[rhs_i],
        _offset_dofdata[rhs_i + 1] - _offset_dofdata[rhs_i]);
  }

  /// Get list of boundary markers
  /// @param[in] rhs_i Index of RHS
  /// @return List of boundary markers (sorted by DOF-ids)
  std::span<const std::int8_t> boundary_markers(const int rhs_i) const
  {
    return std::span<const std::int8_t>(
        _boundary_markers.data() + _offset_dofdata[rhs_i],
        _offset_dofdata[rhs_i + 1] - _offset_dofdata[rhs_i]);
  }

  /// Get list of boundary values
  /// @param[in] rhs_i Index of RHS
  /// @return List of boundary values (sorted by DOF-ids)
  std::span<T> boundary_values(const int rhs_i)
  {
    return std::span<T>(_boundary_values.data() + _offset_dofdata[rhs_i],
                        _offset_dofdata[rhs_i + 1] - _offset_dofdata[rhs_i]);
  }

  /// Get list of boundary values
  /// @param[in] rhs_i Index of RHS
  /// @return List of boundary values (sorted by DOF-ids)
  std::span<const T> boundary_values(const int rhs_i) const
  {
    return std::span<const T>(_boundary_values.data() + _offset_dofdata[rhs_i],
                              _offset_dofdata[rhs_i + 1]
                                  - _offset_dofdata[rhs_i]);
  }

protected:
  /// Calculate DOF ids on boundary facet
  /// @param[in] fct Facet id (on current process)
  /// @return List of DOFs on facet
  std::vector<std::int32_t> boundary_dofs(const std::int32_t fct)
  {
    // Get cell adjacent to facet
    const std::int32_t cell = _fct_to_cell->links(fct)[0];

    return boundary_dofs(cell, fct);
  }

  /// Calculate DOF ids on boundary facet
  /// @param[in] cell Cell id (on current process)
  /// @param[in] fct Facet id (on current process)
  /// @return List of DOFs on facet
  std::vector<std::int32_t> boundary_dofs(const std::int32_t cell,
                                          const std::int32_t fct)
  {
    // Cell-local facet id
    const std::int8_t fct_loc = _local_fct_id[fct];

    // Initialise storage for DOFs
    std::vector<std::int32_t> storage(_ndofs_per_fct);

    // Calculate DOFs on facet
    boundary_dofs(cell, fct_loc, storage);

    return std::move(storage);
  }

  /// Calculate DOF ids on boundary facet
  /// (Use this routine in performance relevant situations)
  /// @param[in] cell Cell id (on current process)
  /// @param[in] fct_loc Cell-local facet id
  /// @param[in,out] boundary_dofs Storage for DOFs on facet
  void boundary_dofs(const std::int32_t cell, const std::int8_t fct_loc,
                     std::span<std::int32_t> boundary_dofs);

  /// Calculate the boundary values
  ///
  /// Evaluates RT-DOFs on a subset of the boundary. The calculated values
  /// are then used for evaluating the BCs for each individual patch problem.
  ///
  /// @param initialise_boundary_values If True, all BCs (not just the
  ///                                   time-dependet ones) are evaluated
  void evaluate_boundary_flux(const bool initialise_boundary_values);

  /* Variable definitions */
  // The FunctionSpace
  std::shared_ptr<const fem::FunctionSpace<U>> _V;
  const bool _flux_is_discontinous;

  // The boundary conditions
  std::vector<std::vector<std::shared_ptr<FluxBC<T, U>>>> _bcs;
  std::vector<std::shared_ptr<fem::Function<T, U>>> _boundary_fluxes;

  // The KernelData
  std::shared_ptr<KernelDataBC<T, U>> _kernel_data;

  // --- Counters
  // The number of considered RHS
  const int _num_rhs;

  // The geometric dimension
  const int _gdim;

  // The flux-space
  const int _flux_degree, _ndofs_per_cell, _ndofs_per_fct;

  // --- Data per DOF
  // The boundary values
  std::vector<T> _boundary_values;

  // The boundary markers
  std::vector<std::int8_t> _boundary_markers;

  // Offset for different RHS
  std::vector<std::int32_t> _offset_dofdata;

  // --- Data per node
  // The boundary markers (patch has purely essential BCs)
  std::vector<std::int8_t> _pnt_on_esnt_boundary;

  // --- Data per facet
  // The boundary DOFs (per facet)
  std::vector<std::int8_t> _local_fct_id;

  // Facet types
  std::vector<std::int8_t> _facet_type;

  // Offset for different RHS
  std::vector<std::int32_t> _offset_fctdata;

  // --- Data
  // Mesh conductivities
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> _fct_to_cell;

  // Geometric mapping
  std::array<U, 9> _data_J, _data_K;
  std::array<U, 18> _detJ_scratch;
};
// ------------------------------------------------------------------------------

} // namespace dolfinx_eqlb::base