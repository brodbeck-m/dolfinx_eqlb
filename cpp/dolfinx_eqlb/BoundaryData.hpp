// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "FluxBC.hpp"
#include "KernelData.hpp"
#include "assemble_projection_boundary.hpp"
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Sparse"

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
#include <dolfinx_eqlb/base/Patch.hpp>
#include <dolfinx_eqlb/base/QuadratureRule.hpp>
#include <dolfinx_eqlb/base/mdspan.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <memory>
#include <span>
#include <tuple>
#include <vector>

using namespace dolfinx;

namespace dolfinx_eqlb
{
template <typename T>
class BoundaryData
{
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
  /// @param V_flux_hdiv         The flux FunctionSpace (a sub-space, if
  ///                            equilibration is based on a mixed problem)
  /// @param rtflux_is_custom    Id if the used RT-space is created from the
  ///                            modifed RT definition
  /// @param quadrature_degree   Degree for surface quadrature
  /// @param fct_esntbound_prime List of facets, where essential BCs are applied
  ///                            on the primal problem
  BoundaryData(
      std::vector<std::vector<std::shared_ptr<FluxBC<T>>>>& list_bcs,
      std::vector<std::shared_ptr<fem::Function<T>>>& boundary_flux,
      std::shared_ptr<const fem::FunctionSpace> V_flux_hdiv,
      bool rtflux_is_custom, int quadrature_degree,
      const std::vector<std::vector<std::int32_t>>& fct_esntbound_prime,
      const bool reconstruct_stress);

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
                          base::mdspan_t<const double, 2> J, const double detJ,
                          base::mdspan_t<const double, 2> K);

  /* Getter methods: general */
  /// Get number of considered RHS
  /// @return Number of considered RHS
  int num_rhs() const { return _num_rhs; }

  /* Getter methods */
  /// Get list of facet types
  /// (marked with respect to PatchFacetType)
  /// @param[in] rhs_i Index of RHS
  /// @return List of facet types (sorted by facet-ids)
  std::span<std::int8_t> facet_type(const int rhs_i)
  {
    return std::span<std::int8_t>(_facet_type.data() + _offset_fctdata[rhs_i],
                                  _offset_fctdata[rhs_i + 1]
                                      - _offset_fctdata[rhs_i]);
  }

  /// Get list of facet types
  /// (marked with respect to PatchFacetType)
  /// @return List of all facet types (sorted by facet-ids)
  base::mdspan_t<const std::int8_t, 2> facet_type()
  {
    return base::mdspan_t<const std::int8_t, 2>(_facet_type.data(), _num_rhs,
                                                _num_fcts);
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

  /* Variable definitions */
  // The boundary conditions
  std::vector<std::shared_ptr<const la::Vector<T>>> _x_boundary_flux;

  // --- Counters
  // The number of considered RHS
  const int _num_rhs;

  // The geometric dimension
  const int _gdim;

  // The number of facets (on current processor)
  const std::int32_t _num_fcts;

  // The number of boundary facets (per subproblem)
  std::vector<std::int32_t> _num_bcfcts;

  // The number of facets per cell
  const int _nfcts_per_cell;

  // The degree of the flux-space (Hdiv)
  const int _flux_degree;

  // The number of DOFs per cell-facet
  const int _ndofs_per_cell;

  // The number of DOFs per cell-facet
  const int _ndofs_per_fct;

  // The number of DOFs (on current processor)
  const std::int32_t _num_dofs;

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

  // --- Data for determination of boundary DOFs
  // The flux function space
  std::shared_ptr<const fem::FunctionSpace> _V_flux_hdiv;

  // Id if flux is discontinuous
  const double _flux_is_discontinous;

  // Mesh conductivities
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> _fct_to_cell;

  // --- Data for projection/ interpolation
  // Surface quadrature kernel
  base::QuadratureRule _quadrature_rule;
  KernelDataBC<T> _kernel_data;

  // Geometric mapping
  std::array<double, 9> _data_J, _data_K;
  std::array<double, 18> _detJ_scratch;
};
} // namespace dolfinx_eqlb