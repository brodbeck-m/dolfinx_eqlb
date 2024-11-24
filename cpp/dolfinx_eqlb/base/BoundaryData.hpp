// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "FluxBC.hpp"
#include "KernelData.hpp"
#include "Patch.hpp"
#include "QuadratureRule.hpp"
#include "assemble_bcs_impl.hpp"
#include "eigen3/Eigen/Dense"
// #include "eigen3/Eigen/Sparse"
#include "mdspan.hpp"

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
#include <tuple>
#include <vector>

using namespace dolfinx;

namespace dolfinx_eqlb::base
{
// ------------------------------------------------------------------------------
/* KernelDataBC */
// ------------------------------------------------------------------------------
template <typename T>
class KernelDataBC : public KernelData<T>
{
public:
  /// Kernel data for calculation of patch boundary-conditions
  ///
  /// Holds the required surface-quadrature rule, calculates required mappings,
  /// interpolates from a flux function and calculates DOFs of a flux-function.
  ///
  /// @param[in] mesh                 The mesh
  /// @param[in] quadrature_rule_fct  The quadrature rule on the cell facets
  /// @param[in] basix_element_fluxpw The basix-element for the H(div) flux
  /// @param[in] basix_element_rhs    The basix-element for RHS and proj. flux
  KernelDataBC(std::shared_ptr<const mesh::Mesh> mesh,
               std::shared_ptr<const QuadratureRule> quadrature_rule_fct,
               std::shared_ptr<const fem::FiniteElement> element_flux_hdiv,
               const int nfluxdofs_per_fct, const int nfluxdofs_cell,
               const bool flux_is_custom);

  /* Tabulate/map shape-function sof the flux-space */

  /// Tabulates a basix_element at the surface-quadrature points
  /// @param[in] basix_element The Basix element
  /// @param[in,out] storage   Stoarge vector for the tabulates shape-functions
  /// @param[out] shape_vector The shape for creating an mdspan of the tabulated
  /// functions
  std::array<std::size_t, 5>
  shapefunctions_flux_qpoints(const basix::FiniteElement& basix_element,
                              std::vector<double>& storage)
  {
    return this->tabulate_basis(basix_element,
                                this->_quadrature_rule[0]->points(), storage,
                                false, false);
  }

  /// Map flux-functions from refrence to current cell
  ///
  /// Aplies the contra-variant Piola mapping to the shape-functions of the
  /// flux. The here performed mapping is restriced to the shape-functions of
  /// one cell facet. All other functions will be neglected.
  ///
  /// @param[in] lfct_id      The cell-local Id of the facet
  /// @param[in, out] phi_cur The shape-function on the current cell
  /// @param[in] phi_ref      The shape-functions on the reference cell
  /// @param[in] J            The Jacobina
  /// @param[in] detJ         The determinant of the Jacobian
  void map_shapefunctions_flux(std::int8_t lfct_id, mdspan_t<double, 3> phi_cur,
                               mdspan_t<const double, 5> phi_ref,
                               mdspan_t<const double, 2> J, double detJ);

  /* Calculate flux DOFs based on different inputs */

  /// Calculates a flux (vector) from a given normal-trace on a cell-facet
  ///
  /// Takes over a list of normal-traces at (multiple) points and return the
  /// flux-vectors calculates from facet_normal x trace, where the facte_normal
  /// has a magnitude of 1.
  ///
  /// @param[in] flux_ntrace_cur The vector of flux normal-traces within the
  ///                            current cell
  /// @param[in] lfct_id         The cell-local Id of the facet
  /// @param[in] K               The inverse of the Jacobian
  /// @param[out] flux_vector    A List of flux-vectors recovered from the
  ///                            normal-trace
  mdspan_t<const T, 2> normaltrace_to_flux(std::span<const T> flux_ntrace_cur,
                                           std::int8_t lfct_id,
                                           mdspan_t<const double, 2> K)
  {
    // Perform calculation
    normaltrace_to_vector(flux_ntrace_cur, lfct_id, K);

    return mdspan_t<const T, 2>(_flux_scratch_data.data(),
                                flux_ntrace_cur.size(),
                                (std::size_t)this->_gdim);
  }

  /// Calculates flux DOFs based on the normal-trace
  ///
  /// Claculates flux DOFs on a cell facet based on given normal-trace,
  /// evaluated at the required interpolation points of the flux-space.
  ///
  /// @param[in] flux_ntrace_cur The flux normal-trace on the interpolation
  ///                            points
  /// @param[in, out] flux_dofs  The calculated flux DOFs on the facet
  /// @param[in] lfct_id         The (cell-local) Id of the facet
  /// @param[in] J               The Jacobi matrix of the mapping function
  /// @param[in] detJ            The determinant of the Jacobi matrix
  /// @param[in] K               The inverse of the Jacobi matrix
  void interpolate_flux(std::span<const T> flux_ntrace_cur,
                        std::span<T> flux_dofs, std::int8_t lfct_id,
                        mdspan_t<const double, 2> J, double detJ,
                        mdspan_t<const double, 2> K);

  /// Calculates boundary DOFs for a patch problem
  ///
  /// Calculates the boundary DOFs for a patrch problem from the boundary DOFs
  /// where the hat-function is neglected by inteprolating boundary_function x
  /// hat_function.
  ///
  /// @param[in] flux_dofs_bc         The boundary DOFs without consideration of
  ///                                 the hat-function
  /// @param[in] cell_id              Vector with Ids of the boundary cells
  /// @param[in] lfct_id              Vector with the (cell-local) Ids of the
  ///                                 boundary facets
  /// @param[in] hat_id               Vector of (cell-local) Ids of the
  ///                                 patch-central node
  /// @param[in] cell_info            The informatios required for DOF
  ///                                 transformations on a cell
  /// @param[in] J                    The Jacobi matrix of the mapping function
  /// @param[in] detJ                 The determinant of the Jacobi matrix
  /// @param[in] K                    The inverse of the Jacobi matrix
  /// @param[out] flux_dofs_patch     The boundary DOFs for a patch-problem
  std::vector<T> interpolate_flux(std::span<const T> flux_dofs_bc,
                                  std::int32_t cell_id, std::int8_t lfct_id,
                                  std::int8_t hat_id,
                                  std::span<const std::uint32_t> cell_info,
                                  mdspan_t<const double, 2> J, double detJ,
                                  mdspan_t<const double, 2> K)
  {
    // Initialise storage
    std::vector<T> flux_dofs_patch(flux_dofs_bc.size());

    // Interpolaate flux
    interpolate_flux(flux_dofs_bc, flux_dofs_patch, cell_id, lfct_id, hat_id,
                     cell_info, J, detJ, K);

    return std::move(flux_dofs_patch);
  }

  /// Calculates boundary DOFs for a patch problem
  ///
  /// Calculates the boundary DOFs for a patrch problem from the boundary DOFs
  /// where the hat-function is neglected by inteprolating boundary_function x
  /// hat_function.
  ///
  /// This routine should be called where performace is relevant!
  ///
  /// @param[in] flux_dofs_bc         The boundary DOFs without consideration of
  ///                                 the hat-function
  /// @param[in, out] flux_dofs_patch The boundary DOFs for a patch-problem
  /// @param[in] cell_id              Vector with Ids of the boundary cells
  /// @param[in] lfct_id              Vector with the (cell-local) Ids of the
  ///                                 boundary facets
  /// @param[in] hat_id               Vector of (cell-local) Ids of the
  ///                                 patch-central node
  /// @param[in] cell_info            The informatios required for DOF
  ///                                 transformations on a cell
  /// @param[in] J                    The Jacobi matrix of the mapping function
  /// @param[in] detJ                 The determinant of the Jacobi matrix
  /// @param[in] K                    The inverse of the Jacobi matrix
  void interpolate_flux(std::span<const T> flux_dofs_bc,
                        std::span<T> flux_dofs_patch, std::int32_t cell_id,
                        std::int8_t lfct_id, std::int8_t hat_id,
                        std::span<const std::uint32_t> cell_info,
                        mdspan_t<const double, 2> J, double detJ,
                        mdspan_t<const double, 2> K);

  /// Calculates flux-DOFs from flux-vector at the interpolation points
  ///
  /// Calculates flux DOFs on one cell facte from the flux-values (on the
  /// current cell) on the interpolation points of the flux-space.
  ///
  /// @param[in] flux_cur             The flux (current cell) on the
  ///                                 interpolation points
  /// @param[in, out] flux_dofs_patch The calculated flux DOFs
  /// @param[in] lfct_id              The (cell-local) Id of the facet on which
  ///                                 the DOFs are calculated
  /// @param[in] J                    The Jacobi matrix of the mapping function
  /// @param[in] detJ                 The determinant of the Jacobi matrix
  /// @param[in] K                    The inverse of the Jacobi matrix
  void interpolate_flux(mdspan_t<const T, 2> flux_cur,
                        std::span<T> flux_dofs_patch, std::int8_t lfct_id,
                        mdspan_t<const double, 2> J, double detJ,
                        mdspan_t<const double, 2> K);

  /* Getter methods: Interpolation */

  /// Extract number of interpolation points
  /// @return Number of interpolation points
  int num_interpolation_points() const { return _nipoints; }

  /// Extract number of interpolation points per facet
  /// @return Number of interpolation points
  int num_interpolation_points_per_facet() const { return _nipoints_per_fct; }

protected:
  void normaltrace_to_vector(std::span<const T> normaltrace_cur,
                             std::int8_t lfct_id, mdspan_t<const double, 2> K);

  /* Variable definitions */

  // Interpolation data
  std::size_t _nipoints_per_fct, _nipoints;
  std::vector<double> _ipoints, _data_M;
  mdspan_t<const double, 4> _M; // Indices: facet, dof, gdim, points

  // Tabulated shape-functions H(div) flux (integration points)
  std::vector<double> _basis_flux_values;
  mdspan_t<const double, 5> _basis_flux;
  const int _ndofs_per_fct, _ndofs_fct, _ndofs_cell;

  // Tabulated shape-functions (hat-function)
  basix::FiniteElement _basix_element_hat;
  std::vector<double> _basis_hat_values;
  mdspan_t<const double, 5> _basis_hat;

  // Pull-back H(div) data
  std::size_t _size_flux_scratch;
  std::vector<T> _flux_scratch_data, _mflux_scratch_data;
  mdspan_t<T, 2> _flux_scratch, _mflux_scratch;

  std::array<double, 3> _normal_scratch;

  std::function<void(mdspan_t<T, 2>&, const mdspan_t<const T, 2>&,
                     const mdspan_t<const double, 2>&, double,
                     const mdspan_t<const double, 2>&)>
      _pull_back_flux;

  // Push-forward H(div) shape-functions
  std::vector<double> _mbasis_flux_values, _mbasis_scratch_values;
  mdspan_t<double, 2> _mbasis_flux, _mbasis_scratch;

  std::function<void(mdspan_t<double, 2>&, const mdspan_t<const double, 2>&,
                     const mdspan_t<const double, 2>&, double,
                     const mdspan_t<const double, 2>&)>
      _push_forward_flux;

  std::function<void(const std::span<double>&,
                     const std::span<const std::uint32_t>&, std::int32_t, int)>
      _apply_dof_transformation;
};
// ------------------------------------------------------------------------------

// ------------------------------------------------------------------------------
/* BoundaryData */
// ------------------------------------------------------------------------------
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
                          mdspan_t<const double, 2> J, const double detJ,
                          mdspan_t<const double, 2> K);

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
  mdspan_t<const std::int8_t, 2> facet_type()
  {
    return mdspan_t<const std::int8_t, 2>(_facet_type.data(), _num_rhs,
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
  QuadratureRule _quadrature_rule;
  KernelDataBC<T> _kernel_data;

  // Geometric mapping
  std::array<double, 9> _data_J, _data_K;
  std::array<double, 18> _detJ_scratch;
};
// ------------------------------------------------------------------------------

} // namespace dolfinx_eqlb::base