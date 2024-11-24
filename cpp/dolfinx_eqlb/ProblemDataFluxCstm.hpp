// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx_eqlb/base/BoundaryData.hpp>
#include <dolfinx_eqlb/base/mdspan.hpp>

#include <memory>
#include <span>
#include <vector>

using namespace dolfinx;

namespace dolfinx_eqlb
{
template <typename T>
class ProblemDataFluxCstm
{
public:
  /// Initialize storage of data for equilibration of (multiple) fluxes
  ///
  /// Initializes storage of the boundary-DOF lookup tables, the boundary values
  /// as well as the functions for the reconstructed flux and projected flues
  /// and RHS for the entire set of problems.
  ///
  /// @param fluxes    List of list of flux functions (H(div))
  /// @param fluxed_dg List of list of flux functions (DG)
  /// @param rhs_dg    List of list of projected right-hand-sides
  ProblemDataFluxCstm(std::vector<std::shared_ptr<fem::Function<T>>>& fluxes,
                      std::vector<std::shared_ptr<fem::Function<T>>>& fluxes_dg,
                      std::vector<std::shared_ptr<fem::Function<T>>>& rhs_dg,
                      std::shared_ptr<base::BoundaryData<T>> boundary_data)
      : _nrhs(fluxes.size()), _flux_hdiv(fluxes), _flux_dg(fluxes_dg),
        _rhs_dg(rhs_dg), _boundary_data(boundary_data)
  {
  }

  /* Setter functions*/

  /* Getter functions: General */
  /// Extract number of equilibrated fluxes
  /// @return Number of equilibrated fluxes
  int nrhs() const { return _nrhs; }

  /// Extract mesh
  /// @return The mesh
  std::shared_ptr<const mesh::Mesh> mesh()
  {
    return _flux_hdiv[0]->function_space()->mesh();
  }

  /* Getter functions: Functions ans FunctionSpaces */
  /// Extract FunctionSpace of H(div) flux
  /// @return The FunctionSpace
  std::shared_ptr<const fem::FunctionSpace> fspace_flux_hdiv() const
  {
    return _flux_hdiv[0]->function_space();
  }

  /// Extract FunctionSpace of projected flux
  /// @return The FunctionSpace
  std::shared_ptr<const fem::FunctionSpace> fspace_flux_dg() const
  {
    return _flux_dg[0]->function_space();
  }

  /// Extract FunctionSpace of projected RHS
  /// @return The FunctionSpace
  std::shared_ptr<const fem::FunctionSpace> fspace_rhs_dg() const
  {
    return _rhs_dg[0]->function_space();
  }

  /// Extract flux function (H(div))
  /// @param index Id of subproblem
  /// @return The projected flux (fe function)
  fem::Function<T>& flux(int index) { return *(_flux_hdiv[index]); }

  /// Extract projected primal flux
  /// @param index Id of subproblem
  /// @return The projected flux (fe function)
  fem::Function<T>& projected_flux(int index) { return *(_flux_dg[index]); }

  /// Extract projected RHS
  /// @param index Id of subproblem
  /// @return The projected RHS (fe function)
  fem::Function<T>& projected_rhs(int index) { return *(_rhs_dg[index]); }

  /* Interface BoundaryData */
  /// Calculate BCs for patch-problem
  void calculate_patch_bc(const int rhs_i, const std::int32_t bound_fcts,
                          const std::int8_t patchnode_local,
                          base::mdspan_t<const double, 2> J, const double detJ,
                          base::mdspan_t<const double, 2> K)
  {
    _boundary_data->calculate_patch_bc(rhs_i, bound_fcts, patchnode_local, J,
                                       detJ, K);
  }

  /// Extract facet-types of all sub-problems
  /// @return Mdspan of facet-types
  base::mdspan_t<const std::int8_t, 2> facet_type() const
  {
    return _boundary_data->facet_type();
  }

  /// Marker if mesh-node is on essential boundary of the stress field
  /// @return List of markers for all nodes
  std::span<const std::int8_t> node_on_essnt_boundary_stress() const
  {
    return _boundary_data->node_on_essnt_boundary_stress();
  }

  /// Extract boundary identifiers for l_i
  /// @param index Id of linearform
  /// @return Boundary identifiers of linearform l_i
  std::span<std::int8_t> boundary_markers(int index)
  {
    return _boundary_data->boundary_markers(index);
  }

  /// Extract boundary identifiers for l_i (constant version)
  /// @param index Id of linearform
  /// @return Boundary identifires of linearform l_i
  std::span<const std::int8_t> boundary_markers(int index) const
  {
    return _boundary_data->boundary_markers(index);
  }

  /// Extract boundary values for l_i
  /// @param index Id of linearform
  /// @return Boundary values of linearform l_i
  std::span<T> boundary_values(int index)
  {
    return _boundary_data->boundary_values(index);
  }

  /// Extract boundary values for l_i (constant version)
  /// @param index Id of linearform
  /// @return Boundary values of linearform l_i
  std::span<const T> boundary_values(int index) const
  {
    return _boundary_data->boundary_values(index);
  }

protected:
  /* Variables */
  // Number of equilibrations
  const int _nrhs;

  // Fe functions of projected flux and RHS
  std::vector<std::shared_ptr<fem::Function<T>>>&_flux_hdiv, _flux_dg, _rhs_dg;

  // The boundary data
  std::shared_ptr<base::BoundaryData<T>> _boundary_data;
};
} // namespace dolfinx_eqlb