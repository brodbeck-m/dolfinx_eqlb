#pragma once

#include "BoundaryData.hpp"
#include "utils.hpp"

#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/mesh/Mesh.h>

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
                      std::shared_ptr<BoundaryData<T>> boundary_data)
      : _nrhs(fluxes.size()), _flux_hdiv(fluxes), _flux_dg(fluxes_dg),
        _rhs_dg(rhs_dg), _boundary_data(boundary_data)
  {
    /* Resize storage for solution of flux minimisation */
    // Number of flux-DOFs on current processor
    const std::int32_t num_dofs_flux
        = fluxes[0]->function_space()->dofmap()->index_map->size_local()
          + fluxes[0]->function_space()->dofmap()->index_map->num_ghosts();

    // Calculate offset
    _offset_x.resize(_nrhs + 1, 0);
    std::generate(_offset_x.begin(), _offset_x.end(),
                  [n = 0, num_dofs_flux]() mutable
                  { return num_dofs_flux * (n++); });

    // Initialise storage
    _x_fhdiv_minimisation.resize(num_dofs_flux * _nrhs, 0);
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

  /// Intermediate storage for solution of minimisation step
  /// @param index Id of subproblem
  /// @return The DOF vector
  std::span<T> x_minimisation(int index)
  {
    return std::span<T>(_x_fhdiv_minimisation.data() + _offset_x[index],
                        _offset_x[index + 1] - _offset_x[index]);
  }

  /// Intermediate storage for solution of minimisation step
  /// @param index Id of subproblem
  /// @return The DOF vector
  std::span<const T> x_minimisation(int index) const
  {
    return std::span<const T>(_x_fhdiv_minimisation.data() + _offset_x[index],
                              _offset_x[index + 1] - _offset_x[index]);
  }

  /* Interface BoundaryData */
  /// Extract facet-types of all sub-problems
  /// @return Mdspan of facet-types
  mdspan_t<const std::int8_t, 2> facet_type() const
  {
    return _boundary_data->facet_type();
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
  std::shared_ptr<BoundaryData<T>> _boundary_data;

  // Intermediate storage of minimisation setp
  std::vector<T> _x_fhdiv_minimisation;
  std::vector<std::int32_t> _offset_x;
};
} // namespace dolfinx_eqlb