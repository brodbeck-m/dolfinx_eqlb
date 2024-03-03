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
class ProblemDataStress
{
public:
  /// Initialize storage of data for equilibration stresses
  ///
  /// Initializes storage of the boundary-DOF lookup tables, the boundary values
  /// as well as the functions for the reconstructed flux.
  ///
  /// @param fluxes        List of list of rows of stress tensor (H(div))
  /// @param boundary_data The boundary conditions
  ProblemDataStress(std::vector<std::shared_ptr<fem::Function<T>>>& fluxes,
                    std::shared_ptr<BoundaryData<T>> boundary_data)
      : _flux_hdiv(fluxes), _boundary_data(boundary_data)
  {
    // Spatial dimension
    const int gdim = _flux_hdiv[0]->function_space()->mesh()->topology().dim();

    // Initialise storage for corrector of weak symmetry
    _storage.resize(gdim * _flux_hdiv[0]->x()->array().size(), 0);
  }

  /* Setter functions*/

  /* Getter functions: General */
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

  /// Extract flux function (H(div))
  /// @param index Id of subproblem
  /// @return The projected flux (fe function)
  fem::Function<T>& flux(int index) { return *(_flux_hdiv[index]); }

  /// Extract storage of correctors for stress
  /// @return The storage vector
  std::span<T> stress_corrector()
  {
    return std::span<T>(_storage.data(), _storage.size());
  }

  /// Extract storage of correctors for stress
  /// @return The storage vector
  std::span<const T> stress_corrector() const
  {
    return std::span<const T>(_storage.data(), _storage.size());
  }

  /* Interface BoundaryData */
  /// Extract facet-types of all sub-problems
  /// @return Mdspan of facet-types
  mdspan_t<const std::int8_t, 2> facet_type() const
  {
    return _boundary_data->facet_type();
  }

  /// Extract boundary identifiers for rhs_i
  /// @param index Id of linearform
  /// @return Boundary identifires of linearform l_i
  std::span<const std::int8_t> boundary_markers(int index) const
  {
    return _boundary_data->boundary_markers(index);
  }

protected:
  /* Variables */
  // Fe functions of projected flux and RHS
  std::vector<std::shared_ptr<fem::Function<T>>>& _flux_hdiv;

  // The boundary data
  std::shared_ptr<BoundaryData<T>> _boundary_data;

  // Intermediate storage
  std::vector<T> _storage;
};
} // namespace dolfinx_eqlb