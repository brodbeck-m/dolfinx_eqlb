#pragma once

#include "FluxBC.hpp"
#include "KernelData.hpp"
#include "QuadratureRule.hpp"
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Sparse"
#include "utils.hpp"

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

namespace dolfinx_eqlb
{
enum facet_type_eqlb : std::int8_t
{
  internal = 0,
  essnt_primal = 1,
  essnt_dual = 2
};

template <typename T>
void assemble_projection(mdspan_t<const double, 2> flux_boundary,
                         mdspan_t<double, 3> phi,
                         std::span<const double> quadrature_weights,
                         const double detJ,
                         Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& A_e,
                         Eigen::Matrix<T, Eigen::Dynamic, 1>& L_e);

template <typename T>
class BoundaryData
{
public:
  BoundaryData(
      std::vector<std::vector<std::shared_ptr<FluxBC<T>>>>& list_bcs,
      std::vector<std::shared_ptr<fem::Function<T>>>& boundary_flux,
      std::shared_ptr<const fem::FunctionSpace> V_flux_hdiv,
      bool rtflux_is_custom,
      const std::vector<std::vector<std::int32_t>>& fct_esntbound_prime);

  void calculate_patch_bc(std::span<const std::int32_t> bound_fcts,
                          std::span<const std::int8_t> patchnode_local);

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
  /// (marked with respect to facet_type_eqlb)
  /// @param[in] rhs_i Index of RHS
  /// @return List of facet types (sorted by facet-ids)
  std::span<std::int8_t> facet_type(const int rhs_i)
  {
    return std::span<std::int8_t>(_facet_type.data() + _offset_fctdata[rhs_i],
                                  _offset_fctdata[rhs_i + 1]
                                      - _offset_fctdata[rhs_i]);
  }

  /// Get list of facet types
  /// (marked with respect to facet_type_eqlb)
  /// @return List of all facet types (sorted by facet-ids)
  mdspan_t<const std::int8_t, 2> facet_type()
  {
    return mdspan_t<const std::int8_t, 2>(_facet_type.data(), _num_rhs,
                                          _num_fcts);
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
  /// Get list of cell local facet ids
  /// @param[in] rhs_i Index of RHS
  /// @return List of cell-local facet ids (sorted by facet-ids)
  std::span<std::int8_t> local_facet_id(const int rhs_i)
  {
    return std::span<std::int8_t>(_local_fct_id.data() + _offset_fctdata[rhs_i],
                                  _offset_fctdata[rhs_i + 1]
                                      - _offset_fctdata[rhs_i]);
  }

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
} // namespace dolfinx_eqlb