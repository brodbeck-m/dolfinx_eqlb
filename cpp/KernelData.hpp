#pragma once

#include "QuadratureRule.hpp"
#include "utils.hpp"

#include <basix/cell.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/cell_types.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

using namespace dolfinx;

namespace dolfinx_adaptivity::equilibration
{
class KernelData
{
public:
  /// Kenel data basic constructor
  ///
  /// Generates data required for isoparametric mapping between refernce and
  /// actual element.
  ///
  /// @param mesh The mesh
  KernelData(std::shared_ptr<const mesh::Mesh> mesh,
             std::shared_ptr<const QuadratureRule> qrule);

  double compute_jacobian(dolfinx_adaptivity::mdspan2_t J,
                          dolfinx_adaptivity::mdspan2_t K,
                          std::span<double> detJ_scratch,
                          dolfinx_adaptivity::cmdspan2_t coords);

  /// Calculate physical normal of facet
  /// @param K      The inverse Jacobi-Matrix
  /// @param fct_id The cell-local facet id
  void physical_fct_normal(std::span<double> normal_phys,
                           dolfinx_adaptivity::mdspan2_t K, std::int8_t fct_id);

  /* Setter functions */

  /* Getter functions */
  /// Returns number of nodes, forming a reference cell
  /// @return Number of nodes on reference cell
  int nnodes_cell() { return _num_coordinate_dofs; }

  /// Returns facet normal on reference facet
  /// @param id_fct The cell-local facet id
  /// @return The facet normal (reference cell)
  std::span<double> fct_normal(std::int8_t fct_id)
  {
    std::size_t tdim = _normals_shape[1];
    return std::span<double>(_fct_normals.data() + fct_id * tdim, tdim);
  }

  /// Returns facet normal on reference facet (const. version)
  /// @param id_fct The cell-local facet id
  /// @return The facet normal (reference cell)
  std::span<const double> fct_normal(std::int8_t fct_id) const
  {
    std::size_t tdim = _normals_shape[1];
    return std::span<const double>(_fct_normals.data() + fct_id * tdim, tdim);
  }

  /// Returns id if cell-normal points outward
  /// @param id_fct The cell-local facet id
  /// @return Direction indicator (true->outward)
  bool fct_normal_is_outward(std::int8_t id_fct)
  {
    return _fct_normal_out[id_fct];
  }

  /// Returns id if cell-normal points outward
  /// @param id_fct1 The cell-local facet id
  /// @param id_fct2 The cell-local facet id
  /// @return Direction indicator (true->outward)
  std::pair<bool, bool> fct_normal_is_outward(std::int8_t id_fct1,
                                              std::int8_t id_fct2)
  {
    return {_fct_normal_out[id_fct1], _fct_normal_out[id_fct2]};
  }

protected:
  /* Variable definitions */
  // Dimensions
  std::uint32_t _gdim;
  std::uint32_t _tdim;

  // Description mesh element
  int _num_coordinate_dofs;
  bool _is_affine;

  // Quadrature rule
  std::shared_ptr<const QuadratureRule> _quadrature_rule;

  // Tabulation of geometric element
  std::array<std::size_t, 4> _g_basis_shape;
  std::vector<double> _g_basis_values;

  // Facet normals (reference element)
  std::vector<double> _fct_normals;
  std::array<std::size_t, 2> _normals_shape;
  std::vector<bool> _fct_normal_out;
};

} // namespace dolfinx_adaptivity::equilibration