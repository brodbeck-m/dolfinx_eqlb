#pragma once

#include "QuadratureRule.hpp"
#include "utils.hpp"

#include <basix/cell.h>
#include <basix/e-lagrange.h>
#include <basix/finite-element.h>
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
  /// actual element and tabulates flux element.
  ///
  /// @param mesh  The mesh
  /// @param qrule The quadrature rule
  KernelData(std::shared_ptr<const mesh::Mesh> mesh,
             std::shared_ptr<const QuadratureRule> qrule,
             const basix::FiniteElement& basix_element_fluxpw);

  /// Kenel data constructor
  ///
  /// Generates data required for isoparametric mapping between refernce and
  /// actual element and tabulates flux element, projected fluxes and projected
  /// RHS.
  ///
  /// @param mesh             The mesh
  /// @param qrule            The quadrature rule
  /// @param degree_flux_proj The element degree of the projected flux
  /// @param degree_rhs_proj  The element degree of the projected RHS
  KernelData(std::shared_ptr<const mesh::Mesh> mesh,
             std::shared_ptr<const QuadratureRule> qrule,
             const basix::FiniteElement& basix_element_fluxpw,
             int degree_flux_proj, int degree_rhs_proj);

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

  dolfinx_adaptivity::s_cmdspan3_t shapefunctions_flux() const
  {
    return stdex::submdspan(_flux_fullbasis, 0, stdex::full_extent,
                            stdex::full_extent, stdex::full_extent);
  }

protected:
  /* Variable definitions */
  // Dimensions
  std::uint32_t _gdim;
  std::uint32_t _tdim;

  // Description mesh element
  int _num_coordinate_dofs;
  bool _is_affine;

  // Facet normals (reference element)
  std::vector<double> _fct_normals;
  std::array<std::size_t, 2> _normals_shape;
  std::vector<bool> _fct_normal_out;

  // Quadrature rule
  std::shared_ptr<const QuadratureRule> _quadrature_rule;

  // Tabulated shape-functions (geometry)
  std::array<std::size_t, 4> _g_basis_shape;
  std::vector<double> _g_basis_values;

  // Tabulated shape-functions (pice-wise H(div) flux)
  std::array<std::size_t, 4> _flux_basis_shape;
  std::vector<double> _flux_basis_values;
  dolfinx_adaptivity::cmdspan4_t _flux_fullbasis;

  // Tabulated shape-functions (projected flux)
  std::array<std::size_t, 4> _fluxproj_basis_shape;
  std::vector<double> _fluxproj_basis_values;

  // Tabulated shape-functions (projected RHS)
  std::array<std::size_t, 4> _rhsproj_basis_shape;
  std::vector<double> _rhsproj_basis_values;
};

} // namespace dolfinx_adaptivity::equilibration