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
  /// Kernel data basic constructor
  ///
  /// Generates data required for isoparametric mapping between reference and
  /// actual element and tabulates flux element.
  ///
  /// @param mesh                 The mesh
  /// @param basix_element_fluxpw The basix-element for the H(div) flux
  /// @param basix_element_rhs    The basix-element for RHS and projected flux
  KernelData(std::shared_ptr<const mesh::Mesh> mesh,
             std::shared_ptr<const QuadratureRule> qrule,
             const basix::FiniteElement& basix_element_fluxpw,
             const basix::FiniteElement& basix_element_rhs);

  /// Compute isogeometric mapping for a given cell
  /// @param J            The Jacobian
  /// @param K            The inverse Jacobian
  /// @param detJ_scratch Storage for determinant calculation
  /// @param coords       The cell coordinates
  /// @return             The determinant of the Jacobian
  double compute_jacobian(dolfinx_adaptivity::mdspan2_t J,
                          dolfinx_adaptivity::mdspan2_t K,
                          std::span<double> detJ_scratch,
                          dolfinx_adaptivity::cmdspan2_t coords);

  /// Compute isogeometric mapping for a given cell
  /// @param J            The Jacobian
  /// @param detJ_scratch Storage for determinant calculation
  /// @param coords       The cell coordinates
  /// @return             The determinant of the Jacobian
  double compute_jacobian(dolfinx_adaptivity::mdspan2_t J,
                          std::span<double> detJ_scratch,
                          dolfinx_adaptivity::cmdspan2_t coords);

  /// Calculate physical normal of facet
  /// @param K      The inverse Jacobi-Matrix
  /// @param fct_id The cell-local facet id
  void physical_fct_normal(std::span<double> normal_phys,
                           dolfinx_adaptivity::mdspan2_t K, std::int8_t fct_id);

  /* Setter functions */

  /* Getter functions (Geometry of cell) */

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

  /* Getter functions (Shape functions) */

  /// Extract shape functions (H(div) flux)
  /// Array with indexes i, j and k: phi_j(x_i)[k] is the
  /// shape-function j at point i within direction k.
  /// @return Array of shape functions (reference cell)
  dolfinx_adaptivity::s_cmdspan3_t shapefunctions_flux() const
  {
    return stdex::submdspan(_flux_fullbasis, 0, stdex::full_extent,
                            stdex::full_extent, stdex::full_extent);
  }

  /// Extract mapped shape functions (H(div) flux)
  /// Array with indexes i, j and k: phi_j(x_i)[k] is the
  /// shape-function j at point i within direction k.
  /// @param J     The Jacobian
  /// @param detJ  The determinant of the Jacobian
  /// @return Array of shape functions (current cell)
  dolfinx_adaptivity::s_cmdspan3_t
  shapefunctions_flux(dolfinx_adaptivity::mdspan2_t J, double detJ);

  /// Extract shape functions on cell (RHS, projected flux)
  /// Array with indexes i, j and k:
  /// phi_k(x_j) is the shape-function k at point j while k determins
  /// if function or the derivative is returned.
  /// @return Array of shape functions (reference cell)
  dolfinx_adaptivity::cmdspan3_t shapefunctions_cell_rhs() const
  {
    return stdex::submdspan(_rhs_cell_fullbasis, stdex::full_extent,
                            stdex::full_extent, stdex::full_extent, 0);
  }

  /// Extract mapped shape functions on cell (RHS, projected flux)
  /// Array with indexes i, j and k:
  /// phi_k(x_j) is the shape-function k at point j while k determins
  /// if function or the derivative is returned.
  /// @param J     The Jacobian
  /// @param detJ  The determinant of the Jacobian
  /// @return Array of shape functions (current cell)
  dolfinx_adaptivity::cmdspan3_t
  shapefunctions_cell_rhs(dolfinx_adaptivity::mdspan2_t J, double detJ);

  /// Extract shape functions on facet (RHS, projected flux)
  /// Array with indexes i, j: phi_j(x_i) is the shape-function j
  /// at point i.
  /// @return Array of shape functions (current cell)
  dolfinx_adaptivity::cmdspan2_t shapefunctions_fct_rhs(int fct_id)
  {
    // Offset of shpfkt for current facet
    const int nqpoints = _quadrature_rule->npoints_per_fct();
    const int obgn = fct_id * nqpoints;
    const int oend = obgn + nqpoints + 1;

    return stdex::submdspan(_rhs_fct_fullbasis, 0,
                            std::pair{obgn, (std::size_t)oend},
                            stdex::full_extent, 0);
  }

  /* Getter functions (Quadrature) */

  /// Extract quadrature weights on cell
  /// @return The quadrature weights
  std::span<const double> quadrature_weights_cell() const
  {
    return _quadrature_rule->weights_cell();
  }

  /// Extract quadrature weights on facet
  /// @param fct_id The cell-local facet id
  /// @return The quadrature weights
  std::span<const double> quadrature_weights_facet(int fct_id)
  {
    // Offset of weights for current facet
    const int nqpoints = _quadrature_rule->npoints_per_fct();
    const int offset = fct_id * nqpoints;

    return std::span<const double>(
        _quadrature_rule->weights_fct().data() + offset, nqpoints);
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
  std::vector<double> _flux_basis_values, _flux_basis_current_values;
  dolfinx_adaptivity::cmdspan4_t _flux_fullbasis;
  dolfinx_adaptivity::mdspan4_t _flux_fullbasis_current;

  // Tabulated shape-functions (projected flux, RHS)
  std::vector<double> _rhs_basis_cell_values, _rhs_basis_fct_values,
      _rhs_basis_current_values;
  dolfinx_adaptivity::cmdspan4_t _rhs_cell_fullbasis, _rhs_fct_fullbasis;
  dolfinx_adaptivity::mdspan4_t _rhs_fullbasis_current;
};

} // namespace dolfinx_adaptivity::equilibration