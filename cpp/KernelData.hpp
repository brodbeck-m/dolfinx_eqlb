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
template <typename T>
class KernelDataEqlb
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
  KernelDataEqlb(std::shared_ptr<const mesh::Mesh> mesh,
                 std::shared_ptr<const QuadratureRule> qrule,
                 const basix::FiniteElement& basix_element_fluxpw,
                 const basix::FiniteElement& basix_element_rhs,
                 const basix::FiniteElement& basix_element_hat);

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

  /// Pull back of flux-data from current to reference cell
  /// @param flux_ref The flux data on reference cell
  /// @param flux_cur The flux data on current cell
  /// @param J        The Jacobian
  /// @param detJ     The determinant of the Jacobian
  /// @param K        The inverse of the Jacobian
  void pull_back_flux(dolfinx_adaptivity::mdspan_t<T, 2> flux_ref,
                      dolfinx_adaptivity::mdspan_t<const T, 2> flux_cur,
                      dolfinx_adaptivity::mdspan_t<const double, 2> J,
                      double detJ,
                      dolfinx_adaptivity::mdspan_t<const double, 2> K)
  {
    _pull_back_fluxspace(flux_ref, flux_cur, K, 1.0 / detJ, J);
  }

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
  dolfinx_adaptivity::smdspan_t<double, 3>
  shapefunctions_flux(dolfinx_adaptivity::mdspan2_t J, double detJ)
  {
    // Map shape functions
    contravariant_piola_mapping(
        stdex::submdspan(_flux_fullbasis_current, 0, stdex::full_extent,
                         stdex::full_extent, stdex::full_extent),
        stdex::submdspan(_flux_fullbasis, 0, stdex::full_extent,
                         stdex::full_extent, stdex::full_extent),
        J, detJ);

    return stdex::submdspan(_flux_fullbasis_current, 0, stdex::full_extent,
                            stdex::full_extent, stdex::full_extent);
  }

  /// Extract shape functions on cell (RHS, projected flux)
  /// Array with indexes i, j and k:
  /// phi_k(x_j) is the shape-function k at point j while i determins
  /// if function or the derivative is returned.
  /// @return Array of shape functions (reference cell)
  dolfinx_adaptivity::s_cmdspan3_t shapefunctions_cell_rhs() const
  {
    return stdex::submdspan(_rhs_cell_fullbasis, stdex::full_extent,
                            stdex::full_extent, stdex::full_extent, 0);
  }

  /// Extract mapped shape functions on cell (RHS, projected flux)
  /// Array with indexes i, j and k:
  /// phi_k(x_j) is the shape-function k at point j while k determins
  /// if function or the derivative is returned.
  /// @param K The inverse Jacobian
  /// @return Array of shape functions (current cell)
  dolfinx_adaptivity::s_cmdspan3_t
  shapefunctions_cell_rhs(dolfinx_adaptivity::cmdspan2_t K);

  /// Extract shape functions on facet (RHS, projected flux)
  /// Array with indexes i, j: phi_j(x_i) is the shape-function j
  /// at point i.
  /// @return Array of shape functions (current cell)
  dolfinx_adaptivity::s_cmdspan2_t shapefunctions_fct_rhs(std::int8_t fct_id)
  {
    // Offset of shpfkt for current facet
    std::size_t obgn = fct_id * _nipoints_per_fct;
    std::size_t oend = obgn + _nipoints_per_fct;

    return stdex::submdspan(_rhs_fct_fullbasis, 0, std::pair{obgn, oend},
                            stdex::full_extent, 0);
  }

  /// Extract shape functions on cell (hat-function)
  /// Array with indexes i and j: phi_k(x_j) is the shape-function k
  /// at point j
  /// @return Array of shape functions (reference cell)
  dolfinx_adaptivity::s_cmdspan2_t shapefunctions_cell_hat() const
  {
    return stdex::submdspan(_hat_cell_fullbasis, 0, stdex::full_extent,
                            stdex::full_extent, 0);
  }

  /// Extract shape functions on facet (hat-function)
  /// Array with indexes i, j: phi_j(x_i) is the shape-function j
  /// at point i.
  /// @return Array of shape functions (reference cell)
  dolfinx_adaptivity::s_cmdspan2_t shapefunctions_fct_hat(std::int8_t fct_id)
  {
    // Offset of shpfkt for current facet
    std::size_t obgn = fct_id * _nipoints_per_fct;
    std::size_t oend = obgn + _nipoints_per_fct;

    return stdex::submdspan(_hat_fct_fullbasis, 0, std::pair{obgn, oend},
                            stdex::full_extent, 0);
  }

  /* Getter functions (Quadrature) */

  // Extract number of quadrature points on cell
  /// @return Number of quadrature points
  int nqpoints_cell() const { return _quadrature_rule->num_points(); }

  /// Extract quadrature points on cell
  /// @return The quadrature points
  dolfinx_adaptivity::mdspan_t<const double, 2> quadrature_points_cell() const
  {
    return dolfinx_adaptivity::mdspan_t<const double, 2>(
        _quadrature_rule->points().data(), _quadrature_rule->num_points(),
        _quadrature_rule->tdim());
  }

  /// Extract quadrature weights on cell
  /// @return The quadrature weights
  std::span<const double> quadrature_weights_cell() const
  {
    return _quadrature_rule->weights();
  }

  /* Getter functions (Interpolation) */
  // Extract number of interpolation points per facet
  /// @return Number of interpolation points
  int nipoints_facet() const { return _nipoints_per_fct; }

  /// Extract interpolation matrix on facet for single DOF
  /// Indices of M: nfct x ndofs x spacial dimension x points
  /// @param fct_id The cell-local facet id
  /// @return The interpolation matrix M
  dolfinx_adaptivity::cmdspan4_t interpl_matrix_facte() { return _M_fct; }

  /// Extract interpolation matrix on facet for single DOF
  /// Indices of M: ndofs x spacial dimension x points
  /// @param fct_id The cell-local facet id
  /// @return The interpolation matrix M
  dolfinx_adaptivity::cmdspan3_t interpl_matrix_facte(std::int8_t fct_id)
  {
    return stdex::submdspan(_M_fct, (std::size_t)fct_id, stdex::full_extent,
                            stdex::full_extent, stdex::full_extent);
  }

  /// Extract interpolation matrix on facet
  /// Indices of M: spacial dimension x points
  /// @param fct_id The cell-local facet id
  /// @param dof_id The facet-local DOF id
  /// @return The interpolation matrix M
  dolfinx_adaptivity::cmdspan2_t interpl_matrix_facte(std::int8_t fct_id,
                                                      std::int8_t dof_id)
  {
    return stdex::submdspan(_M_fct, (std::size_t)fct_id, (std::size_t)dof_id,
                            stdex::full_extent, stdex::full_extent);
  }

protected:
  /// Tabulate shape functions of piecewise H(div) flux
  /// @param basix_element_fluxpw The basix-element for the H(div) flux
  void tabulate_flux_basis(const basix::FiniteElement& basix_element_fluxpw);

  /// Tabulate shape functions of RHS and projected flux
  /// @param basix_element_rhs The basix-element for RHS and projected flux
  void tabulate_rhs_basis(const basix::FiniteElement& basix_element_rhs);

  /// Tabulate shape functions of hat-function
  /// @param basix_element_hat The basix-element for the hat-function
  void tabulate_hat_basis(const basix::FiniteElement& basix_element_hat);

  /// Contravariant Piola mapping
  /// Shape of the basis: points x dofs x spacial dimension
  /// @param phi_cur The current basis function values
  /// @param phi_ref The reference basis function values
  /// @param J    The Jacobian matrix
  /// @param detJ The determinant of the Jacobian matrix
  void contravariant_piola_mapping(
      dolfinx_adaptivity::smdspan_t<double, 3> phi_cur,
      dolfinx_adaptivity::smdspan_t<const double, 3> phi_ref,
      dolfinx_adaptivity::mdspan2_t J, double detJ);

  /* Variable definitions */
  // Dimensions
  std::uint32_t _gdim, _tdim;

  // Description mesh element
  std::size_t _num_coordinate_dofs, _nfcts_per_cell;
  bool _is_affine;

  // Facet normals (reference element)
  std::vector<double> _fct_normals;
  std::array<std::size_t, 2> _normals_shape;
  std::vector<bool> _fct_normal_out;

  // Quadrature rule
  std::shared_ptr<const QuadratureRule> _quadrature_rule;

  // Interpolation data
  std::size_t _nipoints_per_fct, _nipoints_fct;
  std::vector<double> _ipoints_fct, _data_M_fct;
  dolfinx_adaptivity::cmdspan4_t _M_fct; // Indices: facet, dof, gdim, points

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

  // Tabulated shape-functions (hat-function)
  std::vector<double> _hat_basis_cell_values, _hat_basis_fct_values;
  dolfinx_adaptivity::cmdspan4_t _hat_cell_fullbasis, _hat_fct_fullbasis;

  // Push-back H(div) data
  std::function<void(dolfinx_adaptivity::mdspan_t<T, 2>&,
                     const dolfinx_adaptivity::mdspan_t<const T, 2>&,
                     const dolfinx_adaptivity::mdspan_t<const double, 2>&,
                     double,
                     const dolfinx_adaptivity::mdspan_t<const double, 2>&)>
      _pull_back_fluxspace;
};

} // namespace dolfinx_adaptivity::equilibration