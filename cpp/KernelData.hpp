#pragma once

#include "QuadratureRule.hpp"
#include "utils.hpp"

#include <basix/cell.h>
#include <basix/e-lagrange.h>
#include <basix/finite-element.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/cell_types.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <numeric>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

using namespace dolfinx;

namespace dolfinx_eqlb
{
template <typename T>
class KernelData
{
public:
  KernelData(
      std::shared_ptr<const mesh::Mesh> mesh,
      std::vector<std::shared_ptr<const QuadratureRule>> quadrature_rule);

  /// Compute isogeometric mapping for a given cell
  /// @param[in,out] J            The Jacobian
  /// @param[in,out] K            The inverse Jacobian
  /// @param[in,out] detJ_scratch Storage for determinant calculation
  /// @param[in] coords           The cell coordinates
  /// @return                     The determinant of the Jacobian
  double compute_jacobian(mdspan_t<double, 2> J, mdspan_t<double, 2> K,
                          std::span<double> detJ_scratch,
                          mdspan_t<const double, 2> coords);

  /* Basic transformations */
  /// Compute isogeometric mapping for a given cell
  /// @param[in,out] J            The Jacobian
  /// @param[in,out] detJ_scratch Storage for determinant calculation
  /// @param[in] coords           The cell coordinates
  /// @return                     The determinant of the Jacobian
  double compute_jacobian(mdspan_t<double, 2> J, std::span<double> detJ_scratch,
                          mdspan_t<const double, 2> coords);

  /// Calculate physical normal of facet
  /// @param[in,out] normal_phys The physical normal
  /// @param[in] K               The inverse Jacobi-Matrix
  /// @param[in] fct_id          The cell-local facet id
  void physical_fct_normal(std::span<double> normal_phys,
                           mdspan_t<const double, 2> K, std::int8_t fct_id);

  /* Tabulate shape function */
  std::array<std::size_t, 5>
  tabulate_basis(const basix::FiniteElement& basix_element,
                 const std::vector<double>& points,
                 std::vector<double>& storage, bool tabulate_gradient,
                 bool stoarge_elmtcur);

  /* Getter functions (Cell geometry) */
  /// Returns number of nodes, forming a reference cell
  /// @param[out] n The number of nodes, forming the cell
  int nnodes_cell() { return _num_coordinate_dofs; }

  /// Returns number of facets, forming a reference cell
  /// @param[out] n The number of facets, forming the cell
  int nfacets_cell() { return _nfcts_per_cell; }

  /// Returns facet normal on reference facet (const. version)
  /// @param[in] id_fct The cell-local facet id
  /// @param[out] normal_ref The reference facet normal
  std::span<const double> fct_normal(std::int8_t fct_id) const
  {
    return std::span<const double>(_fct_normals.data() + fct_id * _tdim, _tdim);
  }

  /// Returns id if cell-normal points outward
  /// @param[in] id_fct      The cell-local facet id
  /// @param[out] is_outward Direction indicator (true->outward)
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

  /* Getter functions (Quadrature) */
  /// Extract quadrature points on all sub-entity of cell
  /// @param[in] id_qspace The id of the quadrature space
  /// @param[out] points   The quadrature points (flattened storage)
  const std::vector<double>& quadrature_points_flattened(int id_qspace) const
  {
    return _quadrature_rule[id_qspace]->points();
  }

  /// Extract quadrature points on all sub-entity of cell
  /// @param[in] id_qspace The id of the quadrature space
  /// @param[out] points   The quadrature points
  mdspan_t<const double, 2> quadrature_points(int id_qspace)
  {
    // Extract quadrature rule
    std::shared_ptr<const QuadratureRule> quadrature_rule
        = _quadrature_rule[id_qspace];

    // Cast points to mdspan
    return mdspan_t<const double, 2>(quadrature_rule->points().data(),
                                     quadrature_rule->num_points(),
                                     quadrature_rule->tdim());
  }

  /// Extract quadrature points on one sub-entity of cell
  /// @param[in] id_qspace    The id of the quadrature space
  /// @param[in] id_subentity The id of the sub-entity
  /// @param[out] points      The quadrature points
  mdspan_t<const double, 2> quadrature_points(int id_qspace,
                                              std::int8_t id_subentity)
  {
    return _quadrature_rule[id_qspace]->points(id_subentity);
  }

  /// Extract quadrature weights on all sub-entity of cell
  /// @param[in] id_qspace The id of the quadrature space
  /// @param[out] weights  The quadrature weights
  std::span<const double> quadrature_weights(int id_qspace)
  {
    return _quadrature_rule[id_qspace]->weights();
  }

  /// Extract quadrature weights on one sub-entity of cell
  /// @param[in] id_qspace    The id of the quadrature space
  /// @param[in] id_subentity The id of the sub-entity
  /// @param[out] weights     The quadrature weights
  std::span<const double> quadrature_weights(int id_qspace,
                                             std::int8_t id_subentity)
  {
    return _quadrature_rule[id_qspace]->weights(id_subentity);
  }

protected:
  /* Variable definitions */
  // Dimensions
  std::uint32_t _gdim, _tdim;

  // Description mesh element
  int _num_coordinate_dofs, _nfcts_per_cell;
  bool _is_affine;

  // Facet normals (reference element)
  std::vector<double> _fct_normals;
  std::array<std::size_t, 2> _normals_shape;
  std::vector<bool> _fct_normal_out;

  // Quadrature rule
  std::vector<std::shared_ptr<const QuadratureRule>> _quadrature_rule;

  // Tabulated shape-functions (geometry)
  std::vector<double> _g_basis_values;
  mdspan_t<const double, 4> _g_basis;
};

template <typename T>
class KernelDataEqlb : public KernelData<T>
{
public:
  /// Kernel data basic constructor
  ///
  /// Generates data required for isoparametric mapping between reference and
  /// actual element and tabulates flux element.
  ///
  /// @param[in] mesh                 The mesh
  /// @param[in] quadrature_rule_cell The quadrature rule on the cell
  /// @param[in] basix_element_fluxpw The basix-element for the H(div) flux
  /// @param[in] basix_element_rhs    The basix-element for RHS and proj. flux
  /// @param[in] basix_element_hat    The basix-element for the hat-function
  KernelDataEqlb(std::shared_ptr<const mesh::Mesh> mesh,
                 std::shared_ptr<const QuadratureRule> quadrature_rule_cell,
                 const basix::FiniteElement& basix_element_fluxpw,
                 const basix::FiniteElement& basix_element_rhs,
                 const basix::FiniteElement& basix_element_hat);

  /// Pull back of flux-data from current to reference cell
  /// @param flux_ref The flux data on reference cell
  /// @param flux_cur The flux data on current cell
  /// @param J        The Jacobian
  /// @param detJ     The determinant of the Jacobian
  /// @param K        The inverse of the Jacobian
  void pull_back_flux(mdspan_t<T, 2> flux_ref, mdspan_t<const T, 2> flux_cur,
                      mdspan_t<const double, 2> J, double detJ,
                      mdspan_t<const double, 2> K)
  {
    _pull_back_fluxspace(flux_ref, flux_cur, K, 1.0 / detJ, J);
  }

  /* Setter functions */

  /* Getter functions (Shape functions) */

  /// Extract shape functions (H(div) flux)
  /// Array with indexes i, j and k: phi_j(x_i)[k] is the
  /// shape-function j at point i within direction k.
  /// @return Array of shape functions (reference cell)
  s_cmdspan3_t shapefunctions_flux() const
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
  smdspan_t<double, 3> shapefunctions_flux(mdspan2_t J, double detJ)
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
  s_cmdspan3_t shapefunctions_cell_rhs() const
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
  s_cmdspan3_t shapefunctions_cell_rhs(cmdspan2_t K);

  /// Extract shape functions on facet (RHS, projected flux)
  /// Array with indexes i, j: phi_j(x_i) is the shape-function j
  /// at point i.
  /// @return Array of shape functions (current cell)
  s_cmdspan2_t shapefunctions_fct_rhs(std::int8_t fct_id)
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
  s_cmdspan2_t shapefunctions_cell_hat() const
  {
    return stdex::submdspan(_hat_cell_fullbasis, 0, stdex::full_extent,
                            stdex::full_extent, 0);
  }

  /// Extract shape functions on facet (hat-function)
  /// Array with indexes i, j: phi_j(x_i) is the shape-function j
  /// at point i.
  /// @return Array of shape functions (reference cell)
  s_cmdspan2_t shapefunctions_fct_hat(std::int8_t fct_id)
  {
    // Offset of shpfkt for current facet
    std::size_t obgn = fct_id * _nipoints_per_fct;
    std::size_t oend = obgn + _nipoints_per_fct;

    return stdex::submdspan(_hat_fct_fullbasis, 0, std::pair{obgn, oend},
                            stdex::full_extent, 0);
  }

  /* Getter functions (Interpolation) */
  // Extract number of interpolation points per facet
  /// @return Number of interpolation points
  int nipoints_facet() const { return _nipoints_per_fct; }

  /// Extract interpolation matrix on facet for single DOF
  /// Indices of M: nfct x ndofs x spacial dimension x points
  /// @param fct_id The cell-local facet id
  /// @return The interpolation matrix M
  cmdspan4_t interpl_matrix_facte() { return _M_fct; }

  /// Extract interpolation matrix on facet for single DOF
  /// Indices of M: ndofs x spacial dimension x points
  /// @param fct_id The cell-local facet id
  /// @return The interpolation matrix M
  cmdspan3_t interpl_matrix_facte(std::int8_t fct_id)
  {
    return stdex::submdspan(_M_fct, (std::size_t)fct_id, stdex::full_extent,
                            stdex::full_extent, stdex::full_extent);
  }

  /// Extract interpolation matrix on facet
  /// Indices of M: spacial dimension x points
  /// @param fct_id The cell-local facet id
  /// @param dof_id The facet-local DOF id
  /// @return The interpolation matrix M
  cmdspan2_t interpl_matrix_facte(std::int8_t fct_id, std::int8_t dof_id)
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
  void contravariant_piola_mapping(smdspan_t<double, 3> phi_cur,
                                   smdspan_t<const double, 3> phi_ref,
                                   mdspan2_t J, double detJ);

  /* Variable definitions */
  // Interpolation data
  std::size_t _nipoints_per_fct, _nipoints_fct;
  std::vector<double> _ipoints_fct, _data_M_fct;
  cmdspan4_t _M_fct; // Indices: facet, dof, gdim, points

  // Tabulated shape-functions (pice-wise H(div) flux)
  std::vector<double> _flux_basis_values, _flux_basis_current_values;
  cmdspan4_t _flux_fullbasis;
  mdspan4_t _flux_fullbasis_current;

  // Tabulated shape-functions (projected flux, RHS)
  std::vector<double> _rhs_basis_cell_values, _rhs_basis_fct_values,
      _rhs_basis_current_values;
  cmdspan4_t _rhs_cell_fullbasis, _rhs_fct_fullbasis;
  mdspan4_t _rhs_fullbasis_current;

  // Tabulated shape-functions (hat-function)
  std::vector<double> _hat_basis_cell_values, _hat_basis_fct_values;
  cmdspan4_t _hat_cell_fullbasis, _hat_fct_fullbasis;

  // Push-back H(div) data
  std::function<void(mdspan_t<T, 2>&, const mdspan_t<const T, 2>&,
                     const mdspan_t<const double, 2>&, double,
                     const mdspan_t<const double, 2>&)>
      _pull_back_fluxspace;
};

template <typename T>
class KernelDataBC : public KernelData<T>
{
public:
  /// Kernel data basic constructor
  ///
  /// Generates data required for calculating patch specific boundary conditions
  /// during equilibration.
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

  /* Shape functions (flux) at quadrature points */
  std::array<std::size_t, 5>
  shapefunctions_flux_qpoints(const basix::FiniteElement& basix_element,
                              std::vector<double>& storage)
  {
    return this->tabulate_basis(basix_element,
                                this->_quadrature_rule[0]->points(), storage,
                                false, false);
  }

  void map_shapefunctions_flux(std::int8_t lfct_id, mdspan_t<double, 3> phi_cur,
                               mdspan_t<const double, 5> phi_ref,
                               mdspan_t<const double, 2> J, double detJ);

  /* Recover flux from normal-trace */
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

  /* Interpolate flux function */
  void interpolate_flux(std::span<const T> flux_ntrace_cur,
                        std::span<T> flux_dofs, std::int8_t lfct_id,
                        mdspan_t<const double, 2> J, double detJ,
                        mdspan_t<const double, 2> K);

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

  void interpolate_flux(std::span<const T> flux_dofs_bc,
                        std::span<T> flux_dofs_patch, std::int32_t cell_id,
                        std::int8_t lfct_id, std::int8_t hat_id,
                        std::span<const std::uint32_t> cell_info,
                        mdspan_t<const double, 2> J, double detJ,
                        mdspan_t<const double, 2> K);

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
} // namespace dolfinx_eqlb