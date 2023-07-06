#pragma once

#include "utils.hpp"

#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx/mesh/cell_types.h>

#include <algorithm>
#include <array>
#include <iostream>
#include <span>
#include <vector>

using namespace dolfinx;

namespace dolfinx_adaptivity::equilibration
{
class QuadratureRule
{
public:
  QuadratureRule(mesh::CellType cell_type, int degree,
                 basix::quadrature::type quadrature_type
                 = basix::quadrature::type::Default)
      : _cell_type(cell_type), _type(quadrature_type), _degree(degree)
  {
    // Spacial dimensions
    _dim = mesh::cell_dim(cell_type);
    _dim_fct = _dim - 1;

    // Get basix cell type
    basix::cell::type b_cell_type
        = dolfinx::mesh::cell_type_to_basix_type(cell_type);

    /* Quadrature rule cell */
    // Calculate quadrature points and weights
    std::array<std::vector<double>, 2> qrule_cell
        = basix::quadrature::make_quadrature(quadrature_type, b_cell_type,
                                             degree);

    // Set number of quadrature points
    _npoints_cell = qrule_cell[1].size();

    // Extract quadrature points/ weights
    _points_cell = qrule_cell.front();
    _weights_cell = qrule_cell.back();

    // Extract quadrature weights
    _weights_cell.resize(_npoints_cell);
    std::copy(_weights_cell.begin(), _weights_cell.end(),
              qrule_cell[1].begin());

    /* Quadrature rule facet */
    // Get basix-type of facet
    basix::cell::type b_fct_type
        = basix::cell::sub_entity_type(b_cell_type, _dim_fct, 0);

    // Calculate quadrature points and weights
    std::array<std::vector<double>, 2> qrule_fct
        = basix::quadrature::make_quadrature(quadrature_type, b_fct_type,
                                             degree);

    // Set number of quadrature points
    _npoints_fct = qrule_fct[1].size();

    // Extract quadrature points/ weights
    _points_fct = qrule_fct.front();
    _weights_fct = qrule_fct.back();
  }

  /* Setter functions */

  /* Getter functions */
  /// Return the quadrature degree
  /// @return The quadrature degree
  int degree() const { return _degree; }

  /// Return the number of quadrature points on cell
  /// @return The number of quadrature points
  std::size_t npoints_cell() const { return _npoints_cell; }

  /// Return the number of quadrature points on facet
  /// @return The number of quadrature points
  std::size_t npoints_fct() const { return _npoints_fct; }

  /// Return the quadrature points (flattend structure) for cell integrals
  /// @return The quadrature points
  const std::vector<double>& points_cell() const { return _points_cell; }

  /// Return the quadrature points (flattend structure) for facet integrals
  /// @return The quadrature points
  const std::vector<double>& points_fct() const { return _points_fct; }

  /// Return quadrature weights for cell-integrals
  /// @return The quadrature weights
  const std::vector<double>& weights_cell() const { return _weights_cell; }

  /// Return quadrature weights for facet-integrals
  /// @return The quadrature weights
  const std::vector<double>& weights_fct() const { return _weights_fct; }

private:
  /* Variabe definitions */
  // Spacial dimension
  int _dim, _dim_fct;

  // Cell type
  mesh::CellType _cell_type;

  // Quadrature type
  basix::quadrature::type _type;

  // Quadrature degree
  const int _degree;

  // Quadrature points and weights (cell)
  std::size_t _npoints_cell;
  std::vector<double> _points_cell, _weights_cell;

  // Quadrature points ans weights (facet)
  std::size_t _npoints_fct;
  std::vector<double> _points_fct, _weights_fct;
};
} // namespace dolfinx_adaptivity::equilibration