#pragma once

#include "utils.hpp"

#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx/mesh/cell_types.h>

#include <algorithm>
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
    _pdim_cell = qrule_cell[0].size() / _npoints_cell;

    // Extract quadrature points
    _points_cell.resize(qrule_cell[0].size());
    std::copy(_points_cell.begin(), _points_cell.end(), qrule_cell[0].begin());

    // Extract quadrature weights
    _weights_cell.resize(_npoints_cell);
    std::copy(_weights_cell.begin(), _weights_cell.end(),
              qrule_cell[1].begin());

    /* Quadrature rule facet */
    // Get basix-type of facet
    basix::cell::type b_fct_type
        = basix::cell::sub_entity_type(b_cell_type, _dim_fct, 0);

    // Create basix-element for facet
    basix::FiniteElement b_elmt_fct
        = basix::create_element(basix::element::family::P, b_fct_type, 1,
                                basix::element::lagrange_variant::gll_warped,
                                basix::element::dpc_variant::unset, false);

    // Calculate quadrature points and weights
    std::array<std::vector<double>, 2> qrule_fct
        = basix::quadrature::make_quadrature(quadrature_type, b_cell_type,
                                             degree);

    // Set number of quadrature points
    _npoints_fct = qrule_fct[1].size();
    _pdim_fct = qrule_fct[0].size() / _npoints_fct;

    // Extract quadrature points
    _points_fct.resize(qrule_fct[0].size());
    std::copy(_points_fct.begin(), _points_fct.end(), qrule_fct[0].begin());

    // Extract quadrature weights
    _weights_fct.resize(_npoints_fct);
    std::copy(_weights_fct.begin(), _weights_fct.end(), qrule_fct[1].begin());

    std::cout << "Cell: " << std::endl;
    std::cout << "nqpoints_cell: " << _npoints_cell << std::endl;
    std::cout << "len points: " << qrule_cell[0].size() << std::endl;
    for (auto e : _points_cell)
    {
      std::cout << e << " ";
    }
    std::cout << "\n";
    std::cout << "Facet: " << std::endl;
    std::cout << "nqpoints_cell: " << _npoints_fct << std::endl;
    std::cout << "len points: " << qrule_fct[0].size() << std::endl;
    for (auto e : qrule_fct[0])
    {
      std::cout << e << " ";
    }
    std::cout << "\n";

    throw std::exception();
  }

  /* Setter functions */

  /* Getter functions */
  /// Return the quadrature degree
  /// @return The quadrature degree
  int degree() { return _degree; }

  /// Return the quadrature points (flattend structure) for cell integrals
  /// @return The quadrature points
  const std::vector<double>& points_cell() const { return _points_cell; }

  //   /// Return the quadrature points for cell integrals
  //   /// @return The quadrature points
  //   dolfinx_adaptivity::cmdspan2_t points_cell() const
  //   {
  //     return dolfinx_adaptivity::cmdspan2_t(_points_cell.data(),
  //     _npoints_cell,
  //                                           2);
  //   }

  /// Return the quadrature points (flattend structure) for facet integrals
  /// @return The quadrature points
  const std::vector<double>& points_fct() const { return _points_cell; }

  //   /// Return the quadrature points for facet integrals
  //   /// @return The quadrature points
  //   dolfinx_adaptivity::cmdspan2_t points_fct() const
  //   {
  //     return dolfinx_adaptivity::cmdspan2_t(_points_fct.data(), _npoints_fct,
  //     2);
  //   }

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
  std::size_t _npoints_cell, _lenpoints_cell;
  int _pdim_cell;
  std::vector<double> _points_cell, _weights_cell;

  // Quadrature points ans weights (facet)
  std::size_t _npoints_fct, _lenpoints_fct;
  int _pdim_fct;
  std::vector<double> _points_fct, _weights_fct;
};
} // namespace dolfinx_adaptivity::equilibration