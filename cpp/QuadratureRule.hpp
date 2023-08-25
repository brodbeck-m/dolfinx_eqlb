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
    basix::cell::type b_cell_type = mesh::cell_type_to_basix_type(cell_type);

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

    // Calculate quadrature points and weights (_dim -1)
    std::array<std::vector<double>, 2> qrule_fct
        = basix::quadrature::make_quadrature(quadrature_type, b_fct_type,
                                             degree);

    // Set number of quadrature points
    _npoints_per_fct = qrule_fct[1].size();

    // Extract quadrature points/ weights
    const std::vector<double>& q_points = qrule_fct.front();
    const std::vector<double>& q_weights = qrule_fct.back();

    // Map facet quadrature points to reference cell
    if (b_cell_type == basix::cell::type::triangle)
    {
      // Overall number of quadrature points on facets
      _npoints_fct = _npoints_per_fct * 3;

      // Initialise storage
      _points_fct.resize(_npoints_fct * 2);
      _sloc_fct.resize(_npoints_fct);
      _weights_fct.resize(_npoints_fct);

      // Set reference direction
      std::array<double, 2> ref_dir = {0.0, 0.0};
      std::array<double, 2> ref_pos = {0.0, 0.0};
      double ref_length = 1.0;

      // Loop over all facets
      for (std::size_t f = 0; f < 3; ++f)
      {
        // Set offset for storage
        int offset = f * _npoints_per_fct;

        // Set transformation informations
        if (f == 0)
        {
          ref_dir[0] = -1.0;
          ref_dir[1] = 1.0;
          ref_pos[0] = 1.0;
          ref_pos[1] = 0.0;
          ref_length = std::sqrt(2.0);
        }
        else if (f == 1)
        {
          ref_dir[0] = 0.0;
          ref_dir[1] = 1.0;
          ref_pos[0] = 0.0;
          ref_pos[1] = 0.0;
        }
        else if (f == 2)
        {
          ref_dir[0] = 1.0;
          ref_dir[1] = 0.0;
          ref_pos[0] = 0.0;
          ref_pos[1] = 0.0;
        }

        // Loop over all quadrature points
        for (std::size_t i = 0; i < _npoints_per_fct; ++i)
        {
          // Map quadrature points to reference cell
          int id = offset + i;
          int idb = 2 * id;
          _points_fct[idb] = ref_pos[0] + ref_dir[0] * q_points[i];
          _points_fct[idb + 1] = ref_pos[1] + ref_dir[1] * q_points[i];

          // Set local coordinates
          _sloc_fct[id] = q_points[i];

          // Set quadrature weights
          _weights_fct[id] = q_weights[i];
        }
      }
    }
    else
    {
      throw std::runtime_error(
          "Semi-explicit equilibration only supported on triangles");
    }
  }

  /* Setter functions */

  /* Getter functions */
  /// Return the quadrature degree
  /// @return The quadrature degree
  int degree() const { return _degree; }

  /// Return the number of quadrature points on cell
  /// @return The number of quadrature points
  std::size_t npoints_cell() const { return _npoints_cell; }

  /// Return the number of quadrature points on all facet
  /// @return The overall number of quadrature points
  std::size_t npoints_fct() const { return _npoints_fct; }

  /// Return the number of quadrature points on one facet
  /// @return The number of quadrature points
  std::size_t npoints_per_fct() const { return _npoints_per_fct; }

  /// Return the quadrature points (flattend structure) for cell integrals
  /// @return The quadrature points
  const std::vector<double>& points_cell() const { return _points_cell; }

  /// Return the quadrature points (flattend structure) for facet integrals
  /// @return The quadrature points
  const std::vector<double>& points_fct() const { return _points_fct; }

  /// Return the quadrature points (1D) for facet integrals
  /// @return The 1D quadrature points scaled by edge length
  const std::vector<double>& s_fct() const { return _sloc_fct; }

  /// Return quadrature weights for cell-integrals
  /// @return The quadrature weights
  const std::vector<double>& weights_cell() const { return _weights_cell; }

  /// Return quadrature weights for facet-integrals
  /// @return The quadrature weights
  const std::vector<double>& weights_fct() const { return _weights_fct; }

private:
  /* Variable definitions */
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
  std::size_t _npoints_per_fct, _npoints_fct;
  std::vector<double> _points_fct, _sloc_fct, _weights_fct;
};

class QuadratureRuleNew
{
  // Contains quadrature points and weights on a cell on a set of entities
  // Implementation taken from https://github.com/Wells-Group/asimov-contact/

public:
  /// Constructor
  /// @param[in] ct The cell type
  /// @param[in] degree Degree of quadrature rule
  /// @param[in] Dimension of entity
  /// @param[in] type Type of quadrature rule
  QuadratureRuleNew(mesh::CellType ct, int degree, int dim,
                    basix::quadrature::type type
                    = basix::quadrature::type::Default);

  /// Return a list of quadrature points for each entity in the cell
  const std::vector<double>& points() const { return _points; }

  /// Return the quadrature points for the ith entity
  /// @param[in] i The local entity index
  dolfinx_adaptivity::mdspan_t<const double, 2> points(std::int8_t i) const;

  /// Return a list of quadrature weights for each entity in the cell
  /// (using local entity index as in DOLFINx/Basix)
  const std::vector<double>& weights() const { return _weights; }

  /// Return the quadrature weights for the ith entity
  /// @param[in] i The local entity index
  std::span<const double> weights(std::int8_t i) const;

  /// Return dimension of entity in the quadrature rule
  int dim() const { return _dim; }

  /// Return the cell type for the ith quadrature rule
  /// @param[in] Local entity number
  mesh::CellType cell_type(std::int8_t i) const;

  /// Return degree of quadrature rule
  int degree() const;

  /// Return type of the quadrature rule
  basix::quadrature::type type() const;

  /// Return the number of quadrature points per entity
  std::size_t num_points(std::int8_t i) const;

  /// Return the topological dimension of the quadrature rule
  std::size_t tdim() const { return _tdim; };

  /// Return offset for quadrature rule of the ith entity
  const std::vector<std::size_t>& offset() const { return _entity_offset; }

private:
  /* Variable definition */
  // Spacial dimensions
  std::size_t _tdim;
  int _dim;
  int _num_sub_entities; // Number of sub entities

  // Cell type
  mesh::CellType _cell_type;

  // Quadrature type
  basix::quadrature::type _type;

  // Quadrature degree
  int _degree;

  // Quadrature points and weights
  std::vector<double>
      _points; // Quadrature points for each entity on the cell. Shape (entity,
               // num_points, tdim). Flattened row-major.
  std::vector<double>
      _weights; // Quadrature weights for each entity on the cell

  // Offset for each entity
  std::vector<std::size_t> _entity_offset; // The offset for each entity
};

} // namespace dolfinx_adaptivity::equilibration