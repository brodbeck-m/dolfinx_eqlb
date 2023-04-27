#pragma once

#include "utils.hpp"
#include <basix/cell.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/cell_types.h>

#include <array>
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
  KernelData(std::shared_ptr<const mesh::Mesh> mesh);

  double compute_jacobian(dolfinx_adaptivity::mdspan2_t J,
                          dolfinx_adaptivity::mdspan2_t K,
                          std::span<double> detJ_scratch, cmdspan2_t coords);

  void physical_facet_normal(dolfinx_adaptivity::mdspan2_t K,
                             std::int8_t fct_id);

  /* Setter functions */

  /* Getter functions */

protected:
  /* Variable definitions */
  // Dimensions
  std::uint32_t _gdim;
  std::uint32_t _tdim;

  // Description of mesh element
  int _num_coordinate_dofs;
  bool _is_affine;

  // Tabulation of geometric element
  std::array<std::size_t, 4> _g_basis_shape;
  std::vector<double> _g_basis_values;

  // Facet normals (reference element)
  std::vector<double> _facet_normals;
  std::array<std::size_t, 2> _normals_shape;
};

} // namespace dolfinx_adaptivity::equilibration