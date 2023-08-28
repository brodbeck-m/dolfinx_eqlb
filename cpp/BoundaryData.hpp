#pragma once

#include "FluxBC.hpp"
#include "KernelData.hpp"
#include "QuadratureRule.hpp"
#include "utils.hpp"

#include <basix/e-lagrange.h>
#include <basix/element-families.h>
#include <basix/finite-element.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/cell_types.h>

#include <functional>
#include <memory>
#include <span>
#include <vector>

using namespace dolfinx;

namespace dolfinx_eqlb
{
template <typename T>
class BoundaryData
{
public:
  BoundaryData(
      int flux_degree,
      std::vector<std::vector<std::shared_ptr<FluxBC<T>>>>& bcs_flux,
      std::shared_ptr<const mesh::Mesh> mesh,
      const std::vector<std::vector<std::int32_t>>& fct_esntbound_prime);

  BoundaryData(
      int flux_degree,
      std::vector<std::vector<std::shared_ptr<FluxBC<T>>>>& bcs_flux,
      std::shared_ptr<const mesh::Mesh> mesh,
      const std::vector<std::vector<std::int32_t>>& fct_esntbound_prime,
      const basix::FiniteElement& belmt_flux_proj);

  /* Getter methods */
  std::span<std::int8_t> facet_type(int rhs_i)
  {
    return std::span<std::int8_t>(
        _data_facet_type.data() + _offset_facetdata[rhs_i],
        _offset_facetdata[rhs_i + 1] - _offset_facetdata[rhs_i]);
  }

  std::span<std::int32_t> dof_to_fluxbc(int rhs_i)
  {
    return std::span<std::int32_t>(
        _data_dof_to_fluxbc.data() + _offset_facetdata[rhs_i],
        _offset_facetdata[rhs_i + 1] - _offset_facetdata[rhs_i]);
  }

  std::span<std::int32_t> dof_to_fluxbcid(int rhs_i)
  {
    return std::span<std::int32_t>(
        _data_dof_to_fluxbcid.data() + _offset_facetdata[rhs_i],
        _offset_facetdata[rhs_i + 1] - _offset_facetdata[rhs_i]);
  }

protected:
  void initialise_boundary_data(
      int flux_degree, std::shared_ptr<const mesh::Mesh> mesh,
      const std::vector<std::vector<std::int32_t>>& fct_esntbound_prime,
      const basix::FiniteElement& belmt_flux_proj);

  void initialise_boundary_conditions(
      std::shared_ptr<const mesh::Mesh> mesh,
      const std::vector<std::vector<std::int32_t>>& fct_esntbound_prime,
      mdspan_t<const double, 5> basis_flux,
      const graph::AdjacencyList<std::int32_t>& dofmap_flux);

  /* Variable definitions */
  // Counters
  int _gdim, _flux_degree;
  std::vector<std::int32_t> _num_fcts_fluxbc;
  std::size_t _num_rhs;

  // The boundary conditions
  std::vector<std::vector<std::shared_ptr<FluxBC<T>>>>& _flux_bcs;

  // Boundary markers/values in global vector
  std::vector<T> _boundary_values;
  std::vector<std::int8_t> _boundary_markers;

  // Facet types
  // (Facet type: 0->internal, 1->essent. BC primal problem, 2->essent. BCflux)
  std::vector<std::int8_t> _data_facet_type;
  std::vector<std::int32_t> _offset_facetdata;

  // Connectivity between global facet id and FluxBC
  std::vector<std::int32_t> _data_dof_to_fluxbc;

  // Connectivity between global facet id and position in FluxBC
  std::vector<std::int32_t> _data_dof_to_fluxbcid;

  // ID if (some) BCs require projection
  bool _projection_required;

  // --- Data for projection/ interpolation
  // Surface quadrature kernel
  QuadratureRule _quadrature_rule;
  KernelData<T> _kernel_data;

  // Geometric mapping
  std::array<double, 9> _data_J, data_K;
  mdspan_t<double, 2> _J, _K;

  // Interpolation on facets
  // (Indices M: facet, dof, gdim, points)
  std::size_t _num_ipoints_per_fct, _num_ipoints;
  std::vector<double> _ipoints, _data_M;
  mdspan_t<const double, 4> _M;

  // Push-back H(div) data
  std::function<void(mdspan_t<T, 2>&, const mdspan_t<const T, 2>&,
                     const mdspan_t<const double, 2>&, double,
                     const mdspan_t<const double, 2>&)>
      _pull_back_fluxspace;

  // Shape-functions projected flux
  std::vector<double> _basis_bc_values;
  mdspan_t<const double, 5> _basis_bc;

  // Shape-functions hat-function
  basix::FiniteElement _belmt_hat;
  std::vector<double> _basis_hat_values;
  mdspan_t<const double, 5> _basis_hat;
};
} // namespace dolfinx_eqlb