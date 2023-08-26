#pragma once

#include "FluxBC.hpp"
#include "KernelData.hpp"
#include "utils.hpp"

#include <dolfinx/graph/AdjacencyList.h>

#include <functional>
#include <memory>
#include <span>
#include <vector>

using namespace dolfinx;

namespace dolfinx_adaptivity::equilibration
{
template <typename T>
class BoundaryData
{
public:
  BoundaryData(
      const std::vector<std::vector<std::shared_ptr<const FluxBC<T>>>>&
          bcs_flux,
      const std::vector<std::vector<std::int32_t>>& fct_esntbound_prime);

protected:
  /* Variable definitions */
  // The boundary conditions
  const std::vector<std::vector<std::shared_ptr<const FluxBC<T>>>>& _flux_bcs;

  // Boundary markers/values in global vector
  std::vector<T> _boundary_values;
  std::vector<std::int8_t> _boundary_markers;

  // Facet types
  // (Facet type: 0->internal, 1->essent. BC primal problem, 2->essent. BC flux)
  graph::AdjacencyList<std::int8_t> _facet_type;

  // Connectivity between global facet id and FluxBC
  graph::AdjacencyList<std::int8_t> _fctid_to_fluxbc;

  // Connectivity between global facet id and position in FluxBC
  graph::AdjacencyList<std::int32_t> _fctid_to_fluxbcid;

  // --- Data for projection/ interpolation
  // Surface quadrature kernel
  KernelData<T> _kernel_data;

  // Geometric mapping
  std::array<double, 9> _data_J, data_K;
  dolfinx_adaptivity::mdspan_t<double, 2> _J, _K;

  // Interpolation on facets
  // (Indices M: facet, dof, gdim, points)
  std::size_t _ipoints_per_fct, _num_ipoints;
  std::vector<double> _ipoints, _data_M;
  dolfinx_adaptivity::mdspan_t<const double, 4> _M;

  // Push-back H(div) data
  std::function<void(dolfinx_adaptivity::mdspan_t<T, 2>&,
                     const dolfinx_adaptivity::mdspan_t<const T, 2>&,
                     const dolfinx_adaptivity::mdspan_t<const double, 2>&,
                     double,
                     const dolfinx_adaptivity::mdspan_t<const double, 2>&)>
      _pull_back_fluxspace;

  // Shape-functions lagrange space
  std::vector<double> _basis_values;
  dolfinx_adaptivity::mdspan_t<const double, 4> _basis;
};
} // namespace dolfinx_adaptivity::equilibration