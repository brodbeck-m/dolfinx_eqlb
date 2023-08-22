#pragma once

#include "utils.hpp"

#include <array>
#include <iostream>
#include <span>
#include <vector>

using namespace dolfinx;

namespace dolfinx_adaptivity::equilibration
{
template <typename T>
class FluxBC
{
public:
  FluxBC(const std::vector<std::int32_t>& boundary_facets,
         double boundary_value, int n_bceval_per_fct, int ndof_per_fct,
         bool projection_required)
      : _facets(boundary_facets), _boundary_kernel(boundary_value),
        _cstide_eval_bvalues(n_bceval_per_fct),
        _cstride_proj_bvalues(ndof_per_fct),
        _projection_required(projection_required)
  {
    std::cout << "Test class binding: " << std::endl;
    std::cout << "Boundary facets 0, 1, 2: " << _facets[0] << ", " << _facets[1]
              << ", " << _facets[2] << std::endl;
    std::cout << "Evaluations on boundary: " << _cstide_eval_bvalues
              << std::endl;
    std::cout << "DOFs per facet: " << _cstride_proj_bvalues << std::endl;

    // Extract constants
    // TODO: Add extraction of constants

    // Extract coefficients
    // TODO: Add extraction of coefficients
  }

protected:
  /* Variable definitions */
  // Boundary facets
  const std::vector<std::int32_t> _facets;

  // Kernel (executable c++ code)
  const double _boundary_kernel;

  // Coefficients associated with the BCs
  std::vector<T> _coefficients;
  std::int32_t _cstride_coefficients;

  // Constants associated with the BCs
  std::vector<T> _constants;

  // Number of data-points per facet
  const std::int32_t _cstide_eval_bvalues;

  // Number of DOFs (projected RB) per facet
  const std::int32_t _cstride_proj_bvalues;

  // Projection id (true, if projection is required)
  const bool _projection_required;
};

} // namespace dolfinx_adaptivity::equilibration