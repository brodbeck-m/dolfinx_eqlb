#pragma once

#include "utils.hpp"

#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/graph/AdjacencyList.h>

#include <algorithm>
#include <array>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <span>
#include <vector>

using namespace dolfinx;

namespace dolfinx_eqlb
{
template <typename T>
class FluxBC
{

  template <typename X, typename = void>
  struct scalar_value_type
  {
    typedef X value_type;
  };
  template <typename X>
  struct scalar_value_type<X, std::void_t<typename X::value_type>>
  {
    typedef typename X::value_type value_type;
  };
  using scalar_value_type_t = typename scalar_value_type<T>::value_type;

public:
  FluxBC(std::shared_ptr<const fem::FunctionSpace> function_space,
         const std::vector<std::int32_t>& boundary_facets,
         std::function<void(T*, const T*, const T*, const scalar_value_type_t*,
                            const int*, const std::uint8_t*)>
             boundary_value,
         int n_bceval_per_fct, bool projection_required,
         std::vector<std::shared_ptr<const fem::Function<T>>> coefficients,
         std::vector<int> positions_of_coefficients,
         std::vector<std::shared_ptr<const fem::Constant<T>>> constants)
      : _function_space(function_space), _fcts(boundary_facets),
        _nfcts(boundary_facets.size()), _boundary_kernel(boundary_value),
        _cstide_eval(n_bceval_per_fct),
        _projection_required(projection_required),
        _c_positions(positions_of_coefficients)
  {
    /* Calculate boundary DOFs */
    // Degree of the flux-element
    const int degree = _function_space->element()->basix_element().degree();

    // Number of projected DOFs per facet
    if (_function_space->mesh()->geometry().dim() == 2)
    {
      _ndofs_per_fct = degree;
    }
    else if (_function_space->mesh()->geometry().dim() == 3)
    {
      _ndofs_per_fct = 0.5 * (degree + 1) * (degree + 2);
    }
    else
    {
      throw std::runtime_error("Unsupported dimension");
    }

    // Initialise boundary DOFs
    initialise_boundary_dofs();

    /* Initialise constants */
    if (constants.size() > 0)
    {
      // Number of constants
      std::int32_t size_constants = size_constants_data<T>(constants);

      // Resize storage vector
      _constants.resize(size_constants, 0);

      // Extract constants
      extract_constants_data<T>(constants, _constants);
    }

    /* Initialise coefficients */
    if (coefficients.size() > 0)
    {
      // Number of coefficients per element
      _cstride_coefficients = std::accumulate(
          coefficients.cbegin(), coefficients.cend(), 0,
          [](int sum, auto& f)
          { return sum + f->function_space()->element()->space_dimension(); });

      // Resize storage vector
      _coefficients.resize(_nfcts * _cstride_coefficients, 0);

      // Extract coefficients
      extract_coefficients_data(coefficients);
    }

    /* Initialise storage of projection */
    if (_projection_required)
    {
      // Number of projected DOFs per facet
      _cstride_proj = _ndofs_per_fct;

      // Resize storage vector
      const int size = _nfcts * _cstride_proj;

      _projected_bc.resize(size, 0);
      _dofs_projected_bc.resize(size, 0);
    }
  }

  /* Getter functions */
  /// Return the flux function space
  /// @param[out] projection_id The projection id
  bool projection_required() const { return _projection_required; }

  /// Return the number of boundary facets
  /// @param[out] nfcts The number of boundary facets
  std::int32_t num_facets() const { return _nfcts; }

  /// Return list of boundary facets
  /// @param[out] fcts The list of boundary facets
  std::span<const std::int32_t> facets() const
  {
    return std::span<const std::int32_t>(_fcts.data(), _fcts.size());
  }

  /// Return list of all boundary DOFs
  /// @param[out] dofs The list of all boundary DOFs
  std::span<const std::int32_t> dofs() const
  {
    return std::span<const std::int32_t>(_dofs.data(), _dofs.size());
  }

  /// Return boundary DOFs on current facet
  /// @param[in] fct_i The fact-id within this BC instance
  /// @param[out] dofs The list of all boundary DOFs
  std::span<const std::int32_t> dofs(int fct_i) const
  {
    return std::span<const std::int32_t>(_dofs.data() + fct_i * _ndofs_per_fct,
                                         _ndofs_per_fct);
  }

  /// Return the flux function-space
  /// @param[out] function_space The flux function-space
  std::shared_ptr<const fem::FunctionSpace> function_space() const
  {
    return _function_space;
  }

protected:
  /// Compute size of coefficient for each boundary cell
  ///
  /// Extract coefficients for each boundary cell and store values in flattened
  /// array. Array structure relative to counter on boundary cells for this
  /// boundary.
  ///
  /// @param coefficients Vector with fem::Function objects
  void extract_coefficients_data(
      std::vector<std::shared_ptr<const fem::Function<T>>> coefficients)
  {
    // Extract connectivity facets->cell
    int dim = _function_space->mesh()->geometry().dim();
    std::shared_ptr<const graph::AdjacencyList<std::int32_t>> fct_to_cell
        = _function_space->mesh()->topology().connectivity(dim - 1, dim);

    // Extract coefficients
    std::int32_t offs_cstride = 0;

    for (int i : _c_positions)
    {
      // Extract function
      std::shared_ptr<const fem::Function<T>> function = coefficients[i];

      // Data storage
      std::span<const T> x_coeff = function->x()->array();

      // Function space
      std::shared_ptr<const fem::FunctionSpace> function_space
          = function->function_space();

      // cstride of current coefficients
      const int bs = function_space->element()->block_size();
      const int space_dimension = function_space->element()->space_dimension();
      const int map_dimension = space_dimension / bs;

      for (std::size_t i = 0; i < _nfcts; ++i)
      {
        // Global facet id
        std::int32_t fct = _fcts[i];

        // Get cell, adjacent to facet
        std::int32_t c = fct_to_cell->links(fct)[0];

        // DOFmap of cell
        std::span<const int32_t> dofs
            = function_space->dofmap()->list().links(c);

        // Flattened storage
        std::int32_t offs_coef = i * _cstride_coefficients + offs_cstride;
        std::span<T> data
            = std::span<T>(_coefficients.data() + offs_coef, space_dimension);

        // Extract coefficient data
        for (std::size_t j = 0; j < map_dimension; ++j)
        {
          // DOF id
          std::int32_t offs_dof = bs * dofs[j];
          std::int32_t offs = offs_coef + j * bs;

          // Copy DOF
          for (std::size_t k = 0; k < bs; ++k)
          {
            _coefficients[offs + k] = x_coeff[offs_dof + k];
          }
        }
      }

      // Update cstride
      offs_cstride += space_dimension;
    }
  }

  /// Compute (global) DOF-ids of the DOFs on boundary facets
  void initialise_boundary_dofs()
  {
    /* Extract relevant data */
    // Id if element is discontinuous
    const bool is_discontinous
        = _function_space->element()->basix_element().discontinuous();

    // The spacial dimension
    const int dim = _function_space->mesh()->geometry().dim();

    // Connectivity facets->cell
    std::shared_ptr<const graph::AdjacencyList<std::int32_t>> fct_to_cell
        = _function_space->mesh()->topology().connectivity(dim - 1, dim);

    // Connectivity cell->facet
    std::shared_ptr<const graph::AdjacencyList<std::int32_t>> cell_to_fct
        = _function_space->mesh()->topology().connectivity(dim, dim - 1);

    // The flux-DOFmap
    const int ndofs_per_cell = _function_space->element()->space_dimension();
    const graph::AdjacencyList<std::int32_t>& dofmap
        = _function_space->dofmap()->list();

    // The ElementDofLayout
    const fem::ElementDofLayout& doflayout
        = _function_space->dofmap()->element_dof_layout();

    /* Initialise storage */
    _dofs.resize(_nfcts * _ndofs_per_fct, 0);

    mdspan_t<std::int32_t, 2> dofs
        = mdspan_t<std::int32_t, 2>(_dofs.data(), _nfcts, _ndofs_per_fct);

    /* Extract DOFs */
    for (std::size_t i = 0; i < _nfcts; ++i)
    {
      // Global facet id
      std::int32_t fct = _fcts[i];

      // Get cell, adjacent to facet
      std::int32_t c = fct_to_cell->links(fct)[0];

      // Get local cell-local facet id
      std::span<const std::int32_t> fcts_cell = cell_to_fct->links(c);
      std::size_t fct_loc
          = std::distance(fcts_cell.begin(),
                          std::find(fcts_cell.begin(), fcts_cell.end(), fct));

      // Get facet DOFs
      if (is_discontinous)
      {
        // Get offset of facet DOFs in current cell
        const int offs = c * ndofs_per_cell + fct_loc * _ndofs_per_fct;

        // Set ids of facet DOFs
        for (std::size_t j = 0; j < _ndofs_per_fct; ++j)
        {
          dofs(i, j) = offs + j;
        }
      }
      else
      {
        // Extract DOFs of current cell
        std::span<const std::int32_t> dofs_cell = dofmap.links(c);

        // Local cell ids on current facet
        const std::vector<int>& entity_dofs
            = doflayout.entity_dofs(dim - 1, fct_loc);

        // Set ids of facet DOFs
        for (std::size_t j = 0; j < _ndofs_per_fct; ++j)
        {
          dofs(i, j) = dofs_cell[entity_dofs[j]];
        }
      }
    }
  }

  /* Variable definitions */
  // The flux function space
  std::shared_ptr<const fem::FunctionSpace> _function_space;

  // Boundary facets
  const std::int32_t _nfcts;
  const std::vector<std::int32_t> _fcts;

  // Boundary DOFs
  int _ndofs_per_fct;
  std::vector<std::int32_t> _dofs;

  // Kernel (executable c++ code)
  std::function<void(T*, const T*, const T*, const scalar_value_type_t*,
                     const int*, const std::uint8_t*)>
      _boundary_kernel;

  // Coefficients associated with the BCs
  std::vector<T> _coefficients;
  int _cstride_coefficients;

  const std::vector<int> _c_positions;

  // Constants associated with the BCs
  std::vector<T> _constants;

  // Number of data-points per facet
  const int _cstide_eval;

  // Projection id (true, if projection is required)
  const bool _projection_required;

  // Storage of projected BC
  std::vector<T> _projected_bc;
  std::vector<std::int32_t> _dofs_projected_bc;
  int _cstride_proj;
};

} // namespace dolfinx_eqlb