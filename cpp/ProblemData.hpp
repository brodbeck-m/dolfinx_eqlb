#pragma once

#include <algorithm>
#include <cmath>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/utils.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

namespace dolfinx_adaptivity::equilibration
{
template <typename T>
class ProblemData
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
  /// Initialize storage of data for LHS
  ///
  /// Initializes storage of all constants, the boundary-DOF lookup tables
  //  and the boundary values for all LHS considered within the equilibartion.
  ///
  /// @param l        List of all LHS
  /// @param bcs_flux List of list of BCs for each equilibarted flux
  /// @param fluxes   List of list of flux functions for each sub-problem
  ProblemData(
      const std::vector<std::shared_ptr<const dolfinx::fem::Form<T>>>& l,
      const std::vector<
          std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>>>&
          bcs_flux,
      std::vector<std::shared_ptr<fem::Function<T>>>& fluxes)
      : _nlhs(l.size()), _l(l), _flux(fluxes), _kernel(l.size()),
        _offset_cnst(l.size() + 1, 0), _offset_coef(l.size() + 1, 0),
        _offset_bc(l.size() + 1, 0), _cstride(l.size(), 0),
        _begin_hat(l.size(), 0), _begin_fluxdg(l.size(), 0)
  {
    /* Initialize constants */
    std::int32_t size_cnst = 0, size_bc = 0;

    for (std::size_t i = 0; i < _nlhs; ++i)
    {
      // Get current linear form
      const dolfinx::fem::Form<T>& l_i = *(l[i]);

      // Get sizes (constants, boundary conditions)
      std::int32_t size_cnst_i = size_constants(l_i);
      std::int32_t size_bc_i = size_boundary(l_i, bcs_flux[i].empty());

      // Increment overall size
      size_cnst += size_cnst_i;
      size_bc += size_bc_i;

      // Set offset
      _offset_cnst[i + 1] = size_cnst;
      _offset_bc[i + 1] = size_bc;
    }

    // Resize storage
    _data_cnst.resize(size_cnst, 0.0);
    _bdof_marker.resize(size_bc, false);
    _bdof_values.resize(size_bc, 0.0);

    // Set data
    set_data_constants();
    set_data_boundary(bcs_flux);
  }

  void initialize_kernels(dolfinx::fem::IntegralType integral_type, int id)
  {
    // Get DOF-number of hat-function
    int ndof_hat = dolfinx::mesh::cell_num_entities(
        _l[0]->mesh()->topology().cell_type(), 0);

    if (_nlhs == 1)
    {
      /* Get LHS */
      const dolfinx::fem::Form<T>& l_i = *(_l[0]);

      /* Initialize coefficients */
      // Initialize data
      auto coefficients_i = dolfinx::fem::allocate_coefficient_storage(l_i);
      dolfinx::fem::pack_coefficients(l_i, coefficients_i);

      // Extract and store data
      auto& [coeffs_i, cstride_i] = coefficients_i.at({integral_type, id});
      _data_coef.resize(coeffs_i.size());
      _data_coef = std::move(coeffs_i);

      // Get structure of coefficients
      set_structure_coefficients(0, ndof_hat, l_i.coefficient_offsets());

      // Set offsets
      _offset_coef[1] = _data_coef.size();

      /* Set kernel */
      _kernel[0] = l_i.kernel(integral_type, id);
    }
    else
    {
      // Determine size of all coefficients
      std::int32_t size_coef = 0;

      // Loop over all LHS
      for (std::size_t i = 0; i < _nlhs; ++i)
      {
        /* Get LHS */
        const dolfinx::fem::Form<T>& l_i = *(_l[i]);

        /* Initialize datastructure coefficients */
        const std::vector<std::shared_ptr<const dolfinx::fem::Function<T>>>&
            coefficients_i
            = l_i.coefficients();
        const std::vector<int> offsets_i = l_i.coefficient_offsets();

        // Determine number of coefficients
        std::size_t num_entities = 0;
        int cstride = 0;
        if (!coefficients_i.empty())
        {
          cstride = offsets_i.back();
          switch (integral_type)
          {
          case dolfinx::fem::IntegralType::cell:
            num_entities = l_i.cell_domains(id).size();
            break;
          case dolfinx::fem::IntegralType::exterior_facet:
            num_entities = l_i.exterior_facet_domains(id).size() / 2;
            break;
          case dolfinx::fem::IntegralType::interior_facet:
            num_entities = l_i.interior_facet_domains(id).size() / 2;
            break;
          default:
            throw std::runtime_error("Could not allocate coefficient data. "
                                     "Integral type not supported.");
          }
        }

        // Set offset
        size_coef = size_coef + cstride * num_entities;
        _offset_coef[i + 1] = size_coef;

        // Get structure of coefficients
        set_structure_coefficients(i, ndof_hat, offsets_i);

        /* Exctract Kernel */
        _kernel[i] = _l[i]->kernel(integral_type, id);
      }

      // Set data coefficients
      _data_coef.resize(size_coef);
      set_data_coefficients(integral_type, id);
    }
  }

  /* Setter functions */

  /* Getter functions */
  /// Extract linearform l_i
  /// @param index Id of linearform
  /// @return The linearform
  const dolfinx::fem::Form<T>& l(int index) const { return *(_l[index]); }

  /// Extract integration kernel of l_i
  /// @param index Id of linearform
  /// @return The integration kernel
  const std::function<void(T*, const T*, const T*, const scalar_value_type_t*,
                           const int*, const std::uint8_t*)>&
  kernel(int index) const
  {
    return _kernel[index];
  }

  /// Extract flux function
  /// @param index Id of subproblem
  /// @return The flux-function
  dolfinx::fem::Function<T>& flux(int index) const { return *(_flux[index]); }

  /// Extract constants of l_i
  /// @param index Id of linearform
  /// @return Constants of linearform l_i
  std::span<T> constants(int index)
  {
    return std::span<T>(_data_cnst.data() + _offset_cnst[index],
                        _offset_cnst[index + 1] - _offset_cnst[index]);
  }

  /// Extract constants of l_i (constant version)
  /// @param index Id of linearform
  /// @return Constants of linearform l_i
  std::span<const T> constants(int index) const
  {
    return std::span<const T>(_data_cnst.data() + _offset_cnst[index],
                              _offset_cnst[index + 1] - _offset_cnst[index]);
  }

  /// Extract coefficients of l_i
  /// @param index Id of linearform
  /// @return Coefficients of linearform l_i
  std::span<T> coefficients(int index)
  {
    return std::span<T>(_data_coef.data() + _offset_coef[index],
                        _offset_coef[index + 1] - _offset_coef[index]);
  }

  /// Extract coefficients of l_i (constant version)
  /// @param index Id of linearform
  /// @return Coefficients of linearform l_i
  std::span<const T> coefficients(int index) const
  {
    return std::span<const T>(_data_coef.data() + _offset_coef[index],
                              _offset_coef[index + 1] - _offset_coef[index]);
  }

  /// Extract cstride (coefficients) of l_i
  /// @param index Id of linearform
  /// @return cstride of linearform l_i
  int cstride(int index) { return _cstride[index]; }

  /// Extract begin data hat-function (coefficients) of l_i
  /// @param index Id of linearform
  /// @return Begin of hat-function data of linearform l_i
  int begin_hat(int index) { return _begin_hat[index]; }

  /// Extract begin data flux-function (DG) (coefficients) of l_i
  /// @param index Id of linearform
  /// @return Begin of flux-function (DG) data of linearform l_i
  int begin_fluxdg(int index) { return _begin_fluxdg[index]; }

  /// Extract boundary identifires for l_i
  /// @param index Id of linearform
  /// @return Boundary identifires of linearform l_i
  std::span<std::int8_t> boundary_markers(int index)
  {
    return std::span<std::int8_t>(_bdof_marker.data() + _offset_bc[index],
                                  _offset_bc[index + 1] - _offset_bc[index]);
  }

  /// Extract boundary identifires for l_i (constant version)
  /// @param index Id of linearform
  /// @return Boundary identifires of linearform l_i
  std::span<const std::int8_t> boundary_markers(int index) const
  {
    return std::span<const std::int8_t>(_bdof_marker.data() + _offset_bc[index],
                                        _offset_bc[index + 1]
                                            - _offset_bc[index]);
  }

  /// Extract boundary values for l_i
  /// @param index Id of linearform
  /// @return Boundary values of linearform l_i
  std::span<T> boundary_values(int index)
  {
    return std::span<T>(_bdof_values.data() + _offset_bc[index],
                        _offset_bc[index + 1] - _offset_bc[index]);
  }

  /// Extract boundary values for l_i (constant version)
  /// @param index Id of linearform
  /// @return Boundary values of linearform l_i
  std::span<const T> boundary_values(int index) const
  {
    return std::span<const T>(_bdof_values.data() + _offset_bc[index],
                              _offset_bc[index + 1] - _offset_bc[index]);
  }

private:
  /* Handle constants */
  std::int32_t size_constants(const dolfinx::fem::Form<T>& l_i)
  {
    // Extract constants
    const std::vector<std::shared_ptr<const dolfinx::fem::Constant<T>>>&
        constants_i
        = l_i.constants();

    // Get overall size
    std::int32_t size
        = std::accumulate(constants_i.cbegin(), constants_i.cend(), 0,
                          [](std::int32_t sum, auto& constant)
                          { return sum + constant->value.size(); });

    return size;
  }

  void set_data_constants()
  {
    for (std::size_t i = 0; i < _nlhs; ++i)
    {
      // Extract data for l_i
      const dolfinx::fem::Form<T>& l_i = *(_l[i]);

      const std::vector<std::shared_ptr<const dolfinx::fem::Constant<T>>>&
          constants_i
          = l_i.constants();

      std::span<T> data_cnst = constants(i);

      // Extract data
      std::int32_t offset = 0;
      for (auto& constant : constants_i)
      {
        const std::vector<T>& value = constant->value;
        std::copy(value.begin(), value.end(),
                  std::next(data_cnst.begin(), offset));
        offset += value.size();
      }
    }
  }

  /* Hanlde coefficients */
  void set_structure_coefficients(int index, int ndof_hat,
                                  const std::vector<int>& offsets)
  {
    // Get cstide
    int cstride = offsets.back();
    _cstride[index] = cstride;

    if (cstride - offsets[1] == ndof_hat)
    {
      // Set beginn of _hat-data
      _begin_hat[index] = offsets[1];

      // Set beginn of flux_dg-data
      _begin_fluxdg[index] = 0;
    }
    else
    {
      // Set beginn of _hat-data
      _begin_hat[index] = 0;

      // Set beginn of flux_dg-data
      _begin_fluxdg[index] = offsets[1];
    }
  }

  void set_data_coefficients(dolfinx::fem::IntegralType integral_type, int id)
  {
    for (std::size_t i = 0; i < _nlhs; ++i)
    {
      // Get current linear form
      const dolfinx::fem::Form<T>& l_i = *(_l[i]);

      const std::vector<std::shared_ptr<const dolfinx::fem::Function<T>>>&
          coefficients_i
          = l_i.coefficients();
      const std::vector<int> offsets_i = l_i.coefficient_offsets();

      // Storage for current coefficients
      std::span<T> data_coef = coefficients(i);

      // Extract coefficients by default methode
      auto struct_coefficients = fem::allocate_coefficient_storage(l_i);
      fem::pack_coefficients(l_i, struct_coefficients);

      auto& [coeffs_i, cstride_i] = struct_coefficients.at({integral_type, id});

      // Move data into storage
      std::move(coeffs_i.begin(), coeffs_i.end(), data_coef.begin());
    }
  }

  /* Handle boundary data */
  std::int32_t size_boundary(const dolfinx::fem::Form<T>& l_i, bool bc_is_set)
  {
    std::shared_ptr<const dolfinx::common::IndexMap> index_map
        = l_i.function_spaces().at(0)->dofmap()->index_map;
    int bs = l_i.function_spaces().at(0)->dofmap()->index_map_bs();

    // Check if essential bcs are required
    if (!bc_is_set)
    {
      return bs * (index_map->size_local() + index_map->num_ghosts());
    }
    else
    {
      return 0;
    }
  }

  void set_data_boundary(
      const std::vector<
          std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>>>&
          bcs_flux)
  {
    for (std::size_t i = 0; i < _nlhs; ++i)
    {
      // Extract data for l_i
      const dolfinx::fem::Form<T>& l_i = *(_l[i]);
      const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>>&
          bcs
          = bcs_flux[i];

      std::shared_ptr<const dolfinx::fem::FunctionSpace> function_space
          = l_i.function_spaces().at(0);
      std::shared_ptr<const dolfinx::common::IndexMap> index_map
          = function_space->dofmap()->index_map;
      int bs = function_space->dofmap()->index_map_bs();

      // Exctract data storage of current form
      std::span<std::int8_t> bc_marker_i = boundary_markers(i);
      std::span<T> bc_values_i = boundary_values(i);

      // Set boundary markers
      for (std::size_t k = 0; k < bcs.size(); ++k)
      {
        assert(bcs[k]);
        assert(bcs[k]->function_space());
        if (function_space->contains(*bcs[k]->function_space()))
        {
          // Mark boundary DOFs
          bcs[k]->mark_dofs(bc_marker_i);

          // Write boundary values into local data-structure
          bcs[k]->dof_values(bc_values_i);
        }
      }
    }
  }

  // Number of equilibrations
  const int _nlhs;

  // Linearforms
  const std::vector<std::shared_ptr<const dolfinx::fem::Form<T>>>& _l;

  // Integration kernels
  std::vector<
      std::function<void(T*, const T*, const T*, const scalar_value_type_t*,
                         const int*, const std::uint8_t*)>>
      _kernel;

  // Flux functions
  std::vector<std::shared_ptr<fem::Function<T>>>& _flux;

  /* Storage coefficients and constants */
  // Informations
  std::vector<int> _cstride, _begin_hat, _begin_fluxdg;

  // Data
  std::vector<T> _data_coef, _data_cnst;

  // Offset
  std::vector<std::int32_t> _offset_coef, _offset_cnst;

  /* Storage boundary conditions */
  // Markers for DOFs with essential boundary condition
  std::vector<std::int8_t> _bdof_marker;

  // Values at DOFs with essential boundary condition
  std::vector<T> _bdof_values;

  // Offset
  std::vector<std::int32_t> _offset_bc;
};
} // namespace dolfinx_adaptivity::equilibration