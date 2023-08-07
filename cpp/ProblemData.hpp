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

using namespace dolfinx;

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
  /// Initialize storage of data of (multiple) RHS
  ///
  /// Initializes storage of all forms, constants, the boundary-DOF lookup
  /// tables, the boundary values as well as the actual solution functions
  /// for all RHS within a set of problems.
  ///
  /// @param sol_func List of list solution functions for each problem
  /// @param bcs      List of list of BCs for each problem
  /// @param l        List of all RHS
  ProblemData(std::vector<std::shared_ptr<fem::Function<T>>>& sol_func,
              const std::vector<
                  std::vector<std::shared_ptr<const fem::DirichletBC<T>>>>& bcs,
              const std::vector<std::shared_ptr<const fem::Form<T>>>& l)
      : _nlhs(sol_func.size()), _solfunc(sol_func), _l(l),
        _kernel(sol_func.size()), _offset_cnst(sol_func.size() + 1, 0),
        _offset_coef(sol_func.size() + 1, 0),
        _offset_bc(sol_func.size() + 1, 0), _cstride(sol_func.size(), 0)
  {
    // Check if boundary conditions are set
    bool id_no_bcs = bcs.empty();
    std::int32_t size_bc_i = 0;

    // Extract function space on which BCs are set
    const std::shared_ptr<const fem::FunctionSpace> function_space
        = l[0]->function_spaces().at(0);

    /* Initialize constants */
    std::int32_t size_cnst = 0, size_bc = 0;

    for (std::size_t i = 0; i < _nlhs; ++i)
    {
      // Get current linear form
      const fem::Form<T>& l_i = *(l[i]);

      // Get sizes (constants, boundary conditions)
      std::int32_t size_cnst_i = size_constants(l_i);

      if (!id_no_bcs)
      {
        size_bc_i = size_boundary(function_space, bcs[i].empty());
      }

      // Increment overall size
      size_cnst += size_cnst_i;
      size_bc += size_bc_i;

      // Set offset
      _offset_cnst[i + 1] = size_cnst;
      _offset_bc[i + 1] = size_bc;
    }

    // Resize storage and set values
    _data_cnst.resize(size_cnst, 0.0);
    set_data_constants();

    if (!id_no_bcs)
    {
      _bdof_marker.resize(size_bc, false);
      _bdof_values.resize(size_bc, 0.0);
      set_data_boundary(bcs, function_space);
    }
  }

  /// Initialize integration kernels and related coefficients
  ///
  /// Set vector of function pointers onto integration kernels (LHS)
  /// over a given subdomain and initialize storage of related coefficients
  ///
  /// @param integral_type Integral type
  /// @param id            Id of integration-subdomain
  void initialize_kernels(fem::IntegralType integral_type, int id)
  {
    if (_nlhs == 1)
    {
      /* Get LHS */
      const fem::Form<T>& l_i = *(_l[0]);

      /* Initialize coefficients */
      // Initialize data
      auto coefficients_i = fem::allocate_coefficient_storage(l_i);
      fem::pack_coefficients(l_i, coefficients_i);

      // Extract and store data
      auto& [coeffs_i, cstride_i] = coefficients_i.at({integral_type, id});
      _data_coef.resize(coeffs_i.size());
      _data_coef = std::move(coeffs_i);
      _cstride[0] = cstride_i;

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
        const fem::Form<T>& l_i = *(_l[i]);

        /* Initialize datastructure coefficients */
        const std::vector<std::shared_ptr<const fem::Function<T>>>&
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
          case fem::IntegralType::cell:
            num_entities = l_i.cell_domains(id).size();
            break;
          case fem::IntegralType::exterior_facet:
            num_entities = l_i.exterior_facet_domains(id).size() / 2;
            break;
          case fem::IntegralType::interior_facet:
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

        /* Exctract Kernel */
        _kernel[i] = _l[i]->kernel(integral_type, id);
      }

      // Set data coefficients
      _data_coef.resize(size_coef);
      set_data_coefficients(integral_type, id);
    }
  }

  /* Setter functions */
  /// Set different RHS of a set of problems
  /// @param l Vector with forms of RHS
  void set_rhs(const std::vector<std::shared_ptr<const fem::Form<T>>>& l)
  {
    // Check input
    if (_l.size() != 0)
    {
      throw std::runtime_error("RHS of problems already set!");
    }
    else
    {
      if (l.size() != _nlhs)
      {
        throw std::runtime_error("Number of RHS does not match!");
      }
    }

    // Set storage
    _l.resize(_nlhs);
    _kernel.resize(_nlhs);
    _offset_coef.resize(_nlhs + 1, 0);
    _offset_cnst.resize(_nlhs + 1, 0);
    _cstride.resize(_nlhs);

    /* Initialize constants */
    std::int32_t size_cnst = 0;

    for (std::size_t i = 0; i < _nlhs; ++i)
    {
      // Get current linear form
      const fem::Form<T>& l_i = *(l[i]);
      _l[i] = l[i];

      // Get sizes (constants, boundary conditions)
      std::int32_t size_cnst_i = size_constants(l_i);

      // Increment overall size
      size_cnst += size_cnst_i;

      // Set offset
      _offset_cnst[i + 1] = size_cnst;
    }

    // Resize storage and set values
    _data_cnst.resize(size_cnst, 0.0);
    set_data_constants();
  }

  /* Getter functions */
  /// Extract number linearform l_i
  /// @param index Id of linearform
  /// @return Number of linearforms
  int nlhs() const { return _nlhs; }

  /// Extract linearform l_i
  /// @param index Id of linearform
  /// @return The linearform
  const fem::Form<T>& l(int index) const { return *(_l[index]); }

  /// Extract integration kernel of l_i
  /// @param index Id of linearform
  /// @return The integration kernel
  const std::function<void(T*, const T*, const T*, const scalar_value_type_t*,
                           const int*, const std::uint8_t*)>&
  kernel(int index) const
  {
    return _kernel[index];
  }

  /// Extract solution function
  /// @param index Id of subproblem
  /// @return The solution (fe function)
  fem::Function<T>& solution_function(int index) const
  {
    return *(_solfunc[index]);
  }

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

protected:
  /* Handle constants */
  std::int32_t size_constants(const fem::Form<T>& l_i)
  {
    // Extract constants
    const std::vector<std::shared_ptr<const fem::Constant<T>>>& constants_i
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
      const fem::Form<T>& l_i = *(_l[i]);

      const std::vector<std::shared_ptr<const fem::Constant<T>>>& constants_i
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
  void set_data_coefficients(fem::IntegralType integral_type, int id)
  {
    for (std::size_t i = 0; i < _nlhs; ++i)
    {
      // Get current linear form
      const fem::Form<T>& l_i = *(_l[i]);

      const std::vector<std::shared_ptr<const fem::Function<T>>>& coefficients_i
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
      _cstride[i] = cstride_i;
    }
  }

  /* Handle boundary data */
  std::int32_t
  size_boundary(const std::shared_ptr<const fem::FunctionSpace> function_space,
                bool bc_is_set)
  {
    std::shared_ptr<const common::IndexMap> index_map
        = function_space->dofmap()->index_map;
    int bs = function_space->dofmap()->index_map_bs();

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
          std::vector<std::shared_ptr<const fem::DirichletBC<T>>>>& bcs,
      const std::shared_ptr<const fem::FunctionSpace> function_space)
  {
    for (std::size_t i = 0; i < _nlhs; ++i)
    {
      if (!bcs[i].empty())
      {
        // Extract data for subproblem i
        const fem::Form<T>& l_i = *(_l[i]);
        const std::vector<std::shared_ptr<const fem::DirichletBC<T>>>& bc_i
            = bcs[i];

        std::shared_ptr<const common::IndexMap> index_map
            = function_space->dofmap()->index_map;
        int bs = function_space->dofmap()->index_map_bs();

        // Exctract data storage of current form
        std::span<std::int8_t> bc_marker_i = boundary_markers(i);
        std::span<T> bc_values_i = boundary_values(i);

        // Set boundary markers
        for (std::size_t k = 0; k < bc_i.size(); ++k)
        {
          assert(bc_i[k]);
          assert(bc_i[k]->function_space());
          if (function_space->contains(*bc_i[k]->function_space()))
          {
            // Mark boundary DOFs
            bc_i[k]->mark_dofs(bc_marker_i);

            // Write boundary values into local data-structure
            bc_i[k]->dof_values(bc_values_i);
          }
        }
      }
    }
  }

  // Number of equilibrations
  const int _nlhs;

  // Linearforms
  std::vector<std::shared_ptr<const fem::Form<T>>> _l;

  // Integration kernels
  std::vector<
      std::function<void(T*, const T*, const T*, const scalar_value_type_t*,
                         const int*, const std::uint8_t*)>>
      _kernel;

  // Solution functions
  std::vector<std::shared_ptr<fem::Function<T>>>& _solfunc;

  /* Storage coefficients and constants */
  // Informations
  std::vector<int> _cstride;

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