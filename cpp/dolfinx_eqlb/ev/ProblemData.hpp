// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx_eqlb/base/BoundaryData.hpp>
#include <dolfinx_eqlb/base/ProblemData.hpp>
#include <dolfinx_eqlb/base/mdspan.hpp>

using namespace dolfinx;

namespace base = dolfinx_eqlb::base;

namespace dolfinx_eqlb::ev
{
template <typename T>
class ProblemData : public base::ProblemData<T>
{
public:
  /// Initialize storage of data for equilibration of (multiple) fluxes
  ///
  /// Initializes storage of all forms, constants, the boundary-DOF lookup
  /// tables, the boundary values as well as the actual solution functions
  /// for all RHS within a set of problems.
  ///
  /// @param fluxes   List of list of flux functions for each sub-problem
  /// @param bcs_flux List of list of BCs for each equilibrated flux
  /// @param l        List of all RHS (ufl)
  ProblemData(std::vector<std::shared_ptr<fem::Function<T>>>& fluxes,
              const std::vector<std::shared_ptr<const fem::Form<T>>>& l,
              std::shared_ptr<base::BoundaryData<T>> boundary_data)
      : base::ProblemData<T>(fluxes, {}, l), _boundary_data(boundary_data),
        _begin_hat(fluxes.size(), 0), _begin_fluxdg(fluxes.size(), 0)
  {
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
    if (this->_nlhs == 1)
    {
      /* Get LHS */
      const fem::Form<T>& l_i = *(this->_l[0]);

      /* Initialize coefficients */
      // Initialize data
      auto coefficients_i = fem::allocate_coefficient_storage(l_i);
      fem::pack_coefficients(l_i, coefficients_i);

      // Extract and store data
      auto& [coeffs_i, cstride_i] = coefficients_i.at({integral_type, id});
      this->_data_coef.resize(coeffs_i.size());
      this->_data_coef = std::move(coeffs_i);
      this->_cstride[0] = cstride_i;

      // Get structure of coefficients
      set_structure_coefficients(0, l_i.coefficients(),
                                 l_i.coefficient_offsets());

      // Set offsets
      this->_offset_coef[1] = this->_data_coef.size();

      /* Set kernel */
      this->_kernel[0] = l_i.kernel(integral_type, id);
    }
    else
    {
      // Determine size of all coefficients
      std::int32_t size_coef = 0;

      // Loop over all LHS
      for (std::size_t i = 0; i < this->_nlhs; ++i)
      {
        /* Get LHS */
        const fem::Form<T>& l_i = *(this->_l[i]);

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
        this->_offset_coef[i + 1] = size_coef;

        // Get structure of coefficients
        set_structure_coefficients(i, l_i.coefficients(), offsets_i);

        /* Exctract Kernel */
        this->_kernel[i] = this->_l[i]->kernel(integral_type, id);
      }

      // Set data coefficients
      this->_data_coef.resize(size_coef);
      this->set_data_coefficients(integral_type, id);
    }
  }

  /// Set hat functions to spezified value (all LHS)
  /// @param cell_i Current cell
  /// @param node_i Cell-local ID op patch-central node
  /// @param value  Value of hat-function
  void set_hat_function(const int32_t cell_i, const std::int8_t node_i, T value)
  {
    if (this->_nlhs > 0)
    {
      for (std::size_t index = 0; index < this->_nlhs; ++index)
      {
        // Determine local offset
        std::int32_t offset_local
            = this->_cstride[index] * cell_i + _begin_hat[index] + node_i;

        // Modify coefficients
        this->_data_coef[this->_offset_coef[index] + offset_local] = value;
      }
    }
    else
    {
      // Determine local offset
      std::int32_t offset_local
          = this->_cstride[0] * cell_i + _begin_hat[0] + node_i;

      // Modify coefficients
      this->_data_coef[this->_offset_coef[0] + offset_local] = value;
    }
  }

  /// Set hat function to spezified value (specific LHS)
  /// @param cell_i Current cell
  /// @param node_i Cell-local ID op patch-central node
  /// @param value  Value of hat-function
  /// @param value  Index of LHS
  void set_hat_function(const int32_t cell_i, const std::int8_t node_i, T value,
                        const int index)
  {
    // Determine local offset
    std::int32_t offset_local
        = this->_cstride[index] * cell_i + _begin_hat[index] + node_i;

    // Modify coefficients
    this->_data_coef[this->_offset_coef[index] + offset_local] = value;
  }

  /* Getter functions: Flux functions */
  /// Extract flux function
  /// @param index Id of subproblem
  /// @return The flux (fe function)
  fem::Function<T>& flux(int index) const { return *(this->_solfunc[index]); }

  /* Getter functions: Hat function */
  /// Extract begin data hat-function (coefficients) of l_i
  /// @param index Id of linearform
  /// @return Begin of hat-function data of linearform l_i
  int begin_hat(int index) { return _begin_hat[index]; }

  /// Extract begin data flux-function (DG) (coefficients) of l_i
  /// @param index Id of linearform
  /// @return Begin of flux-function (DG) data of linearform l_i
  int begin_fluxdg(int index)
  {
    throw std::runtime_error('Coefficients of flux currently unavailable');
  }

  /* Interface BoundaryData */
  /// Calculate BCs for patch-problem
  void calculate_patch_bc(std::span<const std::int32_t> bound_fcts,
                          std::span<const std::int8_t> patchnode_local)
  {
    _boundary_data->calculate_patch_bc(bound_fcts, patchnode_local);
  }

  /// Extract facet-types of all sub-problems
  /// @return mdspan of facet-types
  base::mdspan_t<const std::int8_t, 2> facet_type() const
  {
    return _boundary_data->facet_type();
  }

  /// Extract boundary identifiers for l_i
  /// @param index Id of linearform
  /// @return Boundary identifiers of linearform l_i
  std::span<std::int8_t> boundary_markers(int index)
  {
    return _boundary_data->boundary_markers(index);
  }

  /// Extract boundary identifiers for l_i (constant version)
  /// @param index Id of linearform
  /// @return Boundary identifires of linearform l_i
  std::span<const std::int8_t> boundary_markers(int index) const
  {
    return _boundary_data->boundary_markers(index);
  }

  /// Extract boundary values for l_i
  /// @param index Id of linearform
  /// @return Boundary values of linearform l_i
  std::span<T> boundary_values(int index)
  {
    return _boundary_data->boundary_values(index);
  }

  /// Extract boundary values for l_i (constant version)
  /// @param index Id of linearform
  /// @return Boundary values of linearform l_i
  std::span<const T> boundary_values(int index) const
  {
    return _boundary_data->boundary_values(index);
  }

protected:
  /* Hanlde coefficients */
  void set_structure_coefficients(int index, auto& list_coeffs,
                                  const std::vector<int>& offsets)
  {
    // Counter error handling
    int count_hat = 0;

    for (std::size_t i = 0; i < list_coeffs.size(); ++i)
    {
      if (list_coeffs[i]->name == "hat")
      {
        // Set begin within coefficient-vector
        _begin_hat[index] = offsets[i];

        // Set identifier for error handling
        count_hat += 1;
      }
    }

    // Error handling indetification of hat-function
    if (count_hat != 1)
    {
      throw std::runtime_error("hat-function could not be idetified");
    }
  }

  /* Variables */
  // The boundary data (equilibration specific)
  std::shared_ptr<base::BoundaryData<T>> _boundary_data;

  // Infos on constants and coefficients
  std::vector<int> _begin_hat, _begin_fluxdg;
};

} // namespace dolfinx_eqlb::ev