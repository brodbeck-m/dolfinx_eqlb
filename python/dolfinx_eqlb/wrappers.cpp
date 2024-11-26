// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "caster_petsc.h"

#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx_eqlb/base/BoundaryData.hpp>
#include <dolfinx_eqlb/base/FluxBC.hpp>
#include <dolfinx_eqlb/base/local_solver.hpp>
#include <dolfinx_eqlb/ev/reconstruction.hpp>
#include <dolfinx_eqlb/se/reconstruction.hpp>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <ufcx.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace
{

using namespace dolfinx_eqlb;

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

template <typename T>
void declare_lsolver(py::module& m)
{
  m.def(
      "local_solver_lu",
      [](std::vector<std::shared_ptr<dolfinx::fem::Function<T>>>& sol_elmt,
         const dolfinx::fem::Form<T>& a,
         const std::vector<std::shared_ptr<const dolfinx::fem::Form<T>>>& l)
      { base::local_solver_lu<T>(sol_elmt, a, l); },
      py::arg("solution"), py::arg("a"), py::arg("l"),
      "Local solver based on the LU decomposition");

  m.def(
      "local_solver_cholesky",
      [](std::vector<std::shared_ptr<dolfinx::fem::Function<T>>>& sol_elmt,
         const dolfinx::fem::Form<T>& a,
         const std::vector<std::shared_ptr<const dolfinx::fem::Form<T>>>& l)
      { base::local_solver_cholesky<T>(sol_elmt, a, l); },
      py::arg("solution"), py::arg("a"), py::arg("l"),
      "Local solver based on the Cholesky decomposition");

  m.def(
      "local_solver_cg",
      [](std::vector<std::shared_ptr<dolfinx::fem::Function<T>>>& sol_elmt,
         const dolfinx::fem::Form<T>& a,
         const std::vector<std::shared_ptr<const dolfinx::fem::Form<T>>>& l)
      { base::local_solver_cg<T>(sol_elmt, a, l); },
      py::arg("solution"), py::arg("a"), py::arg("l"),
      "Local solver based on a Conjugated-Gradient solver");
}

template <typename T>
void declare_fluxeqlb(py::module& m)
{
  m.def(
      "reconstruct_fluxes_minimisation",
      [](const dolfinx::fem::Form<T>& a, const dolfinx::fem::Form<T>& l_pen,
         const std::vector<std::shared_ptr<const dolfinx::fem::Form<T>>>& l,
         std::vector<std::shared_ptr<dolfinx::fem::Function<T>>>& flux_hdiv,
         std::shared_ptr<base::BoundaryData<T>> boundary_data)
      { ev::reconstruction<T>(a, l_pen, l, flux_hdiv, boundary_data); },
      py::arg("a"), py::arg("l_pen"), py::arg("l"), py::arg("flux_hdiv"),
      py::arg("boundary_data"),
      "Local equilibration of H(div) conforming fluxes, solving patch-wise, "
      "constrained minimisation problems.");

  m.def(
      "reconstruct_fluxes_semiexplt",
      [](std::vector<std::shared_ptr<dolfinx::fem::Function<T>>>& flux_hdiv,
         std::vector<std::shared_ptr<dolfinx::fem::Function<T>>>& flux_dg,
         std::vector<std::shared_ptr<dolfinx::fem::Function<T>>>& rhs_dg,
         std::shared_ptr<base::BoundaryData<T>> boundary_data,
         const bool reconstruct_stress)
      {
        se::reconstruction<T>(flux_hdiv, flux_dg, rhs_dg, boundary_data,
                              reconstruct_stress,
                              std::shared_ptr<fem::Function<T>>());
      },
      py::arg("flux_hdiv"), py::arg("flux_dg"), py::arg("rhs_dg"),
      py::arg("boundary_data"), py::arg("reconstruct_stress"),
      "Local equilibration of H(div) conforming fluxes, using an explicit "
      "determination of the fluxes followed by a ministration on a reduced "
      "space. If apply_weal_symmetry is true, the first gdim fluxes are "
      "treated as rows of a (symmetric) stress tensors, with weak imposition "
      "of the symmetry constraint.");

  m.def(
      "reconstruct_fluxes_semiexplt_with_kornconst",
      [](std::vector<std::shared_ptr<dolfinx::fem::Function<T>>>& flux_hdiv,
         std::vector<std::shared_ptr<dolfinx::fem::Function<T>>>& flux_dg,
         std::vector<std::shared_ptr<dolfinx::fem::Function<T>>>& rhs_dg,
         std::shared_ptr<base::BoundaryData<T>> boundary_data,
         const bool reconstruct_stress,
         std::shared_ptr<dolfinx::fem::Function<T>> cells_kornconst)
      {
        se::reconstruction<T>(flux_hdiv, flux_dg, rhs_dg, boundary_data,
                              reconstruct_stress, cells_kornconst);
      },
      py::arg("flux_hdiv"), py::arg("flux_dg"), py::arg("rhs_dg"),
      py::arg("boundary_data"), py::arg("reconstruct_stress"),
      py::arg("cells_kornconst"),
      "Local equilibration of H(div) conforming fluxes, using an explicit"
      "determination of the fluxes followed by a ministration on a reduced"
      "space. If apply_weal_symmetry is true, the first gdim fluxes are"
      "treated as rows of a (symmetric) stress tensors, with weak imposition"
      "of the symmetry constraint. During the equilibration, a upper bounds of "
      "each cells Korn constant is calculated.");
}

template <typename T>
void declare_bcs(py::module& m)
{
  /* A single boundary-condition for the flux */
  py::class_<base::FluxBC<T>, std::shared_ptr<base::FluxBC<T>>>(m, "FluxBC",
                                                                "FluxBC object")
      .def(
          py::init(
              [](std::shared_ptr<const fem::FunctionSpace> function_space,
                 const std::vector<std::int32_t>& boundary_facets,
                 std::uintptr_t kernel_ptr, int n_bceval_per_fct,
                 std::vector<std::shared_ptr<const fem::Function<T>>>
                     coefficients,
                 std::vector<int> positions_of_coefficients,
                 std::vector<std::shared_ptr<const fem::Constant<T>>> constants)
              {
                using scalar_value_type_t =
                    typename scalar_value_type<T>::value_type;

                using kern = std::function<void(
                    T*, const T*, const T*, const scalar_value_type_t*,
                    const int*, const std::uint8_t*)>;

                // Cast kernel_ptr to ufcx_expression
                ufcx_expression* expression
                    = reinterpret_cast<ufcx_expression*>(kernel_ptr);

                // Extract executable kernel
                kern tabulate_tensor_ptr = nullptr;
                if constexpr (std::is_same_v<T, double>)
                {
                  tabulate_tensor_ptr = expression->tabulate_tensor_float64;
                }
                else
                {
                  throw std::runtime_error("Unsupported data type");
                }

                // Return class
                return base::FluxBC<T>(function_space, boundary_facets,
                                       tabulate_tensor_ptr, n_bceval_per_fct,
                                       coefficients, positions_of_coefficients,
                                       constants);
              }),
          py::arg("function_space"), py::arg("facets"),
          py::arg("pointer_boundary_kernel"), py::arg("nevals_per_fct"),
          py::arg("coefficients"), py::arg("position_of_coefficients"),
          py::arg("constants"))
      .def(
          py::init(
              [](std::shared_ptr<const fem::FunctionSpace> function_space,
                 const std::vector<std::int32_t>& boundary_facets,
                 std::uintptr_t kernel_ptr, int n_bceval_per_fct,
                 int quadrature_degree,
                 std::vector<std::shared_ptr<const fem::Function<T>>>
                     coefficients,
                 std::vector<int> positions_of_coefficients,
                 std::vector<std::shared_ptr<const fem::Constant<T>>> constants)
              {
                using scalar_value_type_t =
                    typename scalar_value_type<T>::value_type;

                using kern = std::function<void(
                    T*, const T*, const T*, const scalar_value_type_t*,
                    const int*, const std::uint8_t*)>;

                // Cast kernel_ptr to ufcx_expression
                ufcx_expression* expression
                    = reinterpret_cast<ufcx_expression*>(kernel_ptr);

                // Extract executable kernel
                kern tabulate_tensor_ptr = nullptr;
                if constexpr (std::is_same_v<T, double>)
                {
                  tabulate_tensor_ptr = expression->tabulate_tensor_float64;
                }
                else
                {
                  throw std::runtime_error("Unsupported data type");
                }

                // Return class
                return base::FluxBC<T>(function_space, boundary_facets,
                                       tabulate_tensor_ptr, n_bceval_per_fct,
                                       quadrature_degree, coefficients,
                                       positions_of_coefficients, constants);
              }),
          py::arg("function_space"), py::arg("facets"),
          py::arg("pointer_boundary_kernel"), py::arg("nevals_per_fct"),
          py::arg("quadrature_degree"), py::arg("coefficients"),
          py::arg("position_of_coefficients"), py::arg("constants"))
      .def_property_readonly("quadrature_degree",
                             &base::FluxBC<T>::quadrature_degree);

  /* The collection of all BCs of all RHS */
  py::class_<base::BoundaryData<T>, std::shared_ptr<base::BoundaryData<T>>>(
      m, "BoundaryData", "BoundaryData object")
      .def(py::init(
               [](std::vector<std::vector<std::shared_ptr<base::FluxBC<T>>>>&
                      list_bcs,
                  std::vector<std::shared_ptr<fem::Function<T>>>& boundary_flux,
                  std::shared_ptr<const fem::FunctionSpace> V_flux_hdiv,
                  bool rtflux_is_custom, int quadrature_degree,
                  const std::vector<std::vector<std::int32_t>>&
                      fct_esntbound_prime,
                  const bool reconstruct_stress)
               {
                 // Return class
                 return base::BoundaryData<T>(
                     list_bcs, boundary_flux, V_flux_hdiv, rtflux_is_custom,
                     quadrature_degree, fct_esntbound_prime,
                     reconstruct_stress);
               }),
           py::arg("list_of_bcs"), py::arg("list_of_boundary_fluxes"),
           py::arg("V_flux_hdiv"), py::arg("rtflux_is_custom"),
           py::arg("quadrature_degree"), py::arg("list_bfcts_prime"),
           py::arg("reconstruct_stress"));
}

PYBIND11_MODULE(cpp, m)
{
  // Create module for C++ wrappers
  m.doc() = "DOLFINx_eqlb Python interface";

  // Local solver
  declare_lsolver<double>(m);

  // Boundary condition for fluxes
  declare_bcs<double>(m);

  // Equilibration of vector-valued quantity
  declare_fluxeqlb<double>(m);
}
} // namespace