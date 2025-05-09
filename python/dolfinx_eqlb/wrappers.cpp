// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <basix/finite-element.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx_eqlb/base/deqlb_base.hpp>
#include <dolfinx_eqlb/base/local_solver.hpp>
// #include <dolfinx_eqlb/ev/reconstruction.hpp>
// #include <dolfinx_eqlb/se/reconstruction.hpp>
#include <ufcx.h>

#include <dolfinx_eqlb/ev/deqlb_ev.hpp>
#include <dolfinx_eqlb/se/deqlb_se.hpp>

#include <caster_petsc.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
// #include <nanobind/stl/array.h>
// #include <nanobind/stl/map.h>
// #include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
// #include <nanobind/stl/string.h>
// #include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
// #include <petsc4py/petsc4py.h>

#include <cstdint>
// #include <functional>
#include <memory>
// #include <stdexcept>
// #include <type_traits>
#include <vector>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace nb = nanobind;

using namespace nb::literals;
using namespace dolfinx_eqlb;
using scalar_t = PetscScalar;

template <dolfinx::scalar T>
void declare_lsolver(nb::module_& m)
{
  using U = typename dolfinx::scalar_value_type_t<T>;

  m.def(
      "local_solver",
      [](std::vector<std::shared_ptr<dolfinx::fem::Function<T, U>>>& solutions,
         const dolfinx::fem::Form<T, U>& a,
         const std::vector<std::shared_ptr<const dolfinx::fem::Form<T, U>>>& ls)
      { base::local_solver<T, U>(solutions, a, ls); },
      nb::arg("solutions"), nb::arg("a"), nb::arg("ls"), "Local solver");
}

template <dolfinx::scalar T>
void declare_bcs(nb::module_& m)
{
  using U = typename dolfinx::scalar_value_type_t<T>;

  nb::enum_<base::TimeType>(m, "TimeType", nb::is_arithmetic(),
                            "Time-dependency of a boundary condition.")
      .value("stationary", base::TimeType::stationary)
      .value("timefunction", base::TimeType::timefunction)
      .value("timedependent", base::TimeType::timedependent);

  nb::class_<base::FluxBC<T, U>>(m, "FluxBC", "FluxBC object")
      .def(
          "__init__",
          [](base::FluxBC<T, U>* fp,
             std::shared_ptr<const fem::Expression<T, U>> value,
             const std::vector<std::int32_t>& facets,
             std::shared_ptr<const fem::FunctionSpace<U>> V,
             const int quadrature_degree, const base::TimeType tbehaviour)
          {
            new (fp) base::FluxBC<T, U>(value, facets, V, quadrature_degree,
                                        tbehaviour);
          },
          nb::arg("boundary_expression"), nb::arg("boundary_facets"),
          nb::arg("FunctionSpace"), nb::arg("quadrature_degree"),
          nb::arg("transient_behaviour"))
      .def(
          "__init__",
          [](base::FluxBC<T, U>* fp, const std::vector<std::int32_t>& facets)
          { new (fp) base::FluxBC<T, U>(facets); }, nb::arg("boundary_facets"))
      .def_prop_ro("quadrature_degree", &base::FluxBC<T, U>::quadrature_degree);

  nb::class_<base::BoundaryData<T, U>>(m, "BoundaryData", "BoundaryData object")
      .def(
          "__init__",
          [](base::BoundaryData<T, U>* fp,
             std::vector<std::vector<std::shared_ptr<base::FluxBC<T, U>>>>&
                 list_bcs,
             std::vector<std::shared_ptr<fem::Function<T, U>>>& boundary_fluxes,
             std::shared_ptr<const fem::FunctionSpace<U>> V,
             const std::vector<std::vector<std::int32_t>>& fct_esntbound_prime,
             base::KernelDataBC<T, U>& kernel_data,
             const base::ProblemType problem_type)
          {
            new (fp) base::BoundaryData<T, U>(list_bcs, boundary_fluxes, V,
                                              fct_esntbound_prime, kernel_data,
                                              problem_type);
          },
          nb::arg("list_bcs"), nb::arg("boundary_fluxes"), nb::arg("V"),
          nb::arg("fct_esntbound_prime"), nb::arg("kernel_data"),
          nb::arg("problem_type"))
      .def(
          "update",
          [](base::BoundaryData<T, U>& self,
             std::vector<std::shared_ptr<const fem::Constant<T>>>&
                 time_functions) { self.update(time_functions); },
          nb::arg("time_functions"));
}

template <dolfinx::scalar T>
void declare_equilibrator(nb::module_& m)
{
  using U = typename dolfinx::scalar_value_type_t<T>;

  nb::enum_<base::ProblemType>(m, "ProblemType", nb::is_arithmetic(),
                               "Type of the equilibration problem.")
      .value("flux", base::ProblemType::flux)
      .value("stress", base::ProblemType::stress)
      .value("stress_and_flux", base::ProblemType::stress_and_flux);

  nb::enum_<base::EqStrategy>(m, "EqStrategy", nb::is_arithmetic(),
                              "The used equilibration strategy.")
      .value("semi_explicit", base::EqStrategy::semi_explicit)
      .value("constrained_minimisation",
             base::EqStrategy::constrained_minimisation);

  nb::class_<base::Equilibrator<T, U>>(m, "Equilibrator",
                                       "Basic Equilibrator object")
      .def(
          "__init__",
          [](base::Equilibrator<T, U>* fp, const base::ProblemType problem_type,
             const base::EqStrategy strategy,
             const basix::FiniteElement<U>& element_geom,
             const basix::FiniteElement<U>& element_hat,
             const basix::FiniteElement<U>& element_flux,
             const int quadrature_degree_bcs)
          {
            new (fp) base::Equilibrator<T, U>(
                problem_type, strategy, element_geom, element_hat, element_flux,
                quadrature_degree_bcs);
          },
          nb::arg("problem_type"), nb::arg("strategy"), nb::arg("element_geom"),
          nb::arg("element_hat"), nb::arg("element_flux"),
          nb::arg("quadrature_degree_bcs"))
      .def_prop_ro("problem_type", &base::Equilibrator<T, U>::problem_type,
                   nb::rv_policy::reference_internal)
      .def_prop_ro("strategy", &base::Equilibrator<T, U>::strategy,
                   nb::rv_policy::reference_internal)
      .def_prop_ro("basix_element_hat",
                   &base::Equilibrator<T, U>::basix_element_hat,
                   nb::rv_policy::reference_internal)
      .def_prop_ro("basix_element_flux",
                   &base::Equilibrator<T, U>::basix_element_hat,
                   nb::rv_policy::reference_internal)
      .def_prop_ro("kernel_data_bcs",
                   &base::Equilibrator<T, U>::kernel_data_bcs,
                   nb::rv_policy::reference_internal);
  nb::class_<base::KernelDataBC<T, U>>(m, "KernelDataBC",
                                       "Kernel data for boundary conditions")
      .def(
          "__init__",
          [](base::KernelDataBC<T, U>* fp,
             const basix::FiniteElement<U>& element_geom,
             std::tuple<int, int> quadrature_rule,
             const basix::FiniteElement<U>& element_hat,
             const basix::FiniteElement<U>& element_flux,
             const base::EqStrategy equilibration_strategy)
          {
            new (fp) base::KernelDataBC<T, U>(element_geom, quadrature_rule,
                                              element_hat, element_flux,
                                              equilibration_strategy);
          },
          nb::arg("element_geom"), nb::arg("quadrature_rule"),
          nb::arg("element_hat"), nb::arg("element_flux"),
          nb::arg("equilibration_strategy"));
}

NB_MODULE(cpp, m)
{
  // Create module for C++ wrappers
  m.doc() = "dolfinx_eqlb Python interface";
  m.attr("__version__") = DOLFINX_EQLB_VERSION;

#ifdef NDEBUG
  nanobind::set_leak_warnings(false);
#endif

  // The local solver
  declare_lsolver<double>(m);
  declare_equilibrator<double>(m);
  declare_bcs<double>(m);
}