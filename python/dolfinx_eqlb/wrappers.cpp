// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

// #include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
// #include <dolfinx_eqlb/base/BoundaryData.hpp>
#include <dolfinx_eqlb/base/FluxBC.hpp>
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

  nb::class_<base::FluxBC<T, U>>(m, "FluxBC", "FluxBC object")
      .def(
          "__init__",
          [](base::FluxBC<T, U>* fp,
             std::shared_ptr<const fem::Expression<T, U>> value,
             const std::vector<std::int32_t>& facets,
             std::shared_ptr<const fem::FunctionSpace<U>> V)
          { new (fp) base::FluxBC<T, U>(value, facets, V); },
          nb::arg("boundary_expression"), nb::arg("boundary_facets"),
          nb::arg("FunctionSpace"))
      .def_prop_ro("quadrature_degree", &base::FluxBC<T, U>::quadrature_degree);

  nb::class_<base::BoundaryData<T, U>>(m, "BoundaryData", "BoundaryData object")
      .def(
          "__init__",
          [](base::BoundaryData<T, U>* fp,
             std::vector<std::vector<std::shared_ptr<base::FluxBC<T, U>>>>&
                 list_bcs,
             std::vector<std::shared_ptr<fem::Function<T, U>>>& boundary_flux,
             std::shared_ptr<const fem::FunctionSpace<U>> V_flux_hdiv,
             bool rtflux_is_custom, int quadrature_degree,
             const std::vector<std::vector<std::int32_t>>& fct_esntbound_prime,
             const bool reconstruct_stress)
          {
            new (fp) base::BoundaryData<T, U>(
                list_bcs, boundary_flux, V_flux_hdiv, rtflux_is_custom,
                quadrature_degree, fct_esntbound_prime, reconstruct_stress);
          },
          nb::arg("list_bcs"), nb::arg("boundary_flux"), nb::arg("V_flux_hdiv"),
          nb::arg("rtflux_is_custom"), nb::arg("quadrature_degree"),
          nb::arg("fct_esntbound_prime"), nb::arg("reconstruct_stress"));
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
  declare_bcs<double>(m);

  // Some simple test functions
  m.def("function_ev", []() { ev::function_ev(); }, "A function from ev");
  m.def("test_eigen", []() { ev::test_eigen(); }, "Test usage of eigen in c++");
  m.def("function_se", []() { se::function_se(); }, "A function from se");
}