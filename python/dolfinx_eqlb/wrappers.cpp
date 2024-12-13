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

template <dolfinx::scalar T, std::floating_point U>
void declare_lsolver(nb::module_& m)
{
  m.def(
      "local_solver",
      [](std::vector<std::shared_ptr<dolfinx::fem::Function<T, U>>>& solutions,
         const dolfinx::fem::Form<T, U>& a,
         const std::vector<std::shared_ptr<const dolfinx::fem::Form<T, U>>>& ls)
      { base::local_solver<T, U>(solutions, a, ls); },
      nb::arg("solutions"), nb::arg("a"), nb::arg("ls"), "Local solver");
}

template <dolfinx::scalar T, std::floating_point U>
void declare_bcs(nb::module_& m)
{
  nb::class_<base::FluxBC<T, U>>(m, "FluxBC", "FluxBC object")
      .def(
          "__init__",
          [](base::FluxBC<T, U>* fp,
             const std::vector<std::int32_t>& boundary_facets,
             std::uintptr_t fn_addr,
             std::vector<std::shared_ptr<const fem::Constant<T>>> constants,
             std::vector<std::shared_ptr<const fem::Function<T, U>>>
                 coefficients,
             bool is_zero, bool is_timedependent, bool has_time_function,
             int quadrature_degree)
          {
            using kern = std::function<void(T*, const T*, const T*, const U*,
                                            const int*, const std::uint8_t*)>;

            auto tabulate_expression_ptr
                = (void (*)(T*, const T*, const T*, const U*, const int*,
                            const std::uint8_t*))fn_addr;

            new (fp) base::FluxBC<T, U>(boundary_facets,
                                        tabulate_expression_ptr, constants,
                                        coefficients, is_zero, is_timedependent,
                                        has_time_function, quadrature_degree);
          },
          nb::arg("boundary_facets"), nb::arg("boundary_expression"),
          nb::arg("constants"), nb::arg("coefficients"), nb::arg("is_zero"),
          nb::arg("is_timedependent"), nb::arg("has_time_function"),
          nb::arg("quadrature_degree"));
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
  declare_lsolver<double, double>(m);
  declare_bcs<double, double>(m);

  // Some simple test functions
  m.def("function_ev", []() { ev::function_ev(); }, "A function from ev");
  m.def("test_eigen", []() { ev::test_eigen(); }, "Test usage of eigen in c++");
  m.def("function_se", []() { se::function_se(); }, "A function from se");
}