#include "caster_petsc.h"
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <iostream>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <dolfinx_eqlb/local_solver.hpp>
#include <dolfinx_eqlb/reconstruction.hpp>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(cpp, m)
{
  using namespace dolfinx_adaptivity;

  // Create module for C++ wrappers
  m.doc() = "DOLFINX equilibration routines Python interface";

  // Test function
  m.def("test_pybind", &test_pybind, "Hello-World from c++");

  // Local solver
  m.def("local_solver", [](dolfinx::fem::Function<PetscScalar>& sol_elmt,
                           const dolfinx::fem::Form<PetscScalar>& a,
                           const dolfinx::fem::Form<PetscScalar>& l)
        { local_solver<PetscScalar>(sol_elmt, a, l); });

  // Equilibartion of vector-valued quantity
  m.def(
      "reconstruct_fluxes",
      [](const dolfinx::fem::Form<PetscScalar>& a,
         const dolfinx::fem::Form<PetscScalar>& l_pen,
         const std::vector<
             std::shared_ptr<const dolfinx::fem::Form<PetscScalar>>>& l,
         const std::vector<std::vector<std::int32_t>>& fct_esntbound_prime,
         const std::vector<std::vector<std::int32_t>>& fct_esntbound_flux,
         const std::vector<std::vector<std::shared_ptr<
             const dolfinx::fem::DirichletBC<PetscScalar>>>>& bcs1,
         dolfinx::fem::Function<PetscScalar>& flux_hdiv,
         dolfinx::fem::Function<PetscScalar>& flux_dg)
      {
        equilibration::reconstruct_fluxes<PetscScalar>(
            a, l_pen, l, fct_esntbound_prime, fct_esntbound_flux, bcs1,
            flux_hdiv, flux_dg);
      },
      py::arg("a"), py::arg("l_pen"), py::arg("l"),
      py::arg("fct_esntbound_prime"), py::arg("fct_esntbound_prime"),
      py::arg("bcs"), py::arg("flux_hdiv"), py::arg("flux_dg"),
      "Local equilibartion of H(div) conforming fluxes.");
  //   m.def("reconstruct_flux_patch",
  //         &equilibration::reconstruct_fluxes<PetscScalar>);
}
