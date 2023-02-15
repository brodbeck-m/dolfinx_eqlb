#include "caster_petsc.h"
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
  //   m.def(
  //       "reconstruct_flux_patch",
  //       [](dolfinx::fem::Function<PetscScalar>& sol_elmt,
  //          const dolfinx::fem::Form<PetscScalar>& a,
  //          const dolfinx::fem::Form<PetscScalar>& l)
  //       { dolfinx_eqlb::reconstruct_flux_patch<PetscScalar>(sol_elmt, a, l);
  //       }, py::return_value_policy::take_ownership);
  m.def("reconstruct_flux_patch",
        &equilibration::reconstruct_fluxes<PetscScalar>);
}
