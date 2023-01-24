#include "caster_petsc.h"
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <iostream>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <dolfinx_eqlb/local_solver.hpp>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(cpp, m)
{
  // Create module for C++ wrappers
  m.doc() = "DOLFINX equilibration routines Python interface";

  // Test function
  m.def("test_pybind", &dolfinx_eqlb::test_pybind, "Hello-World from c++");

  // Local solver
  m.def("local_solver", [](dolfinx::fem::Function<PetscScalar>& sol_elmt,
                           const dolfinx::fem::Form<PetscScalar>& a,
                           const dolfinx::fem::Form<PetscScalar>& l)
        { dolfinx_eqlb::local_solver<PetscScalar>(sol_elmt, a, l); });
}
