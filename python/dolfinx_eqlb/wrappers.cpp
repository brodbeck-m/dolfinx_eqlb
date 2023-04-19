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

  // Local solver
  m.def(
      "local_solver_lu",
      [](std::vector<std::shared_ptr<dolfinx::fem::Function<PetscScalar>>>&
             sol_elmt,
         const dolfinx::fem::Form<PetscScalar>& a,
         const std::vector<
             std::shared_ptr<const dolfinx::fem::Form<PetscScalar>>>& l)
      { local_solver_lu<PetscScalar>(sol_elmt, a, l); },
      py::arg("solution"), py::arg("a"), py::arg("l"),
      "Local solver based on the LU decomposition");

  m.def(
      "local_solver_cholesky",
      [](std::vector<std::shared_ptr<dolfinx::fem::Function<PetscScalar>>>&
             sol_elmt,
         const dolfinx::fem::Form<PetscScalar>& a,
         const std::vector<
             std::shared_ptr<const dolfinx::fem::Form<PetscScalar>>>& l)
      { local_solver_cholesky<PetscScalar>(sol_elmt, a, l); },
      py::arg("solution"), py::arg("a"), py::arg("l"),
      "Local solver based on the Cholesky decomposition");

  m.def(
      "local_solver_cg",
      [](std::vector<std::shared_ptr<dolfinx::fem::Function<PetscScalar>>>&
             sol_elmt,
         const dolfinx::fem::Form<PetscScalar>& a,
         const std::vector<
             std::shared_ptr<const dolfinx::fem::Form<PetscScalar>>>& l)
      { local_solver_cg<PetscScalar>(sol_elmt, a, l); },
      py::arg("solution"), py::arg("a"), py::arg("l"),
      "Local solver based on a Conjugated-Gradient solver");

  // Equilibartion of vector-valued quantity
  m.def(
      "reconstruct_fluxes_minimisation",
      [](const dolfinx::fem::Form<PetscScalar>& a,
         const dolfinx::fem::Form<PetscScalar>& l_pen,
         const std::vector<
             std::shared_ptr<const dolfinx::fem::Form<PetscScalar>>>& l,
         const std::vector<std::vector<std::int32_t>>& fct_esntbound_prime,
         const std::vector<std::vector<std::int32_t>>& fct_esntbound_flux,
         const std::vector<std::vector<std::shared_ptr<
             const dolfinx::fem::DirichletBC<PetscScalar>>>>& bcs1,
         std::vector<std::shared_ptr<dolfinx::fem::Function<PetscScalar>>>&
             flux_hdiv)
      {
        equilibration::reconstruct_fluxes_ev<PetscScalar>(
            a, l_pen, l, fct_esntbound_prime, fct_esntbound_flux, bcs1,
            flux_hdiv);
      },
      py::arg("a"), py::arg("l_pen"), py::arg("l"),
      py::arg("fct_esntbound_prime"), py::arg("fct_esntbound_prime"),
      py::arg("bcs"), py::arg("flux_hdiv"),
      "Local equilibartion of H(div) conforming fluxes, solving patch-wise "
      "constrained minimisation problems.");

  m.def(
      "reconstruct_fluxes_semiexplt",
      [](std::vector<std::shared_ptr<dolfinx::fem::Function<PetscScalar>>>&
             flux_hdiv,
         std::vector<std::shared_ptr<dolfinx::fem::Function<PetscScalar>>>&
             flux_dg,
         std::vector<std::shared_ptr<dolfinx::fem::Function<PetscScalar>>>&
             rhs_dg,
         const std::vector<std::vector<std::int32_t>>& fct_esntbound_prime,
         const std::vector<std::vector<std::int32_t>>& fct_esntbound_flux,
         const std::vector<std::vector<std::shared_ptr<
             const dolfinx::fem::DirichletBC<PetscScalar>>>>& bcs,
         const std::vector<std::shared_ptr<const fem::Form<PetscScalar>>>&
             form_o1)
      {
        equilibration::reconstruct_fluxes_cstm<PetscScalar>(
            flux_hdiv, flux_dg, rhs_dg, fct_esntbound_prime, fct_esntbound_flux,
            bcs, form_o1);
      },
      py::arg("flux_hdiv"), py::arg("flux_dg"), py::arg("rhs_dg"),
      py::arg("fct_esntbound_prime"), py::arg("fct_esntbound_prime"),
      py::arg("bcs"), py::arg("forms"),
      "Local equilibartion of H(div) conforming fluxes, solving patch-wise "
      "constrained minimisation problems within a semi-explisit approach.");
}
