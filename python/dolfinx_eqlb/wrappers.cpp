#include "caster_petsc.h"
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <dolfinx_eqlb/FluxBC.hpp>
#include <dolfinx_eqlb/local_solver.hpp>
#include <dolfinx_eqlb/reconstruction.hpp>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
using namespace dolfinx_adaptivity;

template <typename T>
void declare_lsolver(py::module& m)
{
  m.def(
      "local_solver_lu",
      [](std::vector<std::shared_ptr<dolfinx::fem::Function<T>>>& sol_elmt,
         const dolfinx::fem::Form<T>& a,
         const std::vector<std::shared_ptr<const dolfinx::fem::Form<T>>>& l)
      { local_solver_lu<T>(sol_elmt, a, l); },
      py::arg("solution"), py::arg("a"), py::arg("l"),
      "Local solver based on the LU decomposition");

  m.def(
      "local_solver_cholesky",
      [](std::vector<std::shared_ptr<dolfinx::fem::Function<T>>>& sol_elmt,
         const dolfinx::fem::Form<T>& a,
         const std::vector<std::shared_ptr<const dolfinx::fem::Form<T>>>& l)
      { local_solver_cholesky<T>(sol_elmt, a, l); },
      py::arg("solution"), py::arg("a"), py::arg("l"),
      "Local solver based on the Cholesky decomposition");

  m.def(
      "local_solver_cg",
      [](std::vector<std::shared_ptr<dolfinx::fem::Function<T>>>& sol_elmt,
         const dolfinx::fem::Form<T>& a,
         const std::vector<std::shared_ptr<const dolfinx::fem::Form<T>>>& l)
      { local_solver_cg<T>(sol_elmt, a, l); },
      py::arg("solution"), py::arg("a"), py::arg("l"),
      "Local solver based on a Conjugated-Gradient solver");
}

template <typename T>
void declare_fluxeqlb(py::module& m)
{
  m.def(
      "reconstruct_fluxes_minimisation",
      [](const dolfinx::fem::Form<double>& a,
         const dolfinx::fem::Form<double>& l_pen,
         const std::vector<std::shared_ptr<const dolfinx::fem::Form<double>>>&
             l,
         const std::vector<std::vector<std::int32_t>>& fct_esntbound_prime,
         const std::vector<std::vector<std::int32_t>>& fct_esntbound_flux,
         const std::vector<std::vector<
             std::shared_ptr<const dolfinx::fem::DirichletBC<double>>>>& bcs1,
         std::vector<std::shared_ptr<dolfinx::fem::Function<double>>>&
             flux_hdiv)
      {
        equilibration::reconstruct_fluxes_ev<double>(
            a, l_pen, l, fct_esntbound_prime, fct_esntbound_flux, bcs1,
            flux_hdiv);
      },
      py::arg("a"), py::arg("l_pen"), py::arg("l"),
      py::arg("fct_esntbound_prime"), py::arg("fct_esntbound_prime"),
      py::arg("bcs"), py::arg("flux_hdiv"),
      "Local equilibration of H(div) conforming fluxes, solving patch-wise, "
      "constrained minimisation problems.");

  m.def(
      "reconstruct_fluxes_semiexplt",
      [](std::vector<std::shared_ptr<dolfinx::fem::Function<double>>>&
             flux_hdiv,
         std::vector<std::shared_ptr<dolfinx::fem::Function<double>>>& flux_dg,
         std::vector<std::shared_ptr<dolfinx::fem::Function<double>>>& rhs_dg,
         const std::vector<std::vector<std::int32_t>>& fct_esntbound_prime,
         const std::vector<std::vector<std::int32_t>>& fct_esntbound_flux,
         const std::vector<std::vector<
             std::shared_ptr<const dolfinx::fem::DirichletBC<double>>>>& bcs)
      {
        equilibration::reconstruct_fluxes_cstm<double>(
            flux_hdiv, flux_dg, rhs_dg, fct_esntbound_prime, fct_esntbound_flux,
            bcs);
      },
      py::arg("flux_hdiv"), py::arg("flux_dg"), py::arg("rhs_dg"),
      py::arg("fct_esntbound_prime"), py::arg("fct_esntbound_prime"),
      py::arg("bcs"),
      "Local equilibration of H(div) conforming fluxes, using an explicit "
      "determination of the flues alongside with an unconstrained ministration "
      "problem on a reduced space.");
}

template <typename T>
void declare_bcs(py::module& m)
{
  py::class_<equilibration::FluxBC<T>,
             std::shared_ptr<equilibration::FluxBC<T>>>(m, "FluxBC",
                                                        "FluxBC object")
      .def(py::init<const std::vector<std::int32_t>&, double, int, int, bool>(),
           py::arg("facets"), py::arg("value"), py::arg("nevals_per_fct"),
           py::arg("ndofs_per_fct"), py::arg("projection_required"));
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