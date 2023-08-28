#include "caster_petsc.h"
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <ufcx.h>

#include <dolfinx_eqlb/FluxBC.hpp>
#include <dolfinx_eqlb/local_solver.hpp>
#include <dolfinx_eqlb/reconstruction.hpp>

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
        reconstruct_fluxes_ev<double>(a, l_pen, l, fct_esntbound_prime,
                                      fct_esntbound_flux, bcs1, flux_hdiv);
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
        reconstruct_fluxes_cstm<double>(flux_hdiv, flux_dg, rhs_dg,
                                        fct_esntbound_prime, fct_esntbound_flux,
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
  py::class_<FluxBC<T>, std::shared_ptr<FluxBC<T>>>(m, "FluxBC",
                                                    "FluxBC object")
      .def(
          py::init(
              [](std::shared_ptr<const fem::FunctionSpace> function_space,
                 const std::vector<std::int32_t>& boundary_facets,
                 std::uintptr_t kernel_ptr, int n_bceval_per_fct,
                 bool projection_required,
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
                return FluxBC<T>(function_space, boundary_facets,
                                 tabulate_tensor_ptr, n_bceval_per_fct,
                                 projection_required, coefficients,
                                 positions_of_coefficients, constants);
              }),
          py::arg("function_space"), py::arg("facets"),
          py::arg("pointer_boundary_kernel"), py::arg("nevals_per_fct"),
          py::arg("projection_required"), py::arg("coefficients"),
          py::arg("position_of_coefficients"), py::arg("constants"));
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