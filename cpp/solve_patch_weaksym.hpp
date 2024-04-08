#pragma once

#include "eigen3/Eigen/Dense"

#include "KernelData.hpp"
#include "Patch.hpp"
#include "PatchCstm.hpp"
#include "PatchData.hpp"
#include "ProblemDataStress.hpp"
#include "minimise_flux.hpp"
#include "utils.hpp"

#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/graph/AdjacencyList.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <ios>
#include <iostream>
#include <iterator>
#include <span>
#include <tuple>
#include <vector>

using namespace dolfinx;

namespace dolfinx_eqlb
{

namespace stdex = std::experimental;

template <typename T, int id_flux_order, bool modified_patch>
void impose_weak_symmetry(const mesh::Geometry& geometry,
                          PatchFluxCstm<T, id_flux_order>& patch,
                          PatchDataCstm<T, id_flux_order>& patch_data,
                          ProblemDataFluxCstm<T>& problem_data,
                          KernelDataEqlb<T>& kernel_data,
                          kernel_fn<T, true>& minkernel)
{
  /* Extract data */
  // The spatial dimension
  const int gdim = patch.dim();

  // The patch
  const int ncells = patch.ncells();

  // The flux space
  const int ndofs_flux = patch.ndofs_flux();
  const int ndofs_flux_fct = patch.ndofs_flux_fct();

  /* Initialisations */
  // Patch type and reversion information
  std::vector<PatchType> patch_types(gdim, PatchType::internal);
  std::vector<bool> patch_reversions(gdim, false);

  if (gdim == 2)
  {
    patch_types[0] = patch.type(0);
    patch_types[1] = patch.type(1);

    patch_reversions[0] = patch.reversion_required(0);
    patch_reversions[1] = patch.reversion_required(1);
  }
  else
  {
    patch_types[0] = patch.type(0);
    patch_types[1] = patch.type(1);
    patch_types[2] = patch.type(1);

    patch_reversions[0] = patch.reversion_required(0);
    patch_reversions[1] = patch.reversion_required(1);
    patch_reversions[2] = patch.reversion_required(1);
  }

  // Data for Piola mapping
  if constexpr (modified_patch)
  {
    /* Extract data */
    // The geometry
    const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();
    std::span<const double> x = geometry.x();

    // The patch
    std::span<const std::int32_t> cells = patch.cells();

    /* Initialisation */
    // Isoparametric mapping
    std::array<double, 9> Jb;
    std::array<double, 18> detJ_scratch;
    mdspan_t<double, 2> J(Jb.data(), 2, 2);

    // Storage cell geometry
    std::array<double, 12> coordinate_dofs_e;
    mdspan_t<const double, 2> coords(coordinate_dofs_e.data(),
                                     kernel_data.nnodes_cell(), 3);

    /* Evaluate Piola mapping */
    for (std::size_t a = 1; a < ncells + 1; ++a)
    {
      // Current cell
      std::size_t id_a = a - 1;
      std::int32_t c = cells[a];

      // Copy points of current cell
      std::span<const std::int32_t> x_dofs = x_dofmap.links(c);
      copy_cell_data<double, 3>(x, x_dofs, coordinate_dofs_e, 3);

      // Calculate Jacobi matrix and determinant
      double detJ = kernel_data.compute_jacobian(J, detJ_scratch, coords);
      patch_data.store_piola_mapping(id_a, detJ, J);
    }

    /* Initialise assembly informations */
    patch.set_assembly_informations(kernel_data.fct_normal_is_outward(),
                                    patch_data.jacobi_determinant());
  }

  // Coefficients of the stress tensor
  if (!(modified_patch))
  {
    // Flattened storage of stress coefficients
    std::span<T> stress_coefficients = patch_data.coefficients_stress();

    // Copy the coefficients
    for (std::size_t i_row = 0; i_row < gdim; ++i_row)
    {
      for (std::size_t a = 1; a < ncells + 1; ++a)
      {
        // Set offset
        std::size_t offset = (a - 1) * ndofs_flux * gdim + i_row * ndofs_flux;

        // Coefficients of flux i on cell a
        std::span<const T> coeffs_rowi_cella
            = patch_data.coefficients_flux(i_row, a);

        // Move coefficients to flattened storage
        std::copy_n(coeffs_rowi_cella.begin(), ndofs_flux,
                    stress_coefficients.begin() + offset);
      }
    }
  }

  /* Solve minimisation problem */
  // Set DOFmap on patch
  mdspan_t<std::int32_t, 3> asmbl_info
      = patch_data.assembly_info_constained_minimisation();

  set_flux_dofmap(patch, asmbl_info);

  // Set boundary markers
  set_boundary_markers(patch_data.boundary_markers(true), Kernel::StressMin,
                       patch_types, gdim, ncells, ndofs_flux_fct,
                       patch_reversions);

  // Assemble equation system
  // mdspan_t<const std::int32_t, 3> asmbl_info_base
  //     = patch.assembly_info_minimisation();

  // std::cout << "Cells on patch: " << std::endl;
  // for (auto c : patch.cells())
  // {
  //   std::cout << c << " ";
  // }
  // std::cout << "\n";

  std::cout << "DOFmap (local): " << std::endl;
  for (std::size_t i = 0; i < asmbl_info.extent(1); ++i)
  {
    for (std::size_t j = 0; j < asmbl_info.extent(2); ++j)
    {
      std::cout << asmbl_info(0, i, j) << " ";
    }
    std::cout << "\n";
  }

  // std::cout << "DOFmap (patch, non-vector): " << std::endl;
  // for (std::size_t i = 0; i < asmbl_info_base.extent(1); ++i)
  // {
  //   for (std::size_t j = 0; j < asmbl_info_base.extent(2); ++j)
  //   {
  //     std::cout << asmbl_info_base(2, i, j) << " ";
  //   }
  //   std::cout << "\n";
  // }

  std::cout << "DOFmap (patch): " << std::endl;
  for (std::size_t i = 0; i < asmbl_info.extent(1); ++i)
  {
    for (std::size_t j = 0; j < asmbl_info.extent(2); ++j)
    {
      std::cout << asmbl_info(2, i, j) << " ";
    }
    std::cout << "\n";
  }
  std::cout << "DOFmap (prefactors): " << std::endl;
  for (std::size_t i = 0; i < asmbl_info.extent(1); ++i)
  {
    for (std::size_t j = 0; j < asmbl_info.extent(2); ++j)
    {
      std::cout << asmbl_info(3, i, j) << " ";
    }
    std::cout << "\n";
  }

  std::cout << "detJ: " << std::endl;
  for (auto j : patch_data.jacobi_determinant())
  {
    std::cout << j << " ";
  }
  std::cout << "\n";

  assemble_fluxminimiser<T, id_flux_order, true>(
      minkernel, patch_data, asmbl_info, 0, patch.requires_flux_bcs(), true);

  // Solve equation system
  patch_data.factorise_system(true);
  patch_data.solve_system(true);

  /* Store local solution into global storage */
  Eigen::Matrix<T, Eigen::Dynamic, 1>& u_patch = patch_data.u_patch(true);
  const int ndofs_flux_per_cell
      = gdim * ndofs_flux_fct + patch.ndofs_flux_cell_add();

  std::cout << "Solution patch " << patch.node_i() << std::endl;
  const int size_exp = patch.ndofs_minspace_flux(true) + patch.nfcts() + 1;
  for (std::size_t i = 0; i < size_exp; ++i)
  {
    std::cout << u_patch(i) << " ";
  }
  std::cout << "\n";

  for (std::size_t i_row = 0; i_row < gdim; ++i_row)
  {
    // Global storage of the solution
    std::span<T> x_flux_dhdiv = problem_data.flux(i_row).x()->mutable_array();

    // Loop over cells
    for (std::int32_t a = 1; a < ncells + 1; ++a)
    {
      // Initialise offset
      int offset = i_row;

      // Map solution from H(div=0) to H(div) space
      for (std::size_t i = 0; i < ndofs_flux_per_cell; ++i)
      {
        // Local to global storage
        x_flux_dhdiv[asmbl_info(1, a, offset)]
            += asmbl_info(3, a, offset) * u_patch(asmbl_info(2, a, offset));

        // Increment offset
        offset += gdim;
      }
    }
  }
}
} // namespace dolfinx_eqlb