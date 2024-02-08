#pragma once

#include "eigen3/Eigen/Dense"

#include "KernelData.hpp"
#include "Patch.hpp"
#include "PatchFluxCstm.hpp"
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
#include <iostream>
#include <iterator>
#include <span>
#include <vector>

using namespace dolfinx;

namespace dolfinx_eqlb
{

namespace stdex = std::experimental;

// ------------------------------------------------------------------------------

/* Store maping data (J, K) per patch */

/// Store mapping data (Jacobian or its inverse) in flattened array
/// @param cell_id The patch-local index of a cell
/// @param storage The flattened storage
/// @param matrix  The matrix (J, K) on the current cell
void store_mapping_data(const int cell_id, std::span<double> storage,
                        mdspan2_t matrix)
{
  // Set offset
  const int offset = 4 * cell_id;

  storage[offset] = matrix(0, 0);
  storage[offset + 1] = matrix(0, 1);
  storage[offset + 2] = matrix(1, 0);
  storage[offset + 3] = matrix(1, 1);
}

/// Extract mapping data (Jacobian or its inverse) from flattened array
/// @param cell_id The patch-local index of a cell
/// @param storage The flattened storage
/// @return        The matrix (J, K) on the current cell
cmdspan2_t extract_mapping_data(const int cell_id,
                                std::span<const double> storage)
{
  // Set offset
  const int offset = 4 * cell_id;

  return cmdspan2_t(storage.data() + offset, 2, 2);
}
// ------------------------------------------------------------------------------

// ------------------------------------------------------------------------------

/* Minimise fluxes without constraint */

template <typename T, int id_flux_order = 3>
std::vector<std::int32_t>
set_flux_dofmap(PatchFluxCstm<T, id_flux_order>& patch,
                const std::vector<bool>& facet_orientation,
                std::span<const double> storage_detJ)
{
  // Patch data
  const int ncells = patch.ncells();
  const int nfcts = patch.nfcts();

  const int ndofs_flux_fct = patch.ndofs_flux_fct();
  const int ndofs_flux_cell_div = patch.ndofs_flux_cell_div();
  const int ndofs_flux_cell_add = patch.ndofs_flux_cell_add();
  const int ndofs_cell_local = 2 * ndofs_flux_fct + ndofs_flux_cell_add;

  // Initialise storage
  std::vector<std::int32_t> ddofmap_patch(4 * ncells * ndofs_cell_local, 0);
  mdspan_t<std::int32_t, 3> dofmap_patch(ddofmap_patch.data(), 4,
                                         (std::size_t)ncells,
                                         (std::size_t)ndofs_cell_local);
  std::int8_t fctloc_ea, fctloc_eam1;
  std::int32_t prefactor_ea, prefactor_eam1;

  // Loop over cells
  for (std::size_t a = 1; a < ncells + 1; ++a)
  {
    // Set id for accessing storage
    int id_a = a - 1;

    // Get facet orientations
    std::tie(fctloc_eam1, fctloc_ea) = patch.fctid_local(a);

    // Prefactors due to facet orientation
    if (storage_detJ[a - 1] < 0)
    {
      prefactor_eam1 = (facet_orientation[fctloc_eam1]) ? -1 : 1;
      prefactor_ea = (facet_orientation[fctloc_ea]) ? -1 : 1;
    }
    else
    {
      prefactor_eam1 = (facet_orientation[fctloc_eam1]) ? 1 : -1;
      prefactor_ea = (facet_orientation[fctloc_ea]) ? 1 : -1;
    }

    // Non-zero DOFs on cell
    std::span<const std::int32_t> ldofs_fct = patch.dofs_flux_fct_local(a);
    std::span<const std::int32_t> gdofs_fct = patch.dofs_flux_fct_global(a);

    /* Set DOFmap */
    // DOFs associated with d_0 (cell-local, patch-local, global) and prefactors
    dofmap_patch(0, id_a, 0) = ldofs_fct[0];
    dofmap_patch(1, id_a, 0) = 0;
    dofmap_patch(2, id_a, 0) = gdofs_fct[0];
    dofmap_patch(3, id_a, 0) = prefactor_eam1;

    dofmap_patch(0, id_a, 1) = ldofs_fct[ndofs_flux_fct];
    dofmap_patch(1, id_a, 1) = 0;
    dofmap_patch(2, id_a, 1) = gdofs_fct[ndofs_flux_fct];
    dofmap_patch(3, id_a, 1) = prefactor_ea;

    // Set higher order DOFs
    if constexpr (id_flux_order > 1)
    {
      const PatchType type_patch = patch.type(0);

      if constexpr (id_flux_order == 2)
      {
        int iea = ndofs_flux_fct + 1;

        // DOFs d^l_E (cell-local, global) and prefactors
        dofmap_patch(0, id_a, 2) = ldofs_fct[1];
        dofmap_patch(2, id_a, 2) = gdofs_fct[1];
        dofmap_patch(3, id_a, 2) = prefactor_eam1;

        dofmap_patch(0, id_a, 3) = ldofs_fct[iea];
        dofmap_patch(2, id_a, 3) = gdofs_fct[iea];
        dofmap_patch(3, id_a, 3) = prefactor_ea;

        // Patch-local DOFs d^l_E
        if ((type_patch == PatchType::internal) && (a == ncells))
        {
          dofmap_patch(1, id_a, 2) = a;
          dofmap_patch(1, id_a, 3) = 1;
        }
        else
        {
          dofmap_patch(1, id_a, 2) = a;
          dofmap_patch(1, id_a, 3) = 1 + a;
        }
      }
      else
      {
        // Set facet DOFs
        const int offs_e = ndofs_flux_fct - 1;
        int offs_ea, offs_eam1;

        if ((type_patch == PatchType::internal) && (a == ncells))
        {
          offs_eam1 = (a - 1) * (ndofs_flux_fct - 1) - 1;
          offs_ea = -1;
        }
        else
        {
          offs_eam1 = (a - 1) * (ndofs_flux_fct - 1) - 1;
          offs_ea = offs_eam1 + offs_e;
        }

        for (std::size_t i = 2; i < ndofs_flux_fct + 1; ++i)
        {
          int ieam1 = i - 1;
          int iea = i + offs_e;

          // DOFs d^l_E (cell-local, patch-local, global) and prefactors
          dofmap_patch(0, id_a, i) = ldofs_fct[ieam1];
          dofmap_patch(1, id_a, i) = offs_eam1 + i;
          dofmap_patch(2, id_a, i) = gdofs_fct[ieam1];
          dofmap_patch(3, id_a, i) = prefactor_eam1;

          dofmap_patch(0, id_a, iea) = ldofs_fct[iea];
          dofmap_patch(1, id_a, iea) = offs_ea + i;
          dofmap_patch(2, id_a, iea) = gdofs_fct[iea];
          dofmap_patch(3, id_a, iea) = prefactor_ea;
        }

        // Set cell DOFs
        std::span<const std::int32_t> ldofs_cell
            = patch.dofs_flux_cell_local(a);
        std::span<const std::int32_t> gdofs_cell
            = patch.dofs_flux_cell_global(a);

        const int offs_ta
            = nfcts * (ndofs_flux_fct - 1) + (a - 1) * ndofs_flux_cell_add + 1;

        for (std::size_t i = 0; i < ndofs_flux_cell_add; ++i)
        {
          int offs_dmp1 = 2 * ndofs_flux_fct + i;
          int offs_dmp2 = ndofs_flux_cell_div + i;

          // DOFs d^r_T (cell-local, patch-local, global)
          dofmap_patch(0, id_a, offs_dmp1) = ldofs_cell[offs_dmp2];
          dofmap_patch(1, id_a, offs_dmp1) = offs_ta + i;
          dofmap_patch(2, id_a, offs_dmp1) = gdofs_cell[offs_dmp2];

          // Initialise prefactors
          dofmap_patch(3, id_a, offs_dmp1) = 1;
        }
      }
    }
  }

  return std::move(ddofmap_patch);
}

std::vector<std::int8_t> initialise_boundary_markers(const int size_psystem)
{
  // Create vector
  std::vector<std::int8_t> boundary_markers(size_psystem, false);

  return std::move(boundary_markers);
}

void set_boundary_markers(std::span<std::int8_t> boundary_markers,
                          const PatchType type_patch, const int ncells,
                          const int ndofs_flux_fct,
                          bool reversion_required = false)
{
  // Reinitialise markers
  std::fill(boundary_markers.begin(), boundary_markers.end(), false);

  // Set markers
  if ((type_patch != PatchType::internal)
      && (type_patch != PatchType::bound_essnt_dual))
  {
    // Set boundary markers
    boundary_markers[0] = true;

    for (std::size_t i = 1; i < ndofs_flux_fct; ++i)
    {
      if (type_patch == PatchType::bound_essnt_dual)
      {
        // Mark DOFs on facet E0 and En
        boundary_markers[i] = true;
        boundary_markers[ncells * (ndofs_flux_fct - 1) + i] = true;
      }
      else
      {
        if (reversion_required)
        {
          // Mark DOFs in facet En
          // (Mixed patch with reversed order)
          boundary_markers[ncells * (ndofs_flux_fct - 1) + i] = true;
        }
        else
        {
          // Mark DOFs in facet E0
          // (Mixed patch with original order)
          boundary_markers[i] = true;
        }
      }
    }
  }
}

template <typename T, int id_flux_order = 3, bool asmbl_systmtrx = true>
void kernel_fluxmin(mdspan2_t Te, KernelDataEqlb<T>& kernel_data,
                    std::span<T> coefficients,
                    smdspan_t<std::int32_t, 2> dofmap, const double detJ,
                    cmdspan2_t J)
{
  const int index_load = Te.extent(0) - 1;

  /* Initialise storage of cell prefactors */
  std::int32_t prefactor_eam1, prefactor_ea;

  /* Extract shape functions and quadrature data */
  smdspan_t<double, 3> phi = kernel_data.shapefunctions_flux(J, detJ);

  std::span<const double> quadrature_weights
      = kernel_data.quadrature_weights(0);

  /* Initialisation */
  // Interpolated solution from step 1
  std::array<T, 2> sigtilde_q;

  // Number of non-zero DOFs on cell
  const int nidofs_per_cell = dofmap.extent(1) - 1;

  // Manipulate prefactors
  dofmap(3, 0) = 1;
  dofmap(3, 1) = 1;

  /* Assemble tangents */
  for (std::size_t iq = 0; iq < quadrature_weights.size(); ++iq)
  {
    // Calculate quadrature weight
    double alpha = quadrature_weights[iq] * std::fabs(detJ);

    // Interpolate sigma_tilde
    if constexpr (id_flux_order == 1)
    {
      sigtilde_q[0] = coefficients[0] * phi(iq, 0, 0)
                      + coefficients[1] * phi(iq, 1, 0)
                      + coefficients[2] * phi(iq, 2, 0);
      sigtilde_q[1] = coefficients[0] * phi(iq, 0, 1)
                      + coefficients[1] * phi(iq, 1, 1)
                      + coefficients[2] * phi(iq, 2, 1);
    }
    else
    {
      // Set storage to zero
      sigtilde_q[0] = 0;
      sigtilde_q[1] = 0;

      // Interpolation
      for (std::size_t i = 0; i < phi.extent(1); ++i)
      {
        sigtilde_q[0] += coefficients[i] * phi(iq, i, 0);
        sigtilde_q[1] += coefficients[i] * phi(iq, i, 1);
      }
    }

    // Manipulate shape function for coefficient d_0
    phi(iq, dofmap(0, 1), 0) = prefactor_eam1 * phi(iq, dofmap(0, 0), 0)
                               - prefactor_ea * phi(iq, dofmap(0, 1), 0);
    phi(iq, dofmap(0, 1), 1) = prefactor_eam1 * phi(iq, dofmap(0, 0), 1)
                               - prefactor_ea * phi(iq, dofmap(0, 1), 1);

    // Assemble linear- and bilinear form
    for (std::size_t i = 0; i < nidofs_per_cell; ++i)
    {
      // Auxilary variables
      std::size_t ip1 = i + 1;
      double pialpha = dofmap(3, ip1) * alpha;

      // Linear form
      Te(index_load, i) -= (phi(iq, dofmap(0, ip1), 0) * sigtilde_q[0]
                            + phi(iq, dofmap(0, ip1), 1) * sigtilde_q[1])
                           * pialpha;

      if constexpr (asmbl_systmtrx)
      {
        for (std::size_t j = i; j < nidofs_per_cell; ++j)
        {
          // Auxiliary variables
          std::size_t jp1 = j + 1;
          double sp = phi(iq, dofmap(0, ip1), 0) * phi(iq, dofmap(0, jp1), 0)
                      + phi(iq, dofmap(0, ip1), 1) * phi(iq, dofmap(0, jp1), 1);

          // Bilinear form
          Te(i, j) += sp * dofmap(3, jp1) * pialpha;
        }
      }
    }
  }

  // Set symmetric contributions of element mass-matrix
  if constexpr (id_flux_order > 1)
  {
    for (std::size_t i = 1; i < nidofs_per_cell; ++i)
    {
      for (std::size_t j = 0; j < i; ++j)
      {
        Te(i, j) = Te(j, i);
      }
    }
  }

  // Undo manipulation of prefactors
  dofmap(3, 0) = prefactor_eam1;
  dofmap(3, 1) = prefactor_ea;
}

// ------------------------------------------------------------------------------

/// Minimise flux on patch-wise divergence-free H(div) space
///
/// Minimises fluxes on an patch-wise, divergence-free H(div) space.
/// Explicite ansatz for such a space see [1, Lemma 12]. The DOFs are
/// therefore locally numbered as follows:
///
/// dofs_patch = [d0, d^l_E0, ..., d^l_En, d^r_T1, ..., d^r_Tn]
///
/// [1] Bertrand, F.; Carstensen, C.; Gräßle, B. & Tran, N. T.:
///     Stabilization-free HHO a posteriori error control, 2022
///
/// @tparam T             The scalar type
/// @tparam id_flux_order The flux order (1->RT1, 2->RT2, 3->general)
/// @param geometry     The geometry
/// @param patch        The patch
/// @param problem_data The problem data
/// @param kernel_data  The kernel data
template <typename T, int id_flux_order = 3, bool asmbl_systmtrx = true>
void assemble_fluxminimiser(
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& A_patch,
    Eigen::Matrix<T, Eigen::Dynamic, 1>& L_patch,
    PatchFluxCstm<T, id_flux_order>& patch, KernelDataEqlb<T>& kernel_data,
    std::span<bool> facet_orientation, mdspan_t<std::int32_t, 3> dofmap_patch,
    std::span<T> coefficients, std::span<const std::int8_t> boundary_markers,
    std::span<const double> storage_detJ, std::span<const double> storage_J,
    std::span<const double> storage_K, const PatchType type_patch)
{
  assert(id_flux_order < 0);

  /* Extract data */
  // Number of elements/facets on patch
  const int ncells = patch.ncells();
  const int nfcts = patch.nfcts();

  // Degree of the RT element
  const int degree_rt = patch.degree_raviart_thomas();

  // DOF counters
  const int ndofs = patch.ndofs_flux();
  const int ndofs_per_fct = degree_rt + 1;
  const int ndofs_cell_div = patch.ndofs_flux_cell_div();
  const int ndofs_cell_add = patch.ndofs_flux_cell_add();

  /* Initialisation */
  // Element tangents
  const int ndofs_nz = 2 * ndofs_per_fct + ndofs_cell_add - 1;
  const int index_load = ndofs_nz;
  std::vector<T> dTe(ndofs_nz * (ndofs_nz + 1), 0);
  mdspan2_t Te(dTe.data(), ndofs_nz + 1, ndofs_nz);

  // Initialise storage for facet orientation
  std::int8_t fctloc_ea, fctloc_eam1;
  std::array<bool, 2> fct_orientation;

  /* Calculation and assembly */
  for (std::size_t a = 1; a < ncells + 1; ++a)
  {
    int id_a = a - 1;

    /* Data extraction */
    // Jacobi matrix
    const double detJ = storage_detJ[id_a];
    cmdspan2_t J = extract_mapping_data(id_a, storage_J);

    // DOFmap
    smdspan_t<std::int32_t, 2> dofmap_cell = stdex::submdspan(
        dofmap_patch, stdex::full_extent, id_a, stdex::full_extent);

    // Coefficients
    std::span<T> coefficients_elmt = coefficients.subspan(id_a * ndofs, ndofs);

    /* Evaluate kernel */
    std::fill(dTe.begin(), dTe.end(), 0);
    kernel_fluxmin<T, id_flux_order, asmbl_systmtrx>(
        Te, kernel_data, coefficients_elmt, dofmap_cell, detJ, J);

    /* Assemble equation system */
    if constexpr (id_flux_order == 1)
    {
      if ((type_patch == PatchType::bound_essnt_dual)
          || (type_patch == PatchType::bound_mixed))
      {
        // Assemble linar form
        L_patch(0) = 0;

        if constexpr (asmbl_systmtrx)
        {
          // Assemble bilinear form
          A_patch(0, 0) = 1;
        }
      }
      else
      {
        // Assemble linar form
        L_patch(0) += Te(1, 0);

        if constexpr (asmbl_systmtrx)
        {
          // Assemble bilinear form
          A_patch(0, 0) += Te(0, 0);
        }
      }
    }
    else
    {
      // // Loop over DOFs
      // for (std::size_t i = 0; i < ndofs_nz; ++i)
      // {
      //   if constexpr (asmbl_systmtrx)
      //   {
      //     for (std::size_t j = 0; j < ndofs_nz; ++j)
      //     {
      //     }
      //   }
      // }
      throw std::runtime_error("Not implemented");
    }
  }
}

} // namespace dolfinx_eqlb
