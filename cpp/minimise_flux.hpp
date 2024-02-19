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
                        mdspan_t<const double, 2> matrix)
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
mdspan_t<const double, 2> extract_mapping_data(const int cell_id,
                                               std::span<const double> storage)
{
  // Set offset
  const int offset = 4 * cell_id;

  return mdspan_t<const double, 2>(storage.data() + offset, 2, 2);
}
// ------------------------------------------------------------------------------

// ------------------------------------------------------------------------------

/* Minimise fluxes without constraint */

/// Initialise and set patch DOFemap
///
/// Patch-wise divergence free H(div) space requires special DOFmap and
/// prefacers during assembly. Following [1, Lemma 12], the DOFmap is created
/// based on the following patch-wise ordering:
///
///      d0, {d^l_E0}, ..., {d^l_En}, {d^r_T1}, ..., {d^r_Tn}
///
/// Within the DOFmap these informations are packed in a element-wise structure:
///
///      [d0, d0, {d^l_E0,T1}, {d^l_E1,T1}, {d^r_T1}, ...,
///       d0, d0, {d^l_Eam1,Ta}, {d^l_Ea_Ta}, {d^r_Ta}]
///
/// [1] Bertrand, F.; Carstensen, C.; Gräßle, B. & Tran, N. T.:
///     Stabilization-free HHO a posteriori error control, 2022
///
/// @tparam T               The scalar type
/// @tparam id_flux_order   The flux order (1->RT1, 2->RT2, 3->general)
/// @param patch             The patch
/// @param facet_orientation The facet orientation of one cell
/// @param storage_detJ      The Jacobi determinants of the patch cells
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
  std::vector<std::int32_t> ddofmap_patch(5 * ncells * ndofs_cell_local, 0);
  mdspan_t<std::int32_t, 3> dofmap_patch(ddofmap_patch.data(), 5,
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
    dofmap_patch(4, id_a, 0) = prefactor_eam1;

    dofmap_patch(0, id_a, 1) = ldofs_fct[ndofs_flux_fct];
    dofmap_patch(1, id_a, 1) = 0;
    dofmap_patch(2, id_a, 1) = gdofs_fct[ndofs_flux_fct];
    dofmap_patch(3, id_a, 1) = prefactor_ea;
    dofmap_patch(4, id_a, 1) = -prefactor_ea;

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
        dofmap_patch(4, id_a, 2) = prefactor_eam1;

        dofmap_patch(0, id_a, 3) = ldofs_fct[iea];
        dofmap_patch(2, id_a, 3) = gdofs_fct[iea];
        dofmap_patch(3, id_a, 3) = prefactor_ea;
        dofmap_patch(4, id_a, 3) = -prefactor_ea;

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
          dofmap_patch(4, id_a, i) = prefactor_eam1;

          dofmap_patch(0, id_a, iea) = ldofs_fct[iea];
          dofmap_patch(1, id_a, iea) = offs_ea + i;
          dofmap_patch(2, id_a, iea) = gdofs_fct[iea];
          dofmap_patch(3, id_a, iea) = prefactor_ea;
          dofmap_patch(4, id_a, iea) = -prefactor_ea;
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
          dofmap_patch(4, id_a, offs_dmp1) = 1;
        }
      }
    }
  }

  return std::move(ddofmap_patch);
}

/// Initialise boundary markers for patch-wise H(div=0) space
/// @param ndofs_hdivz_per_cell nDOFs in patch-wise H(div=0) space per cell
std::vector<std::int8_t>
initialise_boundary_markers(const int ndofs_hdivz_per_cell)
{
  // Create vector
  std::vector<std::int8_t> boundary_markers(ndofs_hdivz_per_cell, false);

  return std::move(boundary_markers);
}

/// Set boundary markers for patch-wise H(div=0) space
/// @param boundary_markers   The boundary markers
/// @param type_patch         The patch type
/// @param ncells             Number of cells on patch
/// @param ndofs_flux_fct     nDOFs flux-space space per facet
/// @param reversion_required Patch requires reversion
void set_boundary_markers(std::span<std::int8_t> boundary_markers,
                          const PatchType type_patch, const int ncells,
                          const int ndofs_flux_fct,
                          bool reversion_required = false)
{
  // Reinitialise markers
  std::fill(boundary_markers.begin(), boundary_markers.end(), false);

  // Set markers
  if ((type_patch != PatchType::internal)
      && (type_patch != PatchType::bound_essnt_primal))
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

/// Kernel for unconstrained flux minimisation
///
/// Calculates system-matrix and load vector for unconstrained flux minimisation
/// on patch-wise divergence free H(div) space.
///
/// @tparam T               The scalar type
/// @tparam id_flux_order   The flux order (1->RT1, 2->RT2, 3->general)
/// @tparam asmbl_systmtrx  Flag if entire tangent or only load vector is
///                         assembled
/// @param Te           Storage for tangent arrays
/// @param kernel_data  The KernelData
/// @param coefficients The Coefficients
/// @param dofmap       The DOFmap of the patch-wise H(div=0) space
/// @param detJ         The Jacobi determinant
/// @param J            The Jacobi matrix
template <typename T, int id_flux_order = 3, bool asmbl_systmtrx = true>
void kernel_fluxmin(mdspan_t<double, 2> Te, KernelDataEqlb<T>& kernel_data,
                    std::span<const T> coefficients,
                    smdspan_t<const std::int32_t, 2> asmbl_info,
                    const int ndofs_per_cell, const int ndofs_flux_fct,
                    const double detJ, mdspan_t<const double, 2> J)
{
  const int index_load = Te.extent(0) - 1;

  /* Extract shape functions and quadrature data */
  smdspan_t<double, 3> phi = kernel_data.shapefunctions_flux(J, detJ);

  std::span<const double> quadrature_weights
      = kernel_data.quadrature_weights(0);

  /* Initialisation */
  // Interpolated solution from step 1
  std::array<T, 2> sigtilde_q;

  // Data mainpulation of shapfunction for d0
  std::int32_t ld0_Eam1 = asmbl_info(0, 0),
               ld0_Ea = asmbl_info(0, ndofs_flux_fct);
  std::int32_t p_Eam1 = asmbl_info(3, 0), p_Ea = asmbl_info(3, ndofs_flux_fct);

  /* Assemble tangents */
  for (std::size_t iq = 0; iq < quadrature_weights.size(); ++iq)
  {
    // Interpolate sigma_tilde
    sigtilde_q[0] = 0;
    sigtilde_q[1] = 0;

    // Interpolation
    for (std::size_t i = 0; i < phi.extent(1); ++i)
    {
      sigtilde_q[0] += coefficients[i] * phi(iq, i, 0);
      sigtilde_q[1] += coefficients[i] * phi(iq, i, 1);
    }

    // Manipulate shape function for coefficient d_0
    phi(iq, ld0_Ea, 0)
        = p_Ea * (p_Eam1 * phi(iq, ld0_Eam1, 0) + p_Ea * phi(iq, ld0_Ea, 0));
    phi(iq, ld0_Ea, 1)
        = p_Ea * (p_Eam1 * phi(iq, ld0_Eam1, 1) + p_Ea * phi(iq, ld0_Ea, 1));

    // Assemble linear- and bilinear form
    for (std::size_t i = 0; i < ndofs_per_cell; ++i)
    {
      // Auxilary variables
      std::size_t ip1 = i + 1;
      double alpha
          = asmbl_info(3, ip1) * quadrature_weights[iq] * std::fabs(detJ);

      // Linear form
      Te(index_load, i) -= (phi(iq, asmbl_info(0, ip1), 0) * sigtilde_q[0]
                            + phi(iq, asmbl_info(0, ip1), 1) * sigtilde_q[1])
                           * alpha;

      if constexpr (asmbl_systmtrx)
      {
        for (std::size_t j = i; j < ndofs_per_cell; ++j)
        {
          // Auxiliary variables
          std::size_t jp1 = j + 1;
          double sp
              = phi(iq, asmbl_info(0, ip1), 0) * phi(iq, asmbl_info(0, jp1), 0)
                + phi(iq, asmbl_info(0, ip1), 1)
                      * phi(iq, asmbl_info(0, jp1), 1);

          // Bilinear form
          Te(i, j) += sp * asmbl_info(3, jp1) * alpha;
        }
      }
    }
  }

  // Set symmetric contributions of element mass-matrix
  if constexpr (id_flux_order > 1)
  {
    for (std::size_t i = 1; i < ndofs_per_cell; ++i)
    {
      for (std::size_t j = 0; j < i; ++j)
      {
        Te(i, j) = Te(j, i);
      }
    }
  }
}
// ------------------------------------------------------------------------------

/// Assemble EQS for flux minimisation
///
/// Assembles system-matrix and load vector for unconstrained flux minimisation
/// on patch-wise divergence free H(div) space. Explicit ansatz for such a
/// space see [1, Lemma 12].
///
/// [1] Bertrand, F.; Carstensen, C.; Gräßle, B. & Tran, N. T.:
///     Stabilization-free HHO a posteriori error control, 2022
///
/// @tparam T               The scalar type
/// @tparam id_flux_order   The flux order (1->RT1, 2->RT2, 3->general)
/// @tparam asmbl_systmtrx  Flag if entire tangent or only load vector is
///                         assembled
/// @param A_patch          The patch system matrix (mass matrix)
/// @param L_patch          The patch load vector
/// @param patch            The patch
/// @param kernel_data      The kernel data
/// @param boundary_markers The boundary markers
/// @param coefficients     Flux DOFs on cells
/// @param storage_detJ     The Jacobi determinants of the patch cells
/// @param storage_J        The Jacobi matrices of the patch cells
/// @param storage_K        The invers Jacobi matrices of the patch cells
/// @param requires_flux_bc Marker if flux BCs are required
template <typename T, int id_flux_order = 3, bool asmbl_systmtrx = true>
void assemble_fluxminimiser(
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& A_patch,
    Eigen::Matrix<T, Eigen::Dynamic, 1>& L_patch,
    PatchFluxCstmNew<T, id_flux_order, false>& patch,
    KernelDataEqlb<T>& kernel_data,
    std::span<const std::int8_t> boundary_markers,
    std::span<const T> coefficients, std::span<const double> storage_detJ,
    std::span<const double> storage_J, std::span<const double> storage_K,
    const bool requires_flux_bc)
{
  assert(id_flux_order < 0);

  /* Extract data */
  // Number of elements/facets on patch
  const int ncells = patch.ncells();

  // DOF counters
  const int ndofs = patch.ndofs_flux();
  const int ndofs_per_fct = patch.ndofs_flux_fct();
  const int ndofs_cell_add = patch.ndofs_flux_cell_add();

  // DOFmap minimisation problem on patch
  mdspan_t<const std::int32_t, 3> asmbl_info
      = patch.assembly_info_minimisation();

  /* Initialisation */
  // Element tangents
  const int ndofs_nz = 2 * ndofs_per_fct + ndofs_cell_add - 1;
  const int index_load = ndofs_nz;
  std::vector<T> dTe(ndofs_nz * (ndofs_nz + 1), 0);
  mdspan_t<double, 2> Te(dTe.data(), ndofs_nz + 1, ndofs_nz);

  /* Calculation and assembly */
  for (std::size_t a = 1; a < ncells + 1; ++a)
  {
    int id_a = a - 1;

    // Isoparametric mapping
    const double detJ = storage_detJ[id_a];
    mdspan_t<const double, 2> J = extract_mapping_data(id_a, storage_J);

    // DOFmap on cell
    smdspan_t<const std::int32_t, 2> asmbl_info_cell = stdex::submdspan(
        asmbl_info, stdex::full_extent, a, stdex::full_extent);

    // DOFs on cell
    std::span<const T> coefficients_elmt
        = coefficients.subspan(id_a * ndofs, ndofs);

    // Evaluate linear- and bilinear form
    std::fill(dTe.begin(), dTe.end(), 0);
    kernel_fluxmin<T, id_flux_order, asmbl_systmtrx>(
        Te, kernel_data, coefficients_elmt, asmbl_info_cell, ndofs_nz,
        ndofs_per_fct, detJ, J);

    // Assemble linear- and bilinear form
    if constexpr (id_flux_order == 1)
    {
      if (requires_flux_bc)
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
      if (requires_flux_bc)
      {
        for (std::size_t i = 0; i < ndofs_nz; ++i)
        {
          std::int32_t dof_i = asmbl_info_cell(2, i + 1);
          std::int8_t bmarker_i = boundary_markers[dof_i];

          // Assemble load vector
          if (bmarker_i)
          {
            L_patch(dof_i) = 0;
          }
          else
          {
            L_patch(dof_i) += Te(index_load, i);
          }

          // Assemble bilinear form
          if constexpr (asmbl_systmtrx)
          {
            if (bmarker_i)
            {
              A_patch(dof_i, dof_i) = 1;
            }
            else
            {
              for (std::size_t j = 0; j < ndofs_nz; ++j)
              {
                std::int32_t dof_j = asmbl_info_cell(2, j + 1);
                std::int8_t bmarker_j = boundary_markers[dof_j];

                if (bmarker_j)
                {
                  A_patch(dof_i, dof_j) = 0;
                }
                else
                {
                  A_patch(dof_i, dof_j) += Te(i, j);
                }
              }
            }
          }
        }
      }
      else
      {
        for (std::size_t i = 0; i < ndofs_nz; ++i)
        {
          std::int32_t dof_i = asmbl_info_cell(2, i + 1);

          // Assemble load vector
          L_patch(dof_i) += Te(index_load, i);

          // Assemble bilinear form
          if constexpr (asmbl_systmtrx)
          {
            for (std::size_t j = 0; j < ndofs_nz; ++j)
            {
              A_patch(dof_i, asmbl_info_cell(2, j + 1)) += Te(i, j);
            }
          }
        }
      }
    }
  }
}
} // namespace dolfinx_eqlb
