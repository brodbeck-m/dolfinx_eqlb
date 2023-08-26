#pragma once

#include "KernelData.hpp"
#include "PatchFluxCstm.hpp"
#include "eigen3/Eigen/Dense"
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

namespace dolfinx_adaptivity::equilibration
{
/// Integration kernel for system matrix/load vector per patch element
/// @tparam T              The scalar type
/// @tparam id_flux_order  The flux order (1->RT1, 2->RT2, 3->general)
/// @tparam asmbl_systmtrx Flag if entire tangent or only load vector is
///                        assembled
/// @param Te               The element tangent
/// @param kernel_data      The kernel data
/// @param coefficients     Cell DOFs on flux (after step 1)
/// @param dofmap           Cell DOFmap
/// @param fluxdofs_per_fct Number of flux DOFs per facet
/// @param coordinate_dofs  The coordinate DOFs of current cell
template <typename T, int id_flux_order = 3, bool asmbl_systmtrx = true>
void minimisation_kernel(dolfinx_adaptivity::mdspan2_t Te,
                         KernelDataEqlb<T>& kernel_data,
                         std::span<T> coefficients,
                         dolfinx_adaptivity::smdspan_t<std::int32_t, 2> dofmap,
                         const int fluxdofs_per_fct,
                         dolfinx_adaptivity::cmdspan2_t coordinate_dofs)
{
  const int index_load = Te.extent(0) - 1;

  /* Initialise storage of cell prefactors */
  std::int32_t prefactor_eam1 = dofmap(3, 0);
  std::int32_t prefactor_ea = dofmap(3, 1);

  /* Isoparametric mapping */
  std::array<double, 9> Jb;
  dolfinx_adaptivity::mdspan2_t J(Jb.data(), 2, 2);
  std::array<double, 18> detJ_scratch;

  double detJ = kernel_data.compute_jacobian(J, detJ_scratch, coordinate_dofs);

  /* Extract shape functions and quadrature data */
  dolfinx_adaptivity::smdspan_t<double, 3> phi
      = kernel_data.shapefunctions_flux(J, detJ);

  std::span<const double> quadrature_weights
      = kernel_data.quadrature_weights(0);

  /* Initialisation */
  // Interpolated solution from step 1
  std::array<T, 2> sigtilde_q;

  // Number of non-zero DOFs on cell
  const int nidofs_per_cell = dofmap.extent(1) - 1;

  // Set prefactors
  if constexpr (asmbl_systmtrx)
  {
    // Correct prefactors based on sign of detJ
    if (detJ < 0)
    {
      prefactor_eam1 = -prefactor_eam1;
      prefactor_ea = -prefactor_ea;
    }

    // Move prefactors to storage
    if constexpr (id_flux_order > 1)
    {
      if constexpr (id_flux_order == 2)
      {
        dofmap(3, 2) = prefactor_eam1;
        dofmap(3, 3) = prefactor_ea;
      }
      else
      {
        const int offs_e = fluxdofs_per_fct - 1;

        for (std::size_t i = 2; i < fluxdofs_per_fct + 1; ++i)
        {
          dofmap(3, i) = prefactor_eam1;
          dofmap(3, i + offs_e) = prefactor_ea;
        }
      }
    }
  }

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

/// Assemble EQS for unconstrained flux minimisation
///
/// Assembles system-matrix and load vector for unconstrained flux minimisation
/// on patch-wise divergence free H(div) space. Explicit ansatz for such a
/// space see [1, Lemma 12].
///
/// [1] Bertrand, F.; Carstensen, C.; Gräßle, B. & Tran, N. T.:
///     Stabilization-free HHO a posteriori error control, 2022
///
/// @tparam T              The scalar type
/// @tparam id_flux_order  The flux order (1->RT1, 2->RT2, 3->general)
/// @tparam asmbl_systmtrx Flag if entire tangent or only load vector is
///                        assembled
/// @param A_patch         The patch system matrix (mass matrix)
/// @param L_patch         The patch load vector
/// @param patch           The patch
/// @param kernel_data     The kernel data
/// @param dofmap_patch    The patch dofmap
///                        ([dof_local, dof_patch, dof_global,prefactor] x cell
///                        x dofs_per_cell)
/// @param coefficients    Flux DOFs on cells
/// @param coordinate_dofs The coordinates of patch cells
/// @param type_patch      The patch type
template <typename T, int id_flux_order = 3, bool asmbl_systmtrx = true>
void assemble_minimisation(
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& A_patch,
    Eigen::Matrix<T, Eigen::Dynamic, 1>& L_patch,
    PatchFluxCstm<T, id_flux_order>& patch, KernelDataEqlb<T>& kernel_data,
    dolfinx_adaptivity::mdspan_t<std::int32_t, 3> dofmap_patch,
    std::span<T> coefficients, std::span<double> coordinate_dofs,
    const int type_patch)
{
  assert(id_flux_order < 0);

  /* Extract data */
  // Number of elements/facets on patch
  const int ncells = patch.ncells();
  const int nfcts = patch.nfcts();

  // Node of base element
  int nnodes_cell = kernel_data.nnodes_cell();
  const int cstride_geom = 3 * nnodes_cell;

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
  dolfinx_adaptivity::mdspan2_t Te(dTe.data(), ndofs_nz + 1, ndofs_nz);

  /* Calculation and assembly */
  for (std::size_t a = 1; a < ncells + 1; ++a)
  {
    int id_a = a - 1;

    // Cell coordinates
    dolfinx_adaptivity::cmdspan2_t coordinates_elmt(
        coordinate_dofs.data() + id_a * cstride_geom, nnodes_cell, 3);

    // DOFmap on cell
    dolfinx_adaptivity::smdspan_t<std::int32_t, 2> dofmap_cell
        = stdex::submdspan(dofmap_patch, stdex::full_extent, id_a,
                           stdex::full_extent);

    // DOFs on cell (equilibration step 1)
    std::span<T> coefficients_elmt = coefficients.subspan(id_a * ndofs, ndofs);

    // Pack (cell-local and patch-wise) DOF vector on cell
    if constexpr (asmbl_systmtrx)
    {
      // DOFs on facets
      std::span<const std::int32_t> fdofs_fct_local
          = patch.dofs_flux_fct_local(a);
      std::span<const std::int32_t> fdofs_fct_global
          = patch.dofs_flux_fct_global(a);

      // Cell-local DOFs of zero-order facet moments
      dofmap_cell(0, 0) = fdofs_fct_local[0];
      dofmap_cell(0, 1) = fdofs_fct_local[ndofs_per_fct];

      // Patch-local DOFs of zero-order facet moments
      dofmap_cell(1, 0) = 0;
      dofmap_cell(1, 1) = 0;

      // Global DOFs of zero-order facet moments
      dofmap_cell(2, 0) = fdofs_fct_global[0];
      dofmap_cell(2, 1) = fdofs_fct_global[ndofs_per_fct];

      if constexpr (id_flux_order > 1)
      {
        if constexpr (id_flux_order == 2)
        {
          int iea = ndofs_per_fct + 1;

          // Cell-local DOFs of first-order facet moments
          dofmap_cell(0, 2) = fdofs_fct_local[1];
          dofmap_cell(0, 3) = fdofs_fct_local[iea];

          // Patch-local DOFs of first-order facet moments
          if ((type_patch == 0) && (a == ncells))
          {
            dofmap_cell(1, 2) = a;
            dofmap_cell(1, 3) = 1;
          }
          else
          {
            dofmap_cell(1, 2) = a;
            dofmap_cell(1, 3) = 1 + a;
          }

          // Global DOFs of zero-order facet moments
          dofmap_cell(2, 2) = fdofs_fct_global[1];
          dofmap_cell(2, 3) = fdofs_fct_global[iea];
        }
        else
        {
          // Set facet DOFs
          const int offs_e = ndofs_per_fct - 1;
          int offs_ea, offs_eam1;

          if ((type_patch == 0) && (a == ncells))
          {
            offs_eam1 = (a - 1) * (ndofs_per_fct - 1) - 1;
            offs_ea = -1;
          }
          else
          {
            offs_eam1 = (a - 1) * (ndofs_per_fct - 1) - 1;
            offs_ea = offs_eam1 + offs_e;
          }

          for (std::size_t i = 2; i < ndofs_per_fct + 1; ++i)
          {
            int ieam1 = i - 1;
            int iea = i + offs_e;

            // Cell-local DOFs of higher-order facet moments
            dofmap_cell(0, i) = fdofs_fct_local[ieam1];
            dofmap_cell(0, iea) = fdofs_fct_local[iea];

            // Patch-local DOFs of first-order facet moments
            dofmap_cell(1, i) = offs_eam1 + i;
            dofmap_cell(1, iea) = offs_ea + i;

            // Global DOFs of first-order facet moments
            dofmap_cell(2, i) = fdofs_fct_global[ieam1];
            dofmap_cell(2, iea) = fdofs_fct_global[iea];
          }

          // Set cell DOFs
          std::span<const std::int32_t> fdofs_cell_local
              = patch.dofs_flux_cell_local(a);
          std::span<const std::int32_t> fdofs_cell_global
              = patch.dofs_flux_cell_global(a);

          const int offs_ta
              = nfcts * (ndofs_per_fct - 1) + (a - 1) * ndofs_cell_add + 1;

          for (std::size_t i = 0; i < ndofs_cell_add; ++i)
          {
            int offs_dmp1 = 2 * ndofs_per_fct + i;
            int offs_dmp2 = ndofs_cell_div + i;

            // Cell-local DOFs of higher-order facet moments
            dofmap_cell(0, offs_dmp1) = fdofs_cell_local[offs_dmp2];

            // Patch-local DOFs of first-order facet moments
            dofmap_cell(1, offs_dmp1) = offs_ta + i;

            // Global DOFs of first-order facet moments
            dofmap_cell(2, offs_dmp1) = fdofs_cell_global[offs_dmp2];

            // Initialise prefactors
            dofmap_cell(3, offs_dmp1) = 1;
          }
        }
      }
    }

    // Evaluate linear- and bilinear form
    std::fill(dTe.begin(), dTe.end(), 0);
    minimisation_kernel<T, id_flux_order, asmbl_systmtrx>(
        Te, kernel_data, coefficients_elmt, dofmap_cell, ndofs_per_fct,
        coordinates_elmt);

    // Assemble linear- and bilinear form
    if constexpr (id_flux_order == 1)
    {
      // Assemble linar form
      L_patch(0) += Te(1, 0);

      if constexpr (asmbl_systmtrx)
      {
        // Assemble bilinear form
        A_patch(0, 0) += Te(0, 0);
      }
    }
    else
    {
      for (std::size_t i = 0; i < ndofs_nz; ++i)
      {
        std::int32_t dof_i = dofmap_cell(1, i + 1);

        // Assemble load vector
        L_patch(dof_i) += Te(index_load, i);

        // Assemble bilinear form
        if constexpr (asmbl_systmtrx)
        {
          for (std::size_t j = 0; j < ndofs_nz; ++j)
          {
            A_patch(dof_i, dofmap_cell(1, j + 1)) += Te(i, j);
          }
        }
      }
    }
  }
}

} // namespace dolfinx_adaptivity::equilibration