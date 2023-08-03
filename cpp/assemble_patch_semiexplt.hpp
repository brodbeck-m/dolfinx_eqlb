#pragma once

#include "KernelData.hpp"
#include "PatchFluxCstm.hpp"
#include "PatchFluxEV.hpp"
#include "StorageStiffness.hpp"
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
template <typename T, int id_flux_order = 3, bool asmbl_systmtrx = true>
void minimisation_kernel(
    dolfinx_adaptivity::mdspan2_t Te, KernelData<T>& kernel_data,
    std::span<T> coefficients, std::span<const std::int32_t> dofl_cell,
    dolfinx_adaptivity::smdspan_t<double, 1> prefactors_cell,
    const int ndofs_per_fct, dolfinx_adaptivity::cmdspan2_t coordinate_dofs)
{
  const int index_load = Te.extent(0) - 1;

  /* Isoparametric mapping */
  std::array<double, 9> Jb;
  dolfinx_adaptivity::mdspan2_t J(Jb.data(), 2, 2);
  std::array<double, 18> detJ_scratch;

  double detJ = kernel_data.compute_jacobian(J, detJ_scratch, coordinate_dofs);

  /* Calculate prefactors */
  if (detJ < 0)
  {
    prefactors_cell(0) = -prefactors_cell(0);
    prefactors_cell(1) = -prefactors_cell(1);
  }

  /* Extract shape functions and quadrature data */
  dolfinx_adaptivity::smdspan_t<double, 3> phi
      = kernel_data.shapefunctions_flux(J, detJ);

  std::span<const double> quadrature_weights
      = kernel_data.quadrature_weights_cell();

  /* Initialisation */
  // Interpolated solution from step 1
  std::array<T, 2> sigtilde_q;

  // Number of additional coefficients
  const int size_dolf = dofl_cell.size();

  // Set array with prefactors
  std::vector<double> list_prefactors(size_dolf, 1.0);

  if constexpr (id_flux_order > 1)
  {
    if constexpr (id_flux_order == 2)
    {
      list_prefactors[2] = prefactors_cell(0);
      list_prefactors[3] = prefactors_cell(1);
    }
    else
    {
      for (std::size_t i = 2; i < ndofs_per_fct + 1; ++i)
      {
        list_prefactors[i] = prefactors_cell(0);
        list_prefactors[i + ndofs_per_fct - 1] = prefactors_cell(1);
      }
    }
  }

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
      for (std::size_t i = 0; i < phi.extent(1); ++i)
      {
        sigtilde_q[0] += coefficients[i] * phi(iq, i, 0);
        sigtilde_q[1] += coefficients[i] * phi(iq, i, 1);
      }
    }

    // Manipulate shape function for coefficient d_0
    phi(iq, dofl_cell[1], 0) = prefactors_cell(0) * phi(iq, dofl_cell[0], 0)
                               - prefactors_cell(1) * phi(iq, dofl_cell[1], 0);
    phi(iq, dofl_cell[1], 1) = prefactors_cell(0) * phi(iq, dofl_cell[0], 1)
                               - prefactors_cell(1) * phi(iq, dofl_cell[1], 1);

    // Assemble linear- and bilinear form
    for (std::size_t i = 0; i < size_dolf - 1; ++i)
    {
      // Auxilary variables
      std::size_t ip1 = i + 1;
      double pialpha = list_prefactors[ip1] * alpha;

      // Linear form
      Te(index_load, i) -= (phi(iq, dofl_cell[ip1], 0) * sigtilde_q[0]
                            + phi(iq, dofl_cell[ip1], 1) * sigtilde_q[1])
                           * pialpha;

      if constexpr (asmbl_systmtrx)
      {
        for (std::size_t j = i; j < size_dolf - 1; ++j)
        {
          // Auxiliary variables
          std::size_t jp1 = j + 1;
          double sp = phi(iq, dofl_cell[ip1], 0) * phi(iq, dofl_cell[jp1], 0)
                      + phi(iq, dofl_cell[ip1], 1) * phi(iq, dofl_cell[jp1], 1);

          // Bilinear form
          Te(ip1, j) += sp * list_prefactors[jp1] * pialpha;
        }
      }
    }
  }
}

template <typename T, int id_flux_order = 3, bool asmbl_systmtrx = true>
void assemble_minimisation(
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& A_patch,
    Eigen::Matrix<T, Eigen::Dynamic, 1>& L_patch,
    std::span<const std::int32_t> cells, PatchFluxCstm<T, id_flux_order>& patch,
    KernelData<T>& kernel_data, dolfinx_adaptivity::mdspan2_t prefactors_dof,
    std::span<T> coefficients, const int cstride,
    std::span<double> coordinate_dofs, const int type_patch)
{
  assert(id_flux_order < 0);

  /* Extract data */
  // Node of base element
  int nnodes_cell = kernel_data.nnodes_cell();
  const int cstride_geom = 3 * nnodes_cell;

  // Degree of the RT element
  const int degree_rt = patch.degree_raviart_thomas();

  // DOFs per facet
  const int ndofs_per_fct = degree_rt + 1;
  const int ndofs_cell_add = patch.ndofs_flux_cell_add();

  /* Initialisation */
  // Element tangents
  const int ndofs_nz = 1 + 1.5 * degree_rt + degree_rt * degree_rt;
  std::vector<T> dTe(ndofs_nz * (ndofs_nz + 1), 0);
  dolfinx_adaptivity::mdspan2_t Te(dTe.data(), ndofs_nz + 1, ndofs_nz);

  // DOF vector cell
  const int ndofs_add = 2 * (ndofs_per_fct - 1) + ndofs_cell_add;
  std::vector<std::int32_t> dofl_cell(2 + ndofs_add);

  /* Calculation and assembly */
  for (std::size_t a = 1; a < cells.size() + 1; ++a)
  {
    int id_a = a - 1;

    // Extract cell-wise flux DOFs
    std::span<const std::int32_t> fdofs_fct = patch.dofs_flux_fct_local(a);

    // Element data
    dolfinx_adaptivity::smdspan_t<double, 1> prefactors_cell
        = stdex::submdspan(prefactors_dof, id_a, stdex::full_extent);
    std::span<T> coefficients_elmt
        = coefficients.subspan(id_a * cstride, cstride);
    dolfinx_adaptivity::cmdspan2_t coordinates_elmt(
        coordinate_dofs.data() + id_a * cstride_geom, nnodes_cell, 3);

    // Pack (local) DOF vector on cell (except d_0!)
    dofl_cell[0] = fdofs_fct[0];
    dofl_cell[1] = fdofs_fct[ndofs_per_fct];

    if constexpr (id_flux_order == 2)
    {
      // Set facet DOFs
      dofl_cell[2] = fdofs_fct[1];
      dofl_cell[3] = fdofs_fct[ndofs_per_fct + 1];
    }
    else
    {
      // Set facet DOFs and prefactors
      for (std::size_t i = 2; i < ndofs_per_fct + 1; ++i)
      {
        // Index
        std::int32_t iea = i + ndofs_per_fct - 1;

        // DOFs
        dofl_cell[i] = fdofs_fct[i - 1];
        dofl_cell[iea] = fdofs_fct[iea];
      }

      // Set cell DOFs
      if (a == 1)
      {
        const int ndofs_cell_div = patch.ndofs_flux_cell_div();
        const int ndofs_cell_add = patch.ndofs_flux_cell_add();
        std::span<const std::int32_t> fdofs_cell
            = patch.dofs_flux_cell_local(a);

        std::copy_n(fdofs_cell.begin() + ndofs_cell_div, ndofs_cell_add,
                    dofl_cell.begin() + 2 * ndofs_per_fct);
      }
    }

    // Evaluate linear- and bilinear form
    std::fill(dTe.begin(), dTe.end(), 0);
    minimisation_kernel<T, id_flux_order, asmbl_systmtrx>(
        Te, kernel_data, coefficients_elmt, dofl_cell, prefactors_cell,
        ndofs_per_fct, coordinates_elmt);

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
    else if constexpr (id_flux_order == 2)
    {
    }
    else
    {
    }
  }
}

} // namespace dolfinx_adaptivity::equilibration