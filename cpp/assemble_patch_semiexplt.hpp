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
void minimisation_kernel(dolfinx_adaptivity::mdspan2_t Te,
                         KernelData& kernel_data,
                         dolfinx_adaptivity::s_mdspan1_t prefactors_elmt,
                         std::span<T> coefficients,
                         std::span<const std::int32_t> fct_dofs,
                         std::span<const std::int32_t> cell_dofs,
                         dolfinx_adaptivity::cmdspan2_t coordinate_dofs)
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
    prefactors_elmt(0) = -prefactors_elmt(0);
    prefactors_elmt(1) = -prefactors_elmt(1);
  }

  /* Extract shape functions and quadrature data */
  dolfinx_adaptivity::s_cmdspan3_t phi
      = kernel_data.shapefunctions_flux(J, detJ);

  std::span<const double> quadrature_weights
      = kernel_data.quadrature_weights_cell();

  /* Initialisation */
  const int ndofs_fct = fct_dofs.size() / 2;

  std::array<T, 2> sigtilde_q;

  /* Perform quadrature */
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
      throw std::runtime_error("Minimization only implemented for RTk, k>0");
    }

    // Correct normal orientation
    double p_Eam1 = prefactors_elmt(0), p_Ea = prefactors_elmt(1);

    // Coefficient d_0: Linear form
    std::int32_t dofl_Eam1 = fct_dofs[0];
    std::int32_t dofl_Ea = fct_dofs[ndofs_fct];

    std::array<T, 2> diff_phi;
    diff_phi[0] = p_Eam1 * phi(iq, dofl_Eam1, 0) - p_Ea * phi(iq, dofl_Ea, 0);
    diff_phi[1] = p_Eam1 * phi(iq, dofl_Eam1, 1) - p_Ea * phi(iq, dofl_Ea, 1);

    Te(index_load, 0)
        += -(diff_phi[0] * sigtilde_q[0] + diff_phi[1] * sigtilde_q[1]) * alpha;

    // Coefficient d_0: Bilinear form
    if constexpr (asmbl_systmtrx)
    {
      Te(0, 0)
          += (diff_phi[0] * diff_phi[0] + diff_phi[1] * diff_phi[1]) * alpha;
    }

    // Coefficients d^l_Eam1 and d^l_Ea
    // Interactions with d_0 are also calculated here
    if constexpr (id_flux_order > 1)
    {
      throw std::runtime_error("Minimisation only implemented for RT0");
    }

    // Coefficients d^r_Ta
    // Interactions with d_0 and d^l_E are also calculated here
    if constexpr (id_flux_order > 2)
    {
      throw std::runtime_error("Minimisation only implemented for RT0");
    }
  }
}

template <typename T, int id_flux_order = 3, bool asmbl_systmtrx = true>
void assemble_minimisation(
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& A_patch,
    Eigen::Matrix<T, Eigen::Dynamic, 1>& L_patch,
    std::span<const std::int32_t> cells, PatchFluxCstm<T, id_flux_order>& patch,
    KernelData& kernel_data, dolfinx_adaptivity::mdspan2_t prefactors_dof,
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

  /* Initialisation */
  // Element tangents
  const int ndofs_nz = 1 + 1.5 * degree_rt + degree_rt * degree_rt;
  std::vector<T> dTe(ndofs_nz * (ndofs_nz + 1), 0);
  dolfinx_adaptivity::mdspan2_t Te(dTe.data(), ndofs_nz + 1, ndofs_nz);

  /* Calculation and assembly */
  for (std::size_t a = 1; a < cells.size() + 1; ++a)
  {
    int id_a = a - 1;

    // Extract cell-wise flux DOFs
    std::span<const std::int32_t> fdofs_fct = patch.dofs_flux_fct_local(a);
    std::span<const std::int32_t> fdofs_cell = patch.dofs_flux_cell_local(a);

    // Element data
    dolfinx_adaptivity::s_mdspan1_t prefactors_elmt
        = stdex::submdspan(prefactors_dof, id_a, stdex::full_extent);
    std::span<T> coefficients_elmt
        = coefficients.subspan(id_a * cstride, cstride);
    dolfinx_adaptivity::cmdspan2_t coordinates_elmt(
        coordinate_dofs.data() + id_a * cstride_geom, nnodes_cell, 3);

    // Evaluate linear- and bilinear form
    std::fill(dTe.begin(), dTe.end(), 0);
    minimisation_kernel<T, id_flux_order, asmbl_systmtrx>(
        Te, kernel_data, prefactors_elmt, coefficients_elmt, fdofs_fct,
        fdofs_cell, coordinates_elmt);

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
      throw std::runtime_error("assembly_minimisation: Not implemented!");
    }
    else
    {
      throw std::runtime_error("assembly_minimisation: Not implemented!");
    }
  }
}

} // namespace dolfinx_adaptivity::equilibration