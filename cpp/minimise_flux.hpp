#pragma once

#include "eigen3/Eigen/Dense"

#include "KernelData.hpp"
#include "Patch.hpp"
#include "PatchCstm.hpp"
#include "PatchData.hpp"
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

/* Routines for flux minimisation */

enum class Kernel
{
  FluxMin,
  StressMin,
  StressMinNL,
};

/// Create mixed-space DOFmap
///
/// Patch-wise divergence free H(div) space requires special DOFmap and
/// prefacers during assembly. Following [1, Lemma 12], the DOFmap is
/// created based on the following patch-wise ordering:
///
/// {d0}_i, {{d^l_E0}, ..., {d^l_En}}_i, {d^r_T1}_i, ..., {d^r_Tn}_i,
/// {d_c}_j
///
/// The indexes i resp. j thereby denotes the block.size of flux resp.
/// constraints. Cell wise structure of the DOFmap:
///
/// [{dofs_TaEam1}_i, {dofs_TaEa}_i, {dofs_Ta-add}_i, {dofs_Ta-c}_j]
///
/// [1] Bertrand, F.; Carstensen, C.; Gräßle, B. & Tran, N. T.:
///     Stabilization-free HHO a posteriori error control, 2022
///
/// @tparam T               The scalar type
/// @tparam id_flux_order   The flux order (1->RT1, 2->RT2, 3->general)
/// @param patch            The patch
/// @param asmbl_info       The assembly information of the constrained space
template <typename T, int id_flux_order>
void set_flux_dofmap(PatchFluxCstm<T, id_flux_order>& patch,
                     mdspan_t<std::int32_t, 3> asmbl_info)
{
  /* Extract data */
  // Te spacial dimension
  const int gdim = patch.dim();

  // Number of facets per cell
  const int fcts_per_cell = patch.fcts_per_cell();

  // Nodes on cell
  const int nnodes_per_cell = fcts_per_cell;

  // Number of cells on patch
  const int ncells = patch.ncells();

  // DOF counters
  const int ndofs_flux_fct = patch.ndofs_flux_fct();
  const int ndofs_flux_cell_add = patch.ndofs_flux_cell_add();
  const int ndofs_flux_hdivz = patch.ndofs_minspace_flux(false);

  // Assembly information for non-mixed space
  mdspan_t<const std::int32_t, 3> asmbl_info_base
      = patch.assembly_info_minimisation();

  /* Initialisation */
  // Number of (mixed) DOFs per cell
  const int bsize_flux_per_cell = gdim * ndofs_flux_fct + ndofs_flux_cell_add;
  const int bsize_constr_per_cell = fcts_per_cell;
  const int bsize_per_cell = bsize_flux_per_cell + bsize_constr_per_cell;

  const int size_flux_per_cell = gdim * bsize_flux_per_cell;
  const int size_constr_per_cell
      = (gdim == 2) ? bsize_constr_per_cell : bsize_flux_per_cell * gdim;
  const int size_per_cell = size_flux_per_cell + size_constr_per_cell;

  /* Recreate assembly-informations */
  const std::size_t lbound_a = (patch.is_internal()) ? 0 : 1;
  const std::size_t ubound_a = (patch.is_internal()) ? ncells + 2 : ncells + 1;
  const std::size_t ubound_j
      = (gdim == 2) ? bsize_flux_per_cell : bsize_per_cell;

  for (std::size_t a = lbound_a; a < ubound_a; ++a)
  {
    for (std::size_t i = 0; i < gdim; ++i)
    {
      for (std::size_t j = 0; j < ubound_j; ++j)
      {
        int offs = i + gdim * j;

        asmbl_info(0, a, offs) = asmbl_info_base(0, a, j);
        asmbl_info(1, a, offs) = asmbl_info_base(1, a, j);
        asmbl_info(3, a, offs) = asmbl_info_base(3, a, j);

        // Set new patch-wise DOF
        asmbl_info(2, a, offs) = gdim * asmbl_info_base(2, a, j) + i;
      }
    }

    // Extra handling for constraint DOFs (gdim == 2)
    if (gdim == 2)
    {
      for (std::size_t j = 0; j < nnodes_per_cell; ++j)
      {
        int offs_n = size_flux_per_cell + j;
        int offs_b = bsize_flux_per_cell + j;

        asmbl_info(0, a, offs_n) = asmbl_info_base(0, a, offs_b);
        asmbl_info(3, a, offs_n) = 1;

        // Set new patch-wise DOF
        asmbl_info(2, a, offs_n)
            = asmbl_info_base(2, a, offs_b) + ndofs_flux_hdivz;
      }
    }
  }
}

/// Set boundary markers for patch-wise H(div=0) space
/// @param boundary_markers   The boundary markers
/// @param type_kernel        The kernel type of the minimisation problem
/// @param type_patch         The patch type
/// @param gdim               The geometric dimension
/// @param ncells             Number of cells on patch
/// @param ndofs_flux_fct     nDOFs flux-space space per facet
/// @param reversion_required Patch requires reversion
void set_boundary_markers(std::span<std::int8_t> boundary_markers,
                          const Kernel type_kernel,
                          std::vector<PatchType> type_patch, const int gdim,
                          const int ncells, const int ndofs_flux_fct,
                          std::vector<bool> reversion_required)
{
  // Reinitialise markers
  std::fill(boundary_markers.begin(), boundary_markers.end(), false);

  // Set markers
  if ((type_patch[0] != PatchType::internal))
  {
    // Check if mixed space required
    std::size_t bs = (type_kernel == Kernel::FluxMin) ? 1 : gdim;

    // Auxiliaries
    const int offs_En_base = bs * (ncells * (ndofs_flux_fct - 1));

    // Set boundary markers
    for (std::size_t i = 0; i < bs; ++i)
    {
      if (type_patch[i] != PatchType::bound_essnt_primal)
      {
        // Set boundary markers for d0
        boundary_markers[i] = true;

        for (std::size_t j = 1; j < ndofs_flux_fct; ++j)
        {
          if (type_patch[i] == PatchType::bound_essnt_dual)
          {
            // Mark DOFs on facet E0
            int offs_E0 = i + bs * j;
            boundary_markers[offs_E0] = true;

            // Mark DOFs on facet En
            int offs_En = offs_En_base + offs_E0;
            boundary_markers[offs_En_base + offs_E0] = true;
          }
          else
          {
            if (reversion_required[i])
            {
              // Mark DOFs in facet En
              // (Mixed patch with reversed order)
              int offs_En = offs_En_base + i + bs * j;
              boundary_markers[offs_En] = true;
            }
            else
            {
              // Mark DOFs in facet E0
              // (Mixed patch with original order)
              boundary_markers[i + bs * j] = true;
            }
          }
        }
      }
    }
  }
}

template <typename T, bool asmbl_systmtrx = true>
kernel_fn<T, asmbl_systmtrx>
generate_minimisation_kernel(Kernel type, KernelDataEqlb<T>& kernel_data,
                             const int gdim, const int facets_per_cell,
                             const int degree_rt)
{
  /* DOF counters */
  const int ndofs_flux_fct = degree_rt + 1;
  const int ndofs_flux_cell_add = 0.5 * degree_rt * (degree_rt - 1);

  const int ndofs_hdivzero_per_cell
      = 2 * ndofs_flux_fct + ndofs_flux_cell_add - 1;
  const int ndofs_constr_per_cell
      = (gdim == 2) ? facets_per_cell : gdim * facets_per_cell;

  /* The kernels */

  /// Kernel for unconstrained flux minimisation
  ///
  /// Calculates system-matrix and load vector for unconstrained flux
  /// minimisation on patch-wise divergence free H(div) space.
  ///
  /// @tparam T               The scalar type
  /// @tparam asmbl_systmtrx  Flag if entire tangent or only load vector is
  ///                         assembled
  /// @param Te           Storage for tangent arrays
  /// @param coefficients The Coefficients
  /// @param asmbl_info   Information to create the patch-wise H(div=0) space
  /// @param detJ         The Jacobi determinant
  /// @param J            The Jacobi matrix
  kernel_fn<T, asmbl_systmtrx> unconstrained_flux_minimisation
      = [kernel_data, ndofs_hdivzero_per_cell, ndofs_flux_fct](
            mdspan_t<double, 2> Te, std::span<const T> coefficients,
            smdspan_t<const std::int32_t, 2> asmbl_info, const double detJ,
            mdspan_t<const double, 2> J) mutable
  {
    const int index_load = Te.extent(1);

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
    std::int32_t p_Eam1 = asmbl_info(3, 0),
                 p_Ea = asmbl_info(3, ndofs_flux_fct);

    /* Assemble tangents */
    for (std::size_t iq = 0; iq < quadrature_weights.size(); ++iq)
    {
      // Interpolate flux_tilde
      sigtilde_q[0] = 0;
      sigtilde_q[1] = 0;

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

      // Volume integrator
      double dvol = quadrature_weights[iq] * std::fabs(detJ);

      // Assemble linear- and bilinear form
      for (std::size_t i = 0; i < ndofs_hdivzero_per_cell; ++i)
      {
        // Offset
        std::size_t ip1 = i + 1;

        // Manipulate shape functions
        double alpha = asmbl_info(3, ip1) * dvol;

        double phi_i0 = phi(iq, asmbl_info(0, ip1), 0) * alpha;
        double phi_i1 = phi(iq, asmbl_info(0, ip1), 1) * alpha;

        // Linear form
        Te(index_load, i) -= phi_i0 * sigtilde_q[0] + phi_i1 * sigtilde_q[1];

        if constexpr (asmbl_systmtrx)
        {
          for (std::size_t j = i; j < ndofs_hdivzero_per_cell; ++j)
          {
            // Offset
            std::size_t jp1 = j + 1;

            // Manipulate shape functions
            double alpha = asmbl_info(3, ip1) * dvol;

            double phi_j0 = phi(iq, asmbl_info(0, jp1), 0) * asmbl_info(3, jp1);
            double phi_j1 = phi(iq, asmbl_info(0, jp1), 1) * asmbl_info(3, jp1);

            // Bilinear form
            Te(i, j) += phi_i0 * phi_j0 + phi_i1 * phi_j1;
          }
        }
      }
    }

    // Set symmetric contributions of element mass-matrix
    if constexpr (asmbl_systmtrx)
    {
      if (ndofs_hdivzero_per_cell > 1)
      {
        for (std::size_t i = 1; i < ndofs_hdivzero_per_cell; ++i)
        {
          for (std::size_t j = 0; j < i; ++j)
          {
            Te(i, j) = Te(j, i);
          }
        }
      }
    }
  };

  kernel_fn<T, asmbl_systmtrx> constrained_stress_minimisation_2D =
      [kernel_data, ndofs_hdivzero_per_cell, ndofs_constr_per_cell,
       ndofs_flux_fct](mdspan_t<double, 2> Te, std::span<const T> coefficients,
                       smdspan_t<const std::int32_t, 2> asmbl_info,
                       const double detJ, mdspan_t<const double, 2> J) mutable
  {
    const int index_load = Te.extent(1);
    const int index_lmp = index_load + 1;

    // std::cout << "Index load, index lmp, extent T_e: " << index_load << " "
    //           << index_lmp << " " << Te.extent(0) << std::endl;

    /* Extract shape functions and quadrature data */
    smdspan_t<double, 3> phi_f = kernel_data.shapefunctions_flux(J, detJ);
    smdspan_t<const double, 2> phi_c = kernel_data.shapefunctions_cell_hat();

    std::span<const double> quadrature_weights
        = kernel_data.quadrature_weights(0);

    // nDOFs flux space
    const int ndofs_flux = phi_f.extent(1);

    /* Initialisation */
    // Interpolated solution from step 1
    std::array<T, 2> sig_r0, sig_r1;

    // Data manipulation of shape function for d0
    std::int32_t ld0_Eam1 = asmbl_info(0, 0),
                 ld0_Ea = asmbl_info(0, 2 * ndofs_flux_fct);
    std::int32_t p_Eam1 = asmbl_info(3, 0),
                 p_Ea = asmbl_info(3, 2 * ndofs_flux_fct);

    // Offset constrained space
    const int offs_cijbase = 2 * ndofs_hdivzero_per_cell;
    const int offs_cbase = offs_cijbase + 2;

    // std::cout << "Offset 1: " << offs_cijbase << std::endl;
    // std::cout << "Offset 2: " << offs_cbase << std::endl;

    /* Assemble tangents */
    for (std::size_t iq = 0; iq < quadrature_weights.size(); ++iq)
    {
      // Interpolate stress
      sig_r0[0] = 0, sig_r0[1] = 0;
      sig_r1[0] = 0, sig_r1[1] = 0;

      for (std::size_t i = 0; i < ndofs_flux; ++i)
      {
        int offs = ndofs_flux + i;

        sig_r0[0] += coefficients[i] * phi_f(iq, i, 0);
        sig_r0[1] += coefficients[i] * phi_f(iq, i, 1);

        sig_r1[0] += coefficients[offs] * phi_f(iq, i, 0);
        sig_r1[1] += coefficients[offs] * phi_f(iq, i, 1);
      }

      // std::cout << "sig_r0_0, sig_r0_1, sig_r1_0, sig_r1_1: " << sig_r0[0]
      //           << " " << sig_r0[1] << " " << sig_r1[0] << " " << sig_r1[1]
      //           << std::endl;

      // Manipulate shape function for coefficient d_0
      phi_f(iq, ld0_Ea, 0)
          = p_Ea
            * (p_Eam1 * phi_f(iq, ld0_Eam1, 0) + p_Ea * phi_f(iq, ld0_Ea, 0));
      phi_f(iq, ld0_Ea, 1)
          = p_Ea
            * (p_Eam1 * phi_f(iq, ld0_Eam1, 1) + p_Ea * phi_f(iq, ld0_Ea, 1));

      // Volume integrator
      double dvol = quadrature_weights[iq] * std::fabs(detJ);

      // Equations for flux DOFs
      for (std::size_t i = 0; i < ndofs_hdivzero_per_cell; ++i)
      {
        // Offsets due to mixed space
        int offs_i = 2 * i + 2;
        int ir0 = 2 * i;
        int ir1 = ir0 + 1;

        // Cell-local DOF (index i)
        std::int32_t dl_i = asmbl_info(0, offs_i);

        // Manipulate shape functions
        double alpha = asmbl_info(3, offs_i) * dvol;

        double phi_i0 = phi_f(iq, dl_i, 0) * alpha;
        double phi_i1 = phi_f(iq, dl_i, 1) * alpha;

        // Load vector
        Te(index_load, ir0) -= sig_r0[0] * phi_i0 + sig_r0[1] * phi_i1;
        Te(index_load, ir1) -= sig_r1[0] * phi_i0 + sig_r1[1] * phi_i1;

        if constexpr (asmbl_systmtrx)
        {
          // Block k_sig_sig
          for (std::size_t j = i; j < ndofs_hdivzero_per_cell; ++j)
          {
            // Offsets due to mixed space
            int offs_j = 2 * j + 2;
            int jr0 = 2 * j;
            int jr1 = jr0 + 1;

            // Cell-local DOF (index j)
            std::int32_t dl_j = asmbl_info(0, offs_j);

            // Manipulate shape functions
            double phi_j0 = phi_f(iq, dl_j, 0) * asmbl_info(3, offs_j);
            double phi_j1 = phi_f(iq, dl_j, 1) * asmbl_info(3, offs_j);

            // System matrix
            double val = phi_i0 * phi_j0 + phi_i1 * phi_j1;

            Te(ir0, jr0) += val;
            Te(ir1, jr1) += val;
          }

          // Block k_sig_gamma
          for (std::size_t j = 0; j < ndofs_constr_per_cell; ++j)
          {
            // Offsets due to mixed space
            int offs_j = offs_cbase + j;
            int jr = offs_cijbase + j;

            // Cell local DOF (index j)
            std::int32_t dl_j = asmbl_info(0, offs_j);

            // System matrix
            Te(ir0, jr) += phi_i1 * phi_c(iq, dl_j);
            Te(ir1, jr) -= phi_i0 * phi_c(iq, dl_j);
          }
        }
      }

      // Equations for constrained DOFs
      for (std::size_t i = 0; i < ndofs_constr_per_cell; ++i)
      {
        // Offsets due to mixed space
        int offs_i = offs_cbase + i;
        int ir = offs_cijbase + i;

        // Cell local DOF (index j)
        std::int32_t dl_i = asmbl_info(0, offs_i);

        // Manipulate shape functions
        double phi_i = phi_c(iq, dl_i) * dvol;

        // System matrix
        Te(index_load, ir) -= phi_i * (sig_r0[1] - sig_r1[0]);

        // Mean-value zero condition
        Te(index_lmp, i) += phi_i;
      }
    }

    /* Apply symmetrie conditions on system matrix */
    if constexpr (asmbl_systmtrx)
    {
      // Add symmetric part of k_sig_sig
      if (ndofs_hdivzero_per_cell > 1)
      {
        for (std::size_t i = 1; i < 2 * ndofs_hdivzero_per_cell; ++i)
        {
          for (std::size_t j = 0; j < i; ++j)
          {
            Te(i, j) = Te(j, i);
          }
        }
      }

      // Add k_gamma_sig = k_sig_gamma^T
      for (std::size_t i = 0; i < ndofs_constr_per_cell; ++i)
      {
        // Offset due to mixed space
        int ir = offs_cijbase + i;

        for (std::size_t j = 0; j < ndofs_hdivzero_per_cell; ++j)
        {
          // Offset due to mixed space
          int jr0 = 2 * j;
          int jr1 = jr0 + 1;

          Te(ir, jr0) = Te(jr0, ir);
          Te(ir, jr1) = Te(jr1, ir);
        }
      }
    }
  };

  switch (type)
  {
  case Kernel::FluxMin:
    return unconstrained_flux_minimisation;
  case Kernel::StressMin:
    if (gdim == 2)
    {
      return constrained_stress_minimisation_2D;
    }
    else
    {
      throw std::runtime_error(
          "Kernel for weakly symmetric stresses (3D) not implemented");
    }
  case Kernel::StressMinNL:
    if (gdim == 2)
    {
      throw std::runtime_error(
          "Kernel for weakly symmetric 1.PK stress (2D) not implemented");
    }
    else
    {
      throw std::runtime_error(
          "Kernel for weakly symmetric 1.PK stress (3D) not implemented");
    }
  default:
    throw std::invalid_argument("Unrecognised kernel");
  }
}

template <typename T, bool modified_patch>
kernel_fn_schursolver<T, modified_patch>
generate_stress_minimisation_kernel(Kernel type, KernelDataEqlb<T>& kernel_data,
                                    const int gdim, const int facets_per_cell,
                                    const int degree_rt)
{
  /* DOF counters */
  const int ndofs_flux_fct = degree_rt + 1;
  const int ndofs_flux_cell_add = 0.5 * degree_rt * (degree_rt - 1);

  const int ndofs_hdivzero_per_cell
      = 2 * ndofs_flux_fct + ndofs_flux_cell_add - 1;
  const int ndofs_constr_per_cell
      = (gdim == 2) ? facets_per_cell : gdim * facets_per_cell;

  /* The kernels */

  /// Kernel for the (constrained) stress correction (2D)
  ///
  /// Calculates system-matrix and load vector for constrained stress
  /// minimisation on patch-wise divergence free H(div) space. Weak symmetry is
  /// imposed.
  ///
  /// @tparam T               The scalar type
  /// @tparam modified_patch  Flag for kernel for modified boundary patches
  /// @param Te           Storage for tangent arrays
  /// @param coefficients The Coefficients
  /// @param asmbl_info   Information to create the patch-wise H(div=0) space
  /// @param detJ         The Jacobi determinant
  /// @param J            The Jacobi matrix
  kernel_fn_schursolver<T, modified_patch> stress_minimisation_2D
      = [kernel_data, ndofs_hdivzero_per_cell, ndofs_constr_per_cell,
         ndofs_flux_fct](mdspan_t<double, 2> Ae, mdspan_t<double, 2> Be,
                         std::span<double> Ce, std::span<double> Le,
                         std::span<const T> coefficients,
                         smdspan_t<const std::int32_t, 2> asmbl_info,
                         const double detJ, mdspan_t<const double, 2> J) mutable
  {
    /* Extract shape functions and quadrature data */
    smdspan_t<double, 3> phi_f = kernel_data.shapefunctions_flux(J, detJ);
    smdspan_t<const double, 2> phi_c = kernel_data.shapefunctions_cell_hat();

    std::span<const double> quadrature_weights
        = kernel_data.quadrature_weights(0);

    // nDOFs flux space
    const int ndofs_flux = phi_f.extent(1);

    /* Initialisation */
    // Interpolated solution from step 1
    std::array<T, 2> sig_r0, sig_r1;

    // Data manipulation of shape function for d0
    std::int32_t ld0_Eam1 = asmbl_info(0, 0),
                 ld0_Ea = asmbl_info(0, 2 * ndofs_flux_fct);
    std::int32_t p_Eam1 = asmbl_info(3, 0),
                 p_Ea = asmbl_info(3, 2 * ndofs_flux_fct);

    // Offset constrained space
    const int offs_c = ndofs_hdivzero_per_cell + 1;
    const int offs_Lc = 2 * ndofs_hdivzero_per_cell;

    /* Assemble tangents */
    for (std::size_t iq = 0; iq < quadrature_weights.size(); ++iq)
    {
      // Interpolate stress
      sig_r0[0] = 0, sig_r0[1] = 0;
      sig_r1[0] = 0, sig_r1[1] = 0;

      for (std::size_t i = 0; i < ndofs_flux; ++i)
      {
        int offs = ndofs_flux + i;

        sig_r0[0] += coefficients[i] * phi_f(iq, i, 0);
        sig_r0[1] += coefficients[i] * phi_f(iq, i, 1);

        sig_r1[0] += coefficients[offs] * phi_f(iq, i, 0);
        sig_r1[1] += coefficients[offs] * phi_f(iq, i, 1);
      }

      // Manipulate shape function for coefficient d_0
      phi_f(iq, ld0_Ea, 0)
          = p_Ea
            * (p_Eam1 * phi_f(iq, ld0_Eam1, 0) + p_Ea * phi_f(iq, ld0_Ea, 0));
      phi_f(iq, ld0_Ea, 1)
          = p_Ea
            * (p_Eam1 * phi_f(iq, ld0_Eam1, 1) + p_Ea * phi_f(iq, ld0_Ea, 1));

      // Volume integrator
      double dvol = quadrature_weights[iq] * std::fabs(detJ);

      for (std::size_t i = 0; i < ndofs_hdivzero_per_cell; ++i)
      {
        // Offset
        std::size_t ip1 = i + 1;

        // Cell-local DOF (index i)
        std::int32_t dl_i = asmbl_info(0, ip1);

        // Manipulate shape functions
        double alpha = asmbl_info(3, ip1) * dvol;

        double phi_i0 = phi_f(iq, dl_i, 0) * alpha;
        double phi_i1 = phi_f(iq, dl_i, 1) * alpha;

        if constexpr (modified_patch)
        {
          // Load vector L_sigma
          Le[i] -= sig_r0[0] * phi_i0 + sig_r0[1] * phi_i1;
          Le[ndofs_hdivzero_per_cell + i]
              -= sig_r1[0] * phi_i0 + sig_r1[1] * phi_i1;

          // Submatrix A (lower triagular part)
          for (std::size_t j = 0; j < i + 1; ++j)
          {
            // Offset
            std::size_t jp1 = j + 1;

            // Cell-local DOF (index j)
            std::int32_t dl_j = asmbl_info(0, jp1);

            // Manipulate shape functions
            double phi_j0 = phi_f(iq, dl_j, 0) * asmbl_info(3, jp1);
            double phi_j1 = phi_f(iq, dl_j, 1) * asmbl_info(3, jp1);

            // System matrix
            Ae(i, j) += phi_i0 * phi_j0 + phi_i1 * phi_j1;
          }
        }

        // Submatrices B1 and B2
        for (std::size_t j = 0; j < ndofs_constr_per_cell; ++j)
        {
          // Shape function (index j)
          double phi_j = phi_c(iq, asmbl_info(0, offs_c + j));

          // System matrix
          Be(i, j) += phi_i1 * phi_j;
          Be(i, ndofs_constr_per_cell + j) -= phi_i0 * phi_j;
        }
      }

      for (std::size_t i = 0; i < ndofs_constr_per_cell; ++i)
      {
        // Manipulate shape function (index i)
        double phi_i = phi_c(iq, asmbl_info(0, offs_c + i)) * dvol;

        // Submatrix Ce
        Ce[i] += phi_i;

        // Load vector L_c
        Le[offs_Lc + i] -= phi_i * (sig_r0[1] - sig_r1[0]);
      }
    }
  };

  switch (type)
  {
  case Kernel::StressMin:
    if (gdim == 2)
    {
      return stress_minimisation_2D;
    }
    else
    {
      throw std::runtime_error(
          "Kernel for weakly symmetric stresses (3D) not implemented");
    }
  case Kernel::StressMinNL:
    if (gdim == 2)
    {
      throw std::runtime_error(
          "Kernel for weakly symmetric 1.PK stress (2D) not implemented");
    }
    else
    {
      throw std::runtime_error(
          "Kernel for weakly symmetric 1.PK stress (3D) not implemented");
    }
  default:
    throw std::invalid_argument("Unrecognised kernel");
  }
}

// ------------------------------------------------------------------------------

/// Assemble EQS for flux minimisation
///
/// Assembles system-matrix and load vector for unconstrained flux
/// minimisation on patch-wise divergence free H(div) space. Explicit ansatz
/// for such a space see [1, Lemma 12].
///
/// [1] Bertrand, F.; Carstensen, C.; Gräßle, B. & Tran, N. T.:
///     Stabilization-free HHO a posteriori error control, 2022
///
/// @tparam T                       The scalar type
/// @tparam id_flux_order           The flux order (1->RT1, 2->RT2, 3->general)
/// @tparam asmbl_systmtrx          Flag if entire tangent or only load
///                                 vector is assembled
/// @param minimisation_kernel      The kernel for minimisation
/// @param patch_data               The temporary storage for the patch
/// @param asmbl_info               Informations to create the patch-wise
///                                 H(div=0) space
/// @param i_rhs                    Index of the right-hand side
/// @param requires_flux_bc         Marker if flux BCs are required
/// @param constrained_minimisation Flag if constarined system is assembeled
template <typename T, int id_flux_order, bool asmbl_systmtrx>
void assemble_fluxminimiser(kernel_fn<T, asmbl_systmtrx>& minimisation_kernel,
                            PatchDataCstm<T, id_flux_order>& patch_data,
                            mdspan_t<const std::int32_t, 3> asmbl_info,
                            const int i_rhs, const bool requires_flux_bc)
{
  assert(id_flux_order < 0);

  /* Extract data */
  // The spatial dimension
  const int gdim = patch_data.gdim();

  // Number of elements/facets on patch
  const int ncells = patch_data.ncells();

  // Tangent storage
  mdspan_t<T, 2> Te = patch_data.Te();

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& A = patch_data.matrix_A();
  Eigen::Matrix<T, Eigen::Dynamic, 1>& L = patch_data.vector_L_sigma();
  std::span<const std::int8_t> boundary_markers
      = patch_data.boundary_markers(false);

  /* Initialisation */
  if constexpr (asmbl_systmtrx)
  {
    A.setZero();
    L.setZero();
  }
  else
  {
    L.setZero();
  }

  /* Calculation and assembly */
  const int ndofs_per_cell = Te.extent(1);
  const int ndofs_constr_per_cell = gdim + 1;

  const int index_load = ndofs_per_cell;

  for (std::size_t a = 1; a < ncells + 1; ++a)
  {
    int id_a = a - 1;

    // Isoparametric mapping
    const double detJ = patch_data.jacobi_determinant(id_a);
    mdspan_t<const double, 2> J = patch_data.jacobian(id_a);

    // DOFmap on cell
    smdspan_t<const std::int32_t, 2> asmbl_info_cell = stdex::submdspan(
        asmbl_info, stdex::full_extent, a, stdex::full_extent);

    // DOFs on cell
    std::span<const T> coefficients = patch_data.coefficients_flux(i_rhs, a);

    // Evaluate linear- and bilinear form
    patch_data.reinitialise_Te();
    minimisation_kernel(Te, coefficients, asmbl_info_cell, detJ, J);

    // Assemble linear- and bilinear form
    if constexpr (id_flux_order == 1)
    {
      if (requires_flux_bc)
      {
        // Assemble linar form
        L(0) = 0;

        if constexpr (asmbl_systmtrx)
        {
          // Assemble bilinear form
          A(0, 0) = 1;
        }
      }
      else
      {
        // Assemble linar form
        L(0) += Te(1, 0);

        if constexpr (asmbl_systmtrx)
        {
          // Assemble bilinear form
          A(0, 0) += Te(0, 0);
        }
      }
    }
    else
    {
      if (requires_flux_bc)
      {
        for (std::size_t i = 0; i < ndofs_per_cell; ++i)
        {
          std::int32_t dof_i = asmbl_info_cell(2, i + 1);
          std::int8_t bmarker_i = boundary_markers[dof_i];

          // Assemble load vector
          if (bmarker_i)
          {
            L(dof_i) = 0;
          }
          else
          {
            L(dof_i) += Te(index_load, i);
          }

          // Assemble bilinear form
          if constexpr (asmbl_systmtrx)
          {
            if (bmarker_i)
            {
              A(dof_i, dof_i) = 1;
            }
            else
            {
              for (std::size_t j = 0; j < ndofs_per_cell; ++j)
              {
                std::int32_t dof_j = asmbl_info_cell(2, j + 1);
                std::int8_t bmarker_j = boundary_markers[dof_j];

                if (bmarker_j)
                {
                  A(dof_i, dof_j) = 0;
                }
                else
                {
                  A(dof_i, dof_j) += Te(i, j);
                }
              }
            }
          }
        }
      }
      else
      {
        for (std::size_t i = 0; i < ndofs_per_cell; ++i)
        {
          std::int32_t dof_i = asmbl_info_cell(2, i + 1);

          // Assemble load vector
          L(dof_i) += Te(index_load, i);

          // Assemble bilinear form
          if constexpr (asmbl_systmtrx)
          {
            for (std::size_t j = 0; j < ndofs_per_cell; ++j)
            {
              A(dof_i, asmbl_info_cell(2, j + 1)) += Te(i, j);
            }
          }
        }
      }
    }
  }
}

/// Assemble EQS for constrained stress minimisation
///
/// Assembles system-matrix and load vector for constrained flux
/// minimisation on patch-wise divergence free H(div) space. Explicit ansatz
/// for such a space see [1, Lemma 12].
///
/// [1] Bertrand, F.; Carstensen, C.; Gräßle, B. & Tran, N. T.:
///     Stabilization-free HHO a posteriori error control, 2022
///
/// @tparam T                       The scalar type
/// @tparam id_flux_order           The flux order (1->RT1, 2->RT2, 3->general)
/// @tparam modified_patch          Flag if a modified patch is assembeled
/// @param minimisation_kernel      The kernel for minimisation
/// @param patch_data               The temporary storage for the patch
/// @param asmbl_info               Informations to create the patch-wise
///                                 H(div=0) space
/// @param i_rhs                    Index of the right-hand side
/// @param requires_flux_bc         Marker if flux BCs are required
/// @param constrained_minimisation Flag if constarined system is assembeled
// template <typename T, int id_flux_order, bool modified_patch>
// void assemble_stressminimiser(
//     kernel_fn_schursolver<T, modified_patch>& minimisation_kernel,
//     PatchDataCstm<T, id_flux_order>& patch_data,
//     mdspan_t<const std::int32_t, 3> asmbl_info, const int
//     ndofs_flux_minspace, const int i_rhs, const bool requires_flux_bc, const
//     bool constrained_minimisation)
// {
//   assert(id_flux_order < 0);

//   /* Extract data */
//   // The spatial dimension
//   const int gdim = patch_data.gdim();

//   // Number of elements/facets on patch
//   const int ncells = patch_data.ncells();

//   // Tangent storage
//   mdspan_t<T, 2> Te = patch_data.Te(constrained_minimisation);

//   Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& A_patch
//       = patch_data.A_patch(constrained_minimisation);
//   Eigen::Matrix<T, Eigen::Dynamic, 1>& L_patch
//       = patch_data.L_patch(constrained_minimisation);
//   std::span<const std::int8_t> boundary_markers
//       = patch_data.boundary_markers(constrained_minimisation);

//   /* Initialisation */
//   if constexpr (asmbl_systmtrx)
//   {
//     A_patch.setZero();
//     L_patch.setZero();
//   }
//   else
//   {
//     L_patch.setZero();
//   }

//   /* Calculation and assembly */
//   const int ndofs_per_cell = Te.extent(1);
//   const int ndofs_constr_per_cell = gdim + 1;

//   const int index_load = ndofs_per_cell;
//   const int index_lmp = ndofs_per_cell + 1;

//   const int offset_asmblinfo = (constrained_minimisation) ? gdim : 1;
//   const int offset_constr_local = ndofs_per_cell + gdim -
//   ndofs_constr_per_cell; const int offset_constr
//       = patch_data.size_minimisation_system(constrained_minimisation) - 1;

//   const bool requires_lmp = (constrained_minimisation)
//                                 ?
//                                 patch_data.meanvalue_zero_condition_required()
//                                 : false;

//   std::span<const T> coefficients;

//   for (std::size_t a = 1; a < ncells + 1; ++a)
//   {
//     int id_a = a - 1;

//     // Isoparametric mapping
//     const double detJ = patch_data.jacobi_determinant(id_a);
//     mdspan_t<const double, 2> J = patch_data.jacobian(id_a);

//     // DOFmap on cell
//     smdspan_t<const std::int32_t, 2> asmbl_info_cell = stdex::submdspan(
//         asmbl_info, stdex::full_extent, a, stdex::full_extent);

//     // DOFs on cell
//     if (constrained_minimisation)
//     {
//       coefficients = patch_data.coefficients_stress(a);
//     }
//     else
//     {
//       coefficients = patch_data.coefficients_flux(i_rhs, a);
//     }

//     // Evaluate linear- and bilinear form
//     patch_data.reinitialise_Te(constrained_minimisation);
//     minimisation_kernel(Te, coefficients, asmbl_info_cell, detJ, J);

//     // Assemble linear- and bilinear form
//     if constexpr (id_flux_order == 1)
//     {
//       if (requires_flux_bc)
//       {
//         // Assemble linar form
//         L_patch(0) = 0;

//         if constexpr (asmbl_systmtrx)
//         {
//           // Assemble bilinear form
//           A_patch(0, 0) = 1;
//         }
//       }
//       else
//       {
//         // Assemble linar form
//         L_patch(0) += Te(1, 0);

//         if constexpr (asmbl_systmtrx)
//         {
//           // Assemble bilinear form
//           A_patch(0, 0) += Te(0, 0);
//         }
//       }
//     }
//     else
//     {
//       if (requires_flux_bc)
//       {
//         for (std::size_t i = 0; i < ndofs_per_cell; ++i)
//         {
//           std::int32_t dof_i = asmbl_info_cell(2, offset_asmblinfo + i);
//           std::int8_t bmarker_i = boundary_markers[dof_i];

//           // Assemble load vector
//           if (bmarker_i)
//           {
//             L_patch(dof_i) = 0;
//           }
//           else
//           {
//             L_patch(dof_i) += Te(index_load, i);
//           }

//           // Assemble bilinear form
//           if constexpr (asmbl_systmtrx)
//           {
//             if (bmarker_i)
//             {
//               A_patch(dof_i, dof_i) = 1;
//             }
//             else
//             {
//               for (std::size_t j = 0; j < ndofs_per_cell; ++j)
//               {
//                 std::int32_t dof_j = asmbl_info_cell(2, offset_asmblinfo +
//                 j); std::int8_t bmarker_j = boundary_markers[dof_j];

//                 if (bmarker_j)
//                 {
//                   A_patch(dof_i, dof_j) = 0;
//                 }
//                 else
//                 {
//                   A_patch(dof_i, dof_j) += Te(i, j);
//                 }
//               }
//             }
//           }
//         }
//       }
//       else
//       {
//         for (std::size_t i = 0; i < ndofs_per_cell; ++i)
//         {
//           std::int32_t dof_i = asmbl_info_cell(2, offset_asmblinfo + i);

//           // Assemble load vector
//           L_patch(dof_i) += Te(index_load, i);

//           // Assemble bilinear form
//           if constexpr (asmbl_systmtrx)
//           {
//             for (std::size_t j = 0; j < ndofs_per_cell; ++j)
//             {
//               A_patch(dof_i, asmbl_info_cell(2, offset_asmblinfo + j))
//                   += Te(i, j);
//             }
//           }
//         }

//         // Add lagrangian multiplier
//         if (requires_lmp)
//         {
//           // std::cout << "Add lagrangian multiplier!" << std::endl;
//           for (std::size_t i = 0; i < ndofs_constr_per_cell; ++i)
//           {
//             int dof_i = asmbl_info_cell(2, offset_constr_local + i);

//             A_patch(dof_i, offset_constr) += Te(index_lmp, i);
//             A_patch(offset_constr, dof_i) += Te(index_lmp, i);
//           }
//         }
//       }
//     }
//   }
// }
} // namespace dolfinx_eqlb
