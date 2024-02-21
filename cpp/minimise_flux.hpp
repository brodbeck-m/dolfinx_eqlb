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
/* Routines for flux minimisation */

enum class Kernel
{
  UconstrFluxMini,
  ConstrStressMini2D,
  ConstrStressMini3D,
};

std::size_t dimension_minimisation_space(const Kernel type, const int gdim,
                                         const int nnodes_on_patch,
                                         const int ndofs_flux_hdivz)
{
  switch (type)
  {
  case Kernel::UconstrFluxMini:
    return ndofs_flux_hdivz;
    break;
  case Kernel::ConstrStressMini2D:
    return gdim * ndofs_flux_hdivz + nnodes_on_patch;
    break;
  case Kernel::ConstrStressMini3D:
    return gdim * (ndofs_flux_hdivz + nnodes_on_patch);
    break;
  default:
    throw std::invalid_argument("Unrecognized kernel");
  }
}

/// Create mixed-space DOFmap
///
/// Patch-wise divergence free H(div) space requires special DOFmap and
/// prefacers during assembly. Following [1, Lemma 12], the DOFmap is
/// created based on the following patch-wise ordering:
///
/// {d0}_i, {{d^l_E0}, ..., {d^l_En}}_i, {d^r_T1}_i, ..., {d^r_Tn}_i, {d_c}_j
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
/// @param ndofs_flux_hdivz nDOF patch-wise H(div=0) space
template <typename T, int id_flux_order>
std::pair<std::array<std::size_t, 3>, std::vector<std::int32_t>>
set_flux_dofmap(PatchCstm<T, id_flux_order, true>& patch,
                const int ndofs_flux_hdivz)
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

  // Storage for assembly information
  std::vector<std::int32_t> dasmbl_info(
      4 * asmbl_info_base.extent(1) * size_per_cell, 0);

  // Size of new mdspan
  std::array<std::size_t, 3> shape
      = {4, asmbl_info_base.extent(1), size_per_cell};

  /* Recreate assembly-informations */
  mdspan_t<std::int32_t, 3> asmbl_info(dasmbl_info.data(), shape);

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

  return {std::move(shape), std::move(dasmbl_info)};
}

/// Initialise boundary markers for patch-wise H(div=0) space
/// @param type             The kernel type of the minimisation problem
/// @param gdim             The geometric dimension
/// @param nnodes_on_patch  Number of nodes on patch
/// @param ndofs_flux_hdivz nDOFs patch-wise H(div=0) space
/// @return                 Initialised vector for boundary markers
std::vector<std::int8_t> initialise_boundary_markers(const Kernel type,
                                                     const int gdim,
                                                     const int nnodes_on_patch,
                                                     const int ndofs_flux_hdivz)
{
  // Determine dimension of (mixed) fe-space on patch
  const std::size_t size = dimension_minimisation_space(
      type, gdim, nnodes_on_patch, ndofs_flux_hdivz);

  // Create vector
  std::vector<std::int8_t> boundary_markers(size, false);

  return std::move(boundary_markers);
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
    std::size_t bs = (type_kernel == Kernel::UconstrFluxMini) ? 1 : gdim;

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
                             const int ndofs_per_cell, const int ndofs_flux_fct)
{
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
      = [kernel_data, ndofs_per_cell, ndofs_flux_fct](
            mdspan_t<double, 2> Te, std::span<const T> coefficients,
            smdspan_t<const std::int32_t, 2> asmbl_info, const double detJ,
            mdspan_t<const double, 2> J) mutable
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
    std::int32_t p_Eam1 = asmbl_info(3, 0),
                 p_Ea = asmbl_info(3, ndofs_flux_fct);

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
            double sp = phi(iq, asmbl_info(0, ip1), 0)
                            * phi(iq, asmbl_info(0, jp1), 0)
                        + phi(iq, asmbl_info(0, ip1), 1)
                              * phi(iq, asmbl_info(0, jp1), 1);

            // Bilinear form
            Te(i, j) += sp * asmbl_info(3, jp1) * alpha;
          }
        }
      }
    }

    // Set symmetric contributions of element mass-matrix
    if constexpr (asmbl_systmtrx)
    {
      if (ndofs_per_cell > 1)
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
  };

  switch (type)
  {
  case Kernel::UconstrFluxMini:
    return unconstrained_flux_minimisation;
  case Kernel::ConstrStressMini2D:
    throw std::runtime_error("Not implemented yet!");
  case Kernel::ConstrStressMini3D:
    throw std::runtime_error("Not implemented yet!");
  default:
    throw std::invalid_argument("Unrecognized kernel");
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
/// @tparam T               The scalar type
/// @tparam id_flux_order   The flux order (1->RT1, 2->RT2, 3->general)
/// @tparam asmbl_systmtrx  Flag if entire tangent or only load vector is
///                         assembled
/// @param A_patch          The patch system matrix (mass matrix)
/// @param L_patch          The patch load vector
/// @param patch            The patch
/// @param boundary_markers The boundary markers
/// @param asmbl_info       Informations to create the patch-wise H(div=0) space
/// @param coefficients     Flux DOFs on cells
/// @param storage_detJ     The Jacobi determinants of the patch cells
/// @param storage_J        The Jacobi matrices of the patch cells
/// @param storage_K        The invers Jacobi matrices of the patch cells
/// @param requires_flux_bc Marker if flux BCs are required
template <typename T, int id_flux_order = 3, bool asmbl_systmtrx = true>
void assemble_fluxminimiser(
    kernel_fn<T, asmbl_systmtrx>& minimisation_kernel,
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& A_patch,
    Eigen::Matrix<T, Eigen::Dynamic, 1>& L_patch,
    PatchFluxCstm<T, id_flux_order, false>& patch,
    std::span<const std::int8_t> boundary_markers,
    mdspan_t<const std::int32_t, 3> asmbl_info, std::span<const T> coefficients,
    std::span<const double> storage_detJ, std::span<const double> storage_J,
    std::span<const double> storage_K, const bool requires_flux_bc)
{
  assert(id_flux_order < 0);

  /* Extract data */
  // Number of elements/facets on patch
  const int ncells = patch.ncells();

  // DOF counters
  const int ndofs = patch.ndofs_flux();
  const int ndofs_per_fct = patch.ndofs_flux_fct();
  const int ndofs_cell_add = patch.ndofs_flux_cell_add();

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
    minimisation_kernel(Te, coefficients_elmt, asmbl_info_cell, detJ, J);

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
