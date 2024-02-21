#pragma once

#include "eigen3/Eigen/Dense"

#include "KernelData.hpp"
#include "Patch.hpp"
#include "PatchCstm.hpp"
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

template <typename T, int id_flux_order = 3>
void impose_weak_symmetry(const mesh::Geometry& geometry,
                          PatchCstm<T, id_flux_order, true>& patch,
                          ProblemDataStress<T>& problem_data,
                          KernelDataEqlb<T>& kernel_data,
                          kernel_fn<T, true>& minkernel)
{
  /* Extract data */
  // The spatial dimension
  const int gdim = 2;

  // The geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();
  std::span<const double> x = geometry.x();

  // The patch
  std::span<const std::int32_t> cells = patch.cells();

  const int ncells = patch.ncells();
  const int nfcts = patch.nfcts();
  const int nnodes = nfcts + 1;

  // Nodes constructing one element
  const int nnodes_cell = kernel_data.nnodes_cell();

  // DOF-counts function spaces
  const int degree_flux_rt = patch.degree_raviart_thomas();

  const int ndofs_flux = patch.ndofs_flux();
  const int ndofs_flux_fct = patch.ndofs_flux_fct();
  const int ndofs_flux_cell_add = patch.ndofs_flux_cell_add();

  /* Initialise Mappings */
  // Representation/Storage isoparametric mapping
  std::array<double, 9> Jb, Kb;
  mdspan_t<double, 2> J(Jb.data(), 2, 2), K(Kb.data(), 2, 2);
  std::array<double, 18> detJ_scratch;

  std::vector<double> storage_detJ(ncells, 0), storage_J(ncells * 4, 0);

  // Storage cell geometry
  const int cstride_geom = 3 * nnodes_cell;
  std::vector<double> coordinate_dofs_e(cstride_geom, 0);

  /* Initialisations */
  // Coefficients of solution without symmetry
  std::vector<T> dcoefficients(gdim * ncells * ndofs_flux, 0);
  mdspan_t<T, 2> coefficients(dcoefficients.data(), ncells, ndofs_flux);

  // Number of flux-DOFs on patch-wise H(div=0) space
  const int ndofs_flux_hdivz
      = 1 + degree_flux_rt * nfcts
        + 0.5 * degree_flux_rt * (degree_flux_rt - 1) * ncells;

  // The equation system for the minimisation step
  const std::size_t dim_minspace = dimension_minimisation_space(
      Kernel::StressMin, gdim, nnodes, ndofs_flux_hdivz);

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A_patch;
  Eigen::Matrix<T, Eigen::Dynamic, 1> L_patch, u_patch;

  A_patch.resize(dim_minspace, dim_minspace);
  L_patch.resize(dim_minspace);
  u_patch.resize(dim_minspace);

  // Local solver (Cholesky decomposition)
  Eigen::LLT<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> solver;

  // Boundary markers
  std::vector<std::int8_t> boundary_markers = initialise_boundary_markers(
      Kernel::StressMin, gdim, nnodes, ndofs_flux_hdivz);

  // Pre-evaluate J and detJ
  for (std::size_t index = 0; index < ncells; ++index)
  {
    // Index using patch nomenclature
    int a = index + 1;

    // Get current cell
    std::int32_t c = cells[a];

    // Copy points of current cell
    std::span<const std::int32_t> x_dofs = x_dofmap.links(c);
    copy_cell_data<double, 3>(x, x_dofs, coordinate_dofs_e, 3);

    /* Piola mapping */
    // Reshape geometry infos
    mdspan_t<const double, 2> coords(coordinate_dofs_e.data(), nnodes_cell, 3);

    // Calculate Jacobi, inverse, and determinant
    storage_detJ[index] = kernel_data.compute_jacobian(J, detJ_scratch, coords);

    // Storage of (inverse) Jacobian
    store_mapping_data(index, storage_J, J);
  }

  // DOFmap for minimisation problem
  patch.set_assembly_informations(kernel_data.fct_normal_is_outward(),
                                  storage_detJ);

  std::array<std::size_t, 3> shape_asmblinfo;
  std::vector<std::int32_t> dasmblinfo;
  std::tie(shape_asmblinfo, dasmblinfo)
      = set_flux_dofmap(patch, ndofs_flux_hdivz);

  mdspan_t<const std::int32_t, 3> asmbl_info(dasmblinfo.data(),
                                             shape_asmblinfo);
}
} // namespace dolfinx_eqlb