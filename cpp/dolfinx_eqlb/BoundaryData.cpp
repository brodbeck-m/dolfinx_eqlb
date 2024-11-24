// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "BoundaryData.hpp"

using namespace dolfinx;
using namespace dolfinx_eqlb;

template <typename T>
BoundaryData<T>::BoundaryData(
    std::vector<std::vector<std::shared_ptr<FluxBC<T>>>>& list_bcs,
    std::vector<std::shared_ptr<fem::Function<T>>>& boundary_flux,
    std::shared_ptr<const fem::FunctionSpace> V_flux_hdiv,
    bool rtflux_is_custom, int quadrature_degree,
    const std::vector<std::vector<std::int32_t>>& fct_esntbound_prime,
    const bool reconstruct_stress)
    : _flux_degree(V_flux_hdiv->element()->basix_element().degree()),
      _num_rhs(list_bcs.size()), _gdim(V_flux_hdiv->mesh()->geometry().dim()),
      _num_fcts(
          V_flux_hdiv->mesh()->topology().index_map(_gdim - 1)->size_local()),
      _nfcts_per_cell((_gdim == 2) ? 3 : 4), _V_flux_hdiv(V_flux_hdiv),
      _flux_is_discontinous(
          V_flux_hdiv->element()->basix_element().discontinuous()),
      _ndofs_per_cell(V_flux_hdiv->element()->space_dimension()),
      _ndofs_per_fct((_gdim == 2) ? _flux_degree
                                  : 0.5 * _flux_degree * (_flux_degree + 1)),
      _num_dofs(V_flux_hdiv->dofmap()->index_map->size_local()
                + V_flux_hdiv->dofmap()->index_map->num_ghosts()),
      _fct_to_cell(
          V_flux_hdiv->mesh()->topology().connectivity(_gdim - 1, _gdim)),
      _quadrature_rule(base::QuadratureRule(
          V_flux_hdiv->mesh()->topology().cell_type(), quadrature_degree,
          V_flux_hdiv->mesh()->geometry().dim() - 1)),
      _kernel_data(KernelDataBC<T>(
          V_flux_hdiv->mesh(),
          {std::make_shared<base::QuadratureRule>(_quadrature_rule)},
          V_flux_hdiv->element(), _ndofs_per_fct, _ndofs_per_cell,
          rtflux_is_custom))
{
  // Resize storage
  _num_bcfcts.resize(_num_rhs);

  std::int32_t size_dofdata = _num_rhs * _num_dofs;
  _boundary_values.resize(size_dofdata, 0.0);
  _boundary_markers.resize(size_dofdata, false);
  _offset_dofdata.resize(_num_rhs + 1);

  std::int32_t size_fctdata = _num_rhs * _num_fcts;
  _local_fct_id.resize(_num_fcts);
  _facet_type.resize(size_fctdata, base::PatchFacetType::internal);
  _offset_fctdata.resize(_num_rhs + 1);

  if (reconstruct_stress)
  {
    std::int32_t size_nodedata
        = V_flux_hdiv->mesh()->topology().index_map(0)->size_local()
          + V_flux_hdiv->mesh()->topology().index_map(0)->num_ghosts();
    _pnt_on_esnt_boundary.resize(size_nodedata, false);
  }

  // Set offsets
  std::int32_t num_dofs = _num_dofs;
  std::generate(_offset_dofdata.begin(), _offset_dofdata.end(),
                [n = 0, num_dofs]() mutable { return num_dofs * (n++); });

  std::int32_t num_fcts = _num_fcts;
  std::generate(_offset_fctdata.begin(), _offset_fctdata.end(),
                [n = 0, num_fcts]() mutable { return num_fcts * (n++); });

  base::mdspan_t<double, 2> J(_data_J.data(), _gdim, _gdim);
  base::mdspan_t<double, 2> K(_data_K.data(), _gdim, _gdim);

  /* Extract required data */
  // The mesh
  std::shared_ptr<const mesh::Mesh> mesh = V_flux_hdiv->mesh();

  // Required conductivities
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> cell_to_fct
      = mesh->topology().connectivity(_gdim, _gdim - 1);
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> fct_to_pnt
      = mesh->topology().connectivity(_gdim - 1, 0);

  // Geometry DOFmap and nodal coordinates
  const graph::AdjacencyList<std::int32_t>& dofmap_geom
      = mesh->geometry().dofmap();
  std::span<const double> node_coordinates = mesh->geometry().x();

  // Flux DOFmap
  const graph::AdjacencyList<std::int32_t>& dofmap_hdiv
      = V_flux_hdiv->dofmap()->list();
  const fem::ElementDofLayout& doflayout_hdiv
      = V_flux_hdiv->dofmap()->element_dof_layout();

  // Transformation information flux space
  const auto apply_inverse_dof_transform
      = _V_flux_hdiv->element()->get_dof_transformation_function<T>(true, true,
                                                                    false);

  std::span<const std::uint32_t> cell_info;
  if (_V_flux_hdiv->element()->needs_dof_transformations())
  {
    mesh->topology_mutable().create_entity_permutations();
    cell_info = std::span(mesh->topology().get_cell_permutation_info());
  }

  // Counters interpolation- and quadrature points
  const int nipoints = _kernel_data.num_interpolation_points();
  const int nipoints_per_fct
      = _kernel_data.num_interpolation_points_per_facet();
  const int nqpoints = _quadrature_rule.num_points();
  const int nqpoints_per_fct = _quadrature_rule.num_points(0);

  /* Initialise storage */
  // DOF ids and values on boundary facets
  // (values on entire element to enable application of DOF transformations)
  std::vector<std::int32_t> boundary_dofs_fct(_ndofs_per_fct);
  std::vector<T> boundary_values(_ndofs_per_cell);

  // Storage for extraction of cell geometry
  std::vector<double> coordinates_data(mesh->geometry().cmap().dim() * 3);
  base::mdspan_t<const double, 2> coordinates(coordinates_data.data(),
                                              mesh->geometry().cmap().dim(), 3);

  // Storage of normal-traces/normal on boundary
  std::vector<T> values_bkernel(std::max(nipoints, nqpoints));
  std::vector<double> boundary_normal;

  // Constants/coefficients for evaluation of boundary kernel
  std::vector<T> constants_belmts, coefficients_belmts;
  std::int32_t cstride_coefficients;

  // Shape-functions (projection)
  bool initialise_projection = true;

  std::vector<double> basis_projection_values, mbasis_projection_values;
  base::mdspan_t<const double, 5> basis_projection;
  base::mdspan_t<double, 3> mbasis_projection;

  // Linear equation system (projection)
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A_e;
  Eigen::Matrix<T, Eigen::Dynamic, 1> L_e, u_e;
  Eigen::LLT<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> solver;

  /* Calculate boundary DOFs */
  for (int i_rhs = 0; i_rhs < _num_rhs; ++i_rhs)
  {
    // Pointer onto DOF vector of boundary function
    _x_boundary_flux.push_back(boundary_flux[i_rhs]->x());

    // Storage of current RHS
    std::span<std::int8_t> facet_type_i = facet_type(i_rhs);
    std::span<std::int8_t> boundary_markers_i = boundary_markers(i_rhs);

    std::span<T> x_bvals = boundary_flux[i_rhs]->x()->mutable_array();

    // Handle facets with essential BCs on primal problem
    for (std::int32_t fct : fct_esntbound_prime[i_rhs])
    {
      // Set facet type
      facet_type_i[fct] = base::PatchFacetType::essnt_primal;
    }

    // Handle facets with essential BCs on dual problem
    for (std::shared_ptr<FluxBC<T>> bc : list_bcs[i_rhs])
    {
      // Check input
      const int req_eval_per_fct
          = (bc->projection_required()) ? nqpoints_per_fct : nipoints_per_fct;

      // Check input
      if (req_eval_per_fct != bc->num_eval_per_facet())
      {
        throw std::runtime_error("BoundaryData: Number of evaluation points "
                                 "(FluxBC) does not match!");
      }

      // Adjust number of boundary facets
      _num_bcfcts[i_rhs] += bc->num_facets();

      // Extract list of boundary facets
      std::span<const std::int32_t> boundary_fcts = bc->facets();

      // Extract constants and coefficients
      constants_belmts = bc->extract_constants();

      std::tie(cstride_coefficients, coefficients_belmts)
          = bc->extract_coefficients();

      // Extract boundary kernel
      const auto& boundary_kernel = bc->boundary_kernel();

      // Loop over all facets
      for (std::int32_t i_fct = 0; i_fct < bc->num_facets(); ++i_fct)
      {
        // Get facet
        std::int32_t fct = boundary_fcts[i_fct];

        // Get cell adjacent to facet
        std::int32_t cell = _fct_to_cell->links(fct)[0];

        // Get cell-local facet id
        std::span<const std::int32_t> fcts_cell = cell_to_fct->links(cell);
        std::size_t fct_loc
            = std::distance(fcts_cell.begin(),
                            std::find(fcts_cell.begin(), fcts_cell.end(), fct));

        // Get ids of boundary DOFs
        boundary_dofs(cell, fct_loc, boundary_dofs_fct);

        /* Calculate boundary values on current cell */
        // Extract cell geometry
        std::span<const std::int32_t> nodes_cell = dofmap_geom.links(cell);
        for (std::size_t j = 0; j < nodes_cell.size(); ++j)
        {
          std::copy_n(std::next(node_coordinates.begin(), 3 * nodes_cell[j]), 3,
                      std::next(coordinates_data.begin(), 3 * j));
        }

        // Extract DOFs on boundary facet
        std::span<T> boundary_values_fct(
            boundary_values.data() + fct_loc * _ndofs_per_fct, _ndofs_per_fct);

        // Calculate mapping tensors
        double detJ
            = _kernel_data.compute_jacobian(J, K, _detJ_scratch, coordinates);

        // Evaluate boundary kernel
        std::fill(values_bkernel.begin(), values_bkernel.end(), 0.0);
        boundary_kernel(
            values_bkernel.data(),
            coefficients_belmts.data() + i_fct * cstride_coefficients,
            constants_belmts.data(), coordinates_data.data(), nullptr, nullptr);

        // Calculate boundary DOFs
        if (bc->projection_required())
        {
          // Check input
          if (nqpoints_per_fct != bc->num_eval_per_facet())
          {
            throw std::runtime_error(
                "BoundaryData: Quadrature degree of FluxBC does nor match!");
          }

          // Normal-trace on boundary facet
          std::span<T> flux_ntrace(values_bkernel.data()
                                       + fct_loc * nqpoints_per_fct,
                                   nqpoints_per_fct);

          // Initialise projection
          if (initialise_projection)
          {
            // The boundary normal vector
            boundary_normal.resize(_gdim);

            // The shape functions (at quadrature points)
            std::array<std::size_t, 5> shape
                = _kernel_data.shapefunctions_flux_qpoints(
                    _V_flux_hdiv->element()->basix_element(),
                    basis_projection_values);

            basis_projection = base::mdspan_t<const double, 5>(
                basis_projection_values.data(), shape);

            mbasis_projection_values.resize(nqpoints_per_fct * _ndofs_per_fct
                                            * _gdim);
            mbasis_projection = base::mdspan_t<double, 3>(
                mbasis_projection_values.data(), nqpoints_per_fct,
                _ndofs_per_fct, _gdim);

            // The equation system
            A_e.resize(_ndofs_per_fct, _ndofs_per_fct);
            L_e.resize(_ndofs_per_fct);
            u_e.resize(_ndofs_per_fct);

            // Mark projection as initialised
            initialise_projection = false;
          }

          // Extract quadrature weights
          std::span<const double> quadrature_weights
              = _kernel_data.quadrature_weights(0, fct_loc);

          // Calculate the boundary normal vector
          _kernel_data.physical_fct_normal(boundary_normal, K, fct_loc);

          // Map shape functions to current cell
          _kernel_data.map_shapefunctions_flux(fct_loc, mbasis_projection,
                                               basis_projection, J, detJ);

          // Assemble projection operator
          boundary_projection_kernel<T>(flux_ntrace, boundary_normal,
                                        mbasis_projection, quadrature_weights,
                                        detJ, A_e, L_e);

          // Solve linear system
          if (_ndofs_per_fct > 1)
          {
            solver.compute(A_e);
            u_e = solver.solve(L_e);

            for (std::size_t i = 0; i < _ndofs_per_fct; ++i)
            {
              boundary_values_fct[i] = u_e(i);
            }
          }
          else
          {
            boundary_values_fct[0] = L_e(0) / A_e(0, 0);
          }
        }
        else
        {
          // Check input
          if (nipoints_per_fct != bc->num_eval_per_facet())
          {
            throw std::runtime_error(
                "BoundaryData: FunctionSpace of FluxBC does not match!");
          }

          // Normal-trace on boundary facet
          std::span<T> flux_ntrace(values_bkernel.data()
                                       + fct_loc * nipoints_per_fct,
                                   nipoints_per_fct);

          // Perform interpolation
          _kernel_data.interpolate_flux(flux_ntrace, boundary_values_fct,
                                        fct_loc, J, detJ, K);
        }

        // Apply DOF transformations
        apply_inverse_dof_transform(boundary_values, cell_info, cell, 1);

        // Set values and markers
        facet_type_i[fct] = base::PatchFacetType::essnt_dual;
        _local_fct_id[fct] = fct_loc;

        for (std::size_t i = 0; i < _ndofs_per_fct; ++i)
        {
          boundary_markers_i[boundary_dofs_fct[i]] = true;
          x_bvals[boundary_dofs_fct[i]] = boundary_values_fct[i];
        }

        if ((reconstruct_stress) && (i_rhs < _gdim))
        {
          std::span<const std::int32_t> pnts_fct = fct_to_pnt->links(fct);

          for (auto pnt : pnts_fct)
          {
            _pnt_on_esnt_boundary[pnt] += 1;
          }
        }
      }
    }
  }

  // Finalise markers for pure essential stress BCs
  if (reconstruct_stress && (_gdim == 2))
  {
    for (std::size_t i = 0; i < _pnt_on_esnt_boundary.size(); ++i)
    {
      _pnt_on_esnt_boundary[i] = (_pnt_on_esnt_boundary[i] == 4) ? true : false;
    }
  }
}

template <typename T>
void BoundaryData<T>::calculate_patch_bc(
    std::span<const std::int32_t> bound_fcts,
    std::span<const std::int8_t> patchnode_local)
{
  /* Extract data */
  // The mesh
  std::shared_ptr<const mesh::Mesh> mesh = _V_flux_hdiv->mesh();

  // The geometry DOFmap
  const graph::AdjacencyList<std::int32_t>& dofmap_geom
      = mesh->geometry().dofmap();

  // The node coordinates
  std::span<const double> node_coordinates = mesh->geometry().x();

  std::vector<double> coordinates_data(mesh->geometry().cmap().dim() * 3);
  base::mdspan_t<const double, 2> coordinates(coordinates_data.data(),
                                              mesh->geometry().cmap().dim(), 3);

  /* Evaluate BCs for all RHS */
  for (std::size_t i_fct = 0; i_fct < bound_fcts.size(); i_fct++)
  {
    // Facet and cell
    std::int32_t fct = bound_fcts[i_fct];
    std::int32_t cell = _fct_to_cell->links(fct)[0];

    /* Evaluate mapping */
    // Extract cell coordinates
    std::span<const std::int32_t> nodes_cell = dofmap_geom.links(cell);
    for (std::size_t j = 0; j < nodes_cell.size(); ++j)
    {
      std::copy_n(std::next(node_coordinates.begin(), 3 * nodes_cell[j]), 3,
                  std::next(coordinates_data.begin(), 3 * j));
    }

    // Calculate J
    base::mdspan_t<double, 2> J(_data_J.data(), _gdim, _gdim);
    base::mdspan_t<double, 2> K(_data_K.data(), _gdim, _gdim);

    double detJ
        = _kernel_data.compute_jacobian(J, K, _detJ_scratch, coordinates);

    /* Calculate BCs on current facet */
    for (int i_rhs = 0; i_rhs < _num_rhs; ++i_rhs)
    {
      calculate_patch_bc(i_rhs, fct, patchnode_local[i_fct], J, detJ, K);
    }
  }
}

template <typename T>
void BoundaryData<T>::calculate_patch_bc(const int rhs_i,
                                         const std::int32_t fct,
                                         const std::int8_t hat_id,
                                         base::mdspan_t<const double, 2> J,
                                         const double detJ,
                                         base::mdspan_t<const double, 2> K)
{
  if (facet_type(rhs_i)[fct] == base::PatchFacetType::essnt_dual)
  {
    // The cell
    std::int32_t cell = _fct_to_cell->links(fct)[0];

    // The facet
    std::int8_t fct_loc = _local_fct_id[fct];

    /* Extract data */
    // Global DOF ids on facet
    std::vector<std::int32_t> boundary_dofs_fct = boundary_dofs(cell, fct);

    // Storage of (patch) boundary values
    std::span<T> boundary_values_rhs = boundary_values(rhs_i);

    // Storage of (global) boundary values
    std::span<const T> x_bfunc_rhs = _x_boundary_flux[rhs_i]->array();

    /* Calculate patch boundary */
    // Extract boundary DOFs without hat-function
    std::vector<T> flux_dofs_bfunc(_ndofs_per_fct);
    int ndofs_zero = 0;

    for (std::size_t i = 0; i < _ndofs_per_fct; i++)
    {
      flux_dofs_bfunc[i] = x_bfunc_rhs[boundary_dofs_fct[i]];

      if (std::abs(flux_dofs_bfunc[i]) < 1e-7)
      {
        ndofs_zero++;
      }
    }

    // Perform interpolation only on non-zero boundary
    if (ndofs_zero < _ndofs_per_fct)
    {
      // Get cell info
      std::span<const std::uint32_t> cell_info;
      if (_V_flux_hdiv->element()->needs_dof_transformations())
      {
        cell_info = std::span(
            _V_flux_hdiv->mesh()->topology().get_cell_permutation_info());
      }

      // Calculate boundary DOFs
      std::vector<T> bdofs_patch = _kernel_data.interpolate_flux(
          flux_dofs_bfunc, cell, fct_loc, hat_id, cell_info, J, detJ, K);

      for (std::size_t i = 0; i < _ndofs_per_fct; i++)
      {
        boundary_values_rhs[boundary_dofs_fct[i]] = bdofs_patch[i];
      }
    }
  }
}

template <typename T>
void BoundaryData<T>::boundary_dofs(const std::int32_t cell,
                                    const std::int8_t fct_loc,
                                    std::span<std::int32_t> boundary_dofs)
{
  if (_flux_is_discontinous)
  {
    // Get offset of facet DOFs in current cell
    const int offs = cell * _ndofs_per_cell + fct_loc * _ndofs_per_fct;

    // Calculate DOFs on boundary facet
    for (std::size_t i = 0; i < _ndofs_per_fct; ++i)
    {
      boundary_dofs[i] = offs + i;
    }
  }
  else
  {
    // Extract DOFs of current cell
    std::span<const std::int32_t> dofs_cell
        = _V_flux_hdiv->dofmap()->list().links(cell);

    // Local DOF-ids on facet
    const std::vector<int>& entity_dofs
        = _V_flux_hdiv->dofmap()->element_dof_layout().entity_dofs(_gdim - 1,
                                                                   fct_loc);

    // Calculate DOFs on boundary facet
    for (std::size_t i = 0; i < _ndofs_per_fct; ++i)
    {
      boundary_dofs[i] = dofs_cell[entity_dofs[i]];
    }
  }
}

// ------------------------------------------------------------------------------
template class BoundaryData<double>;
// ------------------------------------------------------------------------------
