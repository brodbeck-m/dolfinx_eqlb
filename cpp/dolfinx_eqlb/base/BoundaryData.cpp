// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "BoundaryData.hpp"

using namespace dolfinx;
using namespace dolfinx_eqlb::base;

template <dolfinx::scalar T, std::floating_point U>
BoundaryData<T, U>::BoundaryData(
    std::vector<std::vector<std::shared_ptr<FluxBC<T, U>>>>& list_bcs,
    std::vector<std::shared_ptr<fem::Function<T, U>>>& boundary_fluxes,
    std::shared_ptr<const fem::FunctionSpace<U>> V,
    const std::vector<std::vector<std::int32_t>>& fct_esntbound_prime,
    KernelDataBC<T, U>& kernel_data, const ProblemType problem_type)
    : _bcs(list_bcs), _boundary_fluxes(boundary_fluxes), _V(V),
      _flux_degree(_V->element()->basix_element().degree()),
      _num_rhs(list_bcs.size()), _gdim(_V->mesh()->geometry().dim()),
      _flux_is_discontinous(_V->element()->basix_element().discontinuous()),
      _ndofs_per_cell(_V->element()->space_dimension()),
      _ndofs_per_fct((_gdim == 2) ? _flux_degree
                                  : 0.5 * _flux_degree * (_flux_degree + 1)),
      _fct_to_cell(_V->mesh()->topology()->connectivity(_gdim - 1, _gdim)),
      _kernel_data(std::make_shared<KernelDataBC<T, U>>(kernel_data))
{
  // Counters
  std::int32_t num_fcts
      = _V->mesh()->topology()->index_map(_gdim - 1)->size_local();
  std::int32_t num_dofs = _V->dofmap()->index_map->size_local()
                          + _V->dofmap()->index_map->num_ghosts();
  // Resize storage
  std::int32_t size_dofdata = _num_rhs * num_dofs;
  _boundary_values.resize(size_dofdata, 0.0);
  _boundary_markers.resize(size_dofdata, false);
  _offset_dofdata.resize(_num_rhs + 1);

  std::int32_t size_fctdata = _num_rhs * num_fcts;
  _local_fct_id.resize(num_fcts);
  _facet_type.resize(size_fctdata, FacetType::internal);
  _offset_fctdata.resize(_num_rhs + 1);

  if (problem_type != ProblemType::flux)
  {
    std::int32_t size_nodedata
        = _V->mesh()->topology()->index_map(0)->size_local()
          + _V->mesh()->topology()->index_map(0)->num_ghosts();
    _pnt_on_esnt_boundary.resize(size_nodedata, false);
  }

  // Set offsets
  std::generate(_offset_dofdata.begin(), _offset_dofdata.end(),
                [n = 0, num_dofs]() mutable { return num_dofs * (n++); });
  std::generate(_offset_fctdata.begin(), _offset_fctdata.end(),
                [n = 0, num_fcts]() mutable { return num_fcts * (n++); });

  /* Extract required data */
  // The mesh
  std::shared_ptr<const mesh::Mesh<U>> mesh = _V->mesh();

  // Required conductivities
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> cell_to_fct
      = mesh->topology()->connectivity(_gdim, _gdim - 1);
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> fct_to_pnt
      = mesh->topology()->connectivity(_gdim - 1, 0);

  // Counters interpolation- and quadrature points
  const int nipoints_per_fct
      = _kernel_data->num_interpolation_points_per_facet();
  const int nqpoints_per_fct = _kernel_data->num_points(0, 0);

  /* Initialise storage */
  // DOF ids and values on boundary facets
  // (values on entire element to enable application of DOF transformations)
  std::vector<std::int32_t> boundary_dofs_fct(_ndofs_per_fct);

  /* Set facet- and DOF markers */
  for (int i_rhs = 0; i_rhs < _num_rhs; ++i_rhs)
  {
    // Storage of current RHS
    std::span<std::int8_t> facet_type_i = facet_type(i_rhs);
    std::span<std::int8_t> boundary_markers_i = boundary_markers(i_rhs);

    // Handle facets with essential BCs on primal problem
    for (std::int32_t fct : fct_esntbound_prime[i_rhs])
    {
      // Set facet type
      facet_type_i[fct] = FacetType::essnt_primal;
    }

    // Handle facets with essential BCs on dual problem
    for (std::shared_ptr<FluxBC<T, U>> bc : list_bcs[i_rhs])
    {
      // Check input
      const int req_eval_per_fct
          = (bc->projection_required()) ? nqpoints_per_fct : nipoints_per_fct;

      // Check input
      if (!(bc->is_zero()))
      {
        if (req_eval_per_fct != bc->num_eval_per_facet())
        {
          throw std::runtime_error("BoundaryData: Number of evaluation points "
                                   "(FluxBC) does not match!");
        }
      }

      // Loop over all boundary facets
      for (std::int32_t fct : bc->facets())
      {
        // Get cell adjacent to facet
        std::int32_t cell = _fct_to_cell->links(fct)[0];

        // Get cell-local facet id
        std::span<const std::int32_t> fcts_cell = cell_to_fct->links(cell);
        std::size_t fct_loc
            = std::distance(fcts_cell.begin(),
                            std::find(fcts_cell.begin(), fcts_cell.end(), fct));

        // Get ids of boundary DOFs
        boundary_dofs(cell, fct_loc, boundary_dofs_fct);

        // Set values and markers
        facet_type_i[fct] = FacetType::essnt_dual;
        _local_fct_id[fct] = fct_loc;

        for (std::size_t i = 0; i < _ndofs_per_fct; ++i)
        {
          boundary_markers_i[boundary_dofs_fct[i]] = true;
        }

        if ((problem_type != ProblemType::flux) && (i_rhs < _gdim))
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

  /* Evluate boundary values */
  evaluate_boundary_flux(true);

  // Finalise markers for pure essential stress BCs
  if ((problem_type != ProblemType::flux) && (_gdim == 2))
  {
    for (std::size_t i = 0; i < _pnt_on_esnt_boundary.size(); ++i)
    {
      _pnt_on_esnt_boundary[i] = (_pnt_on_esnt_boundary[i] == 4) ? true : false;
    }
  }
}

template <dolfinx::scalar T, std::floating_point U>
void BoundaryData<T, U>::update(
    std::vector<std::shared_ptr<const fem::Constant<T>>>& time_functions)
{
  // Update the transient BCs without time-function
  evaluate_boundary_flux(false);

  // Update the transient BCs with time-function
  for (int i_rhs = 0; i_rhs < _num_rhs; ++i_rhs)
  {
    // Time-function values of the current RHS
    std::span<const T> tfkts(time_functions[i_rhs]->value);

    // Number of BCs and time-functions
    const int nbcs = _bcs[i_rhs].size();
    const int ntfkts = tfkts.size();

    if (nbcs != ntfkts)
    {
      throw std::runtime_error("Provided time-functions do not match the BCs.");
    }

    // Boundary DOFs of the current RHS
    std::span<T> x_bvals = _boundary_fluxes[i_rhs]->x()->mutable_array();

    // Handle facets with essential BCs on dual problem
    for (std::size_t j = 0; j < _bcs[i_rhs].size(); ++j)
    {
      // The boundary condition
      std::shared_ptr<FluxBC<T, U>> bc = _bcs[i_rhs][j];

      if (bc->transient_behaviour() == TimeType::timefunction)
      {
        // The boundary facets
        std::span<const std::int32_t> bfcts = bc->facets();

        // The initial bounday DOFs
        mdspan_t<const T, 2> initial_bvals
            = bc->initial_boundary_dofs(_ndofs_per_fct);

        // Loop over all boundary factes
        for (std::size_t e = 0; e < bc->num_facets(); ++e)
        {
          // The facet
          std::int32_t bfct = bfcts[e];

          // The boundary entity
          std::int32_t c = _fct_to_cell->links(bfct)[0];
          std::int32_t f = _local_fct_id[bfct];

          // Copy values to global storage
          auto cell_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
              _V->dofmap()->map(), c,
              MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
          std::size_t offs = f * _ndofs_per_fct;

          for (std::size_t k = 0; k < _ndofs_per_fct; ++k)
          {
            x_bvals[cell_dofs[offs + k]] = tfkts[j] * initial_bvals(e, k);
          }
        }
      }
    }
  }
}

template <dolfinx::scalar T, std::floating_point U>
void BoundaryData<T, U>::calculate_patch_bc(
    std::span<const std::int32_t> bound_fcts,
    std::span<const std::int8_t> patchnode_local)
{
  /* Extract data */
  // The mesh
  std::shared_ptr<const mesh::Mesh<U>> mesh = _V->mesh();

  // The geometry DOFmap
  mdspan_t<const std::int32_t, 2> dofmap_geom = mesh->geometry().dofmap();

  // The node coordinates
  std::span<const U> node_coordinates = mesh->geometry().x();

  std::vector<U> coordinates_data(mesh->geometry().cmap().dim() * 3);
  mdspan_t<const U, 2> coordinates(coordinates_data.data(),
                                   mesh->geometry().cmap().dim(), 3);

  /* Evaluate BCs for all RHS */
  for (std::size_t i_fct = 0; i_fct < bound_fcts.size(); i_fct++)
  {
    // Facet and cell
    std::int32_t fct = bound_fcts[i_fct];
    std::int32_t cell = _fct_to_cell->links(fct)[0];

    /* Evaluate mapping */
    // Extract cell coordinates
    auto nodes_cell = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        dofmap_geom, cell, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);

    for (std::size_t j = 0; j < nodes_cell.size(); ++j)
    {
      std::copy_n(std::next(node_coordinates.begin(), 3 * nodes_cell[j]), 3,
                  std::next(coordinates_data.begin(), 3 * j));
    }

    // Calculate J
    mdspan_t<U, 2> J(_data_J.data(), _gdim, _gdim);
    mdspan_t<U, 2> K(_data_K.data(), _gdim, _gdim);

    U detJ = _kernel_data->compute_jacobian(J, K, _detJ_scratch, coordinates);

    /* Calculate BCs on current facet */
    for (int i_rhs = 0; i_rhs < _num_rhs; ++i_rhs)
    {
      calculate_patch_bc(i_rhs, fct, patchnode_local[i_fct], J, detJ, K);
    }
  }
}

template <dolfinx::scalar T, std::floating_point U>
void BoundaryData<T, U>::calculate_patch_bc(
    const int rhs_i, const std::int32_t fct, const std::int8_t hat_id,
    mdspan_t<const U, 2> J, const U detJ, mdspan_t<const U, 2> K)
{
  if (facet_type(rhs_i)[fct] == FacetType::essnt_dual)
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
    std::span<const T> x_bfunc_rhs = _boundary_fluxes[rhs_i]->x()->array();

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
      if (_V->element()->needs_dof_transformations())
      {
        cell_info
            = std::span(_V->mesh()->topology()->get_cell_permutation_info());
      }

      // Calculate boundary DOFs
      std::vector<T> bdofs_patch = _kernel_data->interpolate_flux(
          flux_dofs_bfunc, cell, fct_loc, hat_id, cell_info, J, detJ, K);

      for (std::size_t i = 0; i < _ndofs_per_fct; i++)
      {
        boundary_values_rhs[boundary_dofs_fct[i]] = bdofs_patch[i];
      }
    }
  }
}

template <dolfinx::scalar T, std::floating_point U>
void BoundaryData<T, U>::boundary_dofs(const std::int32_t cell,
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
    smdspan_t<const std::int32_t, 1> dofs_cell
        = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
            _V->dofmap()->map(), cell,
            MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);

    // Local DOF-ids on facet
    const std::vector<int>& entity_dofs
        = _V->dofmap()->element_dof_layout().entity_dofs(_gdim - 1, fct_loc);

    // Calculate DOFs on boundary facet
    for (std::size_t i = 0; i < _ndofs_per_fct; ++i)
    {
      boundary_dofs[i] = dofs_cell[entity_dofs[i]];
    }
  }
}

template <dolfinx::scalar T, std::floating_point U>
void BoundaryData<T, U>::evaluate_boundary_flux(
    const bool initialise_boundary_values)
{
  /* Extract required data */
  // The mesh
  std::shared_ptr<const mesh::Mesh<U>> mesh = _V->mesh();

  // Geometry DOFmap and nodal coordinates
  mdspan_t<const std::int32_t, 2> x_dofmap = mesh->geometry().dofmap();
  std::span<const U> x_g = mesh->geometry().x();

  // The transformation function of the flux space
  const auto apply_inverse_dof_transform
      = _V->element()->template dof_transformation_fn<T>(
          fem::doftransform::inverse_transpose, false);

  std::span<const std::uint32_t> cell_info;
  if (_V->element()->needs_dof_transformations())
  {
    mesh->topology_mutable()->create_entity_permutations();
    cell_info = std::span(mesh->topology()->get_cell_permutation_info());
  }

  // The counters of interpolation- and quadrature points
  const int nipoints = _kernel_data->num_interpolation_points();
  const int nipoints_per_fct
      = _kernel_data->num_interpolation_points_per_facet();
  const int nqpoints = _kernel_data->num_points(0);
  const int nqpoints_per_fct = _kernel_data->num_points(0, 0);

  /* Initialisations */
  // Geometry mapping
  std::vector<U> coord_dofs(x_dofmap.extent(1) * 3);
  mdspan_t<const U, 2> coord(coord_dofs.data(), x_dofmap.extent(1), 3);

  mdspan_t<U, 2> J(_data_J.data(), _gdim, _gdim);
  mdspan_t<U, 2> K(_data_K.data(), _gdim, _gdim);

  // The DOF ids and values on boundary facets
  // (values on entire element to enable application of DOF transformations)
  std::vector<T> x_bvals_cell(_ndofs_per_cell);

  // Projection of boundary values
  std::vector<U> boundary_normal(_gdim);

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A_e(_ndofs_per_fct,
                                                       _ndofs_per_fct);
  Eigen::Matrix<T, Eigen::Dynamic, 1> L_e(_ndofs_per_fct), u_e(_ndofs_per_fct);
  Eigen::LLT<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> solver;

  std::vector<double> basis_projection_values, mbasis_projection_values;

  std::array<std::size_t, 5> shape = _kernel_data->shapefunctions_flux_qpoints(
      _V->element()->basix_element(), basis_projection_values);
  mdspan_t<const double, 5> basis_projection
      = mdspan_t<const U, 5>(basis_projection_values.data(), shape);

  mbasis_projection_values.resize(nqpoints_per_fct * _ndofs_per_fct * _gdim);
  mdspan_t<double, 3> mbasis_projection = mdspan_t<U, 3>(
      mbasis_projection_values.data(), nqpoints_per_fct, _ndofs_per_fct, _gdim);

  /* Evaluate the boundary DOFs */
  for (int i_rhs = 0; i_rhs < _num_rhs; ++i_rhs)
  {
    // Storage of current RHS
    std::span<T> x_bvals = _boundary_fluxes[i_rhs]->x()->mutable_array();

    // Handle facets with essential BCs on dual problem
    for (std::shared_ptr<FluxBC<T, U>> bc : _bcs[i_rhs])
    {
      if ((initialise_boundary_values
           || (bc->transient_behaviour() == TimeType::timedependent))
          && !(bc->is_zero()))
      {
        // The boundary facets
        std::span<const std::int32_t> bfcts = bc->facets();

        // Prepare evaluation of boundary expression
        auto [coeffs, cstride] = bc->pack_coefficients(_local_fct_id);
        std::vector<T> constant_data = bc->pack_constants();

        auto bfn = bc->get_tabulate_expression();

        // Evlaute boundary values
        std::vector<T> values_local(bc->num_eval_per_facet(), 0);
        for (std::size_t e = 0; e < bc->num_facets(); ++e)
        {
          // The facet
          std::int32_t bfct = bfcts[e];

          // The boundary entity
          std::int32_t c = _fct_to_cell->links(bfct)[0];
          std::int32_t f = _local_fct_id[bfct];

          // The cell geometry
          auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
              x_dofmap, c, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);

          for (std::size_t i = 0; i < x_dofs.size(); ++i)
          {
            std::copy_n(std::next(x_g.begin(), 3 * x_dofs[i]), 3,
                        std::next(coord_dofs.begin(), 3 * i));
          }

          // Evaluate the boundary expression
          const T* coeff_cell = coeffs.data() + e * cstride;
          std::ranges::fill(values_local, 0);

          bfn(values_local.data(), coeff_cell, constant_data.data(),
              coord_dofs.data(), &f, nullptr);

          // Evalute mapping tensors
          U detJ = _kernel_data->compute_jacobian(J, K, _detJ_scratch, coord);

          // Interpolate BC into flux function
          std::span<T> x_bvals_fct(x_bvals_cell.data() + f * _ndofs_per_fct,
                                   _ndofs_per_fct);

          if (bc->projection_required())
          {
            // The quadrature weights
            std::span<const U> weights = _kernel_data->quadrature_weights(0, f);

            // The boundary normal vector
            _kernel_data->physical_fct_normal(boundary_normal, K, f);

            // Map shape functions to current cell
            _kernel_data->map_shapefunctions_flux(f, mbasis_projection,
                                                  basis_projection, J, detJ);

            // Assemble projection operator
            boundary_projection_kernel<T, U>(values_local, boundary_normal,
                                             mbasis_projection, weights, detJ,
                                             A_e, L_e);

            // Solve the projection
            if (_ndofs_per_fct > 1)
            {
              solver.compute(A_e);
              u_e = solver.solve(L_e);

              for (std::size_t i = 0; i < _ndofs_per_fct; ++i)
              {
                x_bvals_fct[i] = u_e(i);
              }
            }
            else
            {
              x_bvals_fct[0] = L_e(0) / A_e(0, 0);
            }
          }
          else
          {
            _kernel_data->interpolate_flux(values_local, x_bvals_fct, f, J,
                                           detJ, K);
          }

          // Apply DOF transformations
          apply_inverse_dof_transform(x_bvals_cell, cell_info, c, 1);

          // Copy values to global storage
          auto cell_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
              _V->dofmap()->map(), c,
              MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
          std::size_t offs = f * _ndofs_per_fct;

          for (std::size_t i = 0; i < _ndofs_per_fct; ++i)
          {
            x_bvals[cell_dofs[offs + i]] = x_bvals_fct[i];
          }

          // Store the initial boundary DOFs (if required)
          // TODO - This will not work in parallel (update after scatter!)
          bc->set_initial_boundary_dofs(e, x_bvals_fct);
        }
      }
    }
  }
}

// ------------------------------------------------------------------------------
template class BoundaryData<double, double>;
// ------------------------------------------------------------------------------
