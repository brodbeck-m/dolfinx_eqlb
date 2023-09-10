#include "BoundaryData.hpp"

using namespace dolfinx;
using namespace dolfinx_eqlb;

template <typename T>
void assemble_projection(mdspan_t<const double, 2> flux_boundary,
                         mdspan_t<double, 3> phi,
                         std::span<const double> quadrature_weights,
                         const double detJ,
                         Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& A_e,
                         Eigen::Matrix<T, Eigen::Dynamic, 1>& L_e)
{
  // The number of quadrature points
  const int num_qpoints = quadrature_weights.size();

  // Map shape-functions to reference element
  const int ndofs_per_fct = phi.extent(1);

  // Initialise tangent arrays
  A_e.setZero();
  L_e.setZero();

  // Loop over quadrature points
  const int gdim = flux_boundary.extent(1);

  for (std::size_t iq = 0; iq < num_qpoints; ++iq)
  {
    for (std::size_t i = 0; i < ndofs_per_fct; ++i)
    {
      // Set RHS
      T scal_l = 0.0;

      for (std::size_t k = 0; k < gdim; ++k)
      {
        scal_l += flux_boundary(iq, k) * phi(iq, i, k);
      }

      L_e(i) += scal_l * detJ * quadrature_weights[iq];

      for (std::size_t j = i; j < ndofs_per_fct; ++j)
      {
        // Calculate mass matrix
        double scal_m = 0.0;

        for (std::size_t k = 0; k < gdim; ++k)
        {
          scal_m += phi(iq, i, k) * phi(iq, j, k);
        }

        A_e(i, j) += scal_m * detJ * quadrature_weights[iq];
      }
    }
  }

  // Add symmetric entries of mass-matrix
  for (std::size_t i = 1; i < ndofs_per_fct; ++i)
  {
    for (std::size_t j = 0; j < i; ++j)
    {
      // Calculate mass matrix
      A_e(i, j) += A_e(j, i);
    }
  }
}

template <typename T>
BoundaryData<T>::BoundaryData(
    std::vector<std::vector<std::shared_ptr<FluxBC<T>>>>& list_bcs,
    std::vector<std::shared_ptr<fem::Function<T>>>& boundary_flux,
    std::shared_ptr<const fem::FunctionSpace> V_flux_hdiv,
    bool rtflux_is_custom,
    const std::vector<std::vector<std::int32_t>>& fct_esntbound_prime)
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
      _quadrature_rule(QuadratureRule(
          V_flux_hdiv->mesh()->topology().cell_type(), 2 * _flux_degree,
          V_flux_hdiv->mesh()->geometry().dim() - 1)),
      _kernel_data(
          KernelDataBC<T>(V_flux_hdiv->mesh(),
                          {std::make_shared<QuadratureRule>(_quadrature_rule)},
                          V_flux_hdiv->element(), _ndofs_per_fct,
                          _ndofs_per_cell, rtflux_is_custom))
{
  // Resize storage
  _num_bcfcts.resize(_num_rhs);

  std::int32_t size_dofdata = _num_rhs * _num_dofs;
  _boundary_values.resize(size_dofdata, 0.0);
  _boundary_markers.resize(size_dofdata, false);
  _offset_dofdata.resize(_num_rhs + 1);

  std::int32_t size_fctdata = _num_rhs * _num_fcts;
  _local_fct_id.resize(size_fctdata);
  _facet_type.resize(size_fctdata, facet_type_eqlb::internal);
  _offset_fctdata.resize(_num_rhs + 1);

  // Set offsets
  std::int32_t num_dofs = _num_dofs;
  std::generate(_offset_dofdata.begin(), _offset_dofdata.end(),
                [n = 0, num_dofs]() mutable { return num_dofs * (n++); });

  std::int32_t num_fcts = _num_fcts;
  std::generate(_offset_fctdata.begin(), _offset_fctdata.end(),
                [n = 0, num_fcts]() mutable { return num_fcts * (n++); });

  mdspan_t<double, 2> J(_data_J.data(), _gdim, _gdim);
  mdspan_t<double, 2> K(_data_K.data(), _gdim, _gdim);

  /* Extract required data */
  // The mesh
  std::shared_ptr<const mesh::Mesh> mesh = V_flux_hdiv->mesh();

  // Required conductivities
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> cell_to_fct
      = mesh->topology().connectivity(_gdim, _gdim - 1);

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

  /* Initialise storage */
  // DOF ids and values on boundary facets
  // (values on entire element to enable application of DOF transformations)
  std::vector<std::int32_t> boundary_dofs_fct(_ndofs_per_fct);
  std::vector<T> boundary_values(_ndofs_per_cell);

  // Storage for extraction of cell geometry
  std::vector<double> coordinates_data(mesh->geometry().cmap().dim() * 3);
  mdspan_t<const double, 2> coordinates(coordinates_data.data(),
                                        mesh->geometry().cmap().dim(), 3);

  // Number of interpolation points
  const int nipoints = _kernel_data.num_interpolation_points();
  const int nipoints_per_fct
      = _kernel_data.num_interpolation_points_per_facet();

  // Number of quadrature points
  const int nqpoints = _quadrature_rule.num_points();
  const int nqpoints_per_fct = _quadrature_rule.num_points(0);

  // Storage of normal-traces on boundary
  std::vector<T> values_bkernel(std::max(nipoints, nqpoints));

  // Constants/coefficients for evaluation of boundary kernel
  std::vector<T> constants_belmts, coefficients_belmts;
  std::int32_t cstride_coefficients;

  // Shape-functions (projection)
  bool initialise_projection = true;

  std::vector<double> basis_projection_values, mbasis_projection_values;
  mdspan_t<const double, 5> basis_projection;
  mdspan_t<double, 3> mbasis_projection;

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
    std::span<std::int8_t> local_fct_id_i = local_facet_id(i_rhs);
    std::span<std::int8_t> boundary_markers_i = boundary_markers(i_rhs);

    std::span<T> x_bvals = boundary_flux[i_rhs]->x()->mutable_array();

    // Handle facets with essential BCs on primal problem
    for (std::int32_t fct : fct_esntbound_prime[i_rhs])
    {
      // Set facet type
      facet_type_i[fct] = facet_type_eqlb::essnt_primal;
    }

    // Handle facets with essential BCs on dual problem
    for (std::shared_ptr<FluxBC<T>> bc : list_bcs[i_rhs])
    {
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

        std::span<T> flux_ntrace(values_bkernel.data()
                                     + fct_loc * nipoints_per_fct,
                                 nipoints_per_fct);

        // Calculate boundary DOFs
        if (bc->projection_required())
        {
          // Initialise projection
          if (initialise_projection)
          {
            // Storage for shape functions
            std::array<std::size_t, 5> shape
                = _kernel_data.shapefunctions_flux_qpoints(
                    _V_flux->element()->basix_element(),
                    basis_projection_values);

            basis_projection = mdspan_t<const double, 5>(
                basis_projection_values.data(), shape);

            mbasis_projection_values.resize(nqpoints_per_fct * _ndofs_per_fct
                                            * _gdim);
            mbasis_projection
                = mdspan_t<double, 3>(mbasis_projection_values.data(),
                                      nqpoints_per_fct, _ndofs_per_fct, _gdim);

            // Storage equation system
            A_e.resize(_ndofs_per_fct, _ndofs_per_fct);
            L_e.resize(_ndofs_per_fct);
            u_e.resize(_ndofs_per_fct);

            // Mark projection as initialised
            initialise_projection = false;
          }

          // Extract quadrature weights
          std::span<const double> quadrature_weights
              = kernel_data.extract_quadrature_weights(fct_loc);

          // Calculate flux on boundary
          mdspan_t<const double, 2> flux_vector
              = kernel_data.normaltrace_to_flux(flux_ntrace, fct_loc, K);

          // Assemble linear system
          assemble_projection(kernel_data, flux_ntrace, A_e, L_e, fct_loc, J,
                              detJ, K);

          // Solve linear system

          // Move boundary DOFs into element vector

          throw std::runtime_error(
              "Projection for boundary conditions not implemented");
        }
        else
        {
          // Perform interpolation
          _kernel_data.interpolate_flux(flux_ntrace, boundary_values_fct,
                                        fct_loc, J, detJ, K);
        }

        // Apply DOF transformations
        apply_inverse_dof_transform(boundary_values, cell_info, cell, 1);

        // Set values and markers
        facet_type_i[fct] = facet_type_eqlb::essnt_dual;
        local_fct_id_i[fct] = fct_loc;

        for (std::size_t i = 0; i < _ndofs_per_fct; ++i)
        {
          boundary_markers_i[boundary_dofs_fct[i]] = true;
          x_bvals[boundary_dofs_fct[i]] = boundary_values_fct[i];
        }
      }
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
  mdspan_t<const double, 2> coordinates(coordinates_data.data(),
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
    mdspan_t<double, 2> J(_data_J.data(), _gdim, _gdim);
    mdspan_t<double, 2> K(_data_K.data(), _gdim, _gdim);

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
void BoundaryData<T>::calculate_patch_bc(
    const int rhs_i, const std::int32_t fct, const std::int8_t hat_id,
    mdspan_t<const double, 2> J, const double detJ, mdspan_t<const double, 2> K)
{
  if (facet_type(rhs_i)[fct] == facet_type_eqlb::essnt_dual)
  {
    // The cell
    std::int32_t cell = _fct_to_cell->links(fct)[0];

    // The facet
    std::int8_t fct_loc = local_facet_id(rhs_i)[fct];

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
