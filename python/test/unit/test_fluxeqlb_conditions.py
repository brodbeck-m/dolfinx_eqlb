import numpy as np
from petsc4py import PETSc
import pytest

import basix
import dolfinx.mesh as dmesh
import dolfinx.fem as dfem
import ufl

from dolfinx_eqlb.eqlb import FluxEqlbEV, FluxEqlbSE
from dolfinx_eqlb.lsolver import local_projection

from utils import (
    create_unitsquare_builtin,
    create_unitsquare_gmsh,
    points_boundary_unitsquare,
    initialise_evaluate_function,
)
from testcase_poisson import (
    set_arbitrary_rhs,
    set_arbitrary_bcs,
    solve_poisson_problem,
    equilibrate_poisson,
)

""" Utility routines """


def check_divergence_condition(
    sigma_eq: dfem.Function, sigma_proj: dfem.Function, rhs_proj: dfem.Function
):
    # --- Extract solution data
    # the mesh
    mesh = sigma_eq.function_space.mesh

    # the geometry DOFmap
    gdmap = mesh.geometry.dofmap

    # number of cells in mesh
    n_cells = mesh.topology.index_map(2).size_local

    # degree of the flux space
    degree = sigma_eq.function_space.element.basix_element.degree

    # check if flux space is discontinuous
    flux_is_dg = sigma_eq.function_space.element.basix_element.discontinuous

    # --- Calculate divergence of the equilibrated flux
    V_div = dfem.FunctionSpace(mesh, ("DG", degree - 1))

    if flux_is_dg:
        div_sigeq = local_projection(V_div, [ufl.div(sigma_eq + sigma_proj)])[0]
    else:
        div_sigeq = local_projection(V_div, [ufl.div(sigma_eq)])[0]

    # points for checking divergence condition
    n_points = int(0.5 * (degree + 3) * (degree + 4))
    points_3d = np.zeros((n_points, 3))

    # loop over cells
    for c in range(0, n_cells):
        # points on current element
        x = np.sort(np.random.rand(2, n_points), axis=0)
        points = (
            np.column_stack([x[0], x[1] - x[0], 1.0 - x[1]])
            @ mesh.geometry.x[gdmap.links(c), :2]
        )
        points_3d[:, :2] = points

        # evaluate div(sig_eqlb)
        val_div_sigeq = div_sigeq.eval(points_3d, n_points * [c])

        # evaluate RHS
        val_rhs = rhs_proj.eval(points_3d, n_points * [c])

        assert np.allclose(val_div_sigeq, val_rhs)


def check_jump_condition(sigma_eq: dfem.Function, sig_proj: dfem.Function):
    # --- Extract geometry data
    # the mesh
    domain = sigma_eq.function_space.mesh

    # number of cells/ factes in mesh
    n_fcts = domain.topology.index_map(1).size_local
    n_cells = domain.topology.index_map(2).size_local

    # facet/cell connectivity
    fct_to_cell = domain.topology.connectivity(1, 2)
    cell_to_fct = domain.topology.connectivity(2, 1)

    # --- Interpolate proj./equilibr. flux into Baisx RT space
    # the flux degree
    degree = sigma_eq.function_space.element.basix_element.degree

    # create discontinuous RT space
    P_drt = basix.create_element(
        basix.ElementFamily.RT,
        basix.CellType.triangle,
        degree,
        basix.LagrangeVariant.equispaced,
        True,
    )

    V_drt = dfem.FunctionSpace(domain, basix.ufl_wrapper.BasixElement(P_drt))
    dofmap_sigrt = V_drt.dofmap.list

    # interpolate functions into space
    sig_eq_rt = dfem.Function(V_drt)
    sig_eq_rt.interpolate(sigma_eq)

    sig_proj_rt = dfem.Function(V_drt)
    sig_proj_rt.interpolate(sig_proj)

    # calculate reconstructed flux (use default RT-space)
    x_sig_rt = sig_proj_rt.x.array[:] + sig_eq_rt.x.array[:]

    # --- Determine sign of detj per cell
    # tabulate shape functions of geometry element
    c_element = basix.create_element(
        basix.ElementFamily.P,
        basix.CellType.triangle,
        1,
        basix.LagrangeVariant.gll_warped,
    )

    dphi_geom = c_element.tabulate(1, np.array([[0, 0]]))[1 : 2 + 1, 0, :, 0]

    # determine sign of detj per cell
    sign_detj = np.zeros(n_cells, dtype=np.float64)
    gdofs = np.zeros((3, 2), dtype=np.float64)

    for c in range(0, n_cells):
        # evaluate detj
        gdofs[:] = domain.geometry.x[domain.geometry.dofmap.links(c), :2]

        J_q = np.dot(gdofs.T, dphi_geom.T)
        detj = np.linalg.det(J_q)

        # determine sign of detj
        sign_detj[c] = np.sign(detj)

    # --- Check jump condition on all facets
    for f in range(0, n_fcts):
        # get cells adjacet to f
        cells = fct_to_cell.links(f)

        if len(cells) > 1:
            # signs of cell jacobis
            sign_plus = sign_detj[cells[0]]
            sign_minus = sign_detj[cells[1]]

            # local facet id of cell
            if_plus = np.where(cell_to_fct.links(cells[0]) == f)[0][0]
            if_minus = np.where(cell_to_fct.links(cells[1]) == f)[0][0]

            for i in range(0, degree):
                # local dof id of facet-normal flux
                dof_plus = dofmap_sigrt.links(cells[0])[if_plus * degree + i]
                dof_minus = dofmap_sigrt.links(cells[1])[if_minus * degree + i]

                # calculate outward flux
                if if_plus == 1:
                    flux_plus = x_sig_rt[dof_plus] * sign_plus
                else:
                    flux_plus = x_sig_rt[dof_plus] * (-sign_plus)

                if if_minus == 1:
                    flux_minus = x_sig_rt[dof_minus] * sign_minus
                else:
                    flux_minus = x_sig_rt[dof_minus] * (-sign_minus)

                # check continuity of facet-normal flux
                assert np.isclose(flux_plus + flux_minus, 0)


""" 
Check if equilibrated flux 
    a.) fullfil the divergence condition div(sigma_eqlb) = f 
    b.) lays in the H(div) space
"""


@pytest.mark.parametrize("mesh_type", ["builtin"])
@pytest.mark.parametrize("degree", [1, 2, 3, 4])
@pytest.mark.parametrize("bc_type", ["pure_dirichlet"])
@pytest.mark.parametrize("equilibrator", [FluxEqlbEV, FluxEqlbSE])
def test_equilibration_conditions(mesh_type, degree, bc_type, equilibrator):
    # Create mesh
    if mesh_type == "builtin":
        geometry = create_unitsquare_builtin(
            3, dmesh.CellType.triangle, dmesh.DiagonalType.crossed
        )
    elif mesh_type == "gmsh":
        raise NotImplementedError("GMSH mesh not implemented yet")
    else:
        raise ValueError("Unknown mesh type")

    for degree_prime in range(1, degree + 1):
        for degree_rhs in range(0, degree):
            # Set function space
            V_prime = dfem.FunctionSpace(geometry.mesh, ("P", degree_prime))

            # Determine degree of projected quantities (primal flux, RHS)
            degree_proj = max(degree_prime - 1, degree_rhs)

            # Set RHS
            rhs, rhs_projected = set_arbitrary_rhs(
                geometry.mesh, degree_rhs, degree_projection=degree_proj
            )

            # Set boundary conditions
            (
                boundary_id_dirichlet,
                boundary_id_neumann,
                dirichlet_functions,
                neumann_functions,
                neumann_projection,
            ) = set_arbitrary_bcs(bc_type, V_prime, degree, degree_rhs)

            # Solve equilibration
            u_prime, sigma_projected = solve_poisson_problem(
                V_prime,
                geometry,
                boundary_id_neumann,
                boundary_id_dirichlet,
                rhs,
                neumann_functions,
                dirichlet_functions,
                degree_projection=degree_proj,
            )

            # Solve equilibration
            sigma_eq = equilibrate_poisson(
                equilibrator,
                degree,
                geometry,
                [sigma_projected],
                [rhs_projected],
                [boundary_id_neumann],
                [boundary_id_dirichlet],
                [neumann_functions],
                [neumann_projection],
            )[0]

            # --- Check divergence condition ---
            check_divergence_condition(sigma_eq, sigma_projected, rhs_projected)

            # --- Check jump condition (only required for semi-explicit equilibrator)
            if equilibrator == FluxEqlbSE:
                check_jump_condition(sigma_eq, sigma_projected)


@pytest.mark.parametrize("mesh_type", ["builtin"])
@pytest.mark.parametrize("degree", [1, 2, 3, 4])
@pytest.mark.parametrize("equilibrator", [FluxEqlbEV])
def test_equilibration_boundary_condition(mesh_type, degree, equilibrator):
    # Create mesh
    if mesh_type == "builtin":
        geometry = create_unitsquare_builtin(
            3, dmesh.CellType.triangle, dmesh.DiagonalType.crossed
        )
    elif mesh_type == "gmsh":
        raise NotImplementedError("GMSH mesh not implemented yet")
    else:
        raise ValueError("Unknown mesh type")

    # Initialise boundary facets and test-points
    points_eval = points_boundary_unitsquare(geometry, [1, 4], degree + 1)

    npoints_per_boundary = int(points_eval.shape[0] / 2)
    plist_eval, clist_eval = initialise_evaluate_function(geometry.mesh, points_eval)

    # FunctionSpace of primal problem
    V_prime = dfem.FunctionSpace(geometry.mesh, ("P", degree))

    # Determine degree of projected quantities (primal flux, RHS)
    degree_proj = degree - 1

    # Set RHS
    rhs, rhs_projected = set_arbitrary_rhs(
        geometry.mesh, degree_proj, degree_projection=degree_proj
    )

    for degree_bc in range(0, degree):
        # Set function space
        V_ref = dfem.VectorFunctionSpace(geometry.mesh, ("DG", degree_bc))

        # Set boundary conditions
        x_ufl = ufl.SpatialCoordinate(geometry.mesh)

        if degree_bc == 0:
            ntrace_ufl = dfem.Constant(geometry.mesh, PETSc.ScalarType(0.15))
        else:
            ntrace_ufl = (
                0.2 * ((x_ufl[0] ** degree_bc) + (x_ufl[1] ** degree_bc)) + 0.15
            )

        boundary_id_dirichlet = [2, 3]
        boundary_id_neumann = [1, 4]
        dirichlet_functions = [dfem.Function(V_prime), dfem.Function(V_prime)]
        neumann_functions = [ntrace_ufl, ntrace_ufl]
        neumann_projection = [False, False]

        # Solve equilibration
        u_prime, sigma_projected = solve_poisson_problem(
            V_prime,
            geometry,
            boundary_id_neumann,
            boundary_id_dirichlet,
            rhs,
            neumann_functions,
            dirichlet_functions,
            degree_projection=degree_proj,
        )

        # Solve equilibration
        sigma_eq = equilibrate_poisson(
            equilibrator,
            degree,
            geometry,
            [sigma_projected],
            [rhs_projected],
            [boundary_id_neumann],
            [boundary_id_dirichlet],
            [neumann_functions],
            [neumann_projection],
        )[0]

        # Exact boundary conditions
        bc_ref = ufl.as_vector([-ntrace_ufl, ntrace_ufl])
        refsol = local_projection(V_ref, [bc_ref], quadrature_degree=2 * degree)[0]

        # Evaluate boundary values
        val_eqlbflux = sigma_eq.eval(plist_eval, clist_eval)
        val_ref = refsol.eval(plist_eval, clist_eval)

        assert np.allclose(
            val_eqlbflux[:npoints_per_boundary, 0], val_ref[:npoints_per_boundary, 0]
        )
        assert np.allclose(
            val_eqlbflux[npoints_per_boundary:, 1], val_ref[npoints_per_boundary:, 1]
        )


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
