import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import pytest

import basix
from basix import CellType
import dolfinx.mesh as dmesh
import dolfinx.fem as dfem
import ufl

from dolfinx_eqlb import equilibration

""" Utility functions """


def solve_equilibration(degree, eqlb_type, n_elmt=5):
    # --- Solve primal problem
    # create mesh
    domain = dmesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0, 0]), np.array([2, 2])],
        [n_elmt, n_elmt],
        cell_type=dmesh.CellType.triangle,
        diagonal=dmesh.DiagonalType.crossed,
    )

    # get all boundary facets
    domain.topology.create_connectivity(1, 2)
    boundary_facets = dmesh.exterior_facet_indices(domain.topology)

    # set function space
    V_cg = dfem.FunctionSpace(domain, ("CG", degree))

    # set source term
    V_dg = dfem.FunctionSpace(domain, ("DG", degree - 1))
    f = dfem.Function(V_dg)
    f.x.array[:] = 2 * (np.random.rand(V_dg.dofmap.index_map.size_local) + 0.1)

    # set weak forms
    u = ufl.TrialFunction(V_cg)
    v = ufl.TestFunction(V_cg)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    l = f * v * ufl.dx

    # set boundary conditions
    dofs = dfem.locate_dofs_topological(V_cg, 1, boundary_facets)
    bc_esnt = [dfem.dirichletbc(PETSc.ScalarType(0), dofs, V_cg)]

    # solve primal problem
    solveoptions = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ksp_rtol": 1e-10,
        "ksp_atol": 1e-10,
    }
    problem_prime = dfem.petsc.LinearProblem(a, l, bc_esnt, petsc_options=solveoptions)
    uh = problem_prime.solve()

    # --- Project flux
    # set function space for flux
    V_flux_proj = dfem.VectorFunctionSpace(domain, ("DG", degree - 1))

    # set weak forms
    u = ufl.TrialFunction(V_flux_proj)
    v = ufl.TestFunction(V_flux_proj)

    a_proj = ufl.inner(u, v) * ufl.dx
    l_proj = ufl.inner(-ufl.grad(uh), v) * ufl.dx

    # solve projection
    problem_proj = dfem.petsc.LinearProblem(
        a_proj, l_proj, bcs=[], petsc_options=solveoptions
    )
    sig_proj = problem_proj.solve()

    # --- Equilibrate flux
    # setup and solve equilibration
    if eqlb_type == "EV":
        equilibrator = equilibration.EquilibratorEV(degree, domain, [f], [sig_proj])
    elif eqlb_type == "SemiExplt":
        equilibrator = equilibration.EquilibratorSemiExplt(
            degree, domain, [f], [sig_proj]
        )
    else:
        raise ValueError("Unknown equilibration type")

    equilibrator.set_boundary_conditions([boundary_facets], [[]], [[]])
    equilibrator.equilibrate_fluxes()

    return uh, f, sig_proj, equilibrator.list_flux[0]


def isoparametric_mapping_triangle(domain, dphi_geom, cell_id):
    # geometry data for current cell
    geometry = np.zeros((3, 2), dtype=np.float64)
    geometry[:] = domain.geometry.x[domain.geometry.dofmap.links(cell_id), :2]

    J_q = np.dot(geometry.T, dphi_geom.T)
    detj = np.linalg.det(J_q)

    return J_q, detj


def fkt_to_drt(list_fkt, degree):
    # extract mesh
    domain = list_fkt[0].function_space.mesh

    # create DRT space
    P_drt = basix.create_element(
        basix.ElementFamily.RT,
        basix.CellType.triangle,
        degree,
        basix.LagrangeVariant.equispaced,
        True,
    )
    V_drt = dfem.FunctionSpace(domain, basix.ufl_wrapper.BasixElement(P_drt))

    # interpolate function into DRT
    list_fkt_drt = []

    for fkt in list_fkt:
        fkt_drt = dfem.Function(V_drt)
        fkt_drt.interpolate(fkt)

        list_fkt_drt.append(fkt_drt)

    return list_fkt_drt


def evaluate_fe_functions(shp_fkt, dofs):
    # initialisation
    values = np.zeros((shp_fkt.shape[0], shp_fkt.shape[2]))

    # loop over all points
    for p in range(0, shp_fkt.shape[0]):
        # loop over all basis functions
        for i in range(0, shp_fkt.shape[1]):
            # loop over all dimensions
            for d in range(0, shp_fkt.shape[2]):
                values[p, d] += shp_fkt[p, i, d] * dofs[i]

    return values


def evalute_div_rt(dphi, detj, dofs):
    # initialisation
    values = np.zeros((dphi.shape[1]))

    # loop over all points
    for p in range(0, dphi.shape[1]):
        # loop over all basis functions
        for i in range(0, dphi.shape[2]):
            values[p] += dphi[0, p, i, 0] * dofs[i] + dphi[1, p, i, 1] * dofs[i]

    return values / detj


""" Test divergence condition: div(sigma) = f """


@pytest.mark.parametrize("eqlb_type", ["EV", "SemiExplt"])
@pytest.mark.parametrize("degree", [1])
def test_div_condition(degree, eqlb_type):
    # --- Solve equilibration
    uh, f_proj, sig_proj, sig_eq = solve_equilibration(degree, eqlb_type, n_elmt=5)

    # interpolate sig_eq and sig_proj into (basix) DRT-space
    list_fkt = fkt_to_drt([sig_eq, sig_proj], degree)
    x_sig_eq = list_fkt[0].x.array[:] + list_fkt[1].x.array[:]
    dofmap_sig = list_fkt[0].function_space.dofmap.list

    # extract relevant data
    domain = uh.function_space.mesh
    n_cells = domain.topology.index_map(2).size_local

    dofmap_f = f_proj.function_space.dofmap.list

    # --- Check divergence condition
    # points for checking divergence condition
    points = basix.create_lattice(
        basix.CellType.triangle, degree + 1, basix.LatticeType.equispaced, True
    )

    # tabulate shape functions of geometry element
    c_element = basix.create_element(
        basix.ElementFamily.P,
        basix.CellType.triangle,
        1,
        basix.LagrangeVariant.gll_warped,
    )
    dphi_geom = c_element.tabulate(1, np.array([[0, 0]]))[1 : 2 + 1, 0, :, 0]

    # tabulate shape function derivatives of flux element
    dphi_sig = sig_eq.function_space.element.basix_element.tabulate(1, points)[
        1 : 2 + 1, :, :, :
    ]

    # tabulate shape functions of projected RHS
    phi_f = f_proj.function_space.element.basix_element.tabulate(1, points)[0, :, :, :]

    # loop over cells
    for c in range(0, n_cells):
        # evaluate jacobi matrix
        J_q, detj = isoparametric_mapping_triangle(domain, dphi_geom, c)

        # extract dofs on current cell
        dofs_flux = x_sig_eq[dofmap_sig.links(c)]
        dofs_f = f_proj.x.array[dofmap_f.links(c)]

        # evaluate divergence
        div_sig = evalute_div_rt(dphi_sig, detj, dofs_flux)

        # evaluate RHS
        rhs = evaluate_fe_functions(phi_f, dofs_f)

        assert np.allclose(div_sig, rhs)


""" Test jump condition on facet-normal fluxes """


@pytest.mark.parametrize("degree", [1])
def test_jump_condition(degree):
    # --- Solve equilibration
    uh, f_proj, sig_proj, sig_eq = solve_equilibration(degree, "SemiExplt", n_elmt=5)

    # interpolate functions into DRT-space
    fkt_drt = fkt_to_drt([sig_proj, sig_eq], degree)
    sig_proj_rt = fkt_drt[0]
    sig_eq_rt = fkt_drt[1]

    # calculate reconstructed flux (use default RT-space)
    x_sig_rt = sig_proj_rt.x.array[:] + sig_eq_rt.x.array[:]

    # extract relevant data
    domain = uh.function_space.mesh
    dofmap_sigrt = sig_proj_rt.function_space.dofmap.list
    dofs_per_fct = degree

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
    n_cells = domain.topology.index_map(2).size_local
    sign_detj = np.zeros(n_cells, dtype=np.float64)

    for c in range(0, n_cells):
        # evaluate detj
        J_q, detj = isoparametric_mapping_triangle(domain, dphi_geom, c)

        # determine sign of detj
        sign_detj[c] = np.sign(detj)

    # --- Check jump condition on all facets
    # create connectivity facet -> cell
    fct_to_cell = domain.topology.connectivity(1, 2)
    cell_to_fct = domain.topology.connectivity(2, 1)

    # check jump condition on all facets
    n_fcts = domain.topology.index_map(1).size_local

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
                dof_plus = dofmap_sigrt.links(cells[0])[if_plus * dofs_per_fct + i]
                dof_minus = dofmap_sigrt.links(cells[1])[if_minus * dofs_per_fct + i]

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


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
