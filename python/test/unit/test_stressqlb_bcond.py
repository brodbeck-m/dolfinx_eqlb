# --- Import ---
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import pytest
import typing

import dolfinx.io
import dolfinx.mesh as dmesh
import dolfinx.fem as dfem
import ufl

from dolfinx_eqlb.lsolver import local_projection
import dolfinx_eqlb.eqlb.check_eqlb_conditions as eqlb_checker

from utils import Geometry, create_unitsquare_builtin

from testcase_general import set_arbitrary_rhs
from testcase_elasticity import equilibrate_stresses

""" 
Check if equilibrated flux 
    a.) fulfills the divergence condition div(sigma_eqlb) = f 
    b.) is in the H(div) space
    c.) fulfills the flux boundary conditions strongly
"""


# --- Solver with general BCs ---
def solve_primal_problem_general_usquare(
    V_prime: dfem.FunctionSpace,
    geometry: Geometry,
    neunann_bcs: typing.List[typing.List[bool]],
    rhs: dfem.Function,
    degree_flux_bc: int,
    degree_projection: int,
) -> typing.Tuple[
    dfem.Function,
    typing.List[dfem.Function],
    typing.List[typing.List[int]],
    typing.List[typing.List[int]],
    typing.List[typing.List[dfem.Function]],
]:
    """Solves linear elasticity based on lagrangian finite elements

    Args:
        V_prime (dolfinx.FunctionSpace):      The function space of the primal problem
        geometry (Geometry):                  The geometry of the domain
        bc_id_neumann (List[List[bool]]):     List of lists (one for each facet) with
                                              boundary ids (true/false) for each spatial direction
        rhs (Any):                            The right-hand side of the primal problem
        degree_flux_bc (int):                 Degree of the stress boundary conditions
        degree_projection (int):              Degree of the projected stress

    Returns:
        u_prime (dolfinx.Function):                       The primal solution
        sig_proj (List[dolfinx.Function]):                List of projected stress rows
        boundary_id_dirichlet (List[List[int]]):          Lists of boundary ids for dirichlet BCs
                                                          (one per spatial dimension)
        boundary_id_neumann (List[List[int]]):            List of boundary ids for Neumann BCs
                                                          (one per spatial dimension)
        neumann_functions (List[List[dolfinx.Function]]): List of Neumann boundary conditions
                                                          (one per spatial dimension)
    """

    # The spatial dimension
    gdim = geometry.mesh.topology.dim

    # --- Generate boundary conditions
    # The boundary IDs
    boundary_id_dirichlet = [[] for _ in range(gdim)]
    boundary_id_neumann = [[] for _ in range(gdim)]

    # The neumann functions
    V_nbc = dfem.FunctionSpace(geometry.mesh, ("DG", degree_flux_bc))
    neumann_functions = [[] for _ in range(gdim)]

    for i, bc in enumerate(neunann_bcs):
        fctid = i + 1

        for j, has_neumann in enumerate(bc):
            if has_neumann:
                boundary_id_neumann[j].append(fctid)
                neumann_functions[j].append(dfem.Function(V_nbc))
                neumann_functions[j][-1].x.array[:] = 2 * (
                    np.random.rand(V_nbc.dofmap.index_map.size_local * V_nbc.dofmap.bs)
                    + 0.1
                )
            else:
                boundary_id_dirichlet[j].append(fctid)

    # --- The variational problem
    # Trial- and test-functions
    u = ufl.TrialFunction(V_prime)
    v = ufl.TestFunction(V_prime)

    # The stress tensor
    sigma = 2 * ufl.sym(ufl.grad(u)) + ufl.div(u) * ufl.Identity(gdim)

    # (Bi-)linear forms
    a_prime = ufl.inner(sigma, ufl.sym(ufl.grad(v))) * ufl.dx
    l_prime = ufl.inner(rhs, v) * ufl.dx

    # Natural BCs
    for i in range(gdim):
        for fctid, bc in zip(boundary_id_neumann[i], neumann_functions[i]):
            l_prime += bc * v[i] * geometry.ds(fctid)

    # Essential BCs
    list_essntbcs = []

    for i in range(gdim):
        fcts = geometry.facet_function.indices[
            np.isin(geometry.facet_function.values, boundary_id_dirichlet[i])
        ]
        dofs = dfem.locate_dofs_topological(V_prime.sub(i), 1, fcts)
        list_essntbcs.append(
            dfem.dirichletbc(PETSc.ScalarType(0), dofs, V_prime.sub(i))
        )

    # --- Solve the primal problem
    solveoptions = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ksp_rtol": 1e-12,
        "ksp_atol": 1e-12,
    }
    problem_prime = dfem.petsc.LinearProblem(
        a_prime, l_prime, list_essntbcs, petsc_options=solveoptions
    )
    u_prime = problem_prime.solve()

    # Project stress tensor
    sigma_h = -2 * ufl.sym(ufl.grad(u_prime)) - ufl.div(u_prime) * ufl.Identity(gdim)

    V_flux = dfem.VectorFunctionSpace(geometry.mesh, ("DG", degree_projection))
    sig_proj = local_projection(
        V_flux,
        [
            ufl.as_vector([sigma_h[0, 0], sigma_h[0, 1]]),
            ufl.as_vector([sigma_h[1, 0], sigma_h[1, 1]]),
        ],
    )

    return (
        u_prime,
        sig_proj,
        boundary_id_dirichlet,
        boundary_id_neumann,
        neumann_functions,
    )


# --- Test cases ---
@pytest.mark.parametrize("id_bc", list(range(1, 13)))
@pytest.mark.parametrize("degree", [2, 3, 4])
def test_boundary_conditions(degree, id_bc):
    # Expected fails for degree 2: BCs 8, 10 and 12
    # TODO - Extend patch grouping to handle these cases
    if id_bc == 1:
        neumann_bcs = [[True, False], [False, False], [False, False], [False, False]]
    elif id_bc == 2:
        neumann_bcs = [[False, True], [False, False], [False, False], [False, False]]
    elif id_bc == 3:
        neumann_bcs = [[False, False], [False, True], [False, False], [False, False]]
    elif id_bc == 4:
        neumann_bcs = [[False, False], [True, False], [False, False], [False, False]]
    elif id_bc == 5:
        neumann_bcs = [[True, False], [False, True], [False, False], [False, False]]
    elif id_bc == 6:
        neumann_bcs = [[True, False], [True, False], [False, False], [False, False]]
    elif id_bc == 7:
        neumann_bcs = [[False, True], [False, True], [False, False], [False, False]]
    elif id_bc == 8:
        neumann_bcs = [[False, True], [True, False], [False, False], [False, False]]
    elif id_bc == 9:
        neumann_bcs = [[True, False], [True, True], [False, False], [False, False]]
    elif id_bc == 10:
        neumann_bcs = [[False, True], [True, True], [False, False], [False, False]]
    elif id_bc == 11:
        neumann_bcs = [[True, True], [False, True], [False, False], [False, False]]
    elif id_bc == 12:
        neumann_bcs = [[True, True], [True, False], [False, False], [False, False]]
    else:
        raise ValueError("Unknown BCs")

    # --- Setup the testcase
    # Create mesh
    gdim = 2
    geometry = create_unitsquare_builtin(
        2, dmesh.CellType.triangle, dmesh.DiagonalType.crossed
    )

    # Set function space
    V_prime = dfem.VectorFunctionSpace(geometry.mesh, ("P", degree))

    # Determine degree of projected quantities (primal flux, RHS)
    degree_proj = degree - 1

    # Set RHS
    rhs, rhs_projected = set_arbitrary_rhs(
        geometry.mesh,
        degree_proj,
        degree_projection=degree_proj,
        vector_valued=True,
    )

    # The primal problem
    (
        u_prime,
        sigma_projected,
        boundary_id_dirichlet,
        boundary_id_neumann,
        neumann_functions,
    ) = solve_primal_problem_general_usquare(
        V_prime,
        geometry,
        neumann_bcs,
        rhs,
        degree - 1,
        degree_proj,
    )

    # --- Solve equilibration
    sigma_eq, boundary_dofvalues = equilibrate_stresses(
        degree,
        geometry,
        sigma_projected,
        [rhs_projected.sub(i).collapse() for i in range(gdim)],
        boundary_id_neumann,
        boundary_id_dirichlet,
        neumann_functions,
        [len(neumann_functions[i]) * [False] for i in range(gdim)],
    )

    # --- Check equilibration conditions
    # The boundary conditions
    for i in range(gdim):
        if len(boundary_id_neumann[i]) > 0:
            bcs_fulfilled = eqlb_checker.check_boundary_conditions(
                sigma_eq[i],
                sigma_projected[i],
                boundary_dofvalues[i],
                geometry.facet_function,
                boundary_id_neumann[i],
            )

            if not bcs_fulfilled:
                raise ValueError("Boundary conditions not fulfilled")

    # The divergence condition
    stress_eq = ufl.as_matrix(
        [[sigma_eq[0][0], sigma_eq[0][1]], [sigma_eq[1][0], sigma_eq[1][1]]]
    )
    stress_projected = ufl.as_matrix(
        [
            [sigma_projected[0][0], sigma_projected[0][1]],
            [sigma_projected[1][0], sigma_projected[1][1]],
        ]
    )

    div_condition_fulfilled = eqlb_checker.check_divergence_condition(
        stress_eq,
        stress_projected,
        rhs_projected,
        mesh=geometry.mesh,
        degree=degree,
        flux_is_dg=True,
    )

    if not div_condition_fulfilled:
        raise ValueError("Divergence conditions not fulfilled")

    # The jump condition
    for i in range(geometry.mesh.geometry.dim):
        jump_condition_fulfilled = eqlb_checker.check_jump_condition(
            sigma_eq[i], sigma_projected[i]
        )

        if not jump_condition_fulfilled:
            raise ValueError("Jump conditions not fulfilled")

    # The weak symmetry condition
    wsym_condition_fulfilled = eqlb_checker.check_weak_symmetry_condition(sigma_eq)

    if not wsym_condition_fulfilled:
        raise ValueError("Weak symmetry conditions not fulfilled")


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
