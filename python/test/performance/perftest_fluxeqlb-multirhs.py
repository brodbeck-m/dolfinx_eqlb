"""Performance test for a given flux equilibration considering multiple RHS

Testcase:
    - Unit square
    - Arbitrary RHS (polynomial, no projection required)
    - Pure Dirichlet BCs (homogenous)
"""

# --- Imports ---
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import random
import time
import typing

import dolfinx.fem as dfem
import dolfinx.mesh as dmesh
import ufl

from dolfinx_eqlb.eqlb import fluxbc, FluxEqlbEV, FluxEqlbSE
from dolfinx_eqlb.lsolver import local_projection

# --- Parameters ---
# Number equilibrated RHS
n_rhs = 3

# The orders of the FE spaces
elmt_order_prime = 1
elmt_order_eqlb = 2

# Timing options
n_refinements = 9
n_repeats = 3


# --- The primal problem ---
def setup_primal_problem(n_elmt: int, elmt_order_prime: int):
    # Unit square
    domain = dmesh.create_unit_square(
        MPI.COMM_WORLD,
        n_elmt,
        n_elmt,
        cell_type=dmesh.CellType.triangle,
        diagonal=dmesh.DiagonalType.crossed,
    )

    # Function space
    V_prime = dfem.FunctionSpace(domain, ("CG", elmt_order_prime))

    # Trial and test functions
    u = ufl.TrialFunction(V_prime)
    v = ufl.TestFunction(V_prime)

    # The spacial coordinates
    x = ufl.SpatialCoordinate(domain)

    # The RHS
    f = (
        random.uniform(-1, 1)
        * ufl.sin(random.uniform(-1, 1) * ufl.pi * x[0])
        * random.uniform(-2, 2)
        * ufl.cos(random.uniform(-1, 1) * ufl.pi * x[1])
    )

    # Set equation system
    a = dfem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    l = dfem.form(ufl.inner(f, v) * ufl.dx)

    # Dirichlet BCs
    uD = dfem.Function(V_prime)

    domain.topology.create_connectivity(1, 2)
    boundary_facets = dmesh.exterior_facet_indices(domain.topology)
    boundary_dofs = dfem.locate_dofs_topological(V_prime, 1, boundary_facets)
    bcs_esnt = dfem.dirichletbc(uD, boundary_dofs)

    return dfem.Function(V_prime), a, l, f, boundary_facets, bcs_esnt


def assemble_eqs_primal(
    a: dfem.FormMetaClass,
    l: dfem.FormMetaClass,
    bc_esnt: dfem.DirichletBCMetaClass,
    solver_type: str = "cg",
):
    # Assemble stiffness matrix
    A = dfem.petsc.assemble_matrix(a, bcs=[bc_esnt])
    A.assemble()

    # Initialise RHS
    L = dfem.petsc.create_vector(l)

    with L.localForm() as loc_L:
        loc_L.set(0)

    # Assemble RHS
    dfem.petsc.assemble_vector(L, l)

    # Apply boundary conditions
    dfem.apply_lifting(L, [a], [[bc_esnt]])
    L.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    dfem.set_bc(L, [bc_esnt])

    # Set solver
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setOperators(A)

    if solver_type == "cg":
        solver.setType(PETSc.KSP.Type.CG)
        pc = solver.getPC()
        pc.setType(PETSc.PC.Type.HYPRE)
        pc.setHYPREType("boomeramg")

        solver.setTolerances(rtol=1e-10, atol=1e-10, max_it=1000)
    elif solver_type == "mumps":
        solver.setType(PETSc.KSP.Type.PREONLY)
        pc = solver.getPC()
        pc.setType(PETSc.PC.Type.LU)
        pc.setFactorSolverType("mumps")

        solver.setTolerances(rtol=1e-10, atol=1e-10)
    elif solver_type == "superlu_dist":
        solver.setType(PETSc.KSP.Type.PREONLY)
        pc = solver.getPC()
        pc.setType(PETSc.PC.Type.LU)
        pc.setFactorSolverType("superlu_dist")

        solver.setTolerances(rtol=1e-10, atol=1e-10)
    elif solver_type == "lu":
        solver.setType(PETSc.KSP.Type.PREONLY)
        pc = solver.getPC()
        pc.setType(PETSc.PC.Type.LU)

        solver.setTolerances(rtol=1e-10, atol=1e-10)
    else:
        raise ValueError("Unknown solver type")

    return solver, L


# --- The equilibration ---
def perform_projections(
    elmt_order_eqlb: int,
    n_rhs: int,
    f_prime: typing.Any,
    uh_prime: dfem.Function,
):
    # Extract mesh
    domain = uh_prime.function_space.mesh

    # Project RHS
    V_flux_proj = dfem.VectorFunctionSpace(domain, ("DG", elmt_order_eqlb - 1))
    sigma_proj = local_projection(
        V_flux_proj, [-ufl.grad(uh_prime) for _ in range(n_rhs)]
    )

    # Project fluxes
    V_rhs_proj = dfem.FunctionSpace(domain, ("DG", elmt_order_eqlb - 1))
    rhs_proj = local_projection(V_rhs_proj, [f_prime for _ in range(n_rhs)])

    return sigma_proj, rhs_proj


def setup_equilibration(
    elmt_order_eqlb: int,
    rhs_proj: typing.List[dfem.Function],
    sigma_proj: typing.List[dfem.Function],
    dirbc_fcts: np.ndarray,
    Equilibrator: typing.Union[FluxEqlbEV, FluxEqlbSE],
):
    # Number of RHS
    n_rhs = len(rhs_proj)

    # Extract the mesh
    domain = rhs_proj[0].function_space.mesh

    # Initialise equilibrator
    equilibrator = Equilibrator(elmt_order_eqlb, domain, rhs_proj, sigma_proj)

    # Specify boundary conditions --> pure primal dirichlet
    equilibrator.set_boundary_conditions(
        [dirbc_fcts] * n_rhs, [[] for _ in range(n_rhs)]
    )

    return equilibrator


# --- Performance testing ---
timings = np.zeros((n_refinements, 14))

for r in range(n_repeats):
    for i in range(n_refinements):
        # Number of elements
        n_elmt = 2**i

        # --- Primal problem
        # Setup
        uh_prime, a, l, f, boundary_facets, bcs_esnt = setup_primal_problem(
            n_elmt, elmt_order_prime
        )

        timings[i, 0] = 1 / n_elmt
        timings[i, 1] = uh_prime.function_space.mesh.topology.index_map(2).size_local

        # Assembly
        timings[i, 2] -= time.perf_counter()
        solver, L = assemble_eqs_primal(a, l, bcs_esnt, solver_type="cg")
        timings[i, 2] += time.perf_counter()

        # Solution
        timings[i, 3] -= time.perf_counter()
        solver(L, uh_prime.vector)
        uh_prime.x.scatter_forward()
        timings[i, 3] += time.perf_counter()

        # --- Equilibration
        for j, n_rhs_i in enumerate([1, n_rhs]):
            # Projections
            timings[i, 4 + j] -= time.perf_counter()
            sigma_proj, rhs_proj = perform_projections(
                elmt_order_eqlb, n_rhs_i, f, uh_prime
            )
            timings[i, 4 + j] += time.perf_counter()

            for k, Equilibrator in enumerate([FluxEqlbEV, FluxEqlbSE]):
                # Setup
                equilibrator = setup_equilibration(
                    elmt_order_eqlb,
                    rhs_proj,
                    sigma_proj,
                    boundary_facets,
                    Equilibrator,
                )

                # Solution
                timings[i, 6 + 2 * k + j] -= time.perf_counter()
                equilibrator.equilibrate_fluxes()
                timings[i, 6 + 2 * k + j] += time.perf_counter()

        # Control output
        print("n_elmt: {}, n_retry: {}".format(n_elmt, r + 1))

# Post-process timings
timings[:, 2:10] /= n_repeats
timings[:, 10] = timings[:, 7] / timings[:, 6]
timings[:, 11] = timings[:, 9] / timings[:, 8]
timings[:, 12] = timings[:, 6] / timings[:, 8]
timings[:, 13] = timings[:, 7] / timings[:, 9]

# Export Results
outname = (
    "TimingFluxEqlb_porder-"
    + str(elmt_order_prime)
    + "_eorder-"
    + str(elmt_order_eqlb)
    + "_nrhs-"
    + str(n_rhs)
    + ".csv"
)


header_protocol = (
    "h_min, n_elmt, assembly_prime, solve_prime, "
    "projection_1, projection_n, eqlb_EV_1, eqlb_EV_n, eqlb_SE_1, eqlb_SE_n,"
    "n_to_1_EV, n_to_1_SE, EV_to_SE_1, EV_to_SE_n"
)

np.savetxt(
    outname,
    timings,
    delimiter=",",
    header=header_protocol,
)
