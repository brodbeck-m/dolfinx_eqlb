# --- Imports ---
from enum import Enum
from mpi4py import MPI
from petsc4py import PETSc
import random

import dolfinx.fem as dfem
import dolfinx.mesh as dmesh
import ufl

from dolfinx_eqlb.cpp import local_solver_cholesky

"""Performance test for flux reconstruction

Physics:
    - Poisson
    - Linear elasticity
    - Biot equations (2-field)
    - Biot equations (3-field)

Specification of the testcases:
    - Unit square
    - Arbitrary RHS
    - Pure Dirichlet BCs (homogenous)
"""


# --- General ---
class SolverType(Enum):
    cg_amg = 0
    lu = 1
    mumps = 1
    superlu_dist = 2


def projection(a_proj, l_proj, res_proj, rhs_eqlb):
    # Solve projection
    res_proj_cpp = [f._cpp_object for f in res_proj]
    local_solver_cholesky(res_proj_cpp, a_proj, l_proj)

    # Map RHS into scalar DG spaces
    for i, rhs in enumerate(rhs_eqlb):
        if i < 2:
            rhs.x.array[:] = res_proj[0].sub(i).collapse().x.array[:]
        else:
            rhs.x.array[:] = res_proj[i - 1].sub(0).collapse().x.array[:]


def initialise_primal_solver(a_prime, l_prime, bcs_esnt, solver_type):
    # Assemble stiffness matrix
    A = dfem.petsc.assemble_matrix(a_prime, bcs=bcs_esnt)

    # Initialise RHS
    L = dfem.petsc.create_vector(l_prime)

    # Set solver
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setOperators(A)

    if solver_type == SolverType.cg_amg:
        solver.setType(PETSc.KSP.Type.CG)
        pc = solver.getPC()
        pc.setType(PETSc.PC.Type.HYPRE)
        pc.setHYPREType("boomeramg")

        solver.setTolerances(rtol=1e-10, atol=1e-10, max_it=1000)
    elif solver_type == SolverType.mumps:
        solver.setType(PETSc.KSP.Type.PREONLY)
        pc = solver.getPC()
        pc.setType(PETSc.PC.Type.LU)
        pc.setFactorSolverType("mumps")

        solver.setTolerances(rtol=1e-10, atol=1e-10)
    elif solver_type == SolverType.superlu_dist:
        solver.setType(PETSc.KSP.Type.PREONLY)
        pc = solver.getPC()
        pc.setType(PETSc.PC.Type.LU)
        pc.setFactorSolverType("superlu_dist")

        solver.setTolerances(rtol=1e-10, atol=1e-10)
    elif solver_type == SolverType.lu:
        solver.setType(PETSc.KSP.Type.PREONLY)
        pc = solver.getPC()
        pc.setType(PETSc.PC.Type.LU)

        solver.setTolerances(rtol=1e-10, atol=1e-10)
    else:
        raise ValueError("Unknown solver type")

    return A, L, solver


def assemble_primal_problem(a_prime, l_prime, bcs_esnt, solver_type):
    # Assemble stiffness matrix
    A = dfem.petsc.assemble_matrix(a_prime, bcs=bcs_esnt)
    A.assemble()

    # Initialise RHS
    L = dfem.petsc.create_vector(l_prime)

    # Assemble RHS
    dfem.petsc.assemble_vector(L, l_prime)

    # Apply boundary conditions
    dfem.apply_lifting(L, [a_prime], [bcs_esnt])
    L.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    dfem.set_bc(L, bcs_esnt)

    # Set solver
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setOperators(A)

    if solver_type == SolverType.cg_amg:
        solver.setType(PETSc.KSP.Type.CG)
        pc = solver.getPC()
        pc.setType(PETSc.PC.Type.HYPRE)
        pc.setHYPREType("boomeramg")

        solver.setTolerances(rtol=1e-10, atol=1e-10, max_it=1000)
    elif solver_type == SolverType.mumps:
        solver.setType(PETSc.KSP.Type.PREONLY)
        pc = solver.getPC()
        pc.setType(PETSc.PC.Type.LU)
        pc.setFactorSolverType("mumps")

        solver.setTolerances(rtol=1e-10, atol=1e-10)
    elif solver_type == SolverType.superlu_dist:
        solver.setType(PETSc.KSP.Type.PREONLY)
        pc = solver.getPC()
        pc.setType(PETSc.PC.Type.LU)
        pc.setFactorSolverType("superlu_dist")

        solver.setTolerances(rtol=1e-10, atol=1e-10)
    elif solver_type == SolverType.lu:
        solver.setType(PETSc.KSP.Type.PREONLY)
        pc = solver.getPC()
        pc.setType(PETSc.PC.Type.LU)

        solver.setTolerances(rtol=1e-10, atol=1e-10)
    else:
        raise ValueError("Unknown solver type")

    return solver, L


# --- Setup testcases ---
class Testcases(Enum):
    Poisson = 0
    Elasticity = 1
    Biot_up = 2
    Biot_upp = 3


def setup_testcase(
    testcase: Testcases, domain: dmesh.Mesh, order_prime: int, order_eqlb: int
):
    if testcase == Testcases.Poisson:
        return poisson_problem(domain, order_prime, order_eqlb)
    elif testcase == Testcases.Elasticity:
        return elasticity_problem(domain, order_prime, order_eqlb)
    elif testcase == Testcases.Biot_upp:
        return poroelasticity_problem_upp(domain, order_prime, order_eqlb)
    else:
        raise ValueError("Unknown testcase")


# --- Poisson
def poisson_problem(domain: dmesh.Mesh, order_prime: int, order_eqlb: int):
    # --- Primal problem
    V_prime = dfem.FunctionSpace(domain, ("CG", order_prime))
    uh = dfem.Function(V_prime)

    # Trial and test functions
    u = ufl.TrialFunction(V_prime)
    v = ufl.TestFunction(V_prime)

    # The spacial coordinates
    x = ufl.SpatialCoordinate(domain)

    # The RHS
    r1 = random.uniform(-1, 1)
    r2 = random.uniform(-2, 2)
    f = r1 * ufl.sin(r1 * ufl.pi * x[0]) * r2 * ufl.cos(r1 * ufl.pi * x[1])

    # Set equation system
    a_prime = dfem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    l_prime = dfem.form(ufl.inner(f, v) * ufl.dx)

    # Dirichlet BCs
    uD = dfem.Function(V_prime)

    domain.topology.create_connectivity(1, 2)
    bfcts = dmesh.exterior_facet_indices(domain.topology)
    boundary_dofs = dfem.locate_dofs_topological(V_prime, 1, bfcts)
    bcs_esnt = [dfem.dirichletbc(uD, boundary_dofs)]

    # --- Projection
    V_proj = dfem.VectorFunctionSpace(domain, ("DG", order_eqlb - 1))

    # Trial and test functions
    u = ufl.TrialFunction(V_proj)
    v = ufl.TestFunction(V_proj)

    # The equation system
    a_proj = dfem.form(ufl.inner(u, v) * ufl.dx)
    l_proj = [
        dfem.form(ufl.inner(ufl.as_vector([f, 0]), v) * ufl.dx),
        dfem.form(ufl.inner(-ufl.grad(uh), v) * ufl.dx),
    ]

    # Initialise storage projected quantities
    res_proj = [dfem.Function(V_proj) for _ in range(len(l_proj))]

    # Initialise storage RHS for equilibration
    V_proj_sub = dfem.FunctionSpace(domain, ("DG", order_eqlb - 1))
    rhs_eqlb = [dfem.Function(V_proj_sub) for _ in range(len(l_proj) - 1)]

    return a_prime, l_prime, uh, a_proj, l_proj, res_proj, rhs_eqlb, bfcts, bcs_esnt


# --- Linear elasticity
def elasticity_problem(domain: dmesh.Mesh, order_prime: int, order_eqlb: int):
    # Check input
    if order_prime < 2 or order_eqlb < 2:
        raise RuntimeError("Incompatible order for stress equilibration!")

    # --- Primal problem
    V_prime = dfem.VectorFunctionSpace(domain, ("CG", order_prime))
    uh = dfem.Function(V_prime)

    # Trial and test functions
    u = ufl.TrialFunction(V_prime)
    v = ufl.TestFunction(V_prime)

    # The spacial coordinates
    x = ufl.SpatialCoordinate(domain)

    # The RHS
    r1 = random.uniform(-1, 1)
    r2 = random.uniform(-2, 1.5)

    f_1 = r1 * ufl.sin(r2 * ufl.pi * x[0]) * r2 * ufl.cos(r1 * ufl.pi * x[1])
    f_2 = r1 * ufl.cos(r2 * ufl.pi * x[0]) * r2 * ufl.sin(r1 * ufl.pi * x[1])

    # Set equation system
    sigma = 2 * ufl.sym(ufl.grad(u)) + ufl.div(u) * ufl.Identity(2)

    a_prime = dfem.form(ufl.inner(sigma, ufl.sym(ufl.grad(v))) * ufl.dx)
    l_prime = dfem.form(ufl.inner(ufl.as_vector([f_1, f_2]), v) * ufl.dx)

    # Dirichlet BCs
    uD = dfem.Function(V_prime)

    domain.topology.create_connectivity(1, 2)
    bfcts = dmesh.exterior_facet_indices(domain.topology)
    boundary_dofs = dfem.locate_dofs_topological(V_prime, 1, bfcts)
    bcs_esnt = [dfem.dirichletbc(uD, boundary_dofs)]

    # --- Projection
    V_proj = dfem.VectorFunctionSpace(domain, ("DG", order_eqlb - 1))

    # Trial and test functions
    u = ufl.TrialFunction(V_proj)
    v = ufl.TestFunction(V_proj)

    # The approximated stress tensor
    sigma_h = -2 * ufl.sym(ufl.grad(uh)) - ufl.div(uh) * ufl.Identity(2)

    # The equation system
    a_proj = dfem.form(ufl.inner(u, v) * ufl.dx)
    l_proj = [
        dfem.form(ufl.inner(ufl.as_vector([f_1, f_2]), v) * ufl.dx),
        dfem.form(ufl.inner(ufl.as_vector([sigma_h[0, 0], sigma_h[0, 1]]), v) * ufl.dx),
        dfem.form(ufl.inner(ufl.as_vector([sigma_h[1, 0], sigma_h[1, 1]]), v) * ufl.dx),
    ]

    # Initialise storage projected quantities
    res_proj = [dfem.Function(V_proj) for _ in range(len(l_proj))]

    # Initialise storage RHS for equilibration
    V_proj_sub = dfem.FunctionSpace(domain, ("DG", order_eqlb - 1))
    rhs_eqlb = [dfem.Function(V_proj_sub) for _ in range(2)]

    return a_prime, l_prime, uh, a_proj, l_proj, res_proj, rhs_eqlb, bfcts, bcs_esnt


# --- Poro-elasticity
def poroelasticity_problem_upp(domain: dmesh.Mesh, order_prime: int, order_eqlb: int):
    # Check input
    if order_prime < 2 or order_eqlb < 2:
        raise RuntimeError("Incompatible order for stress equilibration!")

    # --- Primal problem
    Pu = ufl.VectorElement("CG", domain.ufl_cell(), order_prime)
    Pp = ufl.FiniteElement("CG", domain.ufl_cell(), order_prime)
    Ppt = ufl.FiniteElement("CG", domain.ufl_cell(), order_prime - 1)

    V_prime = dfem.FunctionSpace(domain, ufl.MixedElement([Pu, Pp, Ppt]))
    uh = dfem.Function(V_prime)

    # Trial and test functions
    u, p, pt = ufl.TrialFunctions(V_prime)
    v_u, v_p, v_pt = ufl.TestFunctions(V_prime)

    # The spacial coordinates
    x = ufl.SpatialCoordinate(domain)

    # The RHS
    r1 = random.uniform(-1, 1)
    r2 = random.uniform(-2, 1.5)

    f_1 = r1 * ufl.sin(r2 * ufl.pi * x[0]) * r2 * ufl.cos(r1 * ufl.pi * x[1])
    f_2 = r1 * ufl.cos(r2 * ufl.pi * x[0]) * r2 * ufl.sin(r1 * ufl.pi * x[1])
    g = r2 * ufl.sin(r1 * ufl.pi * x[0]) * r2 * ufl.sin(r1 * ufl.pi * x[1])

    # Set equation system
    sigma = 2 * ufl.sym(ufl.grad(u)) - pt * ufl.Identity(2)

    a_prime = dfem.form(
        ufl.inner(sigma, ufl.sym(ufl.grad(v_u))) * ufl.dx
        + ufl.inner(ufl.div(u) + pt - p, v_pt) * ufl.dx
        + ((p - pt) * v_p + ufl.inner(ufl.grad(p), ufl.grad(v_p))) * ufl.dx
    )
    l_prime = dfem.form(
        ufl.inner(ufl.as_vector([f_1, f_2]), v_u) * ufl.dx + ufl.inner(g, v_p) * ufl.dx
    )

    # Dirichlet BCs
    Vu, _ = V_prime.sub(0).collapse()
    Vp, _ = V_prime.sub(1).collapse()

    uD = dfem.Function(Vu)
    pD = dfem.Function(Vp)

    domain.topology.create_connectivity(1, 2)
    bfcts = dmesh.exterior_facet_indices(domain.topology)

    boundary_dofs = dfem.locate_dofs_topological((V_prime.sub(0), Vu), 1, bfcts)
    bcs_esnt = [dfem.dirichletbc(uD, boundary_dofs, V_prime.sub(0))]

    boundary_dofs = dfem.locate_dofs_topological((V_prime.sub(1), Vp), 1, bfcts)
    bcs_esnt.append(dfem.dirichletbc(pD, boundary_dofs, V_prime.sub(1)))

    # --- Projection
    V_proj = dfem.VectorFunctionSpace(domain, ("DG", order_eqlb - 1))

    # Trial and test functions
    u = ufl.TrialFunction(V_proj)
    v = ufl.TestFunction(V_proj)

    # The approximated stress tensor
    uh_u = uh.sub(0).collapse()
    uh_p = uh.sub(1).collapse()
    uh_pt = uh.sub(2).collapse()

    sigma_h = -2 * ufl.sym(ufl.grad(uh_u)) + (uh_pt - uh_p) * ufl.Identity(2)

    # The equation system
    a_proj = dfem.form(ufl.inner(u, v) * ufl.dx)
    l_proj = [
        dfem.form(ufl.inner(ufl.as_vector([f_1, f_2]) - ufl.grad(uh_p), v) * ufl.dx),
        dfem.form(ufl.inner(ufl.as_vector([g + uh_pt - uh_p, 1]), v) * ufl.dx),
        dfem.form(ufl.inner(ufl.as_vector([sigma_h[0, 0], sigma_h[0, 1]]), v) * ufl.dx),
        dfem.form(ufl.inner(ufl.as_vector([sigma_h[1, 0], sigma_h[1, 1]]), v) * ufl.dx),
        dfem.form(ufl.inner(-ufl.grad(uh_p), v) * ufl.dx),
    ]

    # Initialise storage projected quantities
    res_proj = [dfem.Function(V_proj) for _ in range(len(l_proj))]

    # Initialise storage RHS for equilibration
    V_proj_sub = dfem.FunctionSpace(domain, ("DG", order_eqlb - 1))
    rhs_eqlb = [dfem.Function(V_proj_sub) for _ in range(3)]

    return a_prime, l_prime, uh, a_proj, l_proj, res_proj, rhs_eqlb, bfcts, bcs_esnt
