"""
Demo for H(div) conforming equilibration of fluxes

Implementation of a H(div) conforming flux-equilibartion of a 
Poisson problem
                      -div(grad(u)) = f .

To verify the correctness of the proposed implementation, the
gained solution is compared to the exact solution u_ext. For the
right-hand-side f 
                   f(x,y) = -grad(u_ext)
is enforced.

Different problem setups:
1.) u_ext = sin(2*pi * x) * sin(2*pi * y), Dirichlet-BC on [1,2,3,4]
2.) u_ext = sin(2*pi * x) * sin(2*pi * y), Dirichlet-BC on [1,2]
3.) u_ext = sin(2*pi * x) * cos(2*pi * y), Dirichlet-BC on [1,2,3,4]
4.) u_ext = sin(2*pi * x) * cos(2*pi * y), Dirichlet-BC on [1,3]
5.) u_ext = sin(2*pi * x) * cos(2*pi * y), Dirichlet-BC on [2,4]
6.) u_ext = 0.25 + 0.25 * x^2 + 0.5 * y^2, Dirichlet-BC on [1,2,3,4]
7.) u_ext = 0.25 + 0.25 * x^2 + 0.5 * y^2, Dirichlet-BC on [2,4]
"""

import numpy as np
import math
from mpi4py import MPI
import petsc4py
from petsc4py import PETSc

import dolfinx
import dolfinx.fem as dfem
import dolfinx.mesh as dmesh
import ufl
import time

from dolfinx_eqlb import equilibration, lsolver

# --- Input parameters
# Spacial discretisation
sdisc_eorder = 1
sdisc_nelmt = 1

# Equilibration
eqlb_fluxorder = 1

# Type of manufactured solution
extsol_type = 1

# Linear algebra
lgs_solver = "mumps"

# Convergence study
convstudy_nref = 9
convstudy_reffct = 2

# Timing
timing_nretry = 3

# --- Manufactured solution ---
# Primal problem (trigonometric): Homogenous boundary conditions


def ext_sol_1(pkt):
    return lambda x: pkt.sin(2 * pkt.pi * x[0]) * pkt.sin(2 * pkt.pi * x[1])


def ext_flux_1(x):
    # Initialize flux
    sig = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)

    # Set flux
    sig[0] = -2 * np.pi * np.cos(2 * np.pi * x[0]) * np.sin(2 * np.pi * x[1])
    sig[1] = -2 * np.pi * np.sin(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1])

    return sig


# Primal problem (trigonometric): Inhomogenous boundary conditions


def ext_sol_2(pkt):
    return lambda x: pkt.sin(2 * pkt.pi * x[0]) * pkt.cos(2 * pkt.pi * x[1])


def ext_flux_2(x):
    # Initialize flux
    sig = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)

    # Set flux
    sig[0] = -2 * np.pi * np.cos(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1])
    sig[1] = 2 * np.pi * np.sin(2 * np.pi * x[0]) * np.sin(2 * np.pi * x[1])

    return sig


# Primal problem (polynomial): Inhomogenous boundary conditions
def ext_sol_3(pkt):
    return lambda x: 0.25 + 0.25 * (x[0] ** 2) + 0.5 * (x[1] ** 2)


def ext_flux_3(x):
    # Initialize flux
    sig = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)

    # Set flux
    sig[0] = -0.5 * x[0]
    sig[1] = -x[1]

    return sig


# --- Setup primal problem ---


def setup_poisson_primal(n_elmt, eorder, uext_ufl, boundid_prime_vn):
    # --- Create mesh ---
    msh = dmesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0, 0]), np.array([1, 1])],
        [n_elmt, n_elmt],
        cell_type=dmesh.CellType.triangle,
        diagonal=dmesh.DiagonalType.crossed,
    )
    boundaries = [
        (1, lambda x: np.isclose(x[0], 0)),
        (2, lambda x: np.isclose(x[1], 0)),
        (3, lambda x: np.isclose(x[0], 1)),
        (4, lambda x: np.isclose(x[1], 1)),
    ]

    facet_indices, facet_markers = [], []
    for marker, locator in boundaries:
        facets = dolfinx.mesh.locate_entities(msh, 1, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full(len(facets), marker))

    facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
    facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tag = dolfinx.mesh.meshtags(
        msh, 1, facet_indices[sorted_facets], facet_markers[sorted_facets]
    )

    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tag)

    # --- Function spaces ---
    V_u = dfem.FunctionSpace(msh, ufl.FiniteElement("CG", msh.ufl_cell(), eorder))

    # --- Set weak form
    # Trial and test function
    u = ufl.TrialFunction(V_u)
    v = ufl.TestFunction(V_u)

    # Set source term
    x = ufl.SpatialCoordinate(msh)
    f = -ufl.div(ufl.grad(uext_ufl(x)))

    # Equation system
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    l = f * v * ufl.dx

    # Add neumann BC
    if boundid_prime_vn:
        # Surface normal
        normal = ufl.FacetNormal(msh)

        # Add all boundary contributions
        for vn_id in boundid_prime_vn:
            l += ufl.inner(ufl.grad(uext_ufl(x)), normal) * v * ds(vn_id)

    return msh, facet_tag, V_u, dfem.form(a), dfem.form(l), f


def assemble_poisson_primal(
    facet_tag, V_u, a, l, uext_np, boundid_prime_dir, solver_type="cg"
):
    # --- Set boundary conditions ---
    uD = dfem.Function(V_u)
    uD.interpolate(uext_np)

    # Apply dirichlet conditions
    bc_esnt = []

    for dir_id in boundid_prime_dir:
        facets = facet_tag.indices[facet_tag.values == dir_id]
        dofs = dfem.locate_dofs_topological(V_u, 1, facets)
        bc_esnt.append(dfem.dirichletbc(uD, dofs))

    # --- Assemble linear system ---
    # Assemble stiffness matrix
    A = dfem.petsc.assemble_matrix(a, bcs=bc_esnt)
    A.assemble()

    # Assemble LHS
    L = dfem.petsc.create_vector(l)
    with L.localForm() as loc_L:
        loc_L.set(0)
    dfem.petsc.assemble_vector(L, l)

    # Boundary conditions
    dfem.apply_lifting(L, [a], [bc_esnt])
    L.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    dfem.set_bc(L, bc_esnt)

    # --- Initialize solver ---
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

        solver.setTolerances(rtol=1e-10, atol=1e-10, max_it=1000)
    elif solver_type == "superlu_dist":
        solver.setType(PETSc.KSP.Type.PREONLY)
        pc = solver.getPC()
        pc.setType(PETSc.PC.Type.LU)
        pc.setFactorSolverType("superlu_dist")

        solver.setTolerances(rtol=1e-10, atol=1e-10, max_it=1000)
    else:
        solver.setType(PETSc.KSP.Type.PREONLY)
        pc = solver.getPC()
        pc.setType(PETSc.PC.Type.LU)

        solver.setTolerances(rtol=1e-10, atol=1e-10, max_it=1000)

    u_prime = dfem.Function(V_u)

    return solver, L, u_prime


def solve_poisson_primal(solver, L, u_prime):
    solver(L, u_prime.vector)
    u_prime.x.scatter_forward()


def projection_primal(eorder, fluxorder, u_prime, rhs_prime, mult_rhs=False):
    # Create DG-space for projected flux
    msh = u_prime.function_space.mesh

    # Create function spaces
    V_flux_proj = dfem.FunctionSpace(msh, ufl.VectorElement("DG", msh.ufl_cell(), eorder - 1))
    V_rhs_proj = dfem.FunctionSpace(msh, ufl.FiniteElement("DG", msh.ufl_cell(), fluxorder - 1))

    if mult_rhs:
        # Projection
        list_proj = lsolver.local_projector(V_rhs_proj, [-u_prime.dx(0), -u_prime.dx(1), rhs_prime])

        # Assemble flux
        sig_proj = dfem.Function(V_flux_proj)
        sig_proj.x.array[0::2] = list_proj[0].x.array[:]
        sig_proj.x.array[1::2] = list_proj[1].x.array[:]

        return sig_proj, list_proj[2]
    else:
        # Create function spaces
        V_flux_proj = dfem.FunctionSpace(msh, ufl.VectorElement("DG", msh.ufl_cell(), eorder - 1))
        V_rhs_proj = dfem.FunctionSpace(msh, ufl.FiniteElement("DG", msh.ufl_cell(), fluxorder - 1))

        # Projection
        sig_proj = lsolver.local_projector(V_flux_proj, [-ufl.grad(u_prime)])
        rhs_proj = lsolver.local_projector(V_rhs_proj, [rhs_prime])

        return sig_proj[0], rhs_proj[0]


# --- Equilibration ---


def setup_equilibration(
    W, V_flux, facet_tag, sig_ext, boundid_prime_dir, boundid_prime_vn
):
    # Mark boundary facets
    fct_bcesnt_primal = np.array([], dtype=np.int32)
    for id_esnt in boundid_prime_dir:
        list_fcts = facet_tag.indices[facet_tag.values == id_esnt]
        fct_bcesnt_primal = np.concatenate((fct_bcesnt_primal, list_fcts))

    if fct_bcesnt_primal.size == 0:
        fct_bcesnt_primal = []

    fct_bcesnt_flux = np.array([], dtype=np.int32)
    for id_flux in boundid_prime_vn:
        list_fcts = facet_tag.indices[facet_tag.values == id_flux]
        fct_bcesnt_flux = np.concatenate((fct_bcesnt_flux, list_fcts))

    if fct_bcesnt_flux.size == 0:
        fct_bcesnt_flux = []

    # Set flux-boundaries
    bc_esnt_flux = []

    if boundid_prime_vn:
        # Interpolate exact flux
        vD = dfem.Function(V_flux)
        vD.interpolate(sig_ext)

        # Set boundary conditions
        for id_esnt in boundid_prime_vn:
            list_fcts = facet_tag.indices[facet_tag.values == id_esnt]
            dofs = dfem.locate_dofs_topological((W.sub(0), V_flux), 1, list_fcts)
            bc_esnt_flux.append(dfem.dirichletbc(vD, dofs, W.sub(0)))

    return fct_bcesnt_primal, fct_bcesnt_flux, bc_esnt_flux


# --- Error estimation ---


def estimate_error(rhs_prime, u_prime, sig_eqlb):
    # Extract mesh
    msh = u_prime.function_space.mesh

    # Initialize storage of error
    V_e = dfem.FunctionSpace(msh, ufl.FiniteElement("DG", msh.ufl_cell(), 0))
    v = ufl.TestFunction(V_e)

    # Extract cell diameter
    h_cell = dfem.Function(V_e)
    num_cells = msh.topology.index_map(2).size_local + msh.topology.index_map(2).num_ghosts
    h = dolfinx.cpp.mesh.h(msh, 2, range(num_cells))
    h_cell.x.array[:] = h

    # Forms for error estimation
    err_sig = ufl.grad(u_prime) + sig_eqlb
    err_osc = (h_cell / ufl.pi) * (rhs_prime - ufl.div(sig_eqlb))
    form_eta_sig = dfem.form(ufl.dot(err_sig, err_sig) * v * ufl.dx)
    form_eta_osc = dfem.form(ufl.dot(err_osc, err_osc) * v * ufl.dx)

    # Assemble errors
    Leta_sig = dfem.petsc.create_vector(form_eta_sig)
    Leta_osc = dfem.petsc.create_vector(form_eta_osc)
    with L.localForm() as loc_L:
        loc_L.set(0)
    dfem.petsc.assemble_vector(Leta_sig, form_eta_sig)
    dfem.petsc.assemble_vector(Leta_osc, form_eta_osc)

    return Leta_sig, Leta_osc

# --- Error norms ---
def error_L2(diff_u_uh, qorder=None):
    if qorder is None:
        dvol = ufl.dx
    else:
        dvol = ufl.dx(degree=qorder)
    return dfem.form(ufl.inner(diff_u_uh, diff_u_uh) * dvol)

def error_h1(diff_u_uh, qorder=None):
    if qorder is None:
        dvol = ufl.dx
    else:
        dvol = ufl.dx(degree=qorder)
    return dfem.form(ufl.inner(ufl.grad(diff_u_uh), ufl.grad(diff_u_uh)) * dvol)

def error_hdiv0(diff_u_uh, qorder=None):
    if qorder is None:
        dvol = ufl.dx
    else:
        dvol = ufl.dx(degree=qorder)
    return dfem.form(ufl.inner(ufl.div(diff_u_uh), ufl.div(diff_u_uh)) * dvol)

def calculate_error(uh, u_ex, form_error, degree_raise=2, use_ufl=False):
    # Create higher order function space
    degree = uh.function_space.ufl_element().degree() + degree_raise
    family = uh.function_space.ufl_element().family()
    mesh = uh.function_space.mesh

    # Initialise quadrature degree
    qdegree = None

    if use_ufl:
        # Set quadrature degree
        qdegree = 10

        # Get spacial coodinate and set error functional
        e_W = u_ex - uh
    else:
        if uh.function_space.num_sub_spaces > 1:
            elmt = ufl.VectorElement(family, mesh.ufl_cell(), degree)
        else:
            elmt = ufl.FiniteElement(family, mesh.ufl_cell(), degree)

        W = dfem.FunctionSpace(mesh, elmt)

        # Interpolate approximate solution
        u_W = dfem.Function(W)
        u_W.interpolate(uh)

        # Interpolate exact solution, special handling if exact solution
        # is a ufl expression or a python lambda function
        u_ex_W = dfem.Function(W)
        u_ex_W.interpolate(u_ex)

        # Compute the error in the higher order function space
        e_W = dfem.Function(W)
        e_W.x.array[:] = u_W.x.array - u_ex_W.x.array

    # Integrate the error
    error_local = dfem.assemble_scalar(form_error(e_W, qorder = qdegree))
    error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)


# --- Output ---


def init_protocol(i_conv, storage_protocol, nelmt, ndof_prime, ndof_eqlb):
    storage_protocol[i_conv, 0] = nelmt
    storage_protocol[i_conv, 3] = ndof_prime
    storage_protocol[i_conv, 4] = ndof_eqlb

def convergence_rates(i_conv, storage_protocol, uh, sig_proj, sig_eqlb, eta_h_sig, eta_h_osc, 
                      u_ext_np=None, u_ext_ufl=None, sig_ext_np=None, sig_ext_ufl=None):
    # Evaluate errors (with respect to exact solution)
    if u_ext_np is None:
        x = ufl.SpatialCoordinate(uh.function_space.mesh)
        error_u_i = calculate_error(uh, u_ext_ufl(x), error_L2, use_ufl=True)
        error_gu_i = calculate_error(uh, u_ext_ufl(x), error_h1, use_ufl=True)
    else:
        error_u_i = calculate_error(uh, u_ext_np, error_L2, use_ufl=False)
        error_gu_i = calculate_error(uh, u_ext_np, error_h1, use_ufl=False)

    if sig_ext_np is None:
        # Spacial coordinate
        x = ufl.SpatialCoordinate(uh.function_space.mesh)

        # Flux from function
        if sig_ext_ufl is None:
            sig_ext = -ufl.grad(u_ext_ufl(x))
        else:
            sig_ext = sig_ext_ufl(x)

        # Evaluate errors
        error_sigp_i = calculate_error(sig_proj, sig_ext, error_L2, use_ufl=True)
        error_sige_i = calculate_error(sig_eqlb, sig_ext, error_hdiv0, use_ufl=True)
    else:
        error_sigp_i = calculate_error(sig_proj, sig_ext_np, error_L2, use_ufl=False)
        error_sige_i = calculate_error(sig_eqlb, sig_ext_np, error_hdiv0, use_ufl=False)

    # Evaluate estimated error
    error_gu_estm_i = np.sqrt(np.sum(eta_h_sig + eta_h_osc + 2 * np.multiply(np.sqrt(eta_h_sig), np.sqrt(eta_h_osc))))
    error_gu_sig_estm_i = np.sqrt(np.sum(eta_h_sig))
    error_gu_osc_estm_i = np.sqrt(np.sum(eta_h_osc))

    # Convergece rate
    if i_conv == 0:
        rate_u = 0
        rate_sigp = 0
        rate_sige = 0
        rate_gu = 0
        rate_gu_estm = 0
        rate_gu_sig_estm = 0
        rate_gu_osc_estm = 0
    else:
        # Mesh length
        h_i = 1 / storage_protocol[i_conv, 0]
        h_im1 = 1 / storage_protocol[i_conv - 1, 0]

        # Previous errors
        error_u_im1 = storage_protocol[i_conv - 1, 14]
        error_sigp_im1 = storage_protocol[i_conv - 1, 24]
        error_sige_im1 = storage_protocol[i_conv - 1, 26]

        error_gu_im1 = storage_protocol[i_conv - 1, 16]
        error_gu_estm_im1 = storage_protocol[i_conv - 1, 18]
        error_gu_sig_estm_im1 = storage_protocol[i_conv - 1, 20]
        error_gu_osc_estm_im1 = storage_protocol[i_conv - 1, 22]

        # Convergence rates
        rate_u = np.log(error_u_i / error_u_im1) / np.log(h_i / h_im1)
        rate_sigp = np.log(error_sigp_i / error_sigp_im1) / np.log(h_i / h_im1)
        rate_sige = np.log(error_sige_i / error_sige_im1) / np.log(h_i / h_im1)

        rate_gu = np.log(error_gu_i / error_gu_im1) / np.log(h_i / h_im1)
        rate_gu_estm = np.log(error_gu_estm_i / error_gu_estm_im1) / np.log(h_i / h_im1)
        rate_gu_sig_estm = np.log(error_gu_sig_estm_i / error_gu_sig_estm_im1) / np.log(h_i / h_im1)
        rate_gu_osc_estm = np.log(error_gu_osc_estm_i / error_gu_osc_estm_im1) / np.log(h_i / h_im1)

    # Store results
    storage_protocol[i_conv, 14] = error_u_i
    storage_protocol[i_conv, 15] = rate_u
    storage_protocol[i_conv, 16] = error_gu_i
    storage_protocol[i_conv, 17] = rate_gu
    storage_protocol[i_conv, 18] = error_gu_estm_i
    storage_protocol[i_conv, 19] = rate_gu_estm
    storage_protocol[i_conv, 20] = error_gu_sig_estm_i
    storage_protocol[i_conv, 21] = rate_gu_sig_estm
    storage_protocol[i_conv, 22] = error_gu_osc_estm_i
    storage_protocol[i_conv, 23] = rate_gu_osc_estm
    storage_protocol[i_conv, 24] = error_sigp_i
    storage_protocol[i_conv, 25] = rate_sigp
    storage_protocol[i_conv, 26] = error_sige_i
    storage_protocol[i_conv, 27] = rate_sige

def document_calculation(i_conv, storage_protocol, timing_nretry, extime):
    # Set calculation times
    t_prime_total = (
        extime["prime_setup"] + extime["prime_assemble"] + extime["prime_solve"]
    )
    t_eqlb_total = extime["prime_project"] + extime["eqlb_setup"] + extime["eqlb_solve"]
    storage_protocol[i_conv, 5] += extime["prime_setup"] / timing_nretry
    storage_protocol[i_conv, 6] += extime["prime_assemble"] / timing_nretry
    storage_protocol[i_conv, 7] += extime["prime_solve"] / timing_nretry

    storage_protocol[i_conv, 8] += extime["prime_project"] / timing_nretry
    storage_protocol[i_conv, 9] += extime["eqlb_setup"] / timing_nretry
    storage_protocol[i_conv, 10] += extime["eqlb_solve"] / timing_nretry

    storage_protocol[i_conv, 11] += t_prime_total / timing_nretry
    storage_protocol[i_conv, 12] += t_eqlb_total / timing_nretry

    storage_protocol[i_conv, 13] += extime["eqlb_error"] / timing_nretry

    # Set time measures to zero
    for k in extime.keys():
        extime[k] = 0.0


# --- Execute calculation ---
# Initialize timing
extime = {
    "prime_setup": 0.0,
    "prime_assemble": 0.0,
    "prime_solve": 0.0,
    "prime_project": 0.0,
    "eqlb_setup": 0.0,
    "eqlb_solve": 0.0,
    "eqlb_error": 0.0
}

# Initialize exact solution
if extsol_type == 1:
    # Function handles for u_ext/sig_ext
    u_ext_np = ext_sol_1(np)
    u_ext_ufl = ext_sol_1(ufl)
    sig_ext = ext_flux_1

    # Set dirichlet-ids
    boundid_prime_dir = [1, 2, 3, 4]
    boundid_prime_vn = []
elif extsol_type == 2:
    # Function handles for u_ext/sig_ext
    u_ext_np = ext_sol_1(np)
    u_ext_ufl = ext_sol_1(ufl)
    sig_ext = ext_flux_1

    # Set dirichlet-ids
    boundid_prime_dir = [1, 2]
    boundid_prime_vn = [3, 4]
elif extsol_type == 3:
    # Function handles for u_ext/sig_ext
    u_ext_np = ext_sol_2(np)
    u_ext_ufl = ext_sol_2(ufl)
    sig_ext = ext_flux_2

    # Set dirichlet-ids
    boundid_prime_dir = [1, 2, 3, 4]
    boundid_prime_vn = []
elif extsol_type == 4:
    # Function handles for u_ext/sig_ext
    u_ext_np = ext_sol_2(np)
    u_ext_ufl = ext_sol_2(ufl)
    sig_ext = ext_flux_2

    # Set dirichlet-ids
    boundid_prime_dir = [1, 3]
    boundid_prime_vn = [2, 4]
elif extsol_type == 5:
    # Function handles for u_ext/sig_ext
    u_ext_np = ext_sol_2(np)
    u_ext_ufl = ext_sol_2(ufl)
    sig_ext = ext_flux_2

    # Set dirichlet-ids
    boundid_prime_dir = [2, 4]
    boundid_prime_vn = [1, 3]
elif extsol_type == 6:
    # Function handles for u_ext/sig_ext
    u_ext_np = ext_sol_3(np)
    u_ext_ufl = ext_sol_3(ufl)
    sig_ext = ext_flux_3

    # Set dirichlet-ids
    boundid_prime_dir = [1, 2, 3, 4]
    boundid_prime_vn = []
elif extsol_type == 7:
    # Function handles for u_ext/sig_ext
    u_ext_np = ext_sol_3(np)
    u_ext_ufl = ext_sol_3(ufl)
    sig_ext = ext_flux_3

    # Set dirichlet-ids
    boundid_prime_dir = [2, 4]
    boundid_prime_vn = [1, 3]
else:
    raise RuntimeError("No such solution option!")

# Initialize storage protocol
storage_protocol = np.zeros((convstudy_nref + 1, 28))
storage_protocol[:, 1] = sdisc_eorder
storage_protocol[:, 2] = eqlb_fluxorder

# Execute timing
for i_timing in range(0, timing_nretry):
    # Execute convergence study
    for i_conv in range(0, convstudy_nref + 1):
        # Get mesh resolution
        n_elmt = (convstudy_reffct**i_conv) * sdisc_nelmt

        # Initialize primal problem
        extime["prime_setup"] -= time.perf_counter()
        msh, facets, V_u, form_a, form_l, rhs_prime = setup_poisson_primal(
            n_elmt, sdisc_eorder, u_ext_ufl, boundid_prime_vn
        )
        extime["prime_setup"] += time.perf_counter()

        # Assemble LGS (primal problem)
        extime["prime_assemble"] -= time.perf_counter()
        solver, L, u_prime = assemble_poisson_primal(
            facets,
            V_u,
            form_a,
            form_l,
            u_ext_np,
            boundid_prime_dir,
            solver_type=lgs_solver,
        )
        extime["prime_assemble"] += time.perf_counter()

        # Solve primal problem
        extime["prime_solve"] -= time.perf_counter()
        solve_poisson_primal(solver, L, u_prime)
        extime["prime_solve"] += time.perf_counter()

        # Project fluxes into DG-space
        extime["prime_project"] -= time.perf_counter()
        sig_proj, rhs_proj = projection_primal(
            sdisc_eorder, eqlb_fluxorder, u_prime, rhs_prime, mult_rhs=False
        )
        extime["prime_project"] += time.perf_counter()

        # # --- Equilibrate flux
        extime["eqlb_setup"] -= time.perf_counter()
        # Initialize equilibrator
        equilibrator = equilibration.EquilibratorEV(
            eqlb_fluxorder, msh, [rhs_proj], [sig_proj]
        )

        # Identify boundaries and set essential flux-bcs
        fct_bcesnt_primal, fct_bcesnt_flux, bc_esnt_flux = setup_equilibration(
            equilibrator.V,
            equilibrator.V_flux,
            facets,
            sig_ext,
            boundid_prime_dir,
            boundid_prime_vn,
        )

        equilibrator.set_boundary_conditions(
            [fct_bcesnt_primal], [fct_bcesnt_flux], [bc_esnt_flux]
        )
        extime["eqlb_setup"] += time.perf_counter()

        # Solve equilibration
        extime["eqlb_solve"] -= time.perf_counter()
        equilibrator.equilibrate_fluxes()
        extime["eqlb_solve"] += time.perf_counter()

        # Estimate error
        # --- Evaluate error ---
        extime["eqlb_error"] -= time.perf_counter()
        error_sig, error_osc = estimate_error(rhs_prime, u_prime, equilibrator.list_flux[0])
        extime["eqlb_error"] += time.perf_counter()

        # Export solution to paraview (only in first repetition)
        if i_timing == 0:
            # Evaluate exakt flux
            vD = dfem.Function(equilibrator.V_flux)
            vD.interpolate(sig_ext)

            # Set function names
            u_prime.name = "u_prime"
            sig_proj.name = "sig_proj"
            equilibrator.list_flux[0].name = "sig_eqlb"
            vD.name = 'sig_ext'

            # Name output file
            outname = (
                "DemoEqlb-Poisson_degPrime-"
                + str(sdisc_eorder)
                + "_degFlux"
                + str(eqlb_fluxorder)
                + "_nelmt-"
                + str(n_elmt)
                + ".xdmf"
            )

            

            # Write to xdmf
            outfile = dolfinx.io.XDMFFile(MPI.COMM_WORLD, outname, "w")
            outfile.write_mesh(msh)
            outfile.write_function(u_prime, 1)
            outfile.write_function(sig_proj, 1)
            outfile.write_function(equilibrator.list_flux[0], 1)
            outfile.write_function(vD, 1)
            outfile.close()

        # Output to console
        time_primal = extime["prime_assemble"] + extime["prime_solve"]
        print(
            "n_elmt: {}, n_retry: {}, Prim.Sol.: {}, Eqlb.: {}".format(
                n_elmt, i_timing + 1, time_primal, extime["eqlb_solve"]
            )
        )

        # Store data to calculation protocol
        if i_timing == 0:
            # Initialization
            init_protocol(
                i_conv,
                storage_protocol,
                n_elmt,
                len(u_prime.x.array),
                len(equilibrator.list_flux[0].x.array),
            )

            # Convergence history
            convergence_rates(
                i_conv,
                storage_protocol,
                u_prime,
                sig_proj,
                equilibrator.list_flux[0],
                error_sig.array,
                error_osc.array,
                u_ext_ufl=u_ext_ufl
            )

        document_calculation(i_conv, storage_protocol, timing_nretry, extime)

# Export calculation protocol
header_protocol = "n_elmt, order_prime, order_flux, ndof_prime, ndof_eqlb, tp_setup, tp_assembly, tp_solve, te_project, te_setup, te_solve, tp_tot, te_tot, t_error, error_uh, rate_uh, error_grad-uh, rate_grad-uh, estmerr_grad-uh, rate_estm_grad-uh, estmerr_sig, rate_estm_sig, estmerr_oscl, rate_estm_oscl, error_sigp, rate_sigp, error_sige, rate_sige"
np.savetxt(
    "DemoEqlb-Poisson_ConvStudy.csv",
    storage_protocol,
    delimiter=",",
    header=header_protocol,
)
