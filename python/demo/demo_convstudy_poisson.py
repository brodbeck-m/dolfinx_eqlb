"""Convergence study for a Poisson problem

Implementation of a Poisson problem

                -div(grad(u)) = f ,

with subsequent flux reconstruction. To check the implementation 
a convergence study based on a manufactured solution

                u_ext = sin(2*pi * x) * cos(2*pi * y)

is performed. Inhomogeneous Dirichlet boundary conditions are 
applied on boundary surfaces [2,4].
"""

# --- Imports ---
import numpy as np
from mpi4py import MPI

import dolfinx
import dolfinx.fem as dfem
import ufl

from dolfinx_eqlb.eqlb import FluxEqlbEV, FluxEqlbSE

from demo_reconstruction_poisson import (
    create_unit_square_mesh,
    solve_primal_problem,
    equilibrate_flux,
)

# --- Input parameters ---
# The considered equilibration strategy
Equilibrator = FluxEqlbEV

# The orders of the FE spaces
elmt_order_prime = 1
elmt_order_eqlb = 2

# The mesh resolution
sdisc_nelmt_init = 1
convstudy_nref = 6


# --- Exact solution ---
def exact_solution(pkt):
    return lambda x: pkt.sin(2 * pkt.pi * x[0]) * pkt.cos(2 * pkt.pi * x[1])


# --- Error estimation ---
def estimate_error(rhs_prime, u_prime, sig_eqlb):
    # Extract mesh
    msh = u_prime.function_space.mesh

    # Initialize storage of error
    V_e = dfem.FunctionSpace(msh, ufl.FiniteElement("DG", msh.ufl_cell(), 0))
    v = ufl.TestFunction(V_e)

    # Extract cell diameter
    h_cell = dfem.Function(V_e)
    num_cells = (
        msh.topology.index_map(2).size_local + msh.topology.index_map(2).num_ghosts
    )
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

    dfem.petsc.assemble_vector(Leta_sig, form_eta_sig)
    dfem.petsc.assemble_vector(Leta_osc, form_eta_osc)

    # Evaluate error norms
    error_estm = np.sqrt(
        np.sum(
            Leta_sig.array
            + Leta_osc.array
            + 2 * np.multiply(np.sqrt(Leta_sig.array), np.sqrt(Leta_osc.array))
        )
    )
    error_estm_sig = np.sqrt(np.sum(Leta_sig.array))
    error_estm_osc = np.sqrt(np.sum(Leta_osc.array))

    return error_estm, error_estm_sig, error_estm_osc


# --- Convergence study ---
error_norms = np.zeros((convstudy_nref, 11))

for i in range(convstudy_nref):
    # --- Create mesh
    # Set mesh resolution
    sdisc_nelmt = sdisc_nelmt_init * 2**i

    # Create mesh
    domain, facet_tags, ds = create_unit_square_mesh(sdisc_nelmt)

    # --- Solve problem
    # Solve primal problem
    uh_prime = solve_primal_problem(elmt_order_prime, domain, facet_tags, ds)

    # Solve equilibration
    sigma_proj, sigma_eqlb = equilibrate_flux(
        Equilibrator, elmt_order_prime, elmt_order_eqlb, domain, facet_tags, uh_prime
    )

    # ufl expression of the reconstructed flux
    if Equilibrator == FluxEqlbEV:
        sigma = sigma_eqlb
    else:
        sigma = sigma_eqlb + sigma_proj

    # --- Estimate error
    # RHS
    f = -ufl.div(ufl.grad(exact_solution(ufl)(ufl.SpatialCoordinate(domain))))

    errorestm, errorestm_sig, errorestm_osc = estimate_error(f, uh_prime, sigma)

    # --- Compute real errors
    # Volume integrator
    dvol = ufl.dx(degree=10)

    u_ext = exact_solution(ufl)(ufl.SpatialCoordinate(domain))
    sigma_ext = -ufl.grad(u_ext)

    # H1 error displacement
    diff = ufl.grad(uh_prime - u_ext)
    err_uh1 = np.sqrt(
        domain.comm.allreduce(
            dfem.assemble_scalar(dfem.form(ufl.inner(diff, diff) * dvol)),
            op=MPI.SUM,
        )
    )

    # H(div) error flux
    diff = ufl.div(sigma - sigma_ext)
    err_sighdiv = np.sqrt(
        domain.comm.allreduce(
            dfem.assemble_scalar(dfem.form(diff * diff * dvol)),
            op=MPI.SUM,
        )
    )

    # Store results
    error_norms[i, 0] = 1 / sdisc_nelmt
    error_norms[i, 1] = domain.topology.index_map(2).size_local
    error_norms[i, 2] = err_uh1
    error_norms[i, 4] = err_sighdiv
    error_norms[i, 6] = errorestm
    error_norms[i, 7] = errorestm_sig
    error_norms[i, 8] = errorestm_osc
    error_norms[i, 10] = errorestm / err_uh1

# Calculate convergence rates
error_norms[1:, 3] = np.log(error_norms[1:, 2] / error_norms[:-1, 2]) / np.log(
    error_norms[1:, 0] / error_norms[:-1, 0]
)
error_norms[1:, 5] = np.log(error_norms[1:, 4] / error_norms[:-1, 4]) / np.log(
    error_norms[1:, 0] / error_norms[:-1, 0]
)
error_norms[1:, 9] = np.log(error_norms[1:, 6] / error_norms[:-1, 6]) / np.log(
    error_norms[1:, 0] / error_norms[:-1, 0]
)

# Export results to csv
if Equilibrator == FluxEqlbEV:
    eqlb_name = "EV"
else:
    eqlb_name = "SE"

outname = (
    "ConvStudyFluxEqlb-"
    + eqlb_name
    + "_porder-"
    + str(elmt_order_prime)
    + "_eorder-"
    + str(elmt_order_eqlb)
    + ".csv"
)

header_protocol = (
    "h_min, n_elmt, err_u_h1, convrate_u_h1,"
    "err_sigma_hdiv, convrate_sigma_hdiv, "
    "errestm_u_h1, errestm_u_h1_sig, errestm_u_h1_osc, "
    "convrate_estmu_h1, I_eff"
)

np.savetxt(
    outname,
    error_norms,
    delimiter=",",
    header=header_protocol,
)
