# Copyright (C) 2024 Maximilian Brodbeck
#
# This file is part of dolfinx_eqlb
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Convergence study for linear elasticity

Solution of the quasi-static linear elasticity

        -div(2 * mhS * eps + lhS * div(u) * I) = f ,

with subsequent flux reconstruction. A convergence 
study based on a manufactured solution

        u_ext = [sin(2*pi * x) * cos(2*pi * y),
                -cos(2*pi * x) * sin(2*pi * y)]

is performed. Dirichlet boundary conditions are 
applied on boundary surfaces [1, 2, 3, 4].
"""

# --- Imports ---
import numpy as np
from mpi4py import MPI

import dolfinx.fem as dfem
import ufl

from demo_reconstruction_elasticity import (
    create_unit_square_mesh,
    solve_primal_problem,
    equilibrate_flux,
)

# --- Input parameters ---
# The orders of the FE spaces
elmt_order_prime = 2
elmt_order_eqlb = 2

# The mesh resolution
sdisc_nelmt_init = 1
convstudy_nref = 7


# --- Exact solution ---
def exact_solution(x):
    """Exact solution
    u_ext = [sin(2*pi * x) * cos(2*pi * y), -cos(2*pi * x) * sin(2*pi * y)]

    Args:
        x (ufl.SpatialCoordinate): The position x
    Returns:
        The exact function as ufl-expression
    """
    return ufl.as_vector(
        [
            ufl.sin(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1]),
            -ufl.cos(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1]),
        ]
    )


# --- Error estimation ---
def estimate_error(rhs_prime, u_prime, sig_eqlb):
    raise NotImplementedError("Error estimation not implemented")


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
    uh_prime, stress_ref = solve_primal_problem(
        elmt_order_prime, domain, facet_tags, ds, solver="cg"
    )

    # Solve equilibration
    stress_rw_proj, stress_rw_eqlb = equilibrate_flux(
        elmt_order_eqlb,
        domain,
        facet_tags,
        uh_prime,
        stress_ref,
        weak_symmetry=True,
        check_equilibration=False,
    )

    # ufl expression of the reconstructed flux
    stress_proj = ufl.as_matrix(
        [
            [stress_rw_proj[0][0], stress_rw_proj[0][1]],
            [stress_rw_proj[1][0], stress_rw_proj[1][1]],
        ]
    )

    stress_eqlb = ufl.as_matrix(
        [
            [stress_rw_eqlb[0][0], stress_rw_eqlb[0][1]],
            [stress_rw_eqlb[1][0], stress_rw_eqlb[1][1]],
        ]
    )

    sigma = stress_proj + stress_eqlb

    # --- Estimate error
    # RHS
    f = -ufl.div(ufl.grad(exact_solution(ufl.SpatialCoordinate(domain))))

    # errorestm, errorestm_sig, errorestm_osc = estimate_error(f, uh_prime, sigma_eqlb)
    errorestm = 0
    errorestm_sig = 0
    errorestm_osc = 0

    # --- Compute real errors
    # Volume integrator
    dvol = ufl.dx(degree=10)

    # H1 error displacement
    diff = ufl.grad(uh_prime - exact_solution(ufl.SpatialCoordinate(domain)))
    err_uh1 = np.sqrt(
        domain.comm.allreduce(
            dfem.assemble_scalar(dfem.form(ufl.inner(diff, diff) * dvol)),
            op=MPI.SUM,
        )
    )

    # H(div) error flux
    diff = ufl.div(sigma + stress_ref)
    err_sighdiv = np.sqrt(
        domain.comm.allreduce(
            dfem.assemble_scalar(dfem.form(ufl.inner(diff, diff) * dvol)),
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
# error_norms[1:, 9] = np.log(error_norms[1:, 6] / error_norms[:-1, 6]) / np.log(
#     error_norms[1:, 0] / error_norms[:-1, 0]
# )

# Export results to csv
outname = (
    "ConvStudyStressEqlb"
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
