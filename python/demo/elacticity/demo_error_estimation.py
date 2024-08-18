# Copyright (C) 2024 Maximilian Brodbeck
#
# This file is part of dolfinx_eqlb
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Demonstrate the a-posteriori error estimation for linear elasticity

Solve of the quasi-static linear elasticity equation

     div(sigma) = -f  with sigma = 2 * eps + pi_1 * div(u) * I,

with subsequent stress reconstruction on a series of uniformly refined meshes. 
Dirichlet boundary conditions are applied on the entire boundary. A convergence
study based on the manufactured solution

     u_ext = [sin(pi * x) * sin(pi * y), -sin(2*pi * x) * sin(2*pi * y)]

and 

     f = div(sigma(u_ext))
                
is performed. Alongside with the actual errors, the a-posteriori error 
estimate is evaluated and reported.
"""

import numpy as np
from mpi4py import MPI
import typing

import dolfinx
import dolfinx.fem as dfem
import ufl

from demo_reconstruction import (
    MeshType,
    create_unit_square_builtin,
    create_unit_square_gmesh,
    solve_primal_problem,
    equilibrate_flux,
    exact_solution,
)


# --- Error estimation ---
def estimate_error(
    pi_1: float,
    f: typing.Union[dfem.Function, typing.Any],
    uh: dfem.Function,
    sigma_proj: typing.Any,
    delta_sigma_eqlb: typing.Any,
    korns_constants: dfem.Function,
    guarantied_upper_bound: typing.Optional[bool] = True,
) -> typing.Tuple[float, typing.List[float]]:
    """Estimates the error of a linear elastic problem

    The estimate is derived based on the strategy in [1].

    [1] Bertrand, F. et al., https://doi.org/10.1002/num.22741, 2021

    Args:
        pi_1:             The ratio of lambda and mu
        f:                The body forces
        uh:               The displacement solution
        sigma_proj:       The projected stress tensor (UFL)
        delta_sigma_eqlb: The equilibrated stress tensor (UFL)

    Returns:
        The total error estimate,
        The error components
    """

    # Extract mesh
    domain = uh.function_space.mesh

    # Higher order volume integrator
    dvol = ufl.dx(degree=10)

    # Initialize storage of error
    V_e = dfem.FunctionSpace(domain, ufl.FiniteElement("DG", domain.ufl_cell(), 0))
    v = ufl.TestFunction(V_e)

    # Extract cell diameter
    h_cell = dfem.Function(V_e)
    num_cells = (
        domain.topology.index_map(2).size_local
        + domain.topology.index_map(2).num_ghosts
    )
    h = dolfinx.cpp.mesh.h(domain, 2, range(num_cells))
    h_cell.x.array[:] = h

    # Forms for error estimation
    a_delta_sigma = 0.5 * (
        delta_sigma_eqlb
        - (pi_1 / (2 + 2 * pi_1)) * ufl.tr(delta_sigma_eqlb) * ufl.Identity(2)
    )

    err_osc = (
        korns_constants
        * (h_cell / ufl.pi)
        * (f + ufl.div(sigma_proj + delta_sigma_eqlb))
    )

    err_wsym = 0.5 * korns_constants * (delta_sigma_eqlb[0, 1] - delta_sigma_eqlb[1, 0])

    form_eta_sig = dfem.form(ufl.inner(delta_sigma_eqlb, a_delta_sigma) * v * ufl.dx)
    form_eta_osc = dfem.form(ufl.inner(err_osc, err_osc) * v * dvol)
    form_eta_wsym = dfem.form(ufl.inner(err_wsym, err_wsym) * v * ufl.dx)

    # Assemble errors
    Leta_sig = dfem.petsc.create_vector(form_eta_sig)
    Leta_osc = dfem.petsc.create_vector(form_eta_osc)
    Leta_wsym = dfem.petsc.create_vector(form_eta_wsym)

    dfem.petsc.assemble_vector(Leta_sig, form_eta_sig)
    dfem.petsc.assemble_vector(Leta_osc, form_eta_osc)
    dfem.petsc.assemble_vector(Leta_wsym, form_eta_wsym)

    # Evaluate error norms
    if guarantied_upper_bound:
        error_estm = np.sqrt(
            np.sum(
                Leta_sig.array
                + Leta_osc.array
                + Leta_wsym.array
                + 2 * np.multiply(np.sqrt(Leta_osc.array), np.sqrt(Leta_wsym.array))
            )
        )
    else:
        error_estm = np.sqrt(np.sum(Leta_sig.array + Leta_osc.array))

    error_estm_sig = np.sqrt(np.sum(Leta_sig.array))
    error_estm_wsym = np.sqrt(np.sum(Leta_wsym.array))
    error_estm_osc = np.sqrt(np.sum(Leta_osc.array))

    return error_estm, [error_estm_sig, error_estm_wsym, error_estm_osc]


if __name__ == "__main__":
    # --- Input parameters ---
    # The mesh type
    mesh_type = MeshType.builtin

    # Material: pi_1 = lambda/mu
    pi_1 = 1.0

    # The orders of the FE spaces
    order_prime = 2
    order_eqlb = 2

    # Use guarantied upper bound
    guarantied_upper_bound = True

    # The mesh resolution
    sdisc_nelmt_init = 1
    convstudy_nref = 7

    # --- Convergence study ---
    error_norms = np.zeros((convstudy_nref, 12))

    for i in range(convstudy_nref):
        # --- Create mesh
        # Set mesh resolution
        sdisc_nelmt = sdisc_nelmt_init * 2**i

        # Create mesh
        if mesh_type == MeshType.builtin:
            domain, facet_tags, ds = create_unit_square_builtin(sdisc_nelmt)
        elif mesh_type == MeshType.gmsh:
            domain, facet_tags, ds = create_unit_square_gmesh(1 / sdisc_nelmt)
        else:
            raise ValueError("Unknown mesh type")

        # --- Solve problem
        # Solve primal problem
        degree_proj = 1 if (order_eqlb == 2) else None
        uh, stress_ref = solve_primal_problem(
            order_prime, domain, facet_tags, ds, pi_1, degree_proj, "cg"
        )

        # Solve equilibration
        stress_rw_proj, stress_rw_eqlb, korns_constants = equilibrate_flux(
            order_eqlb, domain, facet_tags, pi_1, uh, stress_ref, True, False
        )

        # ufl expression of the reconstructed flux
        stress_proj = -ufl.as_matrix(
            [
                [stress_rw_proj[0][0], stress_rw_proj[0][1]],
                [stress_rw_proj[1][0], stress_rw_proj[1][1]],
            ]
        )

        stress_eqlb = -ufl.as_matrix(
            [
                [stress_rw_eqlb[0][0], stress_rw_eqlb[0][1]],
                [stress_rw_eqlb[1][0], stress_rw_eqlb[1][1]],
            ]
        )

        sigma = stress_proj + stress_eqlb

        # --- Estimate error
        # RHS
        f = -ufl.div(stress_ref)

        errorestm, componetnts_estm = estimate_error(
            1.0,
            f,
            uh,
            stress_proj,
            stress_eqlb,
            korns_constants,
            guarantied_upper_bound,
        )

        # --- Compute real errors
        # Volume integrator
        dvol = ufl.dx(degree=10)

        # Energy norm of the displacement
        diff_u = uh - exact_solution(ufl.SpatialCoordinate(domain))
        err_ufl = (
            ufl.inner(ufl.grad(diff_u), ufl.grad(diff_u))
            + ufl.inner(ufl.div(diff_u), ufl.div(diff_u))
        ) * dvol

        err_uh1 = np.sqrt(
            domain.comm.allreduce(dfem.assemble_scalar(dfem.form(err_ufl)), op=MPI.SUM)
        )

        # H(div) error flux
        diff = ufl.div(sigma - stress_ref)
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
        error_norms[i, 7] = componetnts_estm[0]
        error_norms[i, 8] = componetnts_estm[1]
        error_norms[i, 9] = componetnts_estm[2]
        error_norms[i, 11] = errorestm / err_uh1

    # Calculate convergence rates
    error_norms[1:, 3] = np.log(error_norms[1:, 2] / error_norms[:-1, 2]) / np.log(
        error_norms[1:, 0] / error_norms[:-1, 0]
    )
    error_norms[1:, 5] = np.log(error_norms[1:, 4] / error_norms[:-1, 4]) / np.log(
        error_norms[1:, 0] / error_norms[:-1, 0]
    )
    error_norms[1:, 10] = np.log(error_norms[1:, 6] / error_norms[:-1, 6]) / np.log(
        error_norms[1:, 0] / error_norms[:-1, 0]
    )

    # Export results to csv
    outname = (
        "ConvStudyStressEqlb"
        + "_porder-"
        + str(order_prime)
        + "_eorder-"
        + str(order_eqlb)
        + ".csv"
    )

    header_protocol = (
        "h_min, n_elmt, err_u_h1, convrate_u_h1,"
        "err_sigma_hdiv, convrate_sigma_hdiv, "
        "errestm_u_h1, errestm_u_h1_sig, errestm_u_h1_asym, errestm_u_h1_osc, "
        "convrate_estmu_h1, I_eff"
    )

    np.savetxt(
        outname,
        error_norms,
        delimiter=",",
        header=header_protocol,
    )
