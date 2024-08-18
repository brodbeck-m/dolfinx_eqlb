# Copyright (C) 2024 Maximilian Brodbeck
#
# This file is part of dolfinx_eqlb
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Demonstrate the a-posteriori error estimation for the Poisson equation

Solve a Poisson problem

     -div(grad(u)) = f ,

with subsequent flux reconstruction on a series of uniformly refined meshes.
Thereby the following boundary conditions are supported:

     - dirichlet:     u = u_ext on boundary surfaces [1,2,3,4]
     - neumann_hom:   u = u_ext on boundary surfaces [1,3]
     - neumann_inhom: u = u_ext on boundary surfaces [2,4]  

A convergence study based on the manufactured solution

     u_ext = sin(2*pi * x) * cos(2*pi * y)

and 
     f = -div(grad(u_ext))

is performed. Alongside with the actual errors, the a-posteriori error 
estimate is evaluated and reported.
"""

import numpy as np
from mpi4py import MPI
import typing

import dolfinx
import dolfinx.fem as dfem
import ufl

from dolfinx_eqlb.eqlb import FluxEqlbEV, FluxEqlbSE

from demo_reconstruction import (
    MeshType,
    BCType,
    create_unit_square_builtin,
    create_unit_square_gmesh,
    solve_primal_problem,
    equilibrate_flux,
)


# --- Exact solution ---
def exact_solution(pkt):
    """Exact solution
    u_ext = sin(pi * x) * cos(pi * y)

    Args:
        pkt: The package

    Returns:
        The function handle oft the exact solution
    """

    return lambda x: pkt.sin(2 * pkt.pi * x[0]) * pkt.cos(2 * pkt.pi * x[1])


# --- Error estimation ---
def estimate_error(
    f: typing.Any,
    uh: dfem.Function,
    sigma_eqlb: dfem.Function,
) -> typing.Tuple[float, float, float]:
    """Estimates the error of a Poisson problem

    The estimate is calculated based on [1].

    [1] Ern, A and Vohralík, M, https://doi.org/10.1137/130950100, 2015

    Args:
        f:          The right-hand-side
        uh:         The solution
        sigma_eqlb: The equilibrated flux

    Returns:
        The total error estimate,
        The error from the flux difference,
        The error from data oscillation
    """

    # Extract mesh
    domain = uh.function_space.mesh

    # Check if equilibrated flux is discontinuous
    flux_is_discontinuous = (
        sigma_eqlb.function_space.element.basix_element.discontinuous
    )

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
    if flux_is_discontinuous:
        sigma = sigma_eqlb - ufl.grad(uh)
        err_sig = sigma_eqlb
    else:
        sigma = sigma_eqlb
        err_sig = ufl.grad(uh) + sigma_eqlb

    err_osc = (h_cell / ufl.pi) * (f - ufl.div(sigma))
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


if __name__ == "__main__":
    # --- Input parameters ---
    # The mesh type
    mesh_type = MeshType.builtin

    # The considered equilibration strategy
    Equilibrator = FluxEqlbEV

    # The orders of the FE spaces
    order_prime = 1
    order_eqlb = 1

    # The boundary conditions
    bc_type = BCType.dirichlet

    # The mesh resolution
    sdisc_nelmt_init = 1
    convstudy_nref = 6

    # --- Convergence study ---
    # Check input
    # TODO - Remove when EV is fixed
    if ((Equilibrator == FluxEqlbEV) and (mesh_type == MeshType.gmsh)) and (
        bc_type == BCType.neumann_inhom
    ):
        raise ValueError("EV with inhomogeneous flux BCs currently not working")

    # Perform study
    error_norms = np.zeros((convstudy_nref, 11))

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
        degree_proj = 0 if (order_eqlb == 1) else None
        uh_prime = solve_primal_problem(
            order_prime, domain, facet_tags, ds, bc_type, pdegree_rhs=degree_proj
        )
        # Solve equilibration
        sigma_proj, sigma_eqlb = equilibrate_flux(
            Equilibrator, order_eqlb, domain, facet_tags, bc_type, uh_prime, False
        )

        # ufl expression of the reconstructed flux
        if Equilibrator == FluxEqlbEV:
            sigma = sigma_eqlb
        else:
            sigma = sigma_eqlb + sigma_proj

        # --- Estimate error
        # RHS
        f = -ufl.div(ufl.grad(exact_solution(ufl)(ufl.SpatialCoordinate(domain))))

        errorestm, errorestm_sig, errorestm_osc = estimate_error(
            f, uh_prime, sigma_eqlb
        )

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
        + str(order_prime)
        + "_eorder-"
        + str(order_eqlb)
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
