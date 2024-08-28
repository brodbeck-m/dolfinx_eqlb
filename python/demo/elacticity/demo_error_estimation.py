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

     u_ext = [ sin(pi * x) * cos(pi * y) + x²/(2*pi_1),
              -cos(pi * x) * sin(pi * y) + y²/(2*pi_1)]

and 

     f = div(sigma(u_ext))
                
is performed. Alongside with the actual errors, the a-posteriori error 
estimate is evaluated and reported.
"""

import numpy as np
from numpy.typing import NDArray
from mpi4py import MPI
import typing

import dolfinx
import dolfinx.fem as dfem
import ufl

from demo_reconstruction import (
    MeshType,
    DiscType,
    SolverType,
    create_unit_square_builtin,
    create_unit_square_gmsh,
    solve,
    equilibrate,
    exact_solution,
)


# --- Error estimation ---
def estimate(
    domain: dolfinx.mesh.Mesh,
    pi_1: float,
    f: typing.Union[dfem.Function, typing.Any],
    sdisc_type: DiscType,
    u_h: typing.List[dfem.Function],
    sigma_h: typing.Any,
    delta_sigmaR: typing.Any,
    korns_constants: dfem.Function,
    guarantied_upper_bound: typing.Optional[bool] = True,
) -> typing.Tuple[float, typing.List[float]]:
    """Estimates the error of a linear elastic problem

    The estimate is derived based on the strategy in [1] (displacement formulation)
    resp. [1] (displacement-pressure formulation).

    [1] Bertrand, F. et al., https://doi.org/10.1002/num.22741, 2021

    Args:
        domain:                 The mesh
        pi_1:                   The ratio of lambda and mu
        f:                      The exact body forces
        sigma_h:                The projected stress tensor (UFL)
        delta_sigmaR:           The equilibrated stress tensor (UFL)
        korns_constants:        The cells Korn's constants
        guarantied_upper_bound: True, if the error estimate is a guarantied upper bound

    Returns:
        The total error estimate,
        The error components
    """

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

    # The error estimate
    a_delta_sigma = 0.5 * (
        delta_sigmaR - (pi_1 / (2 + 2 * pi_1)) * ufl.tr(delta_sigmaR) * ufl.Identity(2)
    )

    err_osc = (
        korns_constants * (h_cell / ufl.pi) * (f + ufl.div(sigma_h + delta_sigmaR))
    )

    err_wsym = 0.5 * korns_constants * (delta_sigmaR[0, 1] - delta_sigmaR[1, 0])

    forms_eta = []
    forms_eta.append(dfem.form(ufl.inner(delta_sigmaR, a_delta_sigma) * v * ufl.dx))

    if sdisc_type == DiscType.displacement_pressure:
        # Error due to pressure approximation
        ca_squared = ((2 * pi_1) / (1 + pi_1)) * (
            1 + (pi_1 / (1 + pi_1)) * ((korns_constants * korns_constants) - 9)
        )
        err_div = ufl.div(u_h[0]) - (1 / pi_1) * u_h[1]

        forms_eta.append(dfem.form(ca_squared * err_div * err_div * v * ufl.dx))

    forms_eta.append(dfem.form(ufl.inner(err_wsym, err_wsym) * v * ufl.dx))
    forms_eta.append(dfem.form(ufl.inner(err_osc, err_osc) * v * dvol))

    # Assemble cell-wise errors
    Li_eta = []
    eta_i = []

    for form in forms_eta:
        Li_eta.append(dfem.petsc.create_vector(form))
        dfem.petsc.assemble_vector(Li_eta[-1], form)

        eta_i.append(np.sqrt(np.sum(Li_eta[-1].array)))

    # Evaluate error norms
    if sdisc_type == DiscType.displacement:
        eta = np.sum(Li_eta[0].array)
    elif sdisc_type == DiscType.displacement_pressure:
        eta = np.sum(Li_eta[0].array + Li_eta[1].array)

    if guarantied_upper_bound:
        eta += np.sum(
            Li_eta[-1].array
            + Li_eta[-2].array
            + 2 * np.multiply(np.sqrt(Li_eta[-1].array), np.sqrt(Li_eta[-2].array))
        )
    else:
        eta += np.sum(Li_eta[-1].array)

    return np.sqrt(eta), eta_i


# --- Postprocessing ---
def post_processing(
    pi_1: float,
    sdisc_type: DiscType,
    u_h: typing.List[dfem.Function],
    sigma_proj: typing.Any,
    delta_sigmaR: typing.Any,
    eta: float,
    eta_i: typing.List[float],
    ref_level: int,
    results: NDArray,
):
    """Postprocess the results

    Args:
        pi_1:           The ratio of lambda and mu
        sdisc_type:     The spatial discretisation
        u_h:            The approximated solution
        sigma_proj:     The projected stress tensor
        delta_sigmaR:   The equilibrated stress tensor
        eta:            The total error estimate
        eta_i:          The components of the error estimate
        ref_level:      The current refinement level
        results:        The results array
    """

    # The domain
    domain = u_h[0].function_space.mesh

    # The Volume integrator
    dvol = ufl.dx(degree=10)

    # The exact solution
    u_ext = exact_solution(ufl.SpatialCoordinate(domain), pi_1)
    sigma_ext = 2 * ufl.sym(ufl.grad(u_ext)) + pi_1 * ufl.div(u_ext) * ufl.Identity(2)

    # Energy norm of the displacement
    diff_u = u_h[0] - u_ext

    if sdisc_type == DiscType.displacement:
        err_ufl = (
            ufl.inner(ufl.sym(ufl.grad(diff_u)), ufl.sym(ufl.grad(diff_u)))
            + ufl.inner(ufl.div(diff_u), ufl.div(diff_u))
        ) * dvol
    elif sdisc_type == DiscType.displacement_pressure:
        diff_p = (u_h[1] / pi_1) - ufl.div(u_ext)

        err_ufl = (
            2 * ufl.inner(ufl.sym(ufl.grad(diff_u)), ufl.sym(ufl.grad(diff_u)))
            + ufl.inner(diff_p, diff_p)
        ) * dvol

    err = np.sqrt(
        domain.comm.allreduce(dfem.assemble_scalar(dfem.form(err_ufl)), op=MPI.SUM)
    )

    # H(div) error stress
    diff = ufl.div(sigma_proj + delta_sigmaR - sigma_ext)
    err_sighdiv = np.sqrt(
        domain.comm.allreduce(
            dfem.assemble_scalar(dfem.form(ufl.inner(diff, diff) * dvol)),
            op=MPI.SUM,
        )
    )

    # Store results
    results[ref_level, 0] = 1 / sdisc_nelmt
    results[ref_level, 1] = domain.topology.index_map(2).size_local
    results[ref_level, 2] = err
    results[ref_level, 4] = err_sighdiv
    results[ref_level, 6] = eta

    for i, val in enumerate(eta_i):
        results[ref_level, 7 + i] = val

    results[ref_level, -1] = eta / err


if __name__ == "__main__":
    # --- Input parameters ---
    # The mesh type
    mesh_type = MeshType.builtin

    # Material: pi_1 = lambda/mu
    pi_1 = 1.0

    # The spatial discretisation
    sdisc_type = DiscType.displacement
    order_prime = 2

    solver_type = SolverType.CG

    # The error estimate
    order_eqlb = 2
    guarantied_upper_bound = True

    # The mesh resolution
    sdisc_nelmt_init = 1
    convstudy_nref = 7

    # --- Convergence study ---
    if sdisc_type == DiscType.displacement:
        results = np.zeros((convstudy_nref, 13))
    elif sdisc_type == DiscType.displacement_pressure:
        results = np.zeros((convstudy_nref, 14))

    for i in range(convstudy_nref):
        # --- Create mesh
        # Set mesh resolution
        sdisc_nelmt = sdisc_nelmt_init * 2**i

        # Create mesh
        if mesh_type == MeshType.builtin:
            domain, facet_tags, ds = create_unit_square_builtin(sdisc_nelmt)
        elif mesh_type == MeshType.gmsh:
            domain, facet_tags, ds = create_unit_square_gmsh(1 / sdisc_nelmt)
        else:
            raise ValueError("Unknown mesh type")

        # --- Solve problem
        # Solve primal problem
        degree_proj = (
            1
            if (
                (sdisc_type == DiscType.displacement and order_prime == 2)
                or (sdisc_type == DiscType.displacement_pressure and order_prime == 1)
            )
            else None
        )

        f, stress_ref, u_h, stress_h = solve(
            domain, facet_tags, pi_1, sdisc_type, order_prime, degree_proj, solver_type
        )

        # Solve equilibration
        stress_proj, stress_eqlb, ckorn = equilibrate(
            domain, facet_tags, f, stress_h, order_eqlb, True, False
        )

        # --- Estimate error
        eta, eta_i = estimate(
            domain,
            pi_1,
            -ufl.div(stress_ref),
            sdisc_type,
            u_h,
            stress_proj,
            stress_eqlb,
            ckorn,
            guarantied_upper_bound,
        )

        # --- Postprocessing
        post_processing(
            pi_1, sdisc_type, u_h, stress_proj, stress_eqlb, eta, eta_i, i, results
        )

    # Calculate convergence rates
    results[1:, 3] = np.log(results[1:, 2] / results[:-1, 2]) / np.log(
        results[1:, 0] / results[:-1, 0]
    )
    results[1:, 5] = np.log(results[1:, 4] / results[:-1, 4]) / np.log(
        results[1:, 0] / results[:-1, 0]
    )
    results[1:, -3] = np.log(results[1:, 6] / results[:-1, 6]) / np.log(
        results[1:, 0] / results[:-1, 0]
    )
    results[1:, -2] = np.log(results[1:, 7] / results[:-1, 7]) / np.log(
        results[1:, 0] / results[:-1, 0]
    )

    # Export results to csv
    if sdisc_type == DiscType.displacement:
        outname = "ManSol-u_P-{}_RT-{}.csv".format(order_prime, order_eqlb)
        header_protocol = (
            "hmin, nelmt, err, rateerr, errsigmahdiv, ratesigmahdiv, "
            "eetot, eedsigR, eeosc, eeasym, rateetot, rateedsigR, ieff"
        )
    elif sdisc_type == DiscType.displacement_pressure:
        outname = "ManSol-up_TH-{}_RT-{}.csv".format(order_prime, order_eqlb)
        header_protocol = (
            "hmin, nelmt, err, rateerr, errsigmahdiv, ratesigmahdiv, "
            "eetot, eedsigR, eediv, eeasym, eeosc, rateetot, rateedsigR, ieff"
        )

    np.savetxt(outname, results, delimiter=",", header=header_protocol)
