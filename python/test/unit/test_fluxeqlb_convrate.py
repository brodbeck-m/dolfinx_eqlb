# Copyright (C) 2024 Maximilian Brodbeck
#
# This file is part of dolfinx_eqlb
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Test the convergence rate of the equilibrated fluxes (Poisson problem)"""

from mpi4py import MPI
import numpy as np
import pytest
import typing

import dolfinx.mesh as dmesh
import dolfinx.fem as dfem
import ufl

from dolfinx_eqlb.eqlb import FluxEqlbEV, FluxEqlbSE

from utils import create_unitsquare_builtin
from testcase_general import BCType, set_manufactured_rhs, set_manufactured_bcs
from testcase_poisson import (
    exact_solution,
    exact_flux,
    solve_primal_problem,
    equilibrate_fluxes,
)


# --- Evaluate flux error in the H(div) norm ---
def flux_error(
    uh: typing.Any,
    u_ex: typing.Union[typing.Callable, typing.Any],
    degree_raise: typing.Optional[int] = 2,
    uex_is_ufl: typing.Optional[bool] = False,
):
    """Calculate convergence rate in H(div)
    Assumption: uh is constructed from a FE-space with block-size 1 and
    the FE-space can be interpolated by dolfinX.

    Args:
        uh:         Approximate solution (DOLFINx-function (if u_ex is callable) or ufl-expression)
        u_ex:       Exact solution (callable function for interpolation or ufl expr.)

    Returns:
        The global error measured in the given norm
    """
    # Initialise quadrature degree
    qdegree = None

    if not uex_is_ufl:
        # Get mesh
        mesh = uh.function_space.mesh

        # Create higher order function space
        degree = uh.function_space.ufl_element().degree() + degree_raise
        family = uh.function_space.ufl_element().family()
        mesh = uh.function_space.mesh

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
    else:
        # Get mesh
        try:
            mesh = uh.function_space.mesh
        except:
            mesh = uh.ufl_operands[0].function_space.mesh

        # Set quadrature degree
        qdegree = 10

        # Get spacial coordinate and set error functional
        e_W = u_ex - uh

    # Integrate the error
    dvol = ufl.dx(degree=qdegree)
    error_local = dfem.assemble_scalar(
        dfem.form(ufl.inner(ufl.div(e_W), ufl.div(e_W)) * dvol)
    )
    error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)


# --- The tests ---
@pytest.mark.parametrize("degree", [1, 2, 3, 4])
@pytest.mark.parametrize(
    "bc_type", [BCType.dirichlet, BCType.neumann_hom, BCType.neumann_inhom]
)
@pytest.mark.parametrize("equilibrator", [FluxEqlbEV, FluxEqlbSE])
def test_convrate(
    degree: int, bc_type: BCType, equilibrator: typing.Union[FluxEqlbEV, FluxEqlbSE]
):
    """Test the convergence rate of the equilibrated fluxes

    Solves a series of Poisson problems on uniformly refined meshes and
    compares the convergence rate in the H(div) norm to the theoretical
    prediction.

    Args:
        degree:       The degree of the equilibrated fluxes
        bc_type:      The type of boundary conditions
        equilibrator: The equilibrator
    """

    # Initialise boundary conditions
    if bc_type == BCType.dirichlet:
        boundary_id_dirichlet = [1, 2, 3, 4]
        boundary_id_neumann = []
    elif bc_type == BCType.neumann_hom:
        boundary_id_dirichlet = [1, 3]
        boundary_id_neumann = [2, 4]
    elif bc_type == BCType.neumann_inhom:
        boundary_id_dirichlet = [2, 4]
        boundary_id_neumann = [1, 3]
    else:
        raise ValueError("Unknown boundary condition type")

    # Parameters for convergence study
    convstudy_nelmt = 4
    convstudy_nref = 3
    convstudy_reffct = 2

    # Initialise data storage
    data_convstudy = np.zeros((convstudy_nref + 1, 2))

    for i in range(0, convstudy_nref + 1):
        # New mesh resolution
        n_elmt = convstudy_nelmt * (convstudy_reffct**i)

        # Create mesh
        geometry = create_unitsquare_builtin(
            n_elmt, dmesh.CellType.triangle, dmesh.DiagonalType.crossed
        )

        # Exact solution
        x = ufl.SpatialCoordinate(geometry.mesh)

        u_ext = exact_solution(x)
        flux_ext = exact_flux(x)

        # Set function space
        V_prime = dfem.FunctionSpace(geometry.mesh, ("P", degree))

        # Determine degree of projected quantities (primal flux, RHS)
        degree_proj = degree - 1

        # Set RHS
        rhs, rhs_projected = set_manufactured_rhs(flux_ext, geometry.mesh, degree_proj)

        # Set boundary conditions
        (
            dirichlet_functions,
            neumann_functions,
            neumann_projection,
        ) = set_manufactured_bcs(
            V_prime,
            boundary_id_dirichlet,
            boundary_id_neumann,
            u_ext,
            flux_ext,
            vector_valued=False,
        )

        # Solve equilibration
        u_prime, sigma_projected = solve_primal_problem(
            V_prime,
            geometry,
            boundary_id_neumann,
            boundary_id_dirichlet,
            rhs,
            neumann_functions,
            dirichlet_functions,
            degree_projection=degree_proj,
        )

        # Solve equilibration
        sigma_eq, _ = equilibrate_fluxes(
            equilibrator,
            degree,
            geometry,
            [sigma_projected],
            [rhs_projected],
            [boundary_id_neumann],
            [boundary_id_dirichlet],
            [neumann_functions],
            [neumann_projection],
        )

        # --- Compute convergence rate ---
        data_convstudy[i, 0] = 1 / n_elmt

        # Calculate erroru
        if equilibrator == FluxEqlbSE:
            data_convstudy[i, 1] = flux_error(
                sigma_eq[0] + sigma_projected, flux_ext, uex_is_ufl=True
            )
        else:
            data_convstudy[i, 1] = flux_error(sigma_eq[0], flux_ext, uex_is_ufl=True)

    # Calculate convergence rate
    rates = np.log(data_convstudy[1:, 1] / data_convstudy[:-1, 1]) / np.log(
        data_convstudy[1:, 0] / data_convstudy[:-1, 0]
    )

    assert (rates > degree - 0.1).all()


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
