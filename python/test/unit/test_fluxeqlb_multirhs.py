import numpy as np
from petsc4py import PETSc
import pytest
from typing import Any, List

import basix
import dolfinx.mesh as dmesh
import dolfinx.fem as dfem
import ufl

from dolfinx_eqlb.eqlb import FluxEqlbEV, FluxEqlbSE
import dolfinx_eqlb.eqlb.check_eqlb_conditions as eqlb_checker


from utils import (
    create_unitsquare_builtin,
)

from testcase_poisson import (
    set_arbitrary_rhs,
    set_arbitrary_bcs,
    solve_poisson_problem,
    equilibrate_poisson,
)

""" 
Check flux equilibration for multiple RHS
"""


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
@pytest.mark.parametrize("equilibrator", [FluxEqlbEV, FluxEqlbSE])
def test_equilibration_multi_rhs(degree, equilibrator):
    # Create mesh
    geometry = create_unitsquare_builtin(
        2, dmesh.CellType.triangle, dmesh.DiagonalType.crossed
    )

    # Set function space primal problem
    V_prime = dfem.FunctionSpace(geometry.mesh, ("P", degree))

    # --- Setup/solve two primal problems
    # Initialise storage
    list_refsol = []
    list_proj_flux = []
    list_proj_rhs = []
    list_bound_id_neumann = []
    list_bound_id_dirichlet = []
    list_neumann_functions = []
    list_neumann_projection = []
    list_boundary_values = []

    # Set different boundary conditions
    # list_boundary_ids = [[1, 4], [1, 3], [2], [1, 3, 4]]
    list_boundary_ids = [[1, 4], [1, 3], [1, 3, 4]]

    # Solve equilibrations
    for bids in list_boundary_ids:
        # Set RHS
        rhs, rhs_projected = set_arbitrary_rhs(
            geometry.mesh, degree - 1, degree_projection=(degree - 1)
        )

        list_proj_rhs.append(rhs_projected)

        # Set boundary conditions
        (
            boundary_id_dirichlet,
            boundary_id_neumann,
            dirichlet_functions,
            neumann_functions,
            neumann_projection,
        ) = set_arbitrary_bcs(
            "neumann_inhom", V_prime, degree, degree_bc=(degree - 1), neumann_ids=bids
        )

        list_bound_id_neumann.append(boundary_id_neumann)
        list_bound_id_dirichlet.append(boundary_id_dirichlet)
        list_neumann_functions.append(neumann_functions)
        list_neumann_projection.append(neumann_projection)

        # Solve primal problem
        u_prime, sigma_projected = solve_poisson_problem(
            V_prime,
            geometry,
            boundary_id_neumann,
            boundary_id_dirichlet,
            rhs,
            neumann_functions,
            dirichlet_functions,
            degree_projection=degree - 1,
        )

        list_proj_flux.append(sigma_projected)

        # Solve equilibration
        sigma_eq, boundary_dofvalues = equilibrate_poisson(
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

        list_refsol.append(sigma_eq[0])
        list_boundary_values.append(boundary_dofvalues[0])

    # --- Equilibrate two RHS
    sigma_eq, boundary_dofvalues = equilibrate_poisson(
        equilibrator,
        degree,
        geometry,
        list_proj_flux,
        list_proj_rhs,
        list_bound_id_neumann,
        list_bound_id_dirichlet,
        list_neumann_functions,
        list_neumann_projection,
    )

    # --- Check equilibrated fluxes ---
    for i in range(0, len(list_boundary_ids)):
        # Extract referece data
        ref_sol = list_refsol[i]
        boundary_data_ref = list_boundary_values[i]

        if equilibrator == FluxEqlbEV:
            # Check boundary values
            if not np.allclose(
                boundary_dofvalues[i].x.array, boundary_data_ref.x.array
            ):
                raise ValueError("Boundary data does not match!")

            # Check equilibration
            if not np.allclose(sigma_eq[i].x.array, ref_sol.x.array):
                raise ValueError("Equilibrated fluxes do not match!")
        else:
            # --- Check boundary conditions ---
            eqlb_checker.check_boundary_conditions(
                sigma_eq[i],
                list_proj_flux[i],
                boundary_dofvalues[i],
                geometry.facet_function,
                list_bound_id_neumann[i],
            )

            # --- Check divergence condition ---
            eqlb_checker.check_divergence_condition(
                sigma_eq[i], list_proj_flux[i], list_proj_rhs[i]
            )

            # --- Check jump condition (only required for semi-explicit equilibrator)
            eqlb_checker.check_jump_condition(sigma_eq[i], list_proj_flux[i])


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
