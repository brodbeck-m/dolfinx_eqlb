# Copyright (C) 2024 Maximilian Brodbeck
#
# This file is part of dolfinx_eqlb
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Test conditions of simultaneously equilibrated fluxes (different primal problems)"""

import numpy as np
import pytest
import typing

import dolfinx.mesh as dmesh
import dolfinx.fem as dfem

from dolfinx_eqlb.eqlb import FluxEqlbEV, FluxEqlbSE
import dolfinx_eqlb.eqlb.check_eqlb_conditions as eqlb_checker


from utils import MeshType, create_unitsquare_builtin, create_unitsquare_gmsh

from testcase_general import BCType, set_arbitrary_rhs, set_arbitrary_bcs
from testcase_poisson import solve_primal_problem, equilibrate_fluxes


# --- The test routine ---
def equilibrate_multi_rhs(
    mesh_type: MeshType,
    degree: int,
    bc_type: BCType,
    equilibrator: typing.Union[FluxEqlbEV, FluxEqlbSE],
):
    """Solve a series of equilibrations (different primal problems) simultaneously and checks

        - the BCs
        - the divergence condition
        - the jump condition (only required for semi-explicit equilibrator)

    for the Poisson equations.

    Args:
        mesh_type:    The mesh type
        degree:       The degree of the equilibrated fluxes
        bc_typ:       The type of BCs
        equilibrator: The equilibrator
    """

    # Create mesh
    if mesh_type == MeshType.builtin:
        geometry = create_unitsquare_builtin(
            2, dmesh.CellType.triangle, dmesh.DiagonalType.crossed
        )
    elif mesh_type == MeshType.gmsh:
        geometry = create_unitsquare_gmsh(0.5)
    else:
        raise ValueError("Unknown mesh type")

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
    list_boundary_ids = [[1, 4], [1, 3], [2], [1, 3, 4]]

    # Solve equilibrations
    for bids in list_boundary_ids:
        # Set RHS
        rhs, rhs_projected = set_arbitrary_rhs(
            geometry.mesh,
            degree - 1,
            degree_projection=(degree - 1),
            vector_valued=False,
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
            bc_type, V_prime, degree, degree_bc=(degree - 1), neumann_ids=bids
        )

        list_bound_id_neumann.append(boundary_id_neumann)
        list_bound_id_dirichlet.append(boundary_id_dirichlet)
        list_neumann_functions.append(neumann_functions)
        list_neumann_projection.append(neumann_projection)

        # Solve primal problem
        u_prime, sigma_projected = solve_primal_problem(
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
        sigma_eq, boundary_dofvalues = equilibrate_fluxes(
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
    sigma_eq, boundary_dofvalues = equilibrate_fluxes(
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
            boundary_condition = eqlb_checker.check_boundary_conditions(
                sigma_eq[i],
                list_proj_flux[i],
                boundary_dofvalues[i],
                geometry.facet_function,
                list_bound_id_neumann[i],
            )

            if not boundary_condition:
                raise ValueError("Boundary conditions not fulfilled")

            # --- Check divergence condition ---
            div_condition = eqlb_checker.check_divergence_condition(
                sigma_eq[i], list_proj_flux[i], list_proj_rhs[i]
            )

            if not div_condition:
                raise ValueError("Divergence conditions not fulfilled")

            # --- Check jump condition (only required for semi-explicit equilibrator)
            jump_condition = eqlb_checker.check_jump_condition(
                sigma_eq[i], list_proj_flux[i]
            )

            if not jump_condition:
                raise ValueError("Jump conditions not fulfilled")


# --- Test equilibration strategy by Ern and Vohralik
# TODO - Fix inhom. Neumann BCs on general meshes
@pytest.mark.parametrize("mesh_type", [MeshType.builtin, MeshType.gmsh])
@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_ern_and_vohralik_mrhs(mesh_type, degree):
    """Check multiple equilibrations based on constrained minimisation (Ern and Vohralik)

    TODO - Fix inhom. Neumann BCs on general meshes

    Args:
        mesh_type: The mesh type
        degree:    The degree of the equilibrated fluxes
    """

    equilibrate_multi_rhs(mesh_type, degree, BCType.neumann_hom, FluxEqlbEV)


# --- Test semi-explicit equilibration strategy
@pytest.mark.parametrize("mesh_type", [MeshType.builtin, MeshType.gmsh])
@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_semi_explicit_mrhs(mesh_type, degree):
    """Check multiple equilibrations based on the semi-explicit strategy

    Args:
        mesh_type: The mesh type
        degree:    The degree of the equilibrated fluxes
        bc_type:   The type of BCs
    """

    equilibrate_multi_rhs(mesh_type, degree, BCType.neumann_inhom, FluxEqlbSE)


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
