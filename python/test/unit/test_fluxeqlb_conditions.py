# Copyright (C) 2024 Maximilian Brodbeck
#
# This file is part of dolfinx_eqlb
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Test conditions of equilibrated fluxes"""

from mpi4py import MPI
import pytest
import typing

import basix
from dolfinx import default_real_type, fem, io, mesh
import ufl

from dolfinx_eqlb import eqlb
import dolfinx_eqlb.eqlb.check_eqlb_conditions as eqlb_checker

from utils import MeshType, create_unitsquare_builtin, create_unitsquare_gmsh

from testcase_general import BCType, set_arbitrary_rhs, set_arbitrary_bcs
from testcase_poisson import solve_primal_problem, equilibrate_fluxes


# --- The test routine ---
def equilibrate_flux(
    equilibrator: typing.Type[eqlb.basics.EquilibratorMetaClass],
    mesh_type: MeshType,
    bc_type: BCType,
):
    """Solve equilibration based on a primal problem and check

        - the BCs
        - the divergence condition
        - the jump condition (only required for semi-explicit equilibrator)

    for the Poisson equations.

    Args:
        equilibrator: The equilibrator
        mesh_type:    The mesh type
        bc_typ:       The type of BCs
    """

    # Basic data
    degree = equilibrator.degree()

    # Create mesh
    if mesh_type == MeshType.builtin:
        domain = create_unitsquare_builtin(
            2, mesh.CellType.triangle, mesh.DiagonalType.crossed
        )
    elif mesh_type == MeshType.gmsh:
        domain = create_unitsquare_gmsh(0.5)
    else:
        raise ValueError("Unknown mesh type")

    outfile = io.XDMFFile(MPI.COMM_WORLD, "mesh-rework.xdmf", "w")
    outfile.write_mesh(domain.mesh)
    outfile.close()

    # Initialise loop over degree of boundary flux
    if bc_type != BCType.neumann_inhom:
        degree_bc = 1
    else:
        degree_bc = degree

    # Perform tests
    for degree_bc in range(0, degree_bc):
        for degree_prime in range(max(1, degree - 1), degree + 1):
            for degree_rhs in range(0, degree):
                # Set function space
                V_prime = fem.functionspace(domain.mesh, ("P", degree_prime))

                # Determine degree of projected quantities (primal flux, RHS)
                degree_proj = max(degree_prime - 1, degree_rhs)

                # Set RHS
                rhs, rhs_projected = set_arbitrary_rhs(
                    domain.mesh,
                    degree_rhs,
                    degree_projection=degree_proj,
                    vector_valued=False,
                )

                # Set boundary conditions
                (
                    boundary_id_dirichlet,
                    boundary_id_neumann,
                    dirichlet_functions,
                    neumann_functions,
                    neumann_projection,
                ) = set_arbitrary_bcs(bc_type, V_prime, degree, degree_bc)

                # Solve primal problem
                u_prime, sigma_projected = solve_primal_problem(
                    V_prime,
                    domain,
                    boundary_id_neumann,
                    boundary_id_dirichlet,
                    rhs,
                    neumann_functions,
                    dirichlet_functions,
                    degree_projection=degree_proj,
                )

                # Solve equilibration
                sigma_eq, boundary_dofvalues = equilibrate_fluxes(
                    equilibrator,
                    domain,
                    [sigma_projected],
                    [rhs_projected],
                    [boundary_id_neumann],
                    [boundary_id_dirichlet],
                    [neumann_functions],
                    [neumann_projection],
                )

                # --- Check boundary conditions ---
                if bc_type != BCType.dirichlet:
                    boundary_condition = eqlb_checker.check_boundary_conditions(
                        sigma_eq[0],
                        sigma_projected,
                        boundary_dofvalues[0],
                        domain.facet_function,
                        boundary_id_neumann,
                    )

                    if not boundary_condition:
                        raise ValueError("Boundary conditions not fulfilled")

                # --- Check divergence condition ---
                div_condition = eqlb_checker.check_divergence_condition(
                    sigma_eq[0],
                    sigma_projected,
                    rhs_projected,
                    print_debug_information=True,
                )

                if not div_condition:
                    raise ValueError("Divergence condition not fulfilled")

                # --- Check jump condition (only required for semi-explicit equilibrator)
                if equilibrator._strategy == eqlb.EqlbStrategy.semi_explicit:
                    jump_condition = eqlb_checker.check_jump_condition(
                        sigma_eq[0], sigma_projected
                    )

                    if not jump_condition:
                        raise ValueError("Jump condition not fulfilled")


# --- The tests ---
@pytest.mark.parametrize("mesh_type", [MeshType.builtin, MeshType.gmsh])
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("bc_type", [BCType.neumann_hom])
def test_ern_and_vohralik(mesh_type, degree, bc_type):
    """Check equilibration based on constrained minimisation (Ern and Vohralik)

    TODO - Fix inhom. Neumann BCs on general meshes

    Args:
        mesh_type: The mesh type
        degree:    The degree of the equilibrated fluxes
        bc_type:   The type of BCs
    """

    # Abstract mesh definition
    c_el = basix.ufl.element(
        "Lagrange", "triangle", 1, shape=(2,), dtype=default_real_type
    )

    # The equilibrator
    from dolfinx_eqlb.eqlb.constrained_minimisation import Equilibrator

    equilibrator = Equilibrator(ufl.Mesh(c_el), eqlb.ProblemType.flux, degree)

    # Check solution
    equilibrate_flux(equilibrator, mesh_type, bc_type)


# @pytest.mark.parametrize("mesh_type", [MeshType.builtin, MeshType.gmsh])
# @pytest.mark.parametrize("degree", [1, 2, 3])
# @pytest.mark.parametrize("bc_type", [BCType.neumann_inhom])
# def test_semi_explicit(mesh_type, degree, bc_type):
#     """Check equilibration based on the semi-explicit strategy

#     Args:
#         mesh_type: The mesh type
#         degree:    The degree of the equilibrated fluxes
#         bc_type:   The type of BCs
#     """

#     equilibrate_flux(mesh_type, degree, bc_type, FluxEqlbSE)


if __name__ == "__main__":
    import sys

    # pytest.main(sys.argv)
    test_ern_and_vohralik(MeshType.builtin, 1, BCType.dirichlet)
