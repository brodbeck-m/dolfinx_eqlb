# Copyright (C) 2024 Maximilian Brodbeck
#
# This file is part of dolfinx_eqlb
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Test conditions of equilibrated stresses"""

import pytest

from dolfinx import fem, mesh
import ufl

import dolfinx_eqlb.eqlb.check_eqlb_conditions as eqlb_checker

from utils import MeshType, create_unitsquare_builtin, create_unitsquare_gmsh
from testcase_general import BCType, set_arbitrary_rhs, set_arbitrary_bcs
from testcase_elasticity import solve_primal_problem, equilibrate_stresses


@pytest.mark.parametrize("mesh_type", [MeshType.builtin, MeshType.gmsh])
@pytest.mark.parametrize("degree", [2, 3, 4])
@pytest.mark.parametrize("bc_type", [BCType.dirichlet, BCType.neumann_inhom])
def test_equilibration_conditions(mesh_type: MeshType, degree: int, bc_type: BCType):
    """Check stress equilibration based on the semi-explicit strategy

    Solve equilibration based on a primal problem (linear elasticity in
    displacement formulation) and check

        - the BCs
        - the divergence condition
        - the jump condition
        - the weak symmetry condition

    Args:
        mesh_type: The mesh type
        degree:    The degree of the equilibrated fluxes
        bc_typ:    The type of BCs
    """

    # Create mesh
    gdim = 2

    if mesh_type == MeshType.builtin:
        geometry = create_unitsquare_builtin(
            2, mesh.CellType.triangle, mesh.DiagonalType.crossed
        )
    elif mesh_type == MeshType.gmsh:
        geometry = create_unitsquare_gmsh(0.5)
    else:
        raise ValueError("Unknown mesh type")

    # Initialise loop over degree of boundary flux
    if bc_type != BCType.neumann_inhom:
        degrees_bc = [0]
    else:
        degrees_bc = list(range(max(0, degree - 3), degree))

    # Perform tests
    for degree_bc in degrees_bc:
        for degree_prime in range(max(2, degree - 1), degree + 1):
            for degree_rhs in range(0, degree):
                # Set function space
                V_prime = fem.VectorFunctionSpace(geometry.mesh, ("P", degree_prime))

                # Determine degree of projected quantities (primal flux, RHS)
                degree_proj = max(degree_prime - 1, degree_rhs)

                # Set RHS
                rhs, rhs_projected = set_arbitrary_rhs(
                    geometry.mesh,
                    degree_rhs,
                    degree_projection=degree_proj,
                    vector_valued=True,
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
                    geometry,
                    boundary_id_neumann,
                    boundary_id_dirichlet,
                    rhs,
                    neumann_functions,
                    dirichlet_functions,
                    degree_projection=degree_proj,
                )

                # --- Solve equilibration
                # RHS and Neumann BCs for each row of the stress tensor
                rhs_projected_row = []
                neumann_functions_row = [[] for _ in range(gdim)]

                V_aux = fem.FunctionSpace(
                    geometry.mesh,
                    ("DG", rhs_projected.function_space.element.basix_element.degree),
                )

                for i in range(geometry.mesh.geometry.dim):
                    rhs_projected_row.append(fem.Function(V_aux))

                    # RHS: Get values from vector values space
                    rhs_projected_row[-1].x.array[:] = (
                        rhs_projected.sub(i).collapse().x.array[:]
                    )

                    # Neumann BCs: Get values from ufl-vector
                    for bc_ufl in neumann_functions:
                        neumann_functions_row[i].append(bc_ufl[i])

                sigma_eq, boundary_dofvalues = equilibrate_stresses(
                    degree,
                    geometry,
                    sigma_projected,
                    rhs_projected_row,
                    [boundary_id_neumann, boundary_id_neumann],
                    [boundary_id_dirichlet, boundary_id_dirichlet],
                    neumann_functions_row,
                    [neumann_projection, neumann_projection],
                )

                # --- Check boundary conditions ---
                if bc_type != BCType.dirichlet:
                    for i in range(gdim):
                        boundary_condition = eqlb_checker.check_boundary_conditions(
                            sigma_eq[i],
                            sigma_projected[i],
                            boundary_dofvalues[i],
                            geometry.facet_function,
                            boundary_id_neumann,
                        )

                        if not boundary_condition:
                            raise ValueError("Boundary conditions not fulfilled")

                # --- Check divergence condition ---
                stress_eq = ufl.as_matrix(
                    [[sigma_eq[0][0], sigma_eq[0][1]], [sigma_eq[1][0], sigma_eq[1][1]]]
                )
                stress_projected = ufl.as_matrix(
                    [
                        [sigma_projected[0][0], sigma_projected[0][1]],
                        [sigma_projected[1][0], sigma_projected[1][1]],
                    ]
                )

                div_condition = eqlb_checker.check_divergence_condition(
                    stress_eq,
                    stress_projected,
                    rhs_projected,
                    mesh=geometry.mesh,
                    degree=degree,
                    flux_is_dg=True,
                )

                if not div_condition:
                    raise ValueError("Divergence conditions not fulfilled")

                # --- Check jump condition (only required for semi-explicit equilibrator)
                for i in range(geometry.mesh.geometry.dim):
                    jump_condition = eqlb_checker.check_jump_condition(
                        sigma_eq[i], sigma_projected[i]
                    )

                    if not jump_condition:
                        raise ValueError("Boundary conditions not fulfilled")

                # --- Check weak symmetry
                wsym_condition = eqlb_checker.check_weak_symmetry_condition(sigma_eq)

                if not wsym_condition:
                    raise ValueError("Weak symmetry conditions not fulfilled")


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
