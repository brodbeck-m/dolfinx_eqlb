# --- Import ---
import pytest

import dolfinx.mesh as dmesh
import dolfinx.fem as dfem

from dolfinx_eqlb.eqlb import FluxEqlbEV, FluxEqlbSE
import dolfinx_eqlb.eqlb.check_eqlb_conditions as eqlb_checker

from utils import (
    create_unitsquare_builtin,
    create_unitsquare_gmsh,
)

from testcase_general import set_arbitrary_rhs, set_arbitrary_bcs
from testcase_poisson import solve_primal_problem, equilibrate_fluxes


""" 
Check if equilibrated flux 
    a.) fulfills the divergence condition div(sigma_eqlb) = f 
    b.) is in the H(div) space
    c.) fulfills the flux boundary conditions strongly
"""


# --- The test routine ---
def equilibrate_flux(mesh_type, degree, bc_type, equilibrator):
    # Create mesh
    if mesh_type == "builtin":
        geometry = create_unitsquare_builtin(
            2, dmesh.CellType.triangle, dmesh.DiagonalType.crossed
        )
    elif mesh_type == "gmsh":
        geometry = create_unitsquare_gmsh(0.5)
    else:
        raise ValueError("Unknown mesh type")

    # Initialise loop over degree of boundary flux
    if bc_type != "neumann_inhom":
        degree_bc = 1
    else:
        if degree == 1:
            degree_bc = degree
        else:
            degree_bc = degree + 1

    # Perform tests
    for degree_bc in range(0, degree_bc):
        for degree_prime in range(max(1, degree - 1), degree + 1):
            for degree_rhs in range(0, degree):
                # Set function space
                V_prime = dfem.FunctionSpace(geometry.mesh, ("P", degree_prime))

                # Determine degree of projected quantities (primal flux, RHS)
                degree_proj = max(degree_prime - 1, degree_rhs)

                # Set RHS
                rhs, rhs_projected = set_arbitrary_rhs(
                    geometry.mesh,
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
                    geometry,
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
                    degree,
                    geometry,
                    [sigma_projected],
                    [rhs_projected],
                    [boundary_id_neumann],
                    [boundary_id_dirichlet],
                    [neumann_functions],
                    [neumann_projection],
                )

                # --- Check boundary conditions ---
                if bc_type != "pure_dirichlet":
                    boundary_condition = eqlb_checker.check_boundary_conditions(
                        sigma_eq[0],
                        sigma_projected,
                        boundary_dofvalues[0],
                        geometry.facet_function,
                        boundary_id_neumann,
                    )

                    if not boundary_condition:
                        raise ValueError("Boundary conditions not fulfilled")

                # --- Check divergence condition ---
                div_condition = eqlb_checker.check_divergence_condition(
                    sigma_eq[0], sigma_projected, rhs_projected
                )

                if not div_condition:
                    raise ValueError("Divergence condition not fulfilled")

                # --- Check jump condition (only required for semi-explicit equilibrator)
                if equilibrator == FluxEqlbSE:
                    jump_condition = eqlb_checker.check_jump_condition(
                        sigma_eq[0], sigma_projected
                    )

                    if not jump_condition:
                        raise ValueError("Jump condition not fulfilled")


# --- Test equilibration strategy by Ern and Vohralik
# TODO - Fix inhom. Neumann BCs on general meshes
@pytest.mark.parametrize("mesh_type", ["builtin", "gmsh"])
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("bc_type", ["neumann_hom"])
def test_ern_and_vorhralik_eqlb(mesh_type, degree, bc_type):
    equilibrate_flux(mesh_type, degree, bc_type, FluxEqlbEV)


# --- Test semi-explicit equilibration strategy
@pytest.mark.parametrize("mesh_type", ["builtin", "gmsh"])
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("bc_type", ["neumann_inhom"])
def test_semi_explicit_eqlb(mesh_type, degree, bc_type):
    equilibrate_flux(mesh_type, degree, bc_type, FluxEqlbSE)


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
