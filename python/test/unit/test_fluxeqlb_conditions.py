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

from testcase_poisson import (
    set_arbitrary_rhs,
    set_arbitrary_bcs,
    solve_poisson_problem,
    equilibrate_poisson,
)


""" 
Check if equilibrated flux 
    a.) fulfills the divergence condition div(sigma_eqlb) = f 
    b.) is in the H(div) space
    c.) fulfills the flux boundary conditions strongly
"""


@pytest.mark.parametrize("mesh_type", ["builtin"])
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("bc_type", ["pure_dirichlet", "neumann_inhom"])
@pytest.mark.parametrize("equilibrator", [FluxEqlbEV, FluxEqlbSE])
def test_equilibration_conditions(mesh_type, degree, bc_type, equilibrator):
    # Create mesh
    if mesh_type == "builtin":
        geometry = create_unitsquare_builtin(
            2, dmesh.CellType.triangle, dmesh.DiagonalType.crossed
        )
    elif mesh_type == "gmsh":
        raise NotImplementedError("GMSH mesh not implemented yet")
    else:
        raise ValueError("Unknown mesh type")

    # Initialise loop over degree of boundary flux
    if bc_type != "neumann_inhom":
        degree_max_rhs = 1
    else:
        degree_max_rhs = degree

    # Perform tests
    for degree_bc in range(0, degree_max_rhs):
        for degree_prime in range(1, degree + 1):
            for degree_rhs in range(0, degree):
                # Set function space
                V_prime = dfem.FunctionSpace(geometry.mesh, ("P", degree_prime))

                # Determine degree of projected quantities (primal flux, RHS)
                degree_proj = max(degree_prime - 1, degree_rhs)

                # Set RHS
                rhs, rhs_projected = set_arbitrary_rhs(
                    geometry.mesh, degree_rhs, degree_projection=degree_proj
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
                u_prime, sigma_projected = solve_poisson_problem(
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

                # --- Check boundary conditions ---
                if bc_type != "pure_dirichlet":
                    eqlb_checker.check_boundary_conditions(
                        sigma_eq[0],
                        sigma_projected,
                        boundary_dofvalues[0],
                        geometry.facet_function,
                        boundary_id_neumann,
                    )

                # --- Check divergence condition ---
                eqlb_checker.check_divergence_condition(
                    sigma_eq[0], sigma_projected, rhs_projected
                )

                # --- Check jump condition (only required for semi-explicit equilibrator)
                if equilibrator == FluxEqlbSE:
                    eqlb_checker.check_jump_condition(sigma_eq[0], sigma_projected)


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
