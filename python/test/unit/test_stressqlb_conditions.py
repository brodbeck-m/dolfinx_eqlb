# --- Import ---
import pytest

import dolfinx.mesh as dmesh
import dolfinx.fem as dfem
import ufl

from dolfinx_eqlb.eqlb import FluxEqlbSE, FluxEqlbEV
import dolfinx_eqlb.eqlb.check_eqlb_conditions as eqlb_checker

from utils import (
    create_unitsquare_builtin,
    create_unitsquare_gmsh,
)

from testcase_general import set_arbitrary_rhs, set_arbitrary_bcs
from testcase_elasticity import solve_primal_problem, equilibrate_stresses

""" 
Check if equilibrated flux 
    a.) fulfills the divergence condition div(sigma_eqlb) = f 
    b.) is in the H(div) space
    c.) fulfills the flux boundary conditions strongly
"""


@pytest.mark.parametrize("mesh_type", ["builtin"])
@pytest.mark.parametrize("degree", [1, 2, 3, 4])
@pytest.mark.parametrize("bc_type", ["pure_dirichlet"])
def test_equilibration_conditions(mesh_type, degree, bc_type):
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
                V_prime = dfem.VectorFunctionSpace(geometry.mesh, ("P", degree_prime))

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

                # Solve equilibration
                rhs_projected_row = []

                V_aux = dfem.FunctionSpace(
                    geometry.mesh,
                    ("DG", rhs_projected.function_space.element.basix_element.degree),
                )

                for i in range(geometry.mesh.geometry.dim):
                    rhs_projected_row.append(dfem.Function(V_aux))

                    # Set values from mixed space
                    rhs_projected_row[-1].x.array[:] = (
                        rhs_projected.sub(i).collapse().x.array[:]
                    )

                sigma_eq, boundary_dofvalues = equilibrate_stresses(
                    FluxEqlbSE,
                    degree,
                    geometry,
                    sigma_projected,
                    rhs_projected_row,
                    [boundary_id_neumann, boundary_id_neumann],
                    [boundary_id_dirichlet, boundary_id_dirichlet],
                    [neumann_functions, neumann_functions],
                    [neumann_projection, neumann_projection],
                )

                # # --- Check boundary conditions ---
                # if bc_type != "pure_dirichlet":
                #     raise NotImplementedError(
                #         "Neumann boundary conditions not implemented yet"
                #     )

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

                eqlb_checker.check_divergence_condition(
                    stress_eq,
                    stress_projected,
                    rhs_projected,
                    mesh=geometry.mesh,
                    degree=degree,
                    flux_is_dg=True,
                )

                # --- Check jump condition (only required for semi-explicit equilibrator)
                for i in range(geometry.mesh.geometry.dim):
                    eqlb_checker.check_jump_condition(sigma_eq[i], sigma_projected[i])

                # --- Check weak symmetry
                eqlb_checker.check_weak_symmetry_condition(sigma_eq)


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
