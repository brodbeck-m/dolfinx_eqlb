import numpy as np
import pytest

import basix
import dolfinx.mesh as dmesh
import dolfinx.fem as dfem
import ufl

from dolfinx_eqlb.eqlb import FluxEqlbEV, FluxEqlbSE

from utils import create_unitsquare_builtin, flux_error, error_hdiv0
from testcase_poisson import (
    exact_solution_poisson,
    exact_flux_ufl_poisson,
    set_manufactured_rhs,
    set_manufactured_bcs,
    solve_poisson_problem,
    equilibrate_poisson,
)


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
@pytest.mark.parametrize("bc_type", ["pure_dirichlet", "neumann_hom", "neumann_inhom"])
@pytest.mark.parametrize("equilibrator", [FluxEqlbEV, FluxEqlbSE])
def test_convrate(degree, bc_type, equilibrator):
    # Initialise exact solution
    uext_ufl = exact_solution_poisson(ufl)
    uext_np = exact_solution_poisson(np)

    # Initialise boundary conditions
    if bc_type == "pure_dirichlet":
        boundary_id_dirichlet = [1, 2, 3, 4]
        boundary_id_neumann = []
    elif bc_type == "neumann_hom":
        boundary_id_dirichlet = [1, 3]
        boundary_id_neumann = [2, 4]
    elif bc_type == "neumann_inhom":
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

        # Exact flux as ufl-argument
        x = ufl.SpatialCoordinate(geometry.mesh)
        sigma_ext_ufl = exact_flux_ufl_poisson(x)

        # Set function space
        V_prime = dfem.FunctionSpace(geometry.mesh, ("P", degree))

        # Determine degree of projected quantities (primal flux, RHS)
        degree_proj = degree - 1

        # Set RHS
        rhs, rhs_projected = set_manufactured_rhs(uext_ufl, geometry.mesh, degree_proj)

        # Set boundary conditions
        (
            dirichlet_functions,
            neumann_functions,
            neumann_projection,
        ) = set_manufactured_bcs(
            V_prime, boundary_id_dirichlet, boundary_id_neumann, uext_np, sigma_ext_ufl
        )

        # Solve equilibration
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
        sigma_eq, _ = equilibrate_poisson(
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

        # Calculate error
        if equilibrator == FluxEqlbSE:
            data_convstudy[i, 1] = flux_error(
                sigma_eq[0] + sigma_projected,
                sigma_ext_ufl,
                error_hdiv0,
                uex_is_ufl=True,
            )
        else:
            data_convstudy[i, 1] = flux_error(
                sigma_eq[0], sigma_ext_ufl, error_hdiv0, uex_is_ufl=True
            )

    # Calculate convergence rate
    rates = np.log(data_convstudy[1:, 1] / data_convstudy[:-1, 1]) / np.log(
        data_convstudy[1:, 0] / data_convstudy[:-1, 0]
    )

    assert (rates > degree - 0.1).all()


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
