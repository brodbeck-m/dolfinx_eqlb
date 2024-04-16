""" Demo for H(div) conforming equilibration of fluxes

Implementation of a H(div) conforming flux-equilibration of a 
Poisson problem
                    -div(grad(u)) = f .

To verify the correctness of the proposed implementation, the
gained solution is compared to the exact solution u_ext. For the
right-hand-side f 
                    f(x,y) = -grad(u_ext)
is enforced, where

                    u_ext = sin(2*pi * x) * cos(2*pi * y)
holds. Dirichlet BCs are applied on the boundaries 2 and 4.
"""

# --- Imports ---
import numpy as np
from mpi4py import MPI
import time
import typing

import dolfinx
import dolfinx.fem as dfem
import dolfinx.mesh as dmesh
import ufl

from dolfinx_eqlb.eqlb import FluxEqlbSE, fluxbc
from dolfinx_eqlb.eqlb.check_eqlb_conditions import (
    check_divergence_condition,
    check_jump_condition,
    check_weak_symmetry_condition,
)
from dolfinx_eqlb.lsolver import local_projection


# --- The exact solution
def exact_solution(x):
    """Exact solution
    u_ext = [sin(2*pi * x) * cos(2*pi * y), -cos(2*pi * x) * sin(2*pi * y)]

    Args:
        x (ufl.SpatialCoordinate): The position x
    Returns:
        The exact function as ufl-expression
    """
    return ufl.as_vector(
        [
            ufl.sin(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1]),
            -ufl.cos(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1]),
        ]
    )


def interpolate_ufl_to_function(f_ufl: typing.Any, f_fe: dfem.Function):
    # Create expression
    expr = dfem.Expression(f_ufl, f_fe.function_space.element.interpolation_points())

    # Perform interpolation
    f_fe.interpolate(expr)


# --- The primal problem
def create_unit_square_mesh(n_elmt: int):
    domain = dmesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0, 0]), np.array([1, 1])],
        [n_elmt, n_elmt],
        cell_type=dmesh.CellType.triangle,
        diagonal=dmesh.DiagonalType.crossed,
    )

    boundaries = [
        (1, lambda x: np.isclose(x[0], 0)),
        (2, lambda x: np.isclose(x[1], 0)),
        (3, lambda x: np.isclose(x[0], 1)),
        (4, lambda x: np.isclose(x[1], 1)),
    ]

    facet_indices, facet_markers = [], []
    for marker, locator in boundaries:
        facets = dolfinx.mesh.locate_entities(domain, 1, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full(len(facets), marker))

    facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
    facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tag = dolfinx.mesh.meshtags(
        domain, 1, facet_indices[sorted_facets], facet_markers[sorted_facets]
    )

    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)

    return domain, facet_tag, ds


def solve_primal_problem(
    elmt_order_prime: int,
    domain: dmesh.Mesh,
    facet_tags: typing.Any,
    ds: typing.Any,
    solver: str = "lu",
):
    """Solves the linear elasticity based on lagrangian finite elements

    Args:
        elmt_order_prime (int):     The order of the FE space
        domain (dolfinx.mesh.Mesh): The mesh
        facet_tags :                The facet tags
        ds:                         The measure for the boundary integrals
    """
    # Check input
    if elmt_order_prime < 2:
        raise ValueError("Consistency condition for weak symmetry not fulfilled!")

    # The spatial dimension
    gdim = domain.geometry.dim

    # Set function space (primal problem)
    V_prime = dfem.VectorFunctionSpace(domain, ("CG", elmt_order_prime))

    # The exact solution
    u_ext = exact_solution(ufl.SpatialCoordinate(domain))
    sigma_ext = 2 * ufl.sym(ufl.grad(u_ext)) + ufl.div(u_ext) * ufl.Identity(gdim)

    # Set variational form
    u = ufl.TrialFunction(V_prime)
    v = ufl.TestFunction(V_prime)

    sigma = 2 * ufl.sym(ufl.grad(u)) + ufl.div(u) * ufl.Identity(gdim)

    a_prime = ufl.inner(sigma, ufl.grad(v)) * ufl.dx
    l_prime = ufl.inner(-ufl.div(sigma_ext), v) * ufl.dx

    # Set dirichlet boundary conditions
    u_dirichlet = dfem.Function(V_prime)
    interpolate_ufl_to_function(u_ext, u_dirichlet)

    dofs = dfem.locate_dofs_topological(V_prime, 1, facet_tags.indices)
    bcs_esnt = [dfem.dirichletbc(u_dirichlet, dofs)]

    # Solve problem
    timing = 0

    if solver == "lu":
        solveoptions = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "ksp_rtol": 1e-12,
            "ksp_atol": 1e-12,
        }
    else:
        solveoptions = {
            "ksp_type": "cg",
            "pc_type": "hypre",
            "hypre_type": "boomeramg",
            "ksp_rtol": 1e-10,
            "ksp_atol": 1e-12,
        }

    problem_prime = dfem.petsc.LinearProblem(
        a_prime, l_prime, bcs_esnt, petsc_options=solveoptions
    )

    timing -= time.perf_counter()
    u_prime = problem_prime.solve()
    timing += time.perf_counter()

    print(f"Primal problem solved in {timing:.4e} s")

    return u_prime, sigma_ext


# --- The flux equilibration
def equilibrate_flux(
    elmt_order_eqlb: int,
    domain: dmesh.Mesh,
    facet_tags: typing.Any,
    uh_prime: typing.Any,
    sigma_ext: typing.Any,
    weak_symmetry: bool = True,
):
    """Equilibrates the stress-tensor of linear elasticity

    The RHS is assumed to be the divergence of the exact stress
    tensor (manufactured solution).

    Args:
        elmt_order_prime (int):     The order of the FE space
        domain (dolfinx.mesh.Mesh): The mesh
        facet_tags :                The facet tags
        uh_prime:                   The primal solution
        sigma_ext:                  The exact stress-tensor
    """

    # Check input
    if elmt_order_eqlb < 2:
        raise ValueError("Stress equilibration only possible for k>1")

    # The spatial dimension
    gdim = domain.geometry.dim

    # The approximate solution
    sigma_h = 2 * ufl.sym(ufl.grad(uh_prime)) + ufl.div(uh_prime) * ufl.Identity(gdim)

    # Set source term
    f = ufl.div(sigma_ext)

    # Projected flux
    # (elmt_order_eqlb - 1 would be sufficient but not implemented for semi-explicit eqlb.)
    V_flux_proj = dfem.VectorFunctionSpace(domain, ("DG", elmt_order_eqlb - 1))
    sigma_proj = local_projection(
        V_flux_proj,
        [
            ufl.as_vector([sigma_h[0, 0], sigma_h[0, 1]]),
            ufl.as_vector([sigma_h[1, 0], sigma_h[1, 1]]),
        ],
    )

    # Project RHS
    V_rhs_proj = dfem.FunctionSpace(domain, ("DG", elmt_order_eqlb - 1))
    rhs_proj = local_projection(V_rhs_proj, [f[0], f[1]])

    # Initialise equilibrator
    equilibrator = FluxEqlbSE(
        elmt_order_eqlb, domain, rhs_proj, sigma_proj, equilibrate_stress=weak_symmetry
    )

    # Set boundary conditions
    equilibrator.set_boundary_conditions(
        [facet_tags.indices, facet_tags.indices],
        [[], []],
        quadrature_degree=3 * elmt_order_eqlb,
    )

    # Solve equilibration
    timing = 0

    timing -= time.perf_counter()
    equilibrator.equilibrate_fluxes()
    timing += time.perf_counter()

    print(f"Equilibration solved in {timing:.4e} s")

    # --- Check equilibration conditions ---
    V_rhs_proj = dfem.VectorFunctionSpace(domain, ("DG", elmt_order_eqlb - 1))
    rhs_proj_vecval = local_projection(V_rhs_proj, [f])[0]

    stress_eqlb = ufl.as_matrix(
        [
            [equilibrator.list_flux[0][0], equilibrator.list_flux[0][1]],
            [equilibrator.list_flux[1][0], equilibrator.list_flux[1][1]],
        ]
    )

    stress_proj = ufl.as_matrix(
        [
            [sigma_proj[0][0], sigma_proj[0][1]],
            [sigma_proj[1][0], sigma_proj[1][1]],
        ]
    )

    # Check divergence condition
    check_divergence_condition(
        stress_eqlb,
        stress_proj,
        rhs_proj_vecval,
        mesh=domain,
        degree=elmt_order_eqlb,
        flux_is_dg=True,
    )

    # Check if flux is H(div)
    for i in range(domain.geometry.dim):
        check_jump_condition(equilibrator.list_flux[i], sigma_proj[i])

    return sigma_proj, equilibrator.list_flux


if __name__ == "__main__":
    # --- Parameters ---
    # The orders of the FE spaces
    elmt_order_prime = 2
    elmt_order_eqlb = 2

    # The mesh resolution
    sdisc_nelmt = 100

    # --- Execute calculation ---
    # Create mesh
    domain, facet_tags, ds = create_unit_square_mesh(sdisc_nelmt)

    # Solve primal problem
    uh_prime, sigma_ref = solve_primal_problem(
        elmt_order_prime, domain, facet_tags, ds, solver="cg"
    )

    # Solve equilibration
    sigma_proj, sigma_eqlb = equilibrate_flux(
        elmt_order_eqlb, domain, facet_tags, uh_prime, sigma_ref, True
    )

    # --- Export results to ParaView ---
    # The exact flux
    V_dg_ref = dfem.TensorFunctionSpace(domain, ("DG", elmt_order_prime))
    sigma_ref = local_projection(V_dg_ref, [sigma_ref], quadrature_degree=10)

    # Project equilibrated flux into appropriate DG space
    V_dg_hdiv = dfem.VectorFunctionSpace(domain, ("DG", elmt_order_eqlb))
    sigma_eqlb_dg = local_projection(
        V_dg_hdiv, [sigma_eqlb[0] + sigma_proj[0], sigma_eqlb[1] + sigma_proj[1]]
    )

    # Export primal solution
    uh_prime.name = "u_h"
    sigma_eqlb_dg[0].name = "sigma_eqlb_row1"
    sigma_eqlb_dg[1].name = "sigma_eqlb_row2"
    sigma_ref[0].name = "sigma_ref"

    outfile = dolfinx.io.XDMFFile(MPI.COMM_WORLD, "demo_equilibration.xdmf", "w")
    outfile.write_mesh(domain)
    outfile.write_function(uh_prime, 1)
    outfile.write_function(sigma_ref[0], 1)
    outfile.write_function(sigma_eqlb_dg[0], 1)
    outfile.write_function(sigma_eqlb_dg[1], 1)
    outfile.close()
