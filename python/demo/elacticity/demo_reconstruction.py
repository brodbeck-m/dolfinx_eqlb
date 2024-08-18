# Copyright (C) 2024 Maximilian Brodbeck
#
# This file is part of dolfinx_eqlb
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Demo for H(div) conforming equilibration of stresses

Solution of the quasi-static linear elasticity equation

        -div(2 * eps + pi_1 * div(u) * I) = f ,

with subsequent stress reconstruction. A convergence 
study based on a manufactured solution

        u_ext = [sin(pi * x) * sin(pi * y),
                -sin(2*pi * x) * sin(2*pi * y)]

is performed. Dirichlet boundary conditions are 
applied on boundary surfaces [1, 2, 3, 4].
"""

from enum import Enum
import gmsh
from mpi4py import MPI
import numpy as np
import time
import typing

import dolfinx
import dolfinx.fem as dfem
from dolfinx.io import gmshio
import dolfinx.mesh as dmesh
import ufl

from dolfinx_eqlb.eqlb import FluxEqlbSE
from dolfinx_eqlb.eqlb.check_eqlb_conditions import (
    check_divergence_condition,
    check_jump_condition,
    check_weak_symmetry_condition,
)
from dolfinx_eqlb.lsolver import local_projection


# --- The exact solution
def exact_solution(x) -> typing.Any:
    """Exact solution
    u_ext = [sin(pi * x) * sin(pi * y), -sin(2*pi * x) * sin(2*pi * y)]

    Args:
        x: The spatial position

    Returns:
        The exact function as ufl-expression
    """
    return ufl.as_vector(
        [
            ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
            -ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1]),
        ]
    )


def interpolate_ufl_to_function(f_ufl: typing.Any, f_fe: dfem.Function):
    """Interpolates a UFL expression to a function

    Args:
        f_ufl: The function in UFL
        f_fe:  The function to interpolate into
    """

    # Create expression
    expr = dfem.Expression(f_ufl, f_fe.function_space.element.interpolation_points())

    # Perform interpolation
    f_fe.interpolate(expr)


# --- Mesh generation
class MeshType(Enum):
    builtin = 0
    gmsh = 1


def create_unit_square_builtin(
    n_elmt: int,
) -> typing.Tuple[dmesh.Mesh, dmesh.MeshTagsMetaClass, ufl.Measure]:
    """Create a unit square using the build-in mesh generator

                    4
      -     |---------------|
      |     |               |
      |     |               |
    1 |   1 |               | 3
      |     |               |
      |     |               |
      -     |---------------|
                    2

            '-------1-------'

    Args:
        n_elmt: The number of elements in each direction

    Returns:
        The mesh,
        The facet tags,
        The tagged surface measure
    """

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


def create_unit_square_gmesh(
    h: float,
) -> typing.Tuple[dmesh.Mesh, dmesh.MeshTagsMetaClass, ufl.Measure]:
    """Create a unit square using gmsh

                    4
      -     |---------------|
      |     |               |
      |     |               |
    1 |   1 |               | 3
      |     |               |
      |     |               |
      -     |---------------|
                    2

            '-------1-------'

    Args:
        h (float): The characteristic mesh length

    Returns:
        The mesh,
        The facet tags,
        The tagged surface measure
    """

    # --- Build model
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.Verbosity", 2)

    # Name of the geometry
    gmsh.model.add("LShape")

    # Points
    list_pnts = [[0, 0], [0, 1], [1, 1], [1, 0]]

    pnts = [gmsh.model.occ.add_point(pnt[0], pnt[1], 0.0) for pnt in list_pnts]

    # Bounding curves and 2D surface
    bfcts = [
        gmsh.model.occ.add_line(pnts[0], pnts[1]),
        gmsh.model.occ.add_line(pnts[1], pnts[2]),
        gmsh.model.occ.add_line(pnts[2], pnts[3]),
        gmsh.model.occ.add_line(pnts[3], pnts[0]),
    ]

    boundary = gmsh.model.occ.add_curve_loop(bfcts)
    surface = gmsh.model.occ.add_plane_surface([boundary])
    gmsh.model.occ.synchronize()

    # Set tag on boundaries and surface
    for i, bfct in enumerate(bfcts):
        gmsh.model.addPhysicalGroup(1, [bfct], i + 1)

    gmsh.model.addPhysicalGroup(2, [surface], 1)

    # --- Generate mesh
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
    gmsh.model.mesh.generate(2)

    domain_init, _, _ = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)

    # --- Test if boundary patches contain at least 2 cells
    # List of refined cells
    refined_cells = []

    # Required connectivity's
    domain_init.topology.create_connectivity(0, 2)
    domain_init.topology.create_connectivity(1, 2)
    pnt_to_cell = domain_init.topology.connectivity(0, 2)

    # The boundary facets
    bfcts = dmesh.exterior_facet_indices(domain_init.topology)

    # Get boundary nodes
    V = dfem.FunctionSpace(domain_init, ("Lagrange", 1))
    bpnts = dfem.locate_dofs_topological(V, 1, bfcts)

    # Check if point is linked with only on cell
    for pnt in bpnts:
        cells = pnt_to_cell.links(pnt)

        if len(cells) == 1:
            refined_cells.append(cells[0])

    # Refine mesh
    list_ref_cells = list(set(refined_cells))

    if len(list_ref_cells) > 0:
        print("Refine mesh on boundary")
        domain = dmesh.refine(
            domain_init,
            np.setdiff1d(
                dmesh.compute_incident_entities(domain_init, list_ref_cells, 2, 1),
                bfcts,
            ),
        )
    else:
        domain = domain_init

    # --- Mark facets
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
    facet_function = dolfinx.mesh.meshtags(
        domain, 1, facet_indices[sorted_facets], facet_markers[sorted_facets]
    )
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_function)

    return domain, facet_function, ds


# --- The primal problem
def solve_primal_problem(
    order_prime: int,
    domain: dmesh.Mesh,
    facet_tags: dmesh.MeshTagsMetaClass,
    ds: ufl.Measure,
    pi_1: float,
    pdegree_rhs: typing.Optional[int] = None,
    solver: str = "lu",
) -> typing.Tuple[dfem.Function, typing.Any]:
    """Solves the problem of linear elasticity based on lagrangian finite elements

    Args:
        order_prime: The order of the FE space
        domain:      The mesh
        facet_tags:  The facet tags
        ds:          The measure for the boundary integrals
        pi_1:        The ratio of lambda and mu
        pdegree_rhs: The degree of the DG space into which the RHS
                     is projected into
        solver:      The solver type (lu or cg)

    Returns:
        The displacement solution,
        The exact stress tensor
    """

    # Check input
    if order_prime < 2:
        raise ValueError("Consistency condition for weak symmetry not fulfilled!")

    # The spatial dimension
    gdim = domain.geometry.dim

    # Set function space (primal problem)
    V_prime = dfem.VectorFunctionSpace(domain, ("CG", order_prime))

    # The exact solution
    u_ext = exact_solution(ufl.SpatialCoordinate(domain))
    sigma_ext = 2 * ufl.sym(ufl.grad(u_ext)) + pi_1 * ufl.div(u_ext) * ufl.Identity(
        gdim
    )

    # The right-hand-side
    f = -ufl.div(sigma_ext)

    if pdegree_rhs is None:
        rhs = f
    else:
        rhs = local_projection(
            dfem.VectorFunctionSpace(domain, ("DG", pdegree_rhs)), [f]
        )[0]

    # Set variational form
    u = ufl.TrialFunction(V_prime)
    v = ufl.TestFunction(V_prime)

    sigma = 2 * ufl.sym(ufl.grad(u)) + pi_1 * ufl.div(u) * ufl.Identity(gdim)

    a_prime = ufl.inner(sigma, ufl.sym(ufl.grad(v))) * ufl.dx
    l_prime = ufl.inner(rhs, v) * ufl.dx

    # Set Dirichlet boundary conditions
    u_dirichlet = dfem.Function(V_prime)
    interpolate_ufl_to_function(u_ext, u_dirichlet)

    dofs = dfem.locate_dofs_topological(V_prime, 1, facet_tags.indices[:])
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
            "ksp_rtol": 1e-12,
            "ksp_atol": 1e-12,
        }

    problem_prime = dfem.petsc.LinearProblem(
        a_prime, l_prime, bcs_esnt, petsc_options=solveoptions
    )

    timing -= time.perf_counter()
    uh = problem_prime.solve()
    timing += time.perf_counter()

    print(f"Primal problem solved in {timing:.4e} s")

    return uh, sigma_ext


# --- The flux equilibration
def equilibrate_flux(
    order_eqlb: int,
    domain: dmesh.Mesh,
    facet_tags: dmesh.MeshTagsMetaClass,
    pi_1: float,
    uh: dfem.Function,
    sigma_ext: typing.Any,
    weak_symmetry: typing.Optional[bool] = True,
    check_equilibration: typing.Optional[bool] = True,
) -> typing.Tuple[
    typing.List[dfem.Function], typing.List[dfem.Function], dfem.Function
]:
    """Equilibrates the negative stress-tensor of linear elasticity

    The RHS is assumed to be the divergence of the exact stress
    tensor (manufactured solution).

    Args:
        order_eqlb:          The order of the RT space
        domain:              The mesh
        facet_tags:          The facet tags
        pi_1:                The ratio of lambda and mu
        uh:                  The primal solution
        sigma_ext:           The exact stress-tensor
        weak_symmetry:       Id if weak symmetry condition is enforced
        check_equilibration: Id if equilibration conditions are checked

    Returns:
        The projected stress tensor (row wise),
        The equilibrated stress tensor (row wise),
        The cells Korns constant
    """

    # Check input
    if order_eqlb < 2:
        raise ValueError("Stress equilibration only possible for k>1")

    # The spatial dimension
    gdim = domain.geometry.dim

    # The approximate solution
    sigma_h = -2 * ufl.sym(ufl.grad(uh)) - pi_1 * ufl.div(uh) * ufl.Identity(gdim)

    # Set source term
    f = -ufl.div(sigma_ext)

    # Projected flux
    # (order_eqlb - 1 would be sufficient but not implemented for semi-explicit eqlb.)
    V_flux_proj = dfem.VectorFunctionSpace(domain, ("DG", order_eqlb - 1))
    sigma_proj = local_projection(
        V_flux_proj,
        [
            ufl.as_vector([sigma_h[0, 0], sigma_h[0, 1]]),
            ufl.as_vector([sigma_h[1, 0], sigma_h[1, 1]]),
        ],
    )

    # Project RHS
    V_rhs_proj = dfem.FunctionSpace(domain, ("DG", order_eqlb - 1))
    rhs_proj = local_projection(V_rhs_proj, [f[0], f[1]])

    # Initialise equilibrator
    equilibrator = FluxEqlbSE(
        order_eqlb,
        domain,
        rhs_proj,
        sigma_proj,
        equilibrate_stress=weak_symmetry,
        estimate_korn_constant=True,
    )

    # Set boundary conditions
    equilibrator.set_boundary_conditions(
        [facet_tags.indices[:], facet_tags.indices[:]],
        [[], []],
        quadrature_degree=3 * order_eqlb,
    )

    # Solve equilibration
    timing = 0

    timing -= time.perf_counter()
    equilibrator.equilibrate_fluxes()
    timing += time.perf_counter()

    print(f"Equilibration solved in {timing:.4e} s")

    # --- Check equilibration conditions ---
    if check_equilibration:
        V_rhs_proj = dfem.VectorFunctionSpace(domain, ("DG", order_eqlb - 1))
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
        div_condition_fulfilled = check_divergence_condition(
            stress_eqlb,
            stress_proj,
            rhs_proj_vecval,
            mesh=domain,
            degree=order_eqlb,
            flux_is_dg=True,
        )

        if not div_condition_fulfilled:
            raise ValueError("Divergence conditions not fulfilled")

        # Check if flux is H(div)
        for i in range(domain.geometry.dim):
            jump_condition_fulfilled = check_jump_condition(
                equilibrator.list_flux[i], sigma_proj[i]
            )

            if not jump_condition_fulfilled:
                raise ValueError("Jump conditions not fulfilled")

        # Check weak symmetry condition
        wsym_condition = check_weak_symmetry_condition(equilibrator.list_flux)

        if not wsym_condition:
            raise ValueError("Weak symmetry conditions not fulfilled")

    return sigma_proj, equilibrator.list_flux, equilibrator.get_korn_constants()


if __name__ == "__main__":
    # --- Parameters ---
    # The mesh type
    mesh_type = MeshType.builtin

    # Material: pi_1 = lambda/mu
    pi_1 = 1.0

    # The orders of the FE spaces
    order_prime = 2
    order_eqlb = 3

    # The mesh resolution
    sdisc_nelmt = 150

    # --- Execute calculation ---
    # Create mesh
    if mesh_type == MeshType.builtin:
        domain, facet_tags, ds = create_unit_square_builtin(sdisc_nelmt)
    elif mesh_type == MeshType.gmsh:
        domain, facet_tags, ds = create_unit_square_gmesh(1 / sdisc_nelmt)
    else:
        raise ValueError("Unknown mesh type")

    # Solve primal problem
    degree_proj = 1 if (order_eqlb == 2) else None
    uh, sigma_ref = solve_primal_problem(
        order_prime, domain, facet_tags, ds, pi_1, pdegree_rhs=degree_proj, solver="cg"
    )

    # Solve equilibration
    sigma_proj, sigma_eqlb, korns_constants = equilibrate_flux(
        order_eqlb, domain, facet_tags, pi_1, uh, sigma_ref
    )

    # --- Export results to ParaView ---
    # The exact flux
    V_dg_ref = dfem.TensorFunctionSpace(domain, ("DG", order_prime))
    sigma_ref = local_projection(V_dg_ref, [sigma_ref], quadrature_degree=10)

    # Project equilibrated flux into appropriate DG space
    V_dg_hdiv = dfem.VectorFunctionSpace(domain, ("DG", order_eqlb))
    sigma_eqlb_dg = local_projection(
        V_dg_hdiv, [sigma_eqlb[0] + sigma_proj[0], sigma_eqlb[1] + sigma_proj[1]]
    )

    # Export primal solution
    uh.name = "uh"
    sigma_eqlb_dg[0].name = "sigma_eqlb_row1"
    sigma_eqlb_dg[1].name = "sigma_eqlb_row2"
    sigma_ref[0].name = "sigma_ref"
    korns_constants.name = "korns_constants"

    outfile = dolfinx.io.XDMFFile(MPI.COMM_WORLD, "demo_equilibrate_stresses.xdmf", "w")
    outfile.write_mesh(domain)
    outfile.write_function(uh, 1)
    outfile.write_function(sigma_ref[0], 1)
    outfile.write_function(sigma_eqlb_dg[0], 1)
    outfile.write_function(sigma_eqlb_dg[1], 1)
    outfile.write_function(korns_constants, 1)
    outfile.close()
