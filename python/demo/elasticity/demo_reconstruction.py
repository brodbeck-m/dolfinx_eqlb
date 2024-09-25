# Copyright (C) 2024 Maximilian Brodbeck
#
# This file is part of dolfinx_eqlb
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Demo for H(div) conforming equilibration of stresses

Solution of the quasi-static linear elasticity equation

        div(sigma) = -f  with sigma = 2 * eps + pi_1 * div(u) * I,

with subsequent stress reconstruction. Dirichlet boundary conditions
are applied on the entire boundary using the exact solution

    u_ext = [ sin(pi * x) * cos(pi * y) + x²/(2*pi_1),
             -cos(pi * x) * sin(pi * y) + y²/(2*pi_1)] .
"""

from enum import Enum
import gmsh
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import time
import typing

from dolfinx import fem, io, mesh
import ufl

from dolfinx_eqlb.eqlb import FluxEqlbSE
from dolfinx_eqlb.eqlb.check_eqlb_conditions import (
    check_divergence_condition,
    check_jump_condition,
    check_weak_symmetry_condition,
)
from dolfinx_eqlb.lsolver import local_projection


# --- The exact solution
def exact_solution(x: typing.Any, pi_1: float) -> typing.Any:
    """Exact solution
    u_ext = [ sin(pi * x) * cos(pi * y) + x²/(2*pi_1),
             -cos(pi * x) * sin(pi * y) + y²/(2*pi_1)]

    Args:
        x:    The spatial position
        pi_1: The ratio of lambda and mu

    Returns:
        The exact function as ufl-expression
    """
    return ufl.as_vector(
        [
            ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])
            + 0.5 * (x[0] * x[0] / pi_1),
            -ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
            + 0.5 * (x[1] * x[1] / pi_1),
        ]
    )


def interpolate_ufl_to_function(f_ufl: typing.Any, f_fe: fem.Function):
    """Interpolates a UFL expression to a function

    Args:
        f_ufl: The function in UFL
        f_fe:  The function to interpolate into
    """

    # Create expression
    expr = fem.Expression(f_ufl, f_fe.function_space.element.interpolation_points())

    # Perform interpolation
    f_fe.interpolate(expr)


# --- Mesh generation
class MeshType(Enum):
    builtin = 0
    gmsh = 1


def create_unit_square_builtin(
    n_elmt: int,
) -> typing.Tuple[mesh.Mesh, mesh.MeshTagsMetaClass, ufl.Measure]:
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

    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0, 0]), np.array([1, 1])],
        [n_elmt, n_elmt],
        cell_type=mesh.CellType.triangle,
        diagonal=mesh.DiagonalType.crossed,
    )

    boundaries = [
        (1, lambda x: np.isclose(x[0], 0)),
        (2, lambda x: np.isclose(x[1], 0)),
        (3, lambda x: np.isclose(x[0], 1)),
        (4, lambda x: np.isclose(x[1], 1)),
    ]

    facet_indices, facet_markers = [], []
    for marker, locator in boundaries:
        facets = mesh.locate_entities(domain, 1, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full(len(facets), marker))

    facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
    facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tag = mesh.meshtags(
        domain, 1, facet_indices[sorted_facets], facet_markers[sorted_facets]
    )

    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)

    return domain, facet_tag, ds


def create_unit_square_gmsh(
    h: float,
) -> typing.Tuple[mesh.Mesh, mesh.MeshTagsMetaClass, ufl.Measure]:
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

    domain_init, _, _ = io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)

    # --- Test if boundary patches contain at least 2 cells
    # List of refined cells
    refined_cells = []

    # Required connectivity's
    domain_init.topology.create_connectivity(0, 2)
    domain_init.topology.create_connectivity(1, 2)
    pnt_to_cell = domain_init.topology.connectivity(0, 2)

    # The boundary facets
    bfcts = mesh.exterior_facet_indices(domain_init.topology)

    # Get boundary nodes
    V = fem.FunctionSpace(domain_init, ("Lagrange", 1))
    bpnts = fem.locate_dofs_topological(V, 1, bfcts)

    # Check if point is linked with only on cell
    for pnt in bpnts:
        cells = pnt_to_cell.links(pnt)

        if len(cells) == 1:
            refined_cells.append(cells[0])

    # Refine mesh
    list_ref_cells = list(set(refined_cells))

    if len(list_ref_cells) > 0:
        print("Refine mesh on boundary")
        domain = mesh.refine(
            domain_init,
            np.setdiff1d(
                mesh.compute_incident_entities(domain_init, list_ref_cells, 2, 1),
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
        facets = mesh.locate_entities(domain, 1, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full(len(facets), marker))

    facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
    facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_function = mesh.meshtags(
        domain, 1, facet_indices[sorted_facets], facet_markers[sorted_facets]
    )
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_function)

    return domain, facet_function, ds


# --- The primal problem
class DiscType(Enum):
    displacement = 0
    displacement_pressure = 1


class SolverType(Enum):
    LU = 0
    CG = 1


def solve(
    domain: mesh.Mesh,
    facet_tags: mesh.MeshTagsMetaClass,
    pi_1: float,
    sdisc_type: DiscType,
    degree: int,
    degree_rhs: typing.Optional[int] = None,
    solver_type: typing.Optional[SolverType] = SolverType.LU,
) -> typing.Tuple[
    typing.Union[fem.Function, typing.Any],
    typing.List[fem.Function],
    typing.Any,
    typing.Any,
]:
    """Solves the problem of linear elasticity based on lagrangian finite elements

    Args:
        domain:      The mesh
        facet_tags:  The facet tags
        pi_1:        The ratio of lambda and mu
        sdisc_type:  The discretisation type
        degree:      The degree of the FE space
        degree_rhs:  The degree of the DG space into which the RHS
                     is projected
        solver:      The solver type

    Returns:
        The right-hand-side,
        The exact stress tensor,
        The approximated solution,
        The approximated stress tensor
    """

    # The exact solution
    u_ext = exact_solution(ufl.SpatialCoordinate(domain), pi_1)
    sigma_ext = 2 * ufl.sym(ufl.grad(u_ext)) + pi_1 * ufl.div(u_ext) * ufl.Identity(2)

    # The right-hand-side
    f = -ufl.div(sigma_ext)

    if degree_rhs is None:
        rhs = f
    else:
        V_rhs = fem.VectorFunctionSpace(domain, ("DG", degree_rhs))
        rhs = local_projection(V_rhs, [f])[0]

    # --- Set weak form and BCs
    if sdisc_type == DiscType.displacement:
        # Check input
        if degree < 2:
            raise ValueError("Consistency condition for weak symmetry not fulfilled!")

        # The function space
        V = fem.VectorFunctionSpace(domain, ("CG", degree))
        uh = fem.Function(V)

        # Trial- and test-functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        # The variational form
        sigma = 2 * ufl.sym(ufl.grad(u)) + pi_1 * ufl.div(u) * ufl.Identity(2)

        a = fem.form(ufl.inner(sigma, ufl.sym(ufl.grad(v))) * ufl.dx)
        l = fem.form(ufl.inner(rhs, v) * ufl.dx)

        # The Dirichlet BCs
        uD = fem.Function(V)
        interpolate_ufl_to_function(u_ext, uD)

        dofs = fem.locate_dofs_topological(V, 1, facet_tags.indices[:])
        bcs_esnt = [fem.dirichletbc(uD, dofs)]
    elif sdisc_type == DiscType.displacement_pressure:
        # Check input
        if solver_type != SolverType.LU:
            raise ValueError(
                "u-p formulation should be solved using a LU decomposition!"
            )

        # Set function space
        elmt_u = ufl.VectorElement("P", domain.ufl_cell(), degree + 1)
        elmt_p = ufl.FiniteElement("P", domain.ufl_cell(), degree)

        V = fem.FunctionSpace(domain, ufl.MixedElement([elmt_u, elmt_p]))
        uh = fem.Function(V)

        # Trial- and test-functions
        u, p = ufl.TrialFunctions(V)
        v_u, v_p = ufl.TestFunctions(V)

        # The variational form
        sigma = 2 * ufl.sym(ufl.grad(u)) + p * ufl.Identity(2)

        a = fem.form(
            ufl.inner(sigma, ufl.sym(ufl.grad(v_u))) * ufl.dx
            + (ufl.div(u) - (1 / pi_1) * p) * v_p * ufl.dx
        )
        l = fem.form(ufl.inner(rhs, v_u) * ufl.dx)

        # The Dirichlet BCs
        Vu, _ = V.sub(0).collapse()

        uD = fem.Function(Vu)
        interpolate_ufl_to_function(u_ext, uD)

        dofs = fem.locate_dofs_topological((V.sub(0), Vu), 1, facet_tags.indices[:])
        bcs_esnt = [fem.dirichletbc(uD, dofs, V.sub(0))]
    else:
        raise ValueError("Unknown discretisation type")

    # --- Solve the equation system
    timing = 0

    timing -= time.perf_counter()
    # The system matrix
    A = fem.petsc.assemble_matrix(a, bcs=bcs_esnt)
    A.assemble()

    # The right-hand-side
    L = fem.petsc.create_vector(l)
    fem.petsc.assemble_vector(L, l)
    fem.apply_lifting(L, [a], [bcs_esnt])
    L.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem.set_bc(L, bcs_esnt)

    # The solver
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setOperators(A)

    if solver_type == SolverType.LU:
        solver.setType(PETSc.KSP.Type.PREONLY)
        pc = solver.getPC()
        pc.setType(PETSc.PC.Type.LU)
        pc.setFactorSolverType("mumps")
    elif solver_type == SolverType.CG:
        solver.setType(PETSc.KSP.Type.CG)
        pc = solver.getPC()
        pc.setType(PETSc.PC.Type.HYPRE)
        pc.setHYPREType("boomeramg")
    else:
        raise ValueError("Unsupported solver type")

    # Solve the system
    solver.setTolerances(rtol=1e-12, atol=1e-12, max_it=1000)
    solver.solve(L, uh.vector)

    timing += time.perf_counter()

    print(f"Primal problem solved in {timing:.4e} s")

    if sdisc_type == DiscType.displacement:
        # The approximated stress tensor
        sigma_h = 2 * ufl.sym(ufl.grad(uh)) + pi_1 * ufl.div(uh) * ufl.Identity(2)

        return rhs, sigma_ext, [uh], sigma_h
    elif sdisc_type == DiscType.displacement_pressure:
        # Split the mixed solution
        uh_u = uh.sub(0).collapse()
        uh_p = uh.sub(1).collapse()

        # The approximated stress tensor
        sigma_h = 2 * ufl.sym(ufl.grad(uh_u)) + uh_p * ufl.Identity(2)

        return rhs, sigma_ext, [uh_u, uh_p], sigma_h


# --- The equilibration
def equilibrate(
    domain: mesh.Mesh,
    facet_tags: mesh.MeshTagsMetaClass,
    f: typing.Any,
    sigma_h: typing.Any,
    degree: int,
    weak_symmetry: typing.Optional[bool] = True,
    check_equilibration: typing.Optional[bool] = True,
) -> typing.Tuple[typing.Any, typing.Any, fem.Function]:
    """Equilibrates the negative stress-tensor of linear elasticity

    The RHS is assumed to be the divergence of the exact stress
    tensor (manufactured solution).

    Args:
        domain:              The mesh
        facet_tags:          The facet tags
        f:                   The right-hand-side
        sigma_h:             The approximated stress tensor
        degree:              The degree of the RT space
        weak_symmetry:       Id if weak symmetry condition is enforced
        check_equilibration: Id if equilibration conditions are checked

    Returns:
        The projected stress tensor (ufl tensor),
        The equilibrated stress tensor (ufl tensor),
        The cells Korns constant
    """

    # Check input
    if degree < 2:
        raise ValueError("Stress equilibration only possible for k>1")

    # Projected flux
    # (degree - 1 would be sufficient but not implemented for semi-explicit eqlb.)
    V_flux_proj = fem.VectorFunctionSpace(domain, ("DG", degree - 1))
    sigma_proj = local_projection(
        V_flux_proj,
        [
            ufl.as_vector([-sigma_h[0, 0], -sigma_h[0, 1]]),
            ufl.as_vector([-sigma_h[1, 0], -sigma_h[1, 1]]),
        ],
    )

    # Project RHS
    V_rhs_proj = fem.FunctionSpace(domain, ("DG", degree - 1))
    rhs_proj = local_projection(V_rhs_proj, [f[0], f[1]])

    # Initialise equilibrator
    equilibrator = FluxEqlbSE(
        degree,
        domain,
        rhs_proj,
        sigma_proj,
        equilibrate_stress=weak_symmetry,
        estimate_korn_constant=True,
    )

    # Set boundary conditions
    equilibrator.set_boundary_conditions(
        [facet_tags.indices[:], facet_tags.indices[:]], [[], []]
    )

    # Solve equilibration
    timing = 0

    timing -= time.perf_counter()
    equilibrator.equilibrate_fluxes()
    timing += time.perf_counter()

    print(f"Equilibration solved in {timing:.4e} s")

    # Cast stresses into ufl tensors
    stress_eqlb = ufl.as_matrix(
        [
            [-equilibrator.list_flux[0][0], -equilibrator.list_flux[0][1]],
            [-equilibrator.list_flux[1][0], -equilibrator.list_flux[1][1]],
        ]
    )

    stress_proj = ufl.as_matrix(
        [
            [-sigma_proj[0][0], -sigma_proj[0][1]],
            [-sigma_proj[1][0], -sigma_proj[1][1]],
        ]
    )

    # --- Check equilibration conditions ---
    if check_equilibration:
        V_rhs_proj = fem.VectorFunctionSpace(domain, ("DG", degree - 1))
        rhs_proj_vecval = local_projection(V_rhs_proj, [-f])[0]

        # Check divergence condition
        div_condition_fulfilled = check_divergence_condition(
            stress_eqlb,
            stress_proj,
            rhs_proj_vecval,
            mesh=domain,
            degree=degree,
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

    return stress_proj, stress_eqlb, equilibrator.get_korn_constants()


if __name__ == "__main__":
    # --- Parameters ---
    # The mesh type
    mesh_type = MeshType.builtin

    # Material: pi_1 = lambda/mu
    pi_1 = 1.0

    # The spatial discretisation
    sdisc_type = DiscType.displacement
    order_prime = 2

    # The stress equilibration
    order_eqlb = 2

    # The mesh resolution
    sdisc_nelmt = 20

    # --- Execute calculation ---
    # Create mesh
    if mesh_type == MeshType.builtin:
        domain, facet_tags, ds = create_unit_square_builtin(sdisc_nelmt)
    elif mesh_type == MeshType.gmsh:
        domain, facet_tags, ds = create_unit_square_gmsh(1 / sdisc_nelmt)
    else:
        raise ValueError("Unknown mesh type")

    # Solve primal problem
    degree_proj = 1 if (order_eqlb == 2) else None

    f, sigma_ref, uh, sigma_h = solve(
        domain, facet_tags, pi_1, sdisc_type, order_prime, degree_rhs=degree_proj
    )

    # Solve equilibration
    sigma_proj, sigma_eqlb, korns_constants = equilibrate(
        domain, facet_tags, f, sigma_h, order_eqlb
    )

    # --- Export results to ParaView ---
    # The exact flux
    V_dg_ref = fem.TensorFunctionSpace(domain, ("DG", order_prime))
    sigma_ref = local_projection(V_dg_ref, [sigma_ref], quadrature_degree=10)

    # Project equilibrated flux into appropriate DG space
    V_dg_hdiv = fem.TensorFunctionSpace(domain, ("DG", order_eqlb))
    sigma_eqlb_dg = local_projection(V_dg_hdiv, [sigma_proj + sigma_eqlb])

    # Export primal solution
    uh[0].name = "uh"
    sigma_eqlb_dg[0].name = "sigma_eqlb"
    sigma_ref[0].name = "sigma_ref"
    korns_constants.name = "korns_constants"

    outfile = io.XDMFFile(MPI.COMM_WORLD, "demo_equilibrate_stresses.xdmf", "w")
    outfile.write_mesh(domain)
    outfile.write_function(uh[0], 1)
    outfile.write_function(sigma_ref[0], 1)
    outfile.write_function(sigma_eqlb_dg[0], 1)
    outfile.write_function(korns_constants, 1)
    outfile.close()
