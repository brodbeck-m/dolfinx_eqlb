# Copyright (C) 2024 Maximilian Brodbeck
#
# This file is part of dolfinx_eqlb
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

""" Demo for H(div) conforming equilibration of fluxes

Implementation of a H(div) conforming flux-equilibration for a 
Poisson problem
                    -div(grad(u)) = f .

To verify the correctness of the proposed implementation, the
gained solution is compared to the exact solution u_ext. Assuming 
                    f(x,y) = -grad(u_ext)
the exact solution
                    u_ext = sin(2*pi * x) * cos(2*pi * y)
is enforced. Possible boundary conditions:
    dirichlet:     u = u_ext on boundary surfaces [1,2,3,4]
    neumann_hom:   u = u_ext on boundary surfaces [1,3]
    neumann_inhom: u = u_ext on boundary surfaces [2,4]
"""

from enum import Enum
import gmsh
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import time
import typing

import dolfinx
from dolfinx.io import gmshio
import dolfinx.fem as dfem
import dolfinx.mesh as dmesh
import ufl

from dolfinx_eqlb.eqlb import fluxbc, FluxEqlbEV, FluxEqlbSE
from dolfinx_eqlb.lsolver import local_projection
from dolfinx_eqlb.eqlb.check_eqlb_conditions import (
    check_divergence_condition,
    check_jump_condition,
)


# --- The exact solution
def exact_solution(pkt):
    """Exact solution
    u_ext = sin(pi * x) * cos(pi * y)

    Args:
        pkt: The package

    Returns:
        The function handle oft the exact solution
    """
    return lambda x: pkt.sin(2 * pkt.pi * x[0]) * pkt.cos(2 * pkt.pi * x[1])


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
class BCType(Enum):
    dirichlet = 0
    neumann_hom = 1
    neumann_inhom = 2


def solve_primal_problem(
    order_prime: int,
    domain: dmesh.Mesh,
    facet_tags: dmesh.MeshTagsMetaClass,
    ds: ufl.Measure,
    bc_type: BCType,
    pdegree_rhs: typing.Optional[int] = None,
) -> dfem.Function:
    """Solves the Poisson problem based on lagrangian finite elements

    Args:
        order_prime: The order of the FE space
        domain:      The mesh
        facet_tags:  The facet tags
        ds:          The measure for the boundary integrals
        pdegree_rhs: The degree of the DG space into which the RHS
                     is projected into

    Returns:
        The solution
    """

    # Set function space (primal problem)
    V_prime = dfem.FunctionSpace(domain, ("CG", order_prime))

    # Set trial and test functions
    u = ufl.TrialFunction(V_prime)
    v = ufl.TestFunction(V_prime)

    # Set source term
    x = ufl.SpatialCoordinate(domain)
    f = -ufl.div(ufl.grad(exact_solution(ufl)(x)))

    if pdegree_rhs is None:
        rhs = f
    else:
        rhs = local_projection(dfem.FunctionSpace(domain, ("DG", pdegree_rhs)), [f])[0]

    # Equation system
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    l = rhs * v * ufl.dx

    # Neumann boundary conditions
    normal = ufl.FacetNormal(domain)

    if bc_type == BCType.neumann_inhom:
        l += ufl.inner(ufl.grad(exact_solution(ufl)(x)), normal) * v * ds(1)
        l += ufl.inner(ufl.grad(exact_solution(ufl)(x)), normal) * v * ds(3)

    # Dirichlet boundary conditions
    uD = dfem.Function(V_prime)
    uD.interpolate(exact_solution(np))

    if bc_type == BCType.dirichlet:
        fcts_essnt = facet_tags.indices[:]
    elif bc_type == BCType.neumann_hom:
        fcts_essnt = facet_tags.indices[
            np.logical_or(facet_tags.values == 1, facet_tags.values == 3)
        ]
    elif bc_type == BCType.neumann_inhom:
        fcts_essnt = facet_tags.indices[
            np.logical_or(facet_tags.values == 2, facet_tags.values == 4)
        ]

    dofs_essnt = dfem.locate_dofs_topological(V_prime, 1, fcts_essnt)
    bc_essnt = [dfem.dirichletbc(uD, dofs_essnt)]

    # Solve primal problem
    timing = 0
    problem = dfem.petsc.LinearProblem(
        a,
        l,
        bcs=bc_essnt,
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "hypre_type": "boomeramg",
            "ksp_rtol": 1e-10,
            "ksp_atol": 1e-12,
        },
    )

    timing -= time.perf_counter()
    uh = problem.solve()
    timing += time.perf_counter()

    print(f"Primal problem solved in {timing:.4e} s")

    return uh


# --- The flux equilibration
def equilibrate_flux(
    Equilibrator: typing.Union[FluxEqlbEV, FluxEqlbSE],
    order_eqlb: int,
    domain: dmesh.Mesh,
    facet_tags: dmesh.MeshTagsMetaClass,
    bc_type: BCType,
    uh: dfem.Function,
    check_equilibration: typing.Optional[bool] = True,
) -> typing.Tuple[dfem.Function, dfem.Function]:
    """Equilibrate the flux

    The RHS is assumed to be the divergence of the exact
    flux (manufactured solution).

    Args:
        Equilibrator:        The flux equilibrator
        order_eqlb:          The order of the RT space
        domain:              The mesh
        facet_tags:          The facet tags
        bc_type:             The type of BCs
        uh:                  The primal solution
        check_equilibration: Id if equilibration conditions are checked

    Returns:
        The projected flux,
        The equilibrated flux
    """

    # Set source term
    x = ufl.SpatialCoordinate(domain)
    f = -ufl.div(ufl.grad(exact_solution(ufl)(x)))

    # Project flux and RHS into required DG space
    V_rhs_proj = dfem.FunctionSpace(domain, ("DG", order_eqlb - 1))
    V_flux_proj = dfem.VectorFunctionSpace(domain, ("DG", order_eqlb - 1))

    sigma_proj = local_projection(V_flux_proj, [-ufl.grad(uh)])
    rhs_proj = local_projection(V_rhs_proj, [f])

    # Initialise equilibrator
    equilibrator = Equilibrator(order_eqlb, domain, rhs_proj, sigma_proj)

    # Set BCs
    bc_dual = []

    if bc_type == BCType.dirichlet:
        # Facets on Dirichlet boundary of primal problem
        fcts_essnt = facet_tags.indices

        # Set flux BCs
        qdegree = None
    elif bc_type == BCType.neumann_hom:
        # Facets on Dirichlet boundary of primal problem
        fcts_essnt = facet_tags.indices[
            np.logical_or(facet_tags.values == 1, facet_tags.values == 3)
        ]

        # Set flux BCs
        qdegree = None

        bc_dual.append(
            fluxbc(
                dfem.Constant(domain, PETSc.ScalarType(0.0)),
                facet_tags.indices[
                    np.logical_or(facet_tags.values == 2, facet_tags.values == 4)
                ],
                equilibrator.V_flux,
            )
        )
    elif bc_type == BCType.neumann_inhom:
        # Facets on Dirichlet boundary of primal problem
        fcts_essnt = facet_tags.indices[
            np.logical_or(facet_tags.values == 2, facet_tags.values == 4)
        ]

        # Set flux BCs
        qdegree = 3 * order_eqlb

        bc_dual.append(
            fluxbc(
                ufl.grad(exact_solution(ufl)(x))[0],
                facet_tags.indices[facet_tags.values == 1],
                equilibrator.V_flux,
                requires_projection=True,
                quadrature_degree=qdegree,
            )
        )
        bc_dual.append(
            fluxbc(
                -ufl.grad(exact_solution(ufl)(x))[0],
                facet_tags.indices[facet_tags.values == 3],
                equilibrator.V_flux,
                requires_projection=True,
                quadrature_degree=qdegree,
            )
        )

    equilibrator.set_boundary_conditions(
        [fcts_essnt], [bc_dual], quadrature_degree=qdegree
    )

    # Solve equilibration
    timing = 0

    timing -= time.perf_counter()
    equilibrator.equilibrate_fluxes()
    timing += time.perf_counter()

    print(f"Equilibration solved in {timing:.4e} s")

    # --- Check equilibration conditions ---
    if check_equilibration:
        # Check if reconstruction is in DRT
        flux_is_dg = equilibrator.V_flux.element.basix_element.discontinuous

        # Divergence condition
        div_condition_fulfilled = check_divergence_condition(
            equilibrator.list_flux[0],
            sigma_proj[0],
            rhs_proj[0],
        )

        if not div_condition_fulfilled:
            raise ValueError("Divergence conditions not fulfilled")

        # The jump condition
        if flux_is_dg:
            jump_condition_fulfilled = check_jump_condition(
                equilibrator.list_flux[0], sigma_proj[0]
            )

            if not jump_condition_fulfilled:
                raise ValueError("Jump conditions not fulfilled")

    return sigma_proj[0], equilibrator.list_flux[0]


if __name__ == "__main__":
    # --- Parameters ---
    # The mesh type
    mesh_type = MeshType.builtin

    # The considered equilibration strategy
    Equilibrator = FluxEqlbEV

    # The orders of the FE spaces
    order_prime = 1
    order_eqlb = 1

    # The boundary conditions
    bc_type = BCType.dirichlet

    # The mesh resolution
    sdisc_nelmt = 10

    # --- Execute calculation ---
    # Check input
    # TODO - Remove when EV is fixed
    if ((Equilibrator == FluxEqlbEV) and (mesh_type == MeshType.gmsh)) and (
        bc_type == BCType.neumann_inhom
    ):
        raise ValueError("EV with inhomogeneous flux BCs currently not working")

    # Create mesh
    if mesh_type == MeshType.builtin:
        domain, facet_tags, ds = create_unit_square_builtin(sdisc_nelmt)
    elif mesh_type == MeshType.gmsh:
        domain, facet_tags, ds = create_unit_square_gmesh(1 / sdisc_nelmt)
    else:
        raise ValueError("Unknown mesh type")

    # Solve primal problem
    degree_proj = 0 if (order_eqlb == 1) else None
    uh = solve_primal_problem(
        order_prime, domain, facet_tags, ds, bc_type, pdegree_rhs=degree_proj
    )

    # Solve equilibration
    sigma_proj, sigma_eqlb = equilibrate_flux(
        Equilibrator, order_eqlb, domain, facet_tags, bc_type, uh, True
    )

    # --- Export results to ParaView ---
    # Project flux into appropriate DG space
    V_dg_hdiv = dfem.VectorFunctionSpace(domain, ("DG", order_eqlb))
    v_dg_ref = dfem.VectorFunctionSpace(domain, ("DG", order_prime))

    sigma_ref = local_projection(
        v_dg_ref,
        [-ufl.grad(exact_solution(ufl)(ufl.SpatialCoordinate(domain)))],
        quadrature_degree=8,
    )

    if Equilibrator == FluxEqlbEV:
        sigma_eqlb_dg = local_projection(V_dg_hdiv, [sigma_eqlb])
    else:
        sigma_eqlb_dg = local_projection(V_dg_hdiv, [sigma_eqlb + sigma_proj])

    # Export primal solution
    uh.name = "uh"
    sigma_proj.name = "sigma_proj"
    sigma_eqlb_dg[0].name = "sigma_eqlb"
    sigma_ref[0].name = "sigma_ref"

    outfile = dolfinx.io.XDMFFile(MPI.COMM_WORLD, "demo_equilibration.xdmf", "w")
    outfile.write_mesh(domain)
    outfile.write_function(uh, 1)
    outfile.write_function(sigma_ref[0], 1)
    outfile.write_function(sigma_proj, 1)
    outfile.write_function(sigma_eqlb_dg[0], 1)
    outfile.close()
