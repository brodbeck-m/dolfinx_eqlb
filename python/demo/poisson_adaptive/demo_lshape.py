# Copyright (C) 2024 Maximilian Brodbeck
#
# This file is part of dolfinx_eqlb
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Demonstrate an adaptive solution procedure for a L-shape problem

Solve a Poisson problem

     div(sigma) = f with sigma = -grad(u)

on an L-shaped domain with homogenous Dirichlet BCs on a series of adaptively 
refined meshes. The error estimate is evaluated using the equilibrated flux 
while the spatial refinement is based on a Dörfler marking strategy. Convergence 
is reported with respect to the analytical solution (see e.g. [1]).

[1] Strang, G. and Fix, G., https://doi.org/10.1137/1.9780980232707, 2008
"""

import gmsh
import numpy as np
from numpy.typing import NDArray
from mpi4py import MPI
from petsc4py import PETSc
import time
import typing

from dolfinx import fem, io, mesh
import ufl

from dolfinx_eqlb.eqlb import FluxEqlbEV, FluxEqlbSE
from dolfinx_eqlb.lsolver import local_projection
from dolfinx_eqlb.eqlb.check_eqlb_conditions import (
    check_divergence_condition,
    check_jump_condition,
)


# --- The exact solution
def exact_solution(x: NDArray) -> NDArray:
    """Exact solution
    u_ext = r^(2/3) * sin(2 * theta / 3)

    Args:
        x: The spatial position

    Returns:
        The exact function
    """

    # Radius r and angle theta
    r = np.sqrt(x[0] * x[0] + x[1] * x[1])
    theta = np.arctan2(x[1], x[0]) + np.pi / 2.0

    # Exact solution
    values = r ** (2.0 / 3.0) * np.sin((2.0 / 3.0) * theta)

    # Correct values re-entrant corner
    values[
        np.where(
            np.logical_or(
                np.logical_and(np.isclose(x[0], 0.0, atol=1e-10), x[1] < 0.0),
                np.logical_and(np.isclose(x[1], 0.0, atol=1e-10), x[0] < 0.0),
            )
        )
    ] = 0.0
    return values


# --- Mesh generation
def create_lshape(h: float) -> mesh.Mesh:
    # Initialise gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.Verbosity", 2)

    # Name of the geometry
    gmsh.model.add("LShape")

    # Points
    list_pnts = [
        [0, 0],
        [0, -1],
        [1, -1],
        [1, 1],
        [-1, 1],
        [-1, 0],
    ]

    pnts = [gmsh.model.occ.add_point(pnt[0], pnt[1], 0.0) for pnt in list_pnts]

    # Bounding curves and 2D surface
    bfcts = [
        gmsh.model.occ.add_line(pnts[0], pnts[1]),
        gmsh.model.occ.add_line(pnts[1], pnts[2]),
        gmsh.model.occ.add_line(pnts[2], pnts[3]),
        gmsh.model.occ.add_line(pnts[3], pnts[4]),
        gmsh.model.occ.add_line(pnts[4], pnts[5]),
        gmsh.model.occ.add_line(pnts[5], pnts[0]),
    ]

    boundary = gmsh.model.occ.add_curve_loop(bfcts)
    surface = gmsh.model.occ.add_plane_surface([boundary])
    gmsh.model.occ.synchronize()

    # Set tag on boundaries and surface
    for i, bfct in enumerate(bfcts):
        gmsh.model.addPhysicalGroup(1, [bfct], i + 1)

    gmsh.model.addPhysicalGroup(2, [surface], 1)

    # Generate mesh
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
    gmsh.model.mesh.generate(2)

    domain_mesh, _, _ = io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)

    return domain_mesh


class AdaptiveLShape:
    """An adaptive LShape
    Create an initial mesh based and refines the mesh based on a the Doerfler strategy.
    """

    def __init__(self, h: int):
        """Constructor

        Args:
            h: The initial mesh size
        """

        # --- Initialise storage
        # The mesh counter
        self.refinement_level = 0

        # --- Create the initial mesh
        self.mesh = create_lshape(h)

        # The boundary markers
        self.boundary_markers = [
            (1, lambda x: np.isclose(x[0], -1)),
            (2, lambda x: np.logical_and(x[0] <= 0, np.isclose(x[1], 0))),
            (3, lambda x: np.logical_and(np.isclose(x[0], 0), x[1] <= 0)),
            (4, lambda x: np.isclose(x[1], -1)),
            (5, lambda x: np.isclose(x[0], 1)),
            (6, lambda x: np.isclose(x[1], 1)),
        ]

        # Set facet function and facet integrator
        self.facet_functions = None
        self.ds = None

        self.mark_boundary()

    # --- Generate the mesh ---
    def mark_boundary(self):
        """Marks the boundary based on the initially defined boundary markers"""

        facet_indices, facet_markers = [], []

        for marker, locator in self.boundary_markers:
            facets = mesh.locate_entities(self.mesh, 1, locator)
            facet_indices.append(facets)
            facet_markers.append(np.full(len(facets), marker))

        facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
        facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
        sorted_facets = np.argsort(facet_indices)
        self.facet_functions = mesh.meshtags(
            self.mesh, 1, facet_indices[sorted_facets], facet_markers[sorted_facets]
        )
        self.ds = ufl.Measure(
            "ds", domain=self.mesh, subdomain_data=self.facet_functions
        )

    def refine(
        self,
        doerfler: float,
        eta_h: typing.Optional[fem.Function] = None,
        outname: typing.Optional[str] = None,
    ):
        """Refine the mesh based on Doerflers marking strategy

        Args:
            doerfler: The Doerfler parameter
            eta_h:    The function of the cells error estimate
            outname:  The name of the output file for the mesh
                      (no output when not specified)
        """
        # The number of current mesh cells
        ncells = self.mesh.topology.index_map(2).size_global

        # The total error (squared!)
        eta_total = np.sum(eta_h.array)

        # Export marker to ParaView
        if outname is not None:
            V_out = fem.FunctionSpace(self.mesh, ("DG", 0))
            eta_h_out = fem.Function(V_out)
            eta_h_out.name = "eta_h"
            eta_h_out.x.array[:] = eta_h.array[:]

            outfile = io.XDMFFile(
                MPI.COMM_WORLD,
                outname + "-mesh" + str(self.refinement_level) + "_error.xdmf",
                "w",
            )
            outfile.write_mesh(self.mesh)
            outfile.write_function(eta_h_out, 0)
            outfile.close()

        # Refine the mesh
        if np.isclose(doerfler, 1.0):
            refined_mesh = mesh.refine(self.mesh)
        else:
            # Check input
            if eta_h is None:
                raise ValueError("Error marker required for adaptive refinement")

            # Cut-off
            cutoff = doerfler * eta_total

            # Sort cell contributions
            sorted_cells = np.argsort(eta_h.array)[::-1]

            # Create list of refined cells
            rolling_sum = 0.0
            breakpoint = ncells

            for i, e in enumerate(eta_h.array[sorted_cells]):
                rolling_sum += e
                if rolling_sum > cutoff:
                    breakpoint = i
                    break

            # List of refined cells
            refine_cells = np.array(
                np.sort(sorted_cells[0 : breakpoint + 1]), dtype=np.int32
            )

            # Refine mesh
            edges = mesh.compute_incident_entities(self.mesh, refine_cells, 2, 1)
            refined_mesh = mesh.refine(self.mesh, edges)

        # Update the mesh
        self.mesh = refined_mesh
        self.mark_boundary()

        # Update counter
        self.refinement_level += 1

        # Print infos
        print(
            "Refinement {} - Total error: {} - ncells: {}".format(
                self.refinement_level,
                np.sqrt(eta_total),
                self.mesh.topology.index_map(2).size_global,
            )
        )


# --- The primal problem
def solve(domain: AdaptiveLShape, degree: int) -> fem.Function:
    """Solves the Poisson problem based on lagrangian finite elements

    Args:
        domain:      The domain
        order_prime: The order of the FE space

    Returns:
        The approximate solution
    """

    # Set function space (primal problem)
    V_prime = fem.FunctionSpace(domain.mesh, ("CG", degree))
    uh = fem.Function(V_prime)

    # Set trial and test functions
    u = ufl.TrialFunction(V_prime)
    v = ufl.TestFunction(V_prime)

    # Equation system
    a = fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    l = fem.form(fem.Constant(domain.mesh, PETSc.ScalarType(0)) * v * ufl.dx)

    # Dirichlet boundary conditions
    uD = fem.Function(V_prime)
    uD.interpolate(exact_solution)

    dofs = fem.locate_dofs_topological(V_prime, 1, domain.facet_functions.indices)
    bc_essnt = [fem.dirichletbc(uD, dofs)]

    # Solve primal problem
    timing = 0

    timing -= time.perf_counter()
    A = fem.petsc.assemble_matrix(a, bcs=bc_essnt)
    A.assemble()

    L = fem.petsc.create_vector(l)
    fem.petsc.assemble_vector(L, l)
    fem.apply_lifting(L, [a], [bc_essnt])
    L.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem.set_bc(L, bc_essnt)

    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    pc.setHYPREType("boomeramg")

    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)

    solver.solve(L, uh.vector)
    timing += time.perf_counter()

    print(f"Primal problem solved in {timing:.4e} s")

    return uh


# --- The flux equilibration
def equilibrate(
    Equilibrator: typing.Union[FluxEqlbEV, FluxEqlbSE],
    domain: AdaptiveLShape,
    sigma_h: typing.Any,
    degree: int,
    check_equilibration: typing.Optional[bool] = True,
) -> fem.Function:
    """Equilibrate the flux

    The RHS is assumed to be the divergence of the exact
    flux (manufactured solution).

    Args:
        Equilibrator:        The equilibrator
        domain:              The domain
        sigma_h:             The flux calculated from the primal solution uh
        degree:              The order of RT elements used for equilibration
        check_equilibration: Id if equilibration conditions are checked

    Returns:
        The equilibrated flux
    """

    # Project flux and RHS into required DG space
    V_rhs_proj = fem.FunctionSpace(domain.mesh, ("DG", degree - 1))
    V_flux_proj = fem.VectorFunctionSpace(domain.mesh, ("DG", degree - 1))

    sigma_proj = local_projection(V_flux_proj, [sigma_h])
    rhs_proj = [fem.Function(V_rhs_proj)]

    # Initialise equilibrator
    equilibrator = Equilibrator(degree, domain.mesh, rhs_proj, sigma_proj)

    # Set BCs
    equilibrator.set_boundary_conditions([domain.facet_functions.indices], [[]])

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

    return equilibrator.list_flux[0]


# --- Estimate the error
def estimate(
    domain: AdaptiveLShape,
    delta_sigmaR: typing.Union[fem.Function, typing.Any],
) -> typing.Tuple[fem.Function, float]:
    """Estimates the error of a Poisson problem

    The estimate is calculated based on [1]. For the given problem the
    error due to data oscitation is zero.

    [1] Ern, A. and Vohralík, M., https://doi.org/10.1051/m2an/2018034, 2015

    Args:
        domain:       The domain
        delta_sigmaR: The difference of equilibrated and projected flux

    Returns:
        The cell-local error estimates
        The total error estimate
    """

    # Initialize storage of error
    V_e = fem.FunctionSpace(
        domain.mesh, ufl.FiniteElement("DG", domain.mesh.ufl_cell(), 0)
    )
    v = ufl.TestFunction(V_e)

    # Extract cell diameter
    form_eta = fem.form(ufl.inner(delta_sigmaR, delta_sigmaR) * v * ufl.dx)

    # Assemble errors
    L_eta = fem.petsc.create_vector(form_eta)

    fem.petsc.assemble_vector(L_eta, form_eta)

    return L_eta, np.sqrt(np.sum(L_eta.array))


# --- Post processing
def post_process(
    domain: AdaptiveLShape,
    uh: fem.Function,
    eta_h_tot: float,
    results: NDArray,
):
    # The function space
    degree_W = uh.function_space.element.basix_element.degree + 3
    W = fem.FunctionSpace(domain.mesh, ("CG", degree_W))

    # Calculate err = uh - uext in W
    uext_W = fem.Function(W)
    uext_W.interpolate(exact_solution)

    err_W = fem.Function(W)
    err_W.interpolate(uh)

    err_W.x.array[:] -= uext_W.x.array[:]

    # Evaluate H1 norm
    err_h1 = np.sqrt(
        fem.assemble_scalar(
            fem.form(ufl.inner(ufl.grad(err_W), ufl.grad(err_W)) * ufl.dx)
        )
    )

    # Store results
    results[domain.refinement_level, 0] = domain.mesh.topology.index_map(2).size_global
    results[domain.refinement_level, 1] = uh.function_space.dofmap.index_map.size_global
    results[domain.refinement_level, 2] = err_h1
    results[domain.refinement_level, 4] = eta_h_tot


if __name__ == "__main__":
    # --- Parameters ---
    # The primal problem
    order_prime = 1

    # The equilibration
    equilibrator = FluxEqlbSE
    order_eqlb = 1

    # The adaptive algorithm
    nref = 10
    doerfler = 0.5

    # --- Execute adaptive calculation ---
    # The domain
    domain = AdaptiveLShape(0.5)

    # Storage of results
    results = np.zeros((nref, 7))

    for n in range(0, nref):
        # Solve
        uh = solve(domain, order_prime)

        # Equilibrate the flux
        delta_sigmaR = equilibrate(
            equilibrator, domain, -ufl.grad(uh), order_eqlb, True
        )

        # Mark
        eta_h, eta_h_tot = estimate(domain, delta_sigmaR)

        # Post processing
        post_process(domain, uh, eta_h_tot, results)

        # Refine
        domain.refine(doerfler, eta_h, outname="LShape")

    # Export results
    results[1:, 3] = np.log(results[1:, 2] / results[:-1, 2]) / np.log(
        results[:-1, 1] / results[1:, 1]
    )
    results[1:, 5] = np.log(results[1:, 4] / results[:-1, 4]) / np.log(
        results[:-1, 1] / results[1:, 1]
    )
    results[:, 6] = results[:, 4] / results[:, 2]

    outname = "LShape_porder-{}_eorder-{}.csv".format(order_prime, order_eqlb)
    header = "nelmt, ndofs, erruh1, rateuh1, eeuh1, rateeeuh1, ieff"

    np.savetxt(outname, results, delimiter=",", header=header)
