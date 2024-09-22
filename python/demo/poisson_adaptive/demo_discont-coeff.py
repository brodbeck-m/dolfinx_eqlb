# Copyright (C) 2024 Maximilian Brodbeck
#
# This file is part of dolfinx_eqlb
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Demonstrate an adaptive solution procedure with discontinuous coefficients

Solve a Poisson problem

     div(sigma) = f with sigma = -kappa * grad(u)

on a squared domain with pure Dirichlet BCs on a series of adaptively 
refined meshes. The coeffitient kappa is continous but different within 
the four quadrants of the domain. The error estimate is evaluated using 
the equilibrated flux while the spatial refinement is based on a Dörfler 
marking strategy. Convergence is reported with respect to the analytical 
solution [1, 2].

[1] Kellogg, R. B., https://doi.org/10.1080/00036817408839086, 1975
[2] Reviere, B. and Wheeler M., https://doi.org/10.1016/S0898-1221(03)90086-1, 2003
"""

from enum import Enum
import numpy as np
from numpy.typing import NDArray
from mpi4py import MPI
from petsc4py import PETSc
import time
import typing

import dolfinx
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
class ExtSolType(Enum):
    riviere_5 = 0
    riviere_100 = 1
    kellogg_161 = 2


class ExactSolutionKellogg:
    """The exact solution following [1]:

    uext = r(x)^gamma * psi(sigma, x)

    [1] Kellogg, R. B., https://doi.org/10.1080/00036817408839086, 1975
    """

    def __init__(self, ratio_k: float) -> None:
        """Constructor

        Args:
            gamma: The parameter gamma
            sigma: The parameter sigma
        """
        # The solution parameters
        self.ratio_k = ratio_k

        if self.ratio_k == 161.4476387975881:
            gamma = 0.1
            sigma = -14.92256510455152
        else:
            raise ValueError("Ratio k not implemented")

        # The exact solution within the 4 quadrants
        def uext_q1(x, gamma, sigma):
            # The radius r
            r = np.sqrt(x[0] * x[0] + x[1] * x[1])

            # The angle theta
            theta = np.arctan2(x[1], x[0])

            # Auxiliaries
            h1 = np.cos((0.5 * np.pi - sigma) * gamma)

            return (r**gamma) * h1 * np.cos((theta - 0.25 * np.pi) * gamma)

        def uext_q2(x, gamma, sigma):
            # The radius r
            r = np.sqrt(x[0] * x[0] + x[1] * x[1])

            # The angle theta
            theta = theta = np.arctan2(x[1], x[0])

            # Auxiliaries
            h1 = np.cos(0.25 * np.pi * gamma)

            return (r**gamma) * h1 * np.cos((theta - np.pi + sigma) * gamma)

        def uext_q3(x, gamma, sigma):
            # The radius r
            r = np.sqrt(x[0] * x[0] + x[1] * x[1])

            # The angle theta
            # theta = np.arctan2(x[1], x[0])
            theta = np.pi + np.arctan(x[1] / x[0])

            # Auxiliaries
            h1 = np.cos(sigma * gamma)

            return (r**gamma) * h1 * np.cos((theta - 1.25 * np.pi) * gamma)

        def uext_q4(x, gamma, sigma):
            # The radius r
            r = np.sqrt(x[0] * x[0] + x[1] * x[1])

            # The angle theta
            theta = 2 * np.pi + np.arctan2(x[1], x[0])

            # Auxiliaries
            h1 = np.cos((0.25 * np.pi) * gamma)

            return (r**gamma) * h1 * np.cos((theta - 1.5 * np.pi - sigma) * gamma)

        self.list_uext = []
        self.list_uext.append(lambda x: uext_q1(x, gamma, sigma))
        self.list_uext.append(lambda x: uext_q2(x, gamma, sigma))
        self.list_uext.append(lambda x: uext_q3(x, gamma, sigma))
        self.list_uext.append(lambda x: uext_q4(x, gamma, sigma))

    def interpolate_to_function(
        self,
        f: dfem.Function,
        quadrants: typing.List[NDArray],
    ):
        """Interpolates the exact solution into a function

        Args:
            f:         The finite-element function
            quadrants: The cells within each quadrant of the squared domain
        """

        # Interpolate the exact solution
        for quadrant, uext in zip(quadrants, self.list_uext):
            f.interpolate(uext, cells=quadrant)


class ExactSolutionRiviere:
    """The exact solution following [1]:

    uext = r(x)^gamma * (a_i * cos(alpha * theta) + b_i * sin(alpha * theta))

    [1] Reviere, B. and Wheeler M., https://doi.org/10.1016/S0898-1221(03)90086-1, 2003
    """

    def __init__(self, ratio_k: float) -> None:
        """Constructor

        Args:
            a:       The parameters a
            b:       The parameters b
            alpha:   The parameter alpha
            ratio_k: The ratio of the diffusion coefficient
        """
        # The solution parameters
        self.ratio_k = ratio_k

        # Set other solution parameters
        if ratio_k == 5:
            alpha = 0.53544095
            a = [0.44721360, -0.74535599, -0.94411759, -2.40170264]
            b = [1.0, 2.33333333, 0.55555556, -0.48148148]
        elif ratio_k == 100:
            alpha = 0.12690207
            a = [0.1, -9.60396040, -0.48035487, 7.70156488]
            b = [1.0, 2.96039604, -0.88275659, -6.45646175]
        else:
            raise ValueError("Ratio k not implemented")

        # The exact solution within the 4 quadrants
        def uext_q1(x, ai, bi, alpha):
            # The radius r
            r = np.sqrt(x[0] * x[0] + x[1] * x[1])

            # The angle theta
            arg = alpha * np.arctan2(x[1], x[0])

            return (r**alpha) * (ai * np.sin(arg) + bi * np.cos(arg))

        def uext_q2(x, ai, bi, alpha):
            # The radius r
            r = np.sqrt(x[0] * x[0] + x[1] * x[1])

            # The angle theta
            arg = alpha * np.arctan2(x[1], x[0])

            return (r**alpha) * (ai * np.sin(arg) + bi * np.cos(arg))

        def uext_q3(x, ai, bi, alpha):
            # The radius r
            r = np.sqrt(x[0] * x[0] + x[1] * x[1])

            # The angle theta
            arg = alpha * (np.pi + np.arctan(x[1] / x[0]))

            return (r**alpha) * (ai * np.sin(arg) + bi * np.cos(arg))

        def uext_q4(x, ai, bi, alpha):
            # The radius r
            r = np.sqrt(x[0] * x[0] + x[1] * x[1])

            # The angle theta
            arg = alpha * (2 * np.pi + np.arctan2(x[1], x[0]))

            return (r**alpha) * (ai * np.sin(arg) + bi * np.cos(arg))

        self.list_uext = []
        self.list_uext.append(lambda x: uext_q1(x, a[0], b[0], alpha))
        self.list_uext.append(lambda x: uext_q2(x, a[1], b[1], alpha))
        self.list_uext.append(lambda x: uext_q3(x, a[2], b[2], alpha))
        self.list_uext.append(lambda x: uext_q4(x, a[3], b[3], alpha))

    def interpolate_to_function(
        self,
        f: dfem.Function,
        quadrants: typing.List[NDArray],
    ):
        """Interpolates the exact solution into a function

        Args:
            f:         The finite-element function
            quadrants: The cells within each quadrant of the squared domain
        """

        # Interpolate the exact solution
        for quadrant, uext in zip(quadrants, self.list_uext):
            f.interpolate(uext, cells=quadrant)


# --- Mesh generation
class AdaptiveSquare:
    """An adaptive squared domain
    Create an initial mesh based on a crossed mesh topology and
    refines the mesh based on a the Doerfler strategy.
    """

    def __init__(self, nelmt: int):
        """Constructor

        Args:
            nelmt: The number of elements in x and y direction
        """

        # --- Initialise storage
        # The mesh counter
        self.refinement_level = 0

        # --- Create the initial mesh
        # The mesh
        self.mesh = dmesh.create_rectangle(
            MPI.COMM_WORLD,
            [np.array([-1, -1]), np.array([1, 1])],
            [nelmt, nelmt],
            cell_type=dmesh.CellType.triangle,
            diagonal=dmesh.DiagonalType.crossed,
        )

        # The boundary markers
        self.boundary_markers = [
            (1, lambda x: np.isclose(x[0], -1)),
            (2, lambda x: np.isclose(x[1], -1)),
            (3, lambda x: np.isclose(x[0], 1)),
            (4, lambda x: np.isclose(x[1], 1)),
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
            facets = dmesh.locate_entities(self.mesh, 1, locator)
            facet_indices.append(facets)
            facet_markers.append(np.full(len(facets), marker))

        facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
        facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
        sorted_facets = np.argsort(facet_indices)
        self.facet_functions = dmesh.meshtags(
            self.mesh, 1, facet_indices[sorted_facets], facet_markers[sorted_facets]
        )
        self.ds = ufl.Measure(
            "ds", domain=self.mesh, subdomain_data=self.facet_functions
        )

    def refine(
        self,
        doerfler: float,
        eta_h: typing.Optional[dfem.Function] = None,
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
            V_out = dfem.FunctionSpace(self.mesh, ("DG", 0))
            eta_h_out = dfem.Function(V_out)
            eta_h_out.name = "eta_h"
            eta_h_out.x.array[:] = eta_h.array[:]

            outfile = dolfinx.io.XDMFFile(
                MPI.COMM_WORLD,
                outname + "-mesh" + str(self.refinement_level) + "_error.xdmf",
                "w",
            )
            outfile.write_mesh(self.mesh)
            outfile.write_function(eta_h_out, 0)
            outfile.close()

        # Refine the mesh
        if np.isclose(doerfler, 1.0):
            refined_mesh = dmesh.refine(self.mesh)
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
            edges = dmesh.compute_incident_entities(self.mesh, refine_cells, 2, 1)
            refined_mesh = dmesh.refine(self.mesh, edges)

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

    # --- The 4 mesh quadrants ---
    def quadrants_to_cellgroups(self) -> typing.List[NDArray]:
        """Get the cells within each quadrant of a squared domain

        Returns:
            The cell groups of the 4 quadrants
        """

        # The identifiers
        quadrants = []
        quadrants.append(lambda x: np.logical_and(x[0] >= 0, x[1] >= 0))
        quadrants.append(lambda x: np.logical_and(x[0] <= 0, x[1] >= 0))
        quadrants.append(lambda x: np.logical_and(x[0] <= 0, x[1] <= 0))
        quadrants.append(lambda x: np.logical_and(x[0] >= 0, x[1] <= 0))

        # Set the cell_groups
        cellgroups = []

        for quadrant in quadrants:
            cellgroups.append(
                dmesh.locate_entities(self.mesh, self.mesh.topology.dim, quadrant)
            )

        return cellgroups


# --- The primal problem
def solve(
    domain: AdaptiveSquare,
    uext: typing.Union[ExactSolutionKellogg, ExactSolutionRiviere],
    degree: int,
) -> typing.Tuple[dfem.Function, typing.Any]:
    """Solves the Poisson problem based on lagrangian finite elements

    Args:
        domain:      The domain
        uext:        The exact solution
        order_prime: The order of the FE space

    Returns:
        The approximate solution,
        The diffusion coefficient
    """

    # The subdomains of the mesh
    quadrants = domain.quadrants_to_cellgroups()

    # Set function space (primal problem)
    V_prime = dfem.FunctionSpace(domain.mesh, ("CG", degree))
    V_k = dfem.FunctionSpace(domain.mesh, ("DG", 0))

    uh = dfem.Function(V_prime)

    # Set trial and test functions
    u = ufl.TrialFunction(V_prime)
    v = ufl.TestFunction(V_prime)

    # The diffusion coefficient
    k = dfem.Function(V_k)
    k.x.array[:] = 1.0
    k.x.array[quadrants[0]] = uext.ratio_k
    k.x.array[quadrants[2]] = uext.ratio_k

    # Equation system
    a = dfem.form(ufl.inner(k * ufl.grad(u), ufl.grad(v)) * ufl.dx)
    l = dfem.form(dfem.Constant(domain.mesh, PETSc.ScalarType(0)) * v * ufl.dx)

    # Dirichlet boundary conditions
    uD = dfem.Function(V_prime)
    uext.interpolate_to_function(uD, quadrants)

    dofs = dfem.locate_dofs_topological(V_prime, 1, domain.facet_functions.indices)
    bc_essnt = [dfem.dirichletbc(uD, dofs)]

    # Solve primal problem
    timing = 0

    timing -= time.perf_counter()
    A = dfem.petsc.assemble_matrix(a, bcs=bc_essnt)
    A.assemble()

    L = dfem.petsc.create_vector(l)
    dfem.petsc.assemble_vector(L, l)
    dfem.apply_lifting(L, [a], [bc_essnt])
    L.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    dfem.set_bc(L, bc_essnt)

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

    return uh, k


# --- The flux equilibration
def equilibrate(
    Equilibrator: typing.Union[FluxEqlbEV, FluxEqlbSE],
    domain: AdaptiveSquare,
    sigma_h: typing.Any,
    degree: int,
    check_equilibration: typing.Optional[bool] = True,
) -> dfem.Function:
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
        The projected flux,
        The equilibrated flux
    """

    # Project flux and RHS into required DG space
    V_rhs_proj = dfem.FunctionSpace(domain.mesh, ("DG", degree - 1))
    V_flux_proj = dfem.VectorFunctionSpace(domain.mesh, ("DG", degree - 1))

    sigma_proj = local_projection(V_flux_proj, [sigma_h])
    rhs_proj = [dfem.Function(V_rhs_proj)]

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
    domain: AdaptiveSquare,
    delta_sigmaR: typing.Union[dfem.Function, typing.Any],
    k: dfem.Function,
) -> typing.Tuple[dfem.Function, float]:
    """Estimates the error of a Poisson problem

    The estimate is calculated based on [???]. For the given problem the
    error due to data oscitation is zero.

    Args:
        delta_sigmaR: The difference of equilibrated and projected flux

    Returns:
        The cell-local error estimates,
        The total error estimate
    """

    # Initialize storage of error
    V_e = dfem.FunctionSpace(
        domain.mesh, ufl.FiniteElement("DG", domain.mesh.ufl_cell(), 0)
    )
    v = ufl.TestFunction(V_e)

    # Extract cell diameter
    form_eta = dfem.form((1 / k) * ufl.dot(delta_sigmaR, delta_sigmaR) * v * ufl.dx)

    # Assemble errors
    L_eta = dfem.petsc.create_vector(form_eta)

    dfem.petsc.assemble_vector(L_eta, form_eta)

    return L_eta, np.sqrt(np.sum(L_eta.array))


# --- Post processing
def post_process(
    domain: AdaptiveSquare,
    k: dfem.Function,
    uext: typing.Union[ExactSolutionKellogg, ExactSolutionRiviere],
    uh: dfem.Function,
    eta_h_tot: float,
    results: NDArray,
):
    # The function space
    degree_W = uh.function_space.element.basix_element.degree + 2
    W = dfem.FunctionSpace(domain.mesh, ("CG", degree_W))

    # Calculate err = uh - uext in W
    uext_W = dfem.Function(W)
    uext.interpolate_to_function(uext_W, domain.quadrants_to_cellgroups())

    err_W = dfem.Function(W)
    err_W.interpolate(uh)

    err_W.x.array[:] -= uext_W.x.array[:]

    # Evaluate H1 norm
    err_h1 = np.sqrt(
        dfem.assemble_scalar(
            dfem.form(k * ufl.inner(ufl.grad(err_W), ufl.grad(err_W)) * ufl.dx)
        )
    )

    # Store results
    id = domain.refinement_level
    results[id, 0] = domain.mesh.topology.index_map(2).size_global
    results[id, 1] = uh.function_space.dofmap.index_map.size_global
    results[id, 2] = err_h1
    results[id, 4] = eta_h_tot


if __name__ == "__main__":
    # --- Parameters ---
    # The primal problem
    order_prime = 1

    # The equilibration
    equilibrator = FluxEqlbSE
    order_eqlb = 2

    # The adaptive algorithm
    nref = 15
    doerfler = 0.5

    # The exact solution
    extsol_type = ExtSolType.riviere_5

    # --- Execute adaptive calculation ---
    # The exact solution
    if extsol_type == ExtSolType.riviere_5:
        uext = ExactSolutionRiviere(5)
        outname_base = "Riviere5"
    elif extsol_type == ExtSolType.riviere_100:
        uext = ExactSolutionRiviere(100)
        outname_base = "Riviere100"
    elif extsol_type == ExtSolType.kellogg_161:
        uext = ExactSolutionKellogg(161.4476387975881)
        outname_base = "Kellogg161"

    # The domain
    domain = AdaptiveSquare(2)

    # Storage of results
    results = np.zeros((nref, 7))

    for n in range(0, nref):
        # Solve
        uh, k = solve(domain, uext, order_prime)

        # Equilibrate the flux
        delta_sigmaR, _ = equilibrate(
            equilibrator, domain, -k * ufl.grad(uh), order_eqlb, False
        )

        # Mark
        eta_h, eta_h_tot = estimate(domain, delta_sigmaR, k)

        # Post processing
        post_process(domain, k, uext, uh, eta_h_tot, results)

        # Refine
        domain.refine(doerfler, eta_h, outname=outname_base)

    # Export results
    results[1:, 3] = np.log(results[1:, 2] / results[:-1, 2]) / np.log(
        results[:-1, 1] / results[1:, 1]
    )
    results[1:, 5] = np.log(results[1:, 4] / results[:-1, 4]) / np.log(
        results[:-1, 1] / results[1:, 1]
    )
    results[:, 6] = results[:, 4] / results[:, 2]

    outname = outname_base + "_porder-{}_eorder-{}.csv".format(order_prime, order_eqlb)
    header = "nelmt, ndofs, erruh1, rateuh1, eeuh1, rateeeuh1, ieff"

    np.savetxt(outname, results, delimiter=",", header=header)
