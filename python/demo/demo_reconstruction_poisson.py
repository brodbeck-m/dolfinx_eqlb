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

import dolfinx
import dolfinx.fem as dfem
import dolfinx.mesh as dmesh
import ufl

from dolfinx_eqlb.eqlb import fluxbc, FluxEqlbEV, FluxEqlbSE
from dolfinx_eqlb.lsolver import local_projection


# --- Parameters ---
# The considered equilibration strategy
Equilibrator = FluxEqlbSE

# The orders of the FE spaces
elmt_order_prime = 1
elmt_order_eqlb = 2

# The mesh resolution
sdisc_nelmt = 20


# --- The exact solution
def exact_solution(pkt):
    return lambda x: pkt.sin(2 * pkt.pi * x[0]) * pkt.cos(2 * pkt.pi * x[1])


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


def solve_primal_problem(elmt_order_prime, domain, facet_tags, ds):
    # Set function space (primal problem)
    V_prime = dfem.FunctionSpace(domain, ("CG", elmt_order_prime))

    # Set trial and test functions
    u = ufl.TrialFunction(V_prime)
    v = ufl.TestFunction(V_prime)

    # Set source term
    x = ufl.SpatialCoordinate(domain)
    f = -ufl.div(ufl.grad(exact_solution(ufl)(x)))

    # Equation system
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    l = f * v * ufl.dx

    # Neumann boundary conditions
    normal = ufl.FacetNormal(domain)
    l += ufl.inner(ufl.grad(exact_solution(ufl)(x)), normal) * v * ds(1)
    l += ufl.inner(ufl.grad(exact_solution(ufl)(x)), normal) * v * ds(3)

    # Dirichlet boundary conditions
    uD = dfem.Function(V_prime)
    uD.interpolate(exact_solution(np))

    fcts_essnt = facet_tags.indices[
        np.logical_or(facet_tags.values == 2, facet_tags.values == 4)
    ]
    dofs_essnt = dfem.locate_dofs_topological(V_prime, 1, fcts_essnt)
    bc_essnt = [dfem.dirichletbc(uD, dofs_essnt)]

    # Solve primal problem
    problem = dfem.petsc.LinearProblem(
        a,
        l,
        bcs=bc_essnt,
        petsc_options={"ksp_type": "cg", "ksp_rtol": 1e-10, "ksp_atol": 1e-12},
    )
    uh_prime = problem.solve()

    return uh_prime


# --- The flux equilibration
def equilibrate_flux(
    Equilibrator, elmt_order_prime, elmt_order_eqlb, domain, facet_tags, uh_prime
):
    # Set source term
    x = ufl.SpatialCoordinate(domain)
    f = -ufl.div(ufl.grad(exact_solution(ufl)(x)))

    # Project flux and RHS into required DG space
    V_rhs_proj = dfem.FunctionSpace(domain, ("DG", elmt_order_eqlb - 1))
    # (elmt_order_eqlb - 1 would be sufficient but not implemented for semi-explicit eqlb.)
    V_flux_proj = dfem.VectorFunctionSpace(domain, ("DG", elmt_order_eqlb - 1))

    sigma_proj = local_projection(V_flux_proj, [-ufl.grad(uh_prime)])
    rhs_proj = local_projection(V_rhs_proj, [f])

    # Initialise equilibrator
    equilibrator = Equilibrator(elmt_order_eqlb, domain, rhs_proj, sigma_proj)

    # Get facets on essential boundary surfaces
    fcts_essnt = facet_tags.indices[
        np.logical_or(facet_tags.values == 2, facet_tags.values == 4)
    ]

    # Specify flux boundary conditions
    bc_dual = []

    bc_dual.append(
        fluxbc(
            ufl.grad(exact_solution(ufl)(x))[0],
            facet_tags.indices[facet_tags.values == 1],
            equilibrator.V_flux,
            requires_projection=True,
            quadrature_degree=3 * elmt_order_eqlb,
        )
    )

    bc_dual.append(
        fluxbc(
            -ufl.grad(exact_solution(ufl)(x))[0],
            facet_tags.indices[facet_tags.values == 3],
            equilibrator.V_flux,
            requires_projection=True,
            quadrature_degree=3 * elmt_order_eqlb,
        )
    )

    equilibrator.set_boundary_conditions(
        [fcts_essnt], [bc_dual], quadrature_degree=3 * elmt_order_eqlb
    )

    # Solve equilibration
    equilibrator.equilibrate_fluxes()

    return sigma_proj[0], equilibrator.list_flux[0]


# --- Execute calculation ---
# Create mesh
domain, facet_tags, ds = create_unit_square_mesh(sdisc_nelmt)

# Solve primal problem
uh_prime = solve_primal_problem(elmt_order_prime, domain, facet_tags, ds)

# Solve equilibration
sigma_proj, sigma_eqlb = equilibrate_flux(
    Equilibrator, elmt_order_prime, elmt_order_eqlb, domain, facet_tags, uh_prime
)

# --- Export results to ParaView ---
# Project flux into appropriate DG space
V_dg_hdiv = dfem.VectorFunctionSpace(domain, ("DG", elmt_order_eqlb))
v_dg_ref = dfem.VectorFunctionSpace(domain, ("DG", elmt_order_prime))

sigma_ref = local_projection(
    v_dg_ref,
    [-ufl.grad(exact_solution(ufl)(ufl.SpatialCoordinate(domain)))],
    quadrature_degree=8,
)

if Equilibrator == FluxEqlbEV:
    sigma_eqlb_dg = local_projection(V_dg_hdiv, sigma_eqlb)
else:
    sigma_eqlb_dg = local_projection(V_dg_hdiv, [sigma_eqlb + sigma_proj])

# Export primal solution
uh_prime.name = "u"
sigma_proj.name = "sigma_proj"
sigma_eqlb_dg[0].name = "sigma_eqlb"
sigma_ref[0].name = "sigma_ref"

outfile = dolfinx.io.XDMFFile(MPI.COMM_WORLD, "demo_equilibration.xdmf", "w")
outfile.write_mesh(domain)
outfile.write_function(uh_prime, 1)
outfile.write_function(sigma_ref[0], 1)
outfile.write_function(sigma_proj, 1)
outfile.write_function(sigma_eqlb_dg[0], 1)
