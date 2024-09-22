# Copyright (C) 2024 Maximilian Brodbeck
#
# This file is part of dolfinx_eqlb
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Linear elasticity

Collection of routines for a pre-defined manufactured solution, the solution of the 
primal problem as well as the equilibration of the stress tensor.
"""

import numpy as np
import typing

from dolfinx import fem
import ufl

from dolfinx_eqlb.lsolver import local_projection
from dolfinx_eqlb.eqlb import fluxbc, FluxEqlbSE

from utils import Geometry


# --- Definition of manufactured solution
def exact_solution(x: typing.Any):
    """Exact solution
    u_ext = [sin(2*pi * x) * cos(2*pi * y), -cos(2*pi * x) * sin(2*pi * y)]

    Args:
        The spatial position x

    Returns:
        The exact function as ufl-expression
    """
    return ufl.as_vector(
        [
            ufl.sin(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1]),
            -ufl.cos(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1]),
        ]
    )


def exact_stress_linelast(x: typing.Any, gdim: typing.Optional[int] = 2):
    """Exact stress

    sigma_ext = 2 * epsilon(u_ext) + pi_1 * div(u_ext) * I

    Args:
        x:    The spatial position x
        gdim: The spatial dimension

    Returns:
        The exact stress at position x as ufl expression
    """

    # The exact displacement
    u_ext = exact_solution(x)

    return 2 * ufl.sym(ufl.grad(u_ext)) + ufl.div(u_ext) * ufl.Identity(gdim)


# --- Solution routines
def solve_primal_problem(
    V_prime: fem.FunctionSpace,
    geometry: Geometry,
    bc_id_neumann: typing.List[int],
    bc_id_dirichlet: typing.List[int],
    ufl_rhs: typing.Any,
    ufl_neumann: typing.List[typing.Any],
    u_dirichlet: typing.List[fem.Function],
    degree_projection: typing.Optional[int] = None,
) -> typing.Tuple[fem.Function, typing.List[fem.Function]]:
    """Solves linear elasticity based on lagrangian finite elements

    Args:
        V_prime:           The function space of the primal problem
        geometry:          The geometry
        bc_id_neumann:     List of boundary ids for Neumann BCs
        bc_id_dirichlet:   List of boundary ids for Dirichlet BCs
        ufl_rhs:           The RHS of the primal problem
        ufl_neumann:       The Neumann BCs
        u_dirichlet:       The Dirichlet BCs
        degree_projection: Degree of projected flux

    Returns:
        The primal solution,
        The rows of the projected stress tensor
    """
    # Check input
    if len(bc_id_dirichlet) == 0:
        raise ValueError("Pure neumann problem not supported!")

    if len(list(set(bc_id_dirichlet) & set(bc_id_neumann))) > 0:
        raise ValueError("Overlapping boundary data!")

    # The spatial dimension
    gdim = geometry.mesh.topology.dim

    # Set variational form
    u = ufl.TrialFunction(V_prime)
    v = ufl.TestFunction(V_prime)

    sigma = 2 * ufl.sym(ufl.grad(u)) + ufl.div(u) * ufl.Identity(gdim)

    a_prime = ufl.inner(sigma, ufl.sym(ufl.grad(v))) * ufl.dx
    l_prime = ufl.inner(ufl_rhs, v) * ufl.dx

    # Set dirichlet boundary conditions
    bcs_esnt = []

    for i, id in enumerate(bc_id_dirichlet):
        fcts = geometry.facet_function.indices[geometry.facet_function.values == id]
        dofs = fem.locate_dofs_topological(V_prime, 1, fcts)
        bcs_esnt.append(fem.dirichletbc(u_dirichlet[i], dofs))

    # Set neumann boundary conditions
    for i, id in enumerate(bc_id_neumann):
        l_prime += ufl.inner(ufl_neumann[i], v) * geometry.ds(id)

    # Solve problem
    solveoptions = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ksp_rtol": 1e-12,
        "ksp_atol": 1e-12,
    }
    problem_prime = fem.petsc.LinearProblem(
        a_prime, l_prime, bcs_esnt, petsc_options=solveoptions
    )
    u_prime = problem_prime.solve()

    # Project flux
    if degree_projection is None:
        degree_projection = V_prime.element.basix_element.degree - 1

    sigma_h = -2 * ufl.sym(ufl.grad(u_prime)) - ufl.div(u_prime) * ufl.Identity(gdim)

    V_flux = fem.VectorFunctionSpace(geometry.mesh, ("DG", degree_projection))
    sig_proj = local_projection(
        V_flux,
        [
            ufl.as_vector([sigma_h[0, 0], sigma_h[0, 1]]),
            ufl.as_vector([sigma_h[1, 0], sigma_h[1, 1]]),
        ],
    )

    return u_prime, sig_proj


def equilibrate_stresses(
    degree_flux: int,
    geometry: Geometry,
    sig_proj: typing.List[fem.Function],
    rhs_proj: typing.List[fem.Function],
    bc_id_neumann: typing.List[typing.List[int]],
    bc_id_dirichlet: typing.List[typing.List[int]],
    flux_neumann: typing.List[typing.Any],
    neumann_projection: typing.List[bool],
) -> typing.Tuple[typing.List[fem.Function], typing.List[fem.Function]]:
    """Equilibrate the stress tensor

    Args:
        degree_flux:        Degree of flux space
        geometry:           The geometry
        sig_proj:           The projected fluxes
        rhs_proj:           The projected RHS
        bc_id_neumann:      The boundary ids for Neumann BCs
        bc_id_dirichlet:    The boundary ids for Dirichlet BCs
        flux_neumann:       The Neumann BCs
        neumann_projection: Ids indicating if the Neumann BCs require projection

    Returns:
        The equilibrated fluxes,
        The flux values on the Neumann boundary
    """

    # Extract facet markers
    fct_values = geometry.facet_function.values

    # Set equilibrator
    equilibrator = FluxEqlbSE(degree_flux, geometry.mesh, rhs_proj, sig_proj, True)

    # Mark dirichlet facets of primal problem
    fct_bcesnt_primal = []

    for list_bc_id in bc_id_dirichlet:
        if len(list_bc_id) > 0:
            fct_numpy = np.array([], dtype=np.int32)

            for id_esnt in list_bc_id:
                list_fcts = geometry.facet_function.indices[fct_values == id_esnt]
                fct_numpy = np.concatenate((fct_numpy, list_fcts))

            fct_bcesnt_primal.append(fct_numpy)

    # Set Neumann conditions on flux space
    bc_esnt_flux = []

    for i in range(0, len(bc_id_neumann)):
        if len(bc_id_neumann[i]) > 0:
            list_flux_bc = []

            for j, id_flux in enumerate(bc_id_neumann[i]):
                list_fcts = geometry.facet_function.indices[fct_values == id_flux]
                list_flux_bc.append(
                    fluxbc(
                        -flux_neumann[i][j],
                        list_fcts,
                        equilibrator.V_flux,
                        neumann_projection[i][j],
                    )
                )

            bc_esnt_flux.append(list_flux_bc)
        else:
            bc_esnt_flux.append([])

    # Set equilibrator
    equilibrator.set_boundary_conditions(fct_bcesnt_primal, bc_esnt_flux)

    # Step 1: Equilibrate stress tensor without symmetry
    equilibrator.equilibrate_fluxes()

    return equilibrator.list_flux, equilibrator.list_bfunctions
