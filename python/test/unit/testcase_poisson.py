# Copyright (C) 2024 Maximilian Brodbeck
#
# This file is part of dolfinx_eqlb
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Poisson problem"""

import numpy as np
import typing

from dolfinx import fem
import dolfinx.fem.petsc
import ufl

from dolfinx_eqlb import eqlb, lsolver

# from dolfinx_eqlb.lsolver import local_projection
# from dolfinx_eqlb.eqlb import fluxbc, FluxEqlbEV, FluxEqlbSE

from utils import Domain


# --- The exact solution
def exact_solution(x: typing.Any):
    """Exact solution (Poisson)
    u_ext = sin(2*pi * x) * cos(2*pi * y)

    Args:
        x: The spatial position x

    Returns:
        The exact function as ufl-expression
    """

    return ufl.sin(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1])


def exact_flux(x: typing.Any):
    """Exact flux
    flux_ext = -Grad[sin(2*pi * x) * cos(2*pi * y)]

    Args:
        x: The spatial position x

    Returns:
        The exact flux at position x as ufl expression
    """

    return -ufl.grad(exact_solution(x))


# --- Solution routines
def solve_primal_problem(
    V_prime: fem.FunctionSpace,
    domain: Domain,
    bc_id_neumann: typing.List[int],
    bc_id_dirichlet: typing.List[int],
    ufl_rhs: typing.Any,
    ufl_neumann: typing.List[typing.Any],
    u_dirichlet: typing.List[fem.Function],
    degree_projection: typing.Optional[int] = None,
) -> typing.Tuple[fem.Function, fem.Function]:
    """Solves a Poisson problem based on lagrangian finite elements

    Args:
        V_prime:           The function space of the primal problem
        domain:            The domain
        bc_id_neumann:     List of boundary ids for Neumann BCs
        bc_id_dirichlet:   List of boundary ids for Dirichlet BCs
        ufl_rhs:           The RHS of the primal problem
        ufl_neumann:       The Neumann BCs
        u_dirichlet:       The Dirichlet BCs
        degree_projection: Degree of projected flux

    Returns:
        The primal solution,
        The projected flux
    """

    # Check input
    if len(bc_id_dirichlet) == 0:
        raise ValueError("Pure neumann problem not supported!")

    if len(list(set(bc_id_dirichlet) & set(bc_id_neumann))) > 0:
        raise ValueError("Overlapping boundary data!")

    # Required connectivity
    domain.mesh.topology.create_connectivity(1, 2)

    # The spatial dimension
    gdim = domain.mesh.geometry.dim

    # Set variational form
    u = ufl.TrialFunction(V_prime)
    v = ufl.TestFunction(V_prime)

    a_prime = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    l_prime = ufl.inner(ufl_rhs, v) * ufl.dx

    # Set dirichlet boundary conditions
    bcs_esnt = []

    for i, id in enumerate(bc_id_dirichlet):
        fcts = domain.facet_function.indices[domain.facet_function.values == id]
        dofs = fem.locate_dofs_topological(V_prime, 1, fcts)
        bcs_esnt.append(fem.dirichletbc(u_dirichlet[i], dofs))

    # Set neumann boundary conditions
    for i, id in enumerate(bc_id_neumann):
        l_prime -= ufl.inner(ufl_neumann[i], v) * domain.ds(id)

    # Solve problem
    solveoptions = {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "pc_hypre_type": "boomeramg",
        "ksp_rtol": 1e-10,
        "ksp_atol": 1e-10,
    }
    problem_prime = fem.petsc.LinearProblem(
        a_prime, l_prime, bcs_esnt, petsc_options=solveoptions
    )
    u_prime = problem_prime.solve()

    # Project flux
    if degree_projection is None:
        degree_projection = V_prime.element.basix_element.degree - 1

    V_flux = fem.functionspace(domain.mesh, ("DG", degree_projection, (gdim,)))
    sig_proj = lsolver.local_projection(V_flux, [-ufl.grad(u_prime)])[0]

    return u_prime, sig_proj


def equilibrate_fluxes(
    equilibrator: typing.Type[eqlb.basics.EquilibratorMetaClass],
    domain: Domain,
    sig_proj: typing.List[fem.Function],
    rhs_proj: typing.List[fem.Function],
    bc_id_neumann: typing.List[typing.List[int]],
    bc_id_dirichlet: typing.List[typing.List[int]],
    flux_neumann: typing.List[typing.Any],
    neumann_projection: typing.List[bool],
) -> typing.Tuple[typing.List[fem.Function], typing.List[fem.Function]]:
    """Equilibrate the fluxes

    Args:
        Equilibrator        The equilibrator class
        domain:             The domain
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
    fct_values = domain.facet_function.values

    # Set equilibrator
    if equilibrator._strategy == eqlb.EqlbStrategy.constrained_minimisation:
        from dolfinx_eqlb.eqlb.constrained_minimisation import equilibrationproblem
    elif equilibrator._strategy == eqlb.EqlbStrategy.semi_explicit:
        raise ValueError("Semi-explicit equilibration not supported.")
    else:
        raise ValueError("Invalid equilibration strategy.")

    problem = equilibrationproblem(equilibrator, domain.mesh, rhs_proj, sig_proj)

    # Mark dirichlet facets of primal problem
    fct_bcesnt_primal = []

    for list_bc_id in bc_id_dirichlet:
        if len(list_bc_id) > 0:
            fct_numpy = np.array([], dtype=np.int32)

            for id_esnt in list_bc_id:
                list_fcts = domain.facet_function.indices[fct_values == id_esnt]
                fct_numpy = np.concatenate((fct_numpy, list_fcts))

            fct_bcesnt_primal.append(fct_numpy)

    # Set Neumann conditions on flux space
    bc_esnt_flux = []

    for i in range(0, len(bc_id_neumann)):
        if len(bc_id_neumann[i]) > 0:
            list_flux_bc = []

            for j, id_flux in enumerate(bc_id_neumann[i]):
                list_fcts = domain.facet_function.indices[fct_values == id_flux]
                list_flux_bc.append(
                    eqlb.fluxbc(
                        flux_neumann[i][j],
                        list_fcts,
                        problem._V,
                        neumann_projection[i][j],
                    )
                )

            bc_esnt_flux.append(list_flux_bc)
        else:
            bc_esnt_flux.append([])

    # Set equilibrator
    problem.create_boundary_data(fct_bcesnt_primal, bc_esnt_flux)

    # Solve equilibration
    problem.solve()

    return problem._fluxes, problem._boundary_values
