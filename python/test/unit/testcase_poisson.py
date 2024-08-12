# Copyright (C) 2024 Maximilian Brodbeck
#
# This file is part of dolfinx_eqlb
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

# --- Includes ---
import numpy as np
import typing

import dolfinx.fem as dfem
import ufl

from dolfinx_eqlb.lsolver import local_projection
from dolfinx_eqlb.eqlb import fluxbc

from utils import Geometry

"""
Setup variable test-cases for the poisson problem

Supported variants:
    - manufactured solution based on u_ext = sin(2*pi * x) * cos(2*pi * y)
    - arbitrary right-hand side
"""


# --- The exact solution
def exact_solution(x):
    """Exact solution (Poisson)
    u_ext = sin(2*pi * x) * cos(2*pi * y)

    Args:
        x (ufl.SpatialCoordinate): The position x
    Returns:
        The exact function as ufl-expression
    """
    return ufl.sin(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1])


def exact_flux(x):
    """Exact flux (Poisson)
    flux_ext = -Grad[sin(2*pi * x) * cos(2*pi * y)]

    Args:
        x (ufl.SpatialCoordinate): The position x
    Returns:
        The exact flux as ufl expression
    """
    return -ufl.grad(exact_solution(x))


# --- Solution routines
def solve_primal_problem(
    V_prime: dfem.FunctionSpace,
    geometry: Geometry,
    bc_id_neumann: typing.List[int],
    bc_id_dirichlet: typing.List[int],
    ufl_rhs: typing.Any,
    ufl_neumann: typing.List[typing.Any],
    u_dirichlet: typing.List[dfem.Function],
    degree_projection: int = -1,
):
    """Solves a poisson problem based on lagrangian finite elements

    Args:
        V_prime (dolfinx.FunctionSpace):      The function space of the primal problem
        geometry (Geometry):                  The geometry of the domain
        bc_id_neumann (List[int]):            List of boundary ids for neumann BCs
        bc_id_dirichlet (List[int]):          List of boundary ids for dirichlet BCs
        ufl_rhs (ufl):                        The right-hand side of the primal problem
        ufl_neumann (List[ufl]):              List of neumann boundary conditions
        u_dirichlet (List[dolfinx.Function]): List of dirichlet boundary conditions
        degree_projection (int):              Degree of projected flux
    """
    # Check input
    if len(bc_id_dirichlet) == 0:
        raise ValueError("Pure neumann problem not supported!")

    if len(list(set(bc_id_dirichlet) & set(bc_id_neumann))) > 0:
        raise ValueError("Overlapping boundary data!")

    # Set variational form
    u = ufl.TrialFunction(V_prime)
    v = ufl.TestFunction(V_prime)

    a_prime = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    l_prime = ufl.inner(ufl_rhs, v) * ufl.dx

    # Set dirichlet boundary conditions
    bcs_esnt = []

    for i, id in enumerate(bc_id_dirichlet):
        fcts = geometry.facet_function.indices[geometry.facet_function.values == id]
        dofs = dfem.locate_dofs_topological(V_prime, 1, fcts)
        bcs_esnt.append(dfem.dirichletbc(u_dirichlet[i], dofs))

    # Set neumann boundary conditions
    for i, id in enumerate(bc_id_neumann):
        l_prime -= ufl.inner(ufl_neumann[i], v) * geometry.ds(id)

    # Solve problem
    solveoptions = {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "pc_hypre_type": "boomeramg",
        "ksp_rtol": 1e-10,
        "ksp_atol": 1e-10,
    }
    problem_prime = dfem.petsc.LinearProblem(
        a_prime, l_prime, bcs_esnt, petsc_options=solveoptions
    )
    u_prime = problem_prime.solve()

    # Project flux
    if degree_projection < 0:
        degree_projection = V_prime.element.basix_element.degree - 1

    V_flux = dfem.VectorFunctionSpace(geometry.mesh, ("DG", degree_projection))
    sig_proj = local_projection(V_flux, [-ufl.grad(u_prime)])[0]

    return u_prime, sig_proj


def equilibrate_fluxes(
    Equilibrator: typing.Any,
    degree_flux: int,
    geometry: Geometry,
    sig_proj: typing.List[dfem.Function],
    rhs_proj: typing.List[dfem.Function],
    bc_id_neumann: typing.List[typing.List[int]],
    bc_id_dirichlet: typing.List[typing.List[int]],
    flux_neumann: typing.List[typing.Any],
    neumann_projection: typing.List[bool],
):
    """Equilibrates the fluxes of the primal problem

    Args:
        Equilibrator (equilibration.FluxEquilibrator): The equilibrator object
        degree_flux (int):                             Degree of flux space
        geometry (Geometry):                           The geometry of the domain
        sig_proj (List[dfem.Function]):                List of projected fluxes
        rhs_proj (List[dfem.Function]):                List of projected right-hand sides
        bc_id_neumann (List[List[int]]):               List of boundary ids for neumann BCs
        bc_id_dirichlet (List[List[int]]):             List of boundary ids for dirichlet BCs
        flux_neumann (List[List[ufl]]):                List of neumann boundary conditions
        neumann_projection (List[List[bool]]):         List of booleans indicating wether the
                                                       neumann BCs require projection

    Returns:
        List[dfem.Function]: List of equilibrated fluxes
        List[dfem.Function]: List boundary-functions
                             (functions, containing the correct boundary values)
    """
    # Extract facet markers
    fct_values = geometry.facet_function.values

    # Set equilibrator
    equilibrator = Equilibrator(degree_flux, geometry.mesh, rhs_proj, sig_proj)

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
                        flux_neumann[i][j],
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

    # Solve equilibration
    equilibrator.equilibrate_fluxes()

    return equilibrator.list_flux, equilibrator.list_bfunctions
