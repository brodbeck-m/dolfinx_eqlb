# Copyright (C) 2024 Maximilian Brodbeck
#
# This file is part of dolfinx_eqlb
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""General set-up for equilibration test-cases

Equilibration test-cases requires the definition of right-hand-sides (RHS) and boundary conditions (BC).
These are set either based on random values or a manufactured solution.

"""

from enum import Enum
import numpy as np
from petsc4py import PETSc
import typing

from dolfinx import fem, mesh
import ufl

from dolfinx_eqlb.lsolver import local_projection


# --- Interpolate from ufl to function
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


# --- Normal trace of a flux function
def exact_normaltrace(
    flux_ext: typing.Any, bound_id: int, vector_valued: typing.Optional[bool] = False
) -> typing.Any:
    """Compute the normal trace of a flux function

    Args:
        flux_ext:      The flux function
        bound_id:      The boundary id on a unit square
        vector_valued: True if the flux function is vector-valued

    Returns:
        The normal trace (sigma x normal) of the boundary flux
    """

    if bound_id == 1:
        pfactor = -1.0
        id = 0
    elif bound_id == 2:
        pfactor = -1.0
        id = 1
    elif bound_id == 3:
        pfactor = 1.0
        id = 0
    else:
        pfactor = 1.0
        id = 1

    if vector_valued:
        return pfactor * ufl.as_vector([flux_ext[0, id], flux_ext[1, id]])
    else:
        return pfactor * flux_ext[id]


# --- Set right-hand side
def set_arbitrary_rhs(
    msh: mesh.Mesh,
    degree_rhs: int,
    degree_projection: typing.Optional[int] = None,
    vector_valued: typing.Optional[bool] = False,
) -> typing.Tuple[fem.Function, fem.Function]:
    """Set polynomial right-hand-side (RHS)

    Args:
        msh:               The mesh
        degree_rhs:        Degree of the RHS
        degree_projection: Degree of the function-space within which the RHS is represented
        vector_valued:     True if the RHS is vector-valued

    Returns:
        The RHS used for calculating the primal solution,
        The projected RHS for the equilibration process
    """

    # Check input
    if degree_projection is None:
        degree_projection = degree_rhs

    # The dimension of the rhs
    dim = (msh.geometry.dim,) if vector_valued else None

    # Set function space
    if degree_projection < degree_rhs:
        raise ValueError("Degree of projection to small!")
    else:
        V_rhs = fem.functionspace(msh, ("DG", degree_projection, dim))
        size_rhs = V_rhs.dofmap.index_map_bs * V_rhs.dofmap.index_map.size_local

    function_rhs = fem.Function(V_rhs)

    # Set random data
    if degree_projection > degree_rhs:
        V_data = fem.functionspace(msh, ("DG", degree_rhs, dim))
        size_data = V_data.dofmap.index_map_bs * V_data.dofmap.index_map.size_local

        function_data = fem.Function(V_data)
        function_data.x.array[:] = 2 * (np.random.rand(size_data) + 0.1)

        function_rhs.interpolate(function_data)
    else:
        function_rhs.x.array[:] = 2 * (np.random.rand(size_rhs) + 0.1)

    return function_rhs, function_rhs


def set_manufactured_rhs(
    flux_ext: typing.Any,
    msh: mesh.Mesh,
    degree_rhs: int,
    vector_valued: typing.Optional[bool] = False,
) -> typing.Tuple[typing.Any, fem.Function]:
    """Set right-hand based on manufactured solution

    RHS is the -div(sigma(u_ext)) of the manufactured solution u_ext.

    Args:
        flux_ext:   ufl-expression of the exact flux
        msh:        The mesh
        degree_rhs: Degree of the right-hand side

    Returns:
        The RHS used for calculating the primal solution,
        The projected RHS for the equilibration process
    """

    # The dimension of the rhs
    dim = (msh.geometry.dim,) if vector_valued else None

    # Set function space
    V_rhs = fem.functionspace(msh, ("DG", degree_rhs, dim))

    # UFL function of u_ext
    rhs_ufl = ufl.div(flux_ext)

    # Project RHS to appropriate DG space
    rhs_projected = local_projection(V_rhs, [rhs_ufl])[0]

    return rhs_ufl, rhs_projected


# --- Set boundary conditions
class BCType(Enum):
    dirichlet = 0
    neumann_hom = 1
    neumann_inhom = 2


def set_arbitrary_bcs(
    bc_type: BCType,
    V_prime: fem.FunctionSpace,
    degree_flux: int,
    degree_bc: typing.Optional[int] = 0,
    neumann_ids: typing.List[int] = None,
) -> typing.Tuple[
    typing.List[int],
    typing.List[int],
    typing.List[fem.Function],
    typing.List[fem.Function],
    typing.List[bool],
]:
    """Set arbitrary Dirichlet and Neumann BCs

    Remark: Dirichlet BCs for primal problem are homogenous.

    Args:
        bc_type:     Type of boundary conditions
        V_prime:     The function space of the primal problem
        degree_flux: Degree of the flux space
        degree_bc:   Polynomial degree of the Neumann BCs
        neumann_ids: List of boundary ids for Neumann BCs

    Returns:
        The boundary ids for Dirichlet BCs,
        The boundary ids for Neumann BCs,
        The Dirichlet BCs,
        The Neumann BCs,
        Booleans indicating if the Neumann BCs require projection
    """

    if bc_type == BCType.dirichlet:
        # Set boundary ids
        boundary_id_dirichlet = [1, 2, 3, 4]
        boundary_id_neumann = []

        # Set homogenous dirichlet boundary conditions
        u_D = [fem.Function(V_prime) for i in range(0, len(boundary_id_dirichlet))]

        # Empty array of Neumann conditions
        func_neumann = []
    elif bc_type == BCType.neumann_hom:
        # The mesh
        msh = V_prime.mesh

        # Set boundary ids
        if neumann_ids is None:
            boundary_id_dirichlet = [2, 3]
            boundary_id_neumann = [1, 4]
        else:
            boundary_id_dirichlet = [i for i in range(1, 5) if i not in neumann_ids]
            boundary_id_neumann = neumann_ids

        # Set homogenous dirichlet boundary conditions
        u_D = [fem.Function(V_prime) for i in range(0, len(boundary_id_dirichlet))]

        # Set homogenous neumann boundary conditions
        if V_prime.num_sub_spaces == 0:
            hom_nbc = fem.Constant(msh, PETSc.ScalarType(0.0))
        else:
            if V_prime.num_sub_spaces == 2:
                hom_nbc = ufl.as_vector([0, 0])
            else:
                hom_nbc = ufl.as_vector([0, 0, 0])

        func_neumann = [hom_nbc for i in range(0, len(boundary_id_neumann))]
    elif bc_type == BCType.neumann_inhom:
        # The mesh
        msh = V_prime.mesh

        # Set boundary ids
        if neumann_ids is None:
            boundary_id_dirichlet = [2, 3]
            boundary_id_neumann = [1, 4]
        else:
            boundary_id_dirichlet = [i for i in range(1, 5) if i not in neumann_ids]
            boundary_id_neumann = neumann_ids

        # Set homogenous dirichlet boundary conditions
        u_D = [fem.Function(V_prime) for i in range(0, len(boundary_id_dirichlet))]

        # Set inhomogenous neumann boundary conditions
        V_bc = fem.functionspace(msh, ("DG", degree_bc, (V_prime.dofmap.bs,)))
        size_bc = V_bc.dofmap.index_map.size_local * V_bc.dofmap.bs

        f_bc = fem.Function(V_bc)
        f_bc.x.array[:] = 2 * (np.random.rand(size_bc) + 0.1)

        func_neumann = [f_bc for i in range(0, len(boundary_id_neumann))]
    else:
        raise ValueError("Not implemented!")

    # Specify if projection is required
    if degree_bc <= degree_flux:
        neumann_projection = [False for i in range(0, len(boundary_id_neumann))]
    else:
        neumann_projection = [True for i in range(0, len(boundary_id_neumann))]

    return (
        boundary_id_dirichlet,
        boundary_id_neumann,
        u_D,
        func_neumann,
        neumann_projection,
    )


def set_manufactured_bcs(
    V_prime: fem.FunctionSpace,
    boundary_id_dirichlet: typing.List[int],
    boundary_id_neumann: typing.List[int],
    u_ext: typing.Any,
    flux_ext: typing.Any,
    vector_valued: typing.Optional[bool] = False,
) -> typing.Tuple[
    typing.List[fem.Function], typing.List[typing.Any], typing.List[bool]
]:
    """Sets Dirichlet and Neumann BCs based on manufactured solution

    Args:
        V_prime:               The function space of the primal problem
        boundary_id_dirichlet: The boundary ids for Dirichlet BCs
        boundary_id_neumann:   The boundary ids for Neumann BCs
        u_ext:                 The manufactured solution (ufl representation)
        flux_ext:              The normal trace of the manufactured flux (ufl representation)

    Returns:
        The Dirichlet BCs,
        The Neumann BCs (ufl of normal trace),
        Booleans indicating if the Neumann BCs require projection
    """

    # Set dirichlet BCs
    list_dirichlet = []
    for id in boundary_id_dirichlet:
        uD = fem.Function(V_prime)
        interpolate_ufl_to_function(u_ext, uD)

        list_dirichlet.append(uD)

    # Set neumann BCs
    list_neumann = []
    list_neumann_projection = []

    for id in boundary_id_neumann:
        list_neumann.append(exact_normaltrace(flux_ext, id, vector_valued))
        list_neumann_projection.append(True)

    return list_dirichlet, list_neumann, list_neumann_projection
