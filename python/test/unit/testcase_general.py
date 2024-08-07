# --- Includes ---
from enum import Enum
import numpy as np
from petsc4py import PETSc
import typing

import dolfinx
import dolfinx.fem as dfem
import ufl

from dolfinx_eqlb.lsolver import local_projection
from dolfinx_eqlb.eqlb import fluxbc

from utils import Geometry, interpolate_ufl_to_function

"""
General setup routines for unit tests

Supported variants:
    - manufactured solution based on u_ext
    - arbitrary right-hand side
"""


# --- Normal trace of a flux function
def exact_normaltrace(
    flux_ext: typing.Any, bound_id: int, vector_valued: typing.Optional[bool] = False
):
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
    domain: dolfinx.mesh.Mesh,
    degree_rhs: int,
    degree_projection: int = -1,
    vector_valued: typing.Optional[bool] = False,
):
    """Set polynomial right-hand side of degree degree_rhs

    RHS has to be used to calculate primal solution and thereafter the projected flux.

    Args:
        domain (dolfinx.mesh.Mesh): The mesh
        degree_rhs (int): Degree of the right-hand side
        degree_projection (int): If >0 the degree of the DG space within which the RHS (of degree_rhs) is represented
        vector_valued (bool): True if the right-hand side is vector-valued

    Returns:
        rhs_ufl (dolfinx.Function): The RHS used for calculating the primal solution
        rhs_projected (dolfinx.Function): The projected RHS for the equilibration process

    """
    # Check input
    if degree_projection < 0:
        degree_projection = degree_rhs

    # Set function space
    if degree_projection < degree_rhs:
        raise ValueError("Degree of projection to small!")
    else:
        if vector_valued:
            V_rhs = dfem.VectorFunctionSpace(domain, ("DG", degree_projection))
        else:
            V_rhs = dfem.FunctionSpace(domain, ("DG", degree_projection))

    function_rhs = dfem.Function(V_rhs)

    # Set random data
    if degree_projection > degree_rhs:
        if vector_valued:
            V_data = dfem.VectorFunctionSpace(domain, ("DG", degree_rhs))
        else:
            V_data = dfem.FunctionSpace(domain, ("DG", degree_rhs))

        function_data = dfem.Function(V_data)
        function_data.x.array[:] = 2 * (
            np.random.rand(
                V_data.dofmap.index_map_bs * V_data.dofmap.index_map.size_local
            )
            + 0.1
        )

        function_rhs.interpolate(function_data)
    else:
        function_rhs.x.array[:] = 2 * (
            np.random.rand(
                V_rhs.dofmap.index_map_bs * V_rhs.dofmap.index_map.size_local
            )
            + 0.1
        )

    return function_rhs, function_rhs


def set_manufactured_rhs(
    flux_ext: typing.Any,
    domain: dolfinx.mesh.Mesh,
    degree_rhs: int,
    vector_valued: typing.Optional[bool] = False,
):
    """Set right-hand based on manufactured solution

    RHS is the -div(sigma(u_ext)) of the manufactured solution u_ext.

    Args:
        flux_ext (Callable): ufl-expression of the exact flux
        domain (dolfinx.mesh.Mesh): The mesh
        degree_rhs (int): Degree of the right-hand side

    Returns:
        rhs_ufl (ufl): The RHS used for calculating the primal solution
        rhs_projected (dolfinx.Function): The projected RHS for the equilibration process
    """
    # Set function space
    if vector_valued:
        V_rhs = dfem.VectorFunctionSpace(domain, ("DG", degree_rhs))
    else:
        V_rhs = dfem.FunctionSpace(domain, ("DG", degree_rhs))

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
    V_prime: dfem.FunctionSpace,
    degree_flux: int,
    degree_bc: int = 0,
    neumann_ids: typing.List[int] = None,
):
    """Set arbitrary dirichlet and neumann BCs

    Remarks:
         1.) Dirichlet BCs for primal problem are homogenous.

    Args:
        bc_type (BCType):        Type of boundary conditions
        V_prime (FunctionSpace): The function space of the primal problem
        degree_flux (int):       Degree of the flux space
        degree_bc (int):         Polynomial degree of the boundary conditions
        neumann_ids (List[int]): List of boundary ids for neumann BCs

    Returns:
        boundary_id_dirichlet (List[int]): List of boundary ids for dirichlet BCs
        boundary_id_neumann (List[int]):   List of boundary ids for neumann BCs
        u_D (List[Function]):              List of dirichlet boundary conditions
        func_neumann (List[ufl]):          List of neumann boundary conditions
        neumann_projection (List[bool]):   List of booleans indicating wether the neumann
                                           BCs require projection
    """
    if bc_type == BCType.dirichlet:
        # Set boundary ids
        boundary_id_dirichlet = [1, 2, 3, 4]
        boundary_id_neumann = []

        # Set homogenous dirichlet boundary conditions
        u_D = [dfem.Function(V_prime) for i in range(0, len(boundary_id_dirichlet))]

        # Empty array of Neumann conditions
        func_neumann = []
    elif bc_type == BCType.neumann_hom:
        # The mesh
        domain = V_prime.mesh

        # Set boundary ids
        if neumann_ids is None:
            boundary_id_dirichlet = [2, 3]
            boundary_id_neumann = [1, 4]
        else:
            boundary_id_dirichlet = [i for i in range(1, 5) if i not in neumann_ids]
            boundary_id_neumann = neumann_ids

        # Set homogenous dirichlet boundary conditions
        u_D = [dfem.Function(V_prime) for i in range(0, len(boundary_id_dirichlet))]

        # Set homogenous neumann boundary conditions
        if V_prime.num_sub_spaces == 0:
            hom_nbc = dfem.Constant(domain, PETSc.ScalarType(0.0))
        else:
            if V_prime.num_sub_spaces == 2:
                hom_nbc = ufl.as_vector([0, 0])
            else:
                hom_nbc = ufl.as_vector([0, 0, 0])

        func_neumann = [hom_nbc for i in range(0, len(boundary_id_neumann))]
    elif bc_type == BCType.neumann_inhom:
        # The mesh
        domain = V_prime.mesh

        # Set boundary ids
        if neumann_ids is None:
            boundary_id_dirichlet = [2, 3]
            boundary_id_neumann = [1, 4]
        else:
            boundary_id_dirichlet = [i for i in range(1, 5) if i not in neumann_ids]
            boundary_id_neumann = neumann_ids

        # Set homogenous dirichlet boundary conditions
        u_D = [dfem.Function(V_prime) for i in range(0, len(boundary_id_dirichlet))]

        # Set inhomogenous neumann boundary conditions
        if V_prime.num_sub_spaces == 0:
            V_bc = dfem.FunctionSpace(domain, ("DG", degree_bc))
        else:
            V_bc = dfem.VectorFunctionSpace(domain, ("DG", degree_bc))

        f_bc = dfem.Function(V_bc)
        f_bc.x.array[:] = 2 * (
            np.random.rand(V_bc.dofmap.index_map.size_local * V_bc.dofmap.bs) + 0.1
        )

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
    V_prime: dfem.FunctionSpace,
    boundary_id_dirichlet: typing.List[int],
    boundary_id_neumann: typing.List[int],
    u_ext: typing.Any,
    flux_ext: typing.Any,
    vector_valued: typing.Optional[bool] = False,
):
    """Sets dirichlet and neumann BCs based on manufactured solution

    Args:
        V_prime (dolfinx.FunctionSpace):   The function space of the primal problem
        boundary_id_dirichlet (List[int]): List of boundary ids for dirichlet BCs
        boundary_id_neumann (List[int]):   List of boundary ids for neumann BCs
        u_ext (ufl):                       The manufactured solution (ufl representation)
        flux_ext (ufl):                    The normal trace of the manufactured flux (ufl representation)

    Returns:
        list_dirichlet (List[dolfinx.Function]): List of dirichlet boundary conditions
        list_neumann (List[ufl]):                List of neumann boundary conditions
        list_neumann_projection (List[bool]):    List of booleans indicating wether the neumann
                                                 BCs require projection
    """

    # Set dirichlet BCs
    list_dirichlet = []
    for id in boundary_id_dirichlet:
        uD = dfem.Function(V_prime)
        interpolate_ufl_to_function(u_ext, uD)

        list_dirichlet.append(uD)

    # Set neumann BCs
    list_neumann = []
    list_neumann_projection = []

    for id in boundary_id_neumann:
        list_neumann.append(exact_normaltrace(flux_ext, id, vector_valued))
        list_neumann_projection.append(True)

    return list_dirichlet, list_neumann, list_neumann_projection
