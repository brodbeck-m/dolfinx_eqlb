# --- Includes ---
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from typing import Any, Callable, List

import dolfinx
import dolfinx.fem as dfem
import dolfinx.mesh as dmesh

import ufl

from dolfinx_eqlb.lsolver import local_projection
from dolfinx_eqlb.eqlb import fluxbc

from python.test.unit.utils import Geometry

"""
Setup variable test-cases for the poisson problem

Supported variants:
    - manufactured solution based on u_ext = sin(2*pi * x) * cos(2*pi * y)
    - arbitrary right-hand side
"""


# --- Definition of manufactured solution
def exact_solution_poisson(pkt):
    """Exact solution
    u_ext = sin(2*pi * x) * cos(2*pi * y)

    Args:
        pkt: Defines wether the function works with numpy or ufl
    Returns:
        lambda: The exact solution as function of the position x
    """
    return lambda x: pkt.sin(2 * pkt.pi * x[0]) * pkt.cos(2 * pkt.pi * x[1])


def exact_flux_np_poisson(x):
    """Exact flux
    flux_ext = -Grad[sin(2*pi * x) * cos(2*pi * y)]

    Args:
        x
    Returns:
        The exact flux at spacial positions x as numpy array
    """
    # Initialize flux
    sig = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)

    # Set flux
    sig[0] = -2 * np.pi * np.cos(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1])
    sig[1] = 2 * np.pi * np.sin(2 * np.pi * x[0]) * np.sin(2 * np.pi * x[1])

    return sig


def exact_flux_ufl_poisson(x):
    """Exact flux
    flux_ext = -Grad[sin(2*pi * x) * cos(2*pi * y)]

    Args:
        x
    Returns:
        The exact flux at spacial positions x as ufl expression
    """
    return -ufl.grad(exact_solution_poisson(ufl)(x))


# --- Set right-hand side
def set_arbitrary_rhs(
    domain: dolfinx.mesh.Mesh, degree_rhs: int, degree_projection: int = -1
):
    """Set polynomial right-hand side of degree degree_rhs

    RHS has to be used to calculate primal solution and thereafter the projected flux.

    Args:
        domain (dolfinx.mesh.Mesh): The mesh
        degree_rhs (int): Degree of the right-hand side
        degree_projection (int): If >0 the degree of the DG space within which the RHS (of degree_rhs) is represented

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
        V_rhs = dfem.FunctionSpace(domain, ("DG", degree_projection))

    function_rhs = dfem.Function(V_rhs)

    # Set random data
    if degree_projection > degree_rhs:
        V_data = dfem.FunctionSpace(domain, ("DG", degree_rhs))
        function_data = dfem.Function(V_data)
        function_data.x.array[:] = 2 * (
            np.random.rand(V_data.dofmap.index_map.size_local) + 0.1
        )
        function_rhs.interpolate(function_data)
    else:
        function_rhs.x.array[:] = 2 * (
            np.random.rand(V_rhs.dofmap.index_map.size_local) + 0.1
        )

    return function_rhs, function_rhs


def set_manufactured_rhs(
    u_ext_ufl: Callable,
    domain: dolfinx.mesh.Mesh,
    degree_rhs: int,
):
    """Set right-hand based on manufactured solution

    RHS is the -div(grad(u_ext)) of the manufactured solution u_ext.

    Args:
        u_ext_ufl (Callable): ufl-expression of the manufactured solution
        domain (dolfinx.mesh.Mesh): The mesh
        degree_rhs (int): Degree of the right-hand side

    Returns:
        rhs_ufl (ufl): The RHS used for calculating the primal solution
        rhs_projected (dolfinx.Function): The projected RHS for the equilibration process

    """
    # Set function space
    V_rhs = dfem.FunctionSpace(domain, ("DG", degree_rhs))

    # UFL function of u_ext
    x_crds = ufl.SpatialCoordinate(domain)
    rhs_ufl = -ufl.div(ufl.grad(u_ext_ufl(x_crds)))

    # Project RHS to appropriate DG space
    rhs_projected = local_projection(V_rhs, [rhs_ufl])[0]

    return rhs_ufl, rhs_projected


# --- Set boundary conditions
def set_arbitrary_bcs(
    bc_type: str, V_prime: dfem.FunctionSpace, degree_flux: int, degree_bc: int = 0
):
    """Set arbitrary dirichlet and neumann BCs

    Remark: Dirichlet BCs for primal problem are homogenous.

    Args:
        bc_type (str):           Type of boundary conditions
                                 (pure_dirichlet, neumann_homogenous, neumann_inhomogenous)
        V_prime (FunctionSpace): The function space of the primal problem
        degree_flux (int):       Degree of the flux space
        degree_bc (int):         Polynomial degree of the boundary conditions

    Returns:
        boundary_id_dirichlet (List[int]): List of boundary ids for dirichlet BCs
        boundary_id_neumann (List[int]):   List of boundary ids for neumann BCs
        u_D (List[Function]):              List of dirichlet boundary conditions
        func_neumann (List[ufl]):          List of neumann boundary conditions
        neumann_projection (List[bool]):   List of booleans indicating wether the neumann
                                           BCs require projection
    """
    if bc_type == "pure_dirichlet":
        # Set boundary ids
        boundary_id_dirichlet = [1, 2, 3, 4]
        boundary_id_neumann = []

        # Set homogenous dirichlet boundary conditions
        u_D = [dfem.Function(V_prime) for i in range(0, len(boundary_id_dirichlet))]

        # Empty array of Neumann conditions
        func_neumann = []
    elif bc_type == "neumann_homogenous":
        # The mesh
        domain = V_prime.mesh

        # Set boundary ids
        boundary_id_dirichlet = [2, 3]
        boundary_id_neumann = [1, 4]

        # Set homogenous dirichlet boundary conditions
        u_D = [dfem.Function(V_prime) for i in range(0, len(boundary_id_dirichlet))]

        # Set homogenous dirichlet boundary conditions
        func_neumann = [
            dfem.Constant(domain, PETSc.ScalarType(0.0))
            for i in range(0, len(boundary_id_neumann))
        ]
    elif bc_type == "neumann_inhomogenous":
        # The mesh
        domain = V_prime.mesh

        # Set boundary ids
        boundary_id_dirichlet = [2, 3]
        boundary_id_neumann = [1, 4]

        # Set homogenous dirichlet boundary conditions
        u_D = [dfem.Function(V_prime) for i in range(0, len(boundary_id_dirichlet))]

        # Set homogenous dirichlet boundary conditions
        V_bc = dfem.FunctionSpace(domain, ("DG", degree_bc))

        for i in range(0, len(boundary_id_neumann)):
            # Set random function on boundary
            func_neumann.append(dfem.Function(V_bc))

            func_neumann[i].x.array[:] = 2 * (
                np.random.rand(V_bc.dofmap.index_map.size_local) + 0.1
            )
    else:
        raise ValueError("Not implemented!")

    if degree_bc > degree_flux - 1:
        neumann_projection = [True for i in range(0, len(boundary_id_neumann))]
    else:
        neumann_projection = [False for i in range(0, len(boundary_id_neumann))]

    return (
        boundary_id_dirichlet,
        boundary_id_neumann,
        u_D,
        func_neumann,
        neumann_projection,
    )


def set_manufactured_bcs(
    V_prime: dfem.FunctionSpace,
    boundary_id_dirichlet: List[int],
    boundary_id_neumann: List[int],
    u_ext: Callable,
    sigma_ext: Any,
):
    """Sets dirichlet and neumann BCs based on manufactured solution

    Args:
        V_prime (dolfinx.FunctionSpace):   The function space of the primal problem
        boundary_id_dirichlet (List[int]): List of boundary ids for dirichlet BCs
        boundary_id_neumann (List[int]):   List of boundary ids for neumann BCs
        u_ext (Callable):                  The manufactured solution (return np.array)
        sigma_ext (ufl):                   The manufactured flux (ufl representation)

    Returns:
        u_D (List[dolfinx.Function]):      List of dirichlet boundary conditions
        func_neumann (List[ufl]):          List of neumann boundary conditions
        neumann_projection (List[bool]):   List of booleans indicating wether the neumann
                                           BCs require projection
    """

    # Set dirichlet BCs
    u_D = []
    for id in boundary_id_dirichlet:
        uD = dfem.Function(V_prime)
        uD.interpolate(u_ext)

        u_D.append(uD)

    # Set neumann BCs
    func_neumann = []

    for id in boundary_id_neumann:
        func_neumann.append(sigma_ext)

    return u_D, func_neumann, len(func_neumann) * [True]


# --- Solution routines
def solve_poisson_problem(
    V_prime: dfem.FunctionSpace,
    geometry: Geometry,
    bc_id_neumann: List[int],
    bc_id_dirichlet: List[int],
    ufl_rhs: Any,
    ufl_neumann: List[Any],
    u_dirichlet: List[dfem.Function],
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
        l_prime += ufl.inner(ufl_neumann[i], v) * geometry.ds(id)

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


def equilibrate_poisson(
    Equilibrator: Any,
    degree_flux: int,
    geometry: Geometry,
    sig_proj: List[dfem.Function],
    rhs_proj: List[dfem.Function],
    bc_id_neumann: List[List[int]],
    bc_id_dirichlet: List[List[int]],
    flux_neumann: List[Any],
    neumann_projection: List[bool],
) -> List[dfem.Function]:
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

    return equilibrator.list_flux
