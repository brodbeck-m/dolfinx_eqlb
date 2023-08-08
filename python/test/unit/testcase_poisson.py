# --- Includes ---
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from typing import Any, Callable, List

import dolfinx
import dolfinx.fem as dfem
import dolfinx.mesh as dmesh

import ufl

from dolfinx_eqlb import equilibration, lsolver

from python.test.unit.utils import Geometry

"""
Setup variable test-cases for the poisson problem

Supported variants:
    - manufactured solution based on u_ext = sin(2*pi * x) * cos(2*pi * y)
    - arbitrary right-hand side
"""


# --- Definition of manufactured solution
def exakt_solution_poisson(pkt):
    return lambda x: pkt.sin(2 * pkt.pi * x[0]) * pkt.cos(2 * pkt.pi * x[1])


def exact_flux_poisson(x):
    # Initialize flux
    sig = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)

    # Set flux
    sig[0] = -2 * np.pi * np.cos(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1])
    sig[1] = 2 * np.pi * np.sin(2 * np.pi * x[0]) * np.sin(2 * np.pi * x[1])

    return sig


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
    degree_flux: int,
    degree_rhs: int,
):
    """Set right-hand based on manufactured solution

    RHS is the -div(grad(u_ext)) of the manufactured solution

            u_ext(x,y) = sin(2*pi * x) * cos(2*pi * y)

    Args:
        domain (dolfinx.mesh.Mesh): The mesh
        degree_flux (int): Degree of the equilibrated flux
        degree_rhs (int): Degree of the right-hand side

    Returns:
        rhs_ufl (ufl): The RHS used for calculating the primal solution
        rhs_projected (dolfinx.Function): The projected RHS for the equilibration process

    """

    # Check input
    if degree_rhs > degree_flux - 1:
        raise ValueError("Degree of RHS to large!")

    # Set function space
    V_rhs = dfem.FunctionSpace(domain, ("DG", degree_rhs))

    # UFL function of u_ext
    x_crds = ufl.SpatialCoordinate(domain)
    rhs_ufl = -ufl.div(ufl.grad(u_ext_ufl(x_crds)))

    # Project RHS to appropriate DG space
    rhs_projected = lsolver.local_projector(V_rhs, [rhs_ufl])[0]

    return rhs_ufl, rhs_projected


# --- Set boundary conditions
def set_arbitrary_bcs(bc_type: str, V_prime: dfem.FunctionSpace):
    if bc_type == "pure_dirichlet":
        # Set boundary ids
        boundary_id_dirichlet = [1, 2, 3, 4]
        boundary_id_neumann = []

        # Set homogenous dirichlet boundary conditions
        u_D = [dfem.Function(V_prime) for i in range(0, len(boundary_id_dirichlet))]

        # Empty array of Neumann conditions
        func_neumann = []
    else:
        raise ValueError("Not implemented!")

    return boundary_id_dirichlet, boundary_id_neumann, u_D, func_neumann


def set_manufactured_bcs(bc_type: str):
    pass


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
    for id in bc_id_neumann:
        l_prime += ufl.inner(ufl_neumann[id], v) * geometry.ds(id)

    # Solve problem
    solveoptions = {
        "ksp_type": "preonly",
        "pc_type": "lu",
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
    sig_proj = lsolver.local_projector(V_flux, [-ufl.grad(u_prime)])[0]

    return u_prime, sig_proj


def equilibrate_poisson(
    Equilibrator: equilibration.FluxEquilibrator,
    degree_flux: int,
    geometry: Geometry,
    sig_proj: List[dfem.Function],
    rhs_proj: List[dfem.Function],
    bc_id_neumann: List[List[int]],
    bc_id_dirichlet: List[List[int]],
    flux_neumann: List[Any],
) -> List[dfem.Function]:
    # Extract facet markers
    fct_values = geometry.facet_function.values

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
    fct_bcesnt_flux = []
    bc_esnt_flux = []

    for list_bc_id in bc_id_neumann:
        fct_numpy = np.array([], dtype=np.int32)

        if len(list_bc_id) > 0:
            for id_flux in list_bc_id:
                list_fcts = geometry.facet_function.indices[fct_values == id_flux]
                fct_numpy = np.concatenate((fct_numpy, list_fcts))

            fct_bcesnt_flux.append(fct_numpy)

            raise NotImplementedError("Neumann boundary conditions not supported!")
        else:
            fct_bcesnt_flux.append([])
            bc_esnt_flux.append([])

    # Set equilibrator
    equilibrator = Equilibrator(degree_flux, geometry.mesh, rhs_proj, sig_proj)
    equilibrator.set_boundary_conditions(
        fct_bcesnt_primal, fct_bcesnt_flux, bc_esnt_flux
    )

    # Solve equilibration
    equilibrator.equilibrate_fluxes()

    return equilibrator.list_flux
