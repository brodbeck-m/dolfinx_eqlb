import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import pytest

import basix
from basix import CellType
import dolfinx.mesh as dmesh
import dolfinx.fem as dfem
import ufl

from dolfinx_eqlb import equilibration

''' Utility functions '''
def solve_equilibration(degree, n_elmt=5):
    # --- Solve primal problem
    # create mesh
    domain = dmesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([2, 2])], [n_elmt, n_elmt],
                                    cell_type=dmesh.CellType.triangle, diagonal=dmesh.DiagonalType.crossed)
    
    # get all boundary facets
    domain.topology.create_connectivity(1, 2)
    boundary_facets = dmesh.exterior_facet_indices(domain.topology)

    # set function space
    V_cg = dfem.FunctionSpace(domain, ("CG", degree))

    # set source term
    V_dg = dfem.FunctionSpace(domain, ("DG", degree - 1))
    f = dfem.Function(V_dg)
    f.x.array[:] = 2 * (np.random.rand(V_dg.dofmap.index_map.size_local) + 0.1)

    # set weak forms
    u = ufl.TrialFunction(V_cg)
    v = ufl.TestFunction(V_cg)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    l = f * v * ufl.dx

    # set boundary conditions
    dofs = dfem.locate_dofs_topological(V_cg, 1, boundary_facets)
    bc_esnt = [dfem.dirichletbc(PETSc.ScalarType(0), dofs, V_cg)]

    # solve primal problem
    solveoptions = {"ksp_type": "preonly", "pc_type": "lu", "ksp_rtol":1e-10, "ksp_atol":1e-10}
    problem_prime = dfem.petsc.LinearProblem(a, l, bc_esnt, petsc_options=solveoptions)
    uh = problem_prime.solve()

    # --- Project flux
    # set function space for flux
    V_flux_proj = dfem.VectorFunctionSpace(domain, ("DG", degree - 1))

    # set weak forms
    u = ufl.TrialFunction(V_flux_proj)
    v = ufl.TestFunction(V_flux_proj)

    a_proj = ufl.inner(u, v) * ufl.dx
    l_proj = ufl.inner(-ufl.grad(uh), v) * ufl.dx

    # solve projection
    problem_proj = dfem.petsc.LinearProblem(a_proj, l_proj, bcs=[], petsc_options=solveoptions)
    sig_proj = problem_proj.solve()

    # --- Equilibrate flux
    # setup and solve equilibration
    equilibrator = equilibration.EquilibratorSemiExplt(degree, domain, [f], [sig_proj])
    equilibrator.set_boundary_conditions([boundary_facets], [[]], [[]])
    equilibrator.equilibrate_fluxes()

    # interpolate projected flux to pice-wise H(div)
    sig_proj_rt = dfem.Function(equilibrator.V_flux)
    sig_proj_rt.interpolate(sig_proj)

    return uh, sig_proj_rt, equilibrator.list_flux[0]
    

def isoparametric_mapping_triangle(domain, dphi_geom, cell_id):
    # geometry data for current cell
    geometry = np.zeros((3, 2), dtype=np.float64)
    geometry[:] = domain.geometry.x[domain.geometry.dofmap.links(cell_id), :2]
        
    J_q = np.dot(geometry.T, dphi_geom.T)
    detj = np.linalg.det(J_q)

    return J_q, detj

''' Test divergence condition: div(sigma) = f '''


''' Test jump condition on facet-normal fluxes '''
@pytest.mark.parametrize("degree", [1])
def test_jump_condition(degree):
    # --- Solve equilibration
    uh, sig_proj_rt, sig_eq = solve_equilibration(degree, n_elmt=5)

    # calculate reconstructed flux
    x_sig_rt = sig_proj_rt.x.array[:] + sig_eq.x.array[:]

    # extract relevant data
    domain = uh.function_space.mesh
    dofmap_sigrt = sig_proj_rt.function_space.dofmap.list
    dofs_per_fct = degree
    
    # --- Determine sign of detj per cell
    # tabulate shape functions of geometry element
    c_element = basix.create_element(basix.ElementFamily.P, 
                                     basix.CellType.triangle, 1, 
                                     basix.LagrangeVariant.gll_warped)
    dphi_geom = c_element.tabulate(1, np.array([[0, 0]]))[1:2 + 1, 0, :, 0]

    # determine sign of detj per cell
    n_cells = domain.topology.index_map(2).size_local
    sign_detj = np.zeros(n_cells, dtype=np.float64)

    for c in range(0, n_cells):
        # evaluate detj
        J_q, detj = isoparametric_mapping_triangle(domain, dphi_geom, c)

        # determine sign of detj
        sign_detj[c] = np.sign(detj)

    # --- Check jump condition on all facets
    # create connectivity facet -> cell
    fct_to_cell = domain.topology.connectivity(1, 2)
    cell_to_fct = domain.topology.connectivity(2, 1)

    # check jump condition on all facets
    n_fcts = domain.topology.index_map(1).size_local

    for f in range(0, n_fcts):
        # get cells adjacet to f
        cells = fct_to_cell.links(f)

        if len(cells) > 1:
            # signs of cell jacobis
            sign_plus = sign_detj[cells[0]]
            sign_minus = sign_detj[cells[1]]

            # local facet id of cell
            if_plus = np.where(cell_to_fct.links(cells[0]) == f)[0][0]
            if_minus = np.where(cell_to_fct.links(cells[1]) == f)[0][0]

            for i in range(0, degree):
                # local dof id of facet-normal flux
                dof_plus = dofmap_sigrt.links(cells[0])[if_plus * dofs_per_fct + i]
                dof_minus = dofmap_sigrt.links(cells[1])[if_minus * dofs_per_fct + i]

                # calculate outward flux
                if if_plus == 1:
                    flux_plus = x_sig_rt[dof_plus] * sign_plus
                else:
                    flux_plus = x_sig_rt[dof_plus] * (-sign_plus)

                if if_minus == 1:
                    flux_minus = x_sig_rt[dof_minus] * sign_minus
                else:    
                    flux_minus = x_sig_rt[dof_minus] * (-sign_minus)

                # check continuity of facet-normal flux
                assert(np.isclose(flux_plus + flux_minus, 0))

if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)