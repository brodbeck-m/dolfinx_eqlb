from mpi4py import MPI
import numpy as np
import pytest

import basix
import basix.ufl_wrapper

import dolfinx
import dolfinx.mesh as dmesh
import dolfinx.fem as dfem

import dolfinx_eqlb.cpp

import ufl

'''Utility functions'''


def create_fespace_discontinous(family_basix, cell_basix, degree):
    elmt_basix = basix.create_element(family_basix, cell_basix, degree,
                                      basix.LagrangeVariant.equispaced, True)
    return basix.ufl_wrapper.BasixElement(elmt_basix)


def setup_problem_projection(cell, n_elmt):
    # Set cell variables
    if cell == ufl.triangle:
        cell_mesh = dmesh.CellType.triangle
        cell_basix = basix.CellType.triangle
    elif cell == ufl.tetrahedron:
        cell_mesh = dmesh.CellType.tetrahedron
        cell_basix = basix.CellType.tetrahedron
    elif cell == ufl.quadrilateral:
        cell_mesh = dmesh.CellType.quadrilateral
        cell_basix = basix.CellType.quadrilateral
    elif cell == ufl.hexahedron:
        cell_mesh = dmesh.CellType.hexahedron
        cell_basix = basix.CellType.hexahedron
    else:
        assert False

    # Create mesh
    if (cell.geometric_dimension() == 1):
        assert False
    elif (cell.geometric_dimension() == 2):
        msh = dmesh.create_unit_square(MPI.COMM_WORLD, n_elmt, n_elmt, cell_mesh,
                                       dmesh.GhostMode.shared_facet)
    else:
        msh = dmesh.create_unit_cube(MPI.COMM_WORLD, n_elmt, n_elmt, n_elmt, cell_mesh,
                                     dmesh.GhostMode.shared_facet)

    return msh, cell_basix


'''Test existance of required discontinous function-spaces (Lagrange, RT, BDM)'''


@pytest.mark.parametrize("cell", [ufl.triangle])
@pytest.mark.parametrize("family_basix",
                         [basix.ElementFamily.P,
                          basix.ElementFamily.RT,
                          basix.ElementFamily.BDM])
@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_elmtbasix_discontinous(cell, family_basix, degree):
    # Create problem
    msh, cell_basix = setup_problem_projection(cell, 5)

    # Create Finite element (discontinous!), FunctionSpace and Function
    V = dfem.FunctionSpace(msh, create_fespace_discontinous(
        family_basix, cell_basix, degree))
    func = dfem.Function(V)

    # Check if overall number of DOFs is correct
    n_elmt = msh.topology.index_map(cell.geometric_dimension()).size_local
    n_dof_dg = V.element.space_dimension * n_elmt * V.dofmap.bs

    assert (V.dofmap.index_map.size_global * V.dofmap.index_map_bs) == n_dof_dg


'''Only ufl-arguments: Test projection into discontinous Lagrange elements'''


@pytest.mark.parametrize("cell", [ufl.triangle,
                                  ufl.tetrahedron,
                                  ufl.quadrilateral,
                                  ufl.hexahedron])
@pytest.mark.parametrize("Element", [ufl.FiniteElement])
@pytest.mark.parametrize("n_elmt", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("test_func", ["const", "sin"])
def test_localprojection_ufl_vector(cell, Element, n_elmt, degree, test_func):
    # Create problem
    msh, cell_basix = setup_problem_projection(cell, n_elmt)

    # Create Function space
    if (cell == ufl.triangle or cell == ufl.tetrahedron):
        elmt = Element('DG', msh.ufl_cell(), degree)
    else:
        elmt = Element('DQ', msh.ufl_cell(), degree)
    V = dfem.FunctionSpace(msh, elmt)
    proj_local = dfem.Function(V)

    # Linear- and bilineaform
    dvol = ufl.Measure("dx", domain=msh,
                       metadata={"quadrature_degree": 3*degree})

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(u, v)*ufl.dx

    if (Element == ufl.FiniteElement):
        if test_func == "const":
            l = 2*v*ufl.dx
        else:
            x = ufl.SpatialCoordinate(msh)

            if (cell.geometric_dimension() == 1):
                l = ufl.sin(x[0]) * v * dvol
            elif (cell.geometric_dimension() == 2):
                l = ufl.sin(x[0]) * ufl.sin(x[1]) * v * dvol
            else:
                l = ufl.sin(x[0]) * ufl.sin(x[1]) * ufl.sin(x[2]) * v * dvol
    elif (Element == ufl.VectorElement):
        if test_func == "const":
            if (cell.geometric_dimension() == 2):
                f_ufl = ufl.as_vector([2, 5])
            elif (cell.geometric_dimension() == 3):
                f_ufl = ufl.as_vector([2, 5, 1])
            else:
                assert False
            l = ufl.inner(f_ufl, v)*ufl.dx
        else:
            x = ufl.SpatialCoordinate(msh)

            if (cell.geometric_dimension() == 2):
                f_ufl = ufl.as_vector([ufl.sin(x[0])*ufl.sin(x[1]),
                                       ufl.cos(x[1])*ufl.cos(x[1])])
            elif (cell.geometric_dimension() == 3):
                f_ufl = ufl.as_vector([ufl.sin(x[0])*ufl.sin(x[1])*ufl.sin(x[2]),
                                       ufl.cos(x[1])*ufl.cos(x[1]) *
                                       ufl.cos(x[2]),
                                       ufl.sin(x[1])*ufl.cos(x[1])*ufl.sin(x[2])])
            else:
                assert False
            l = ufl.inner(f_ufl, v)*dvol
    else:
        assert False

    # Calculate global projection
    problem = dfem.petsc.LinearProblem(a, l, bcs=[], petsc_options={
                                       "ksp_type": "preonly", "pc_type": "lu"})
    proj_global = problem.solve()

    # Calculate local projection
    dolfinx_eqlb.cpp.local_solver(proj_local._cpp_object,
                                  dfem.form(a), dfem.form(l))

    # Compare solutions
    assert np.allclose(proj_global.vector.array, proj_local.vector.array)


'''Only ufl-arguments: Test projection into discontinous RT and BDM elements'''


@pytest.mark.parametrize("cell", [ufl.triangle, ufl.tetrahedron])
@pytest.mark.parametrize("family_basix", [basix.ElementFamily.P])
# @pytest.mark.parametrize("family_basix",
#                          [basix.ElementFamily.P,
#                           basix.ElementFamily.RT,
#                           basix.ElementFamily.BDM])
@pytest.mark.parametrize("n_elmt", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("test_func", ["const", "sin"])
def test_localprojection_ufl_Hdiv(cell, family_basix, n_elmt, degree, test_func):
    # Create problem
    msh, cell_basix = setup_problem_projection(cell, n_elmt)

    # Create Function space
    elmt = create_fespace_discontinous(family_basix, cell_basix, degree)
    V = dfem.FunctionSpace(msh, elmt)
    proj_local = dfem.Function(V)

    # Linear- and bilineaform
    dvol = ufl.Measure("dx", domain=msh,
                       metadata={"quadrature_degree": 3*degree})

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(u, v)*ufl.dx

    if test_func == "const":
        l = 2*v*ufl.dx
    else:
        x = ufl.SpatialCoordinate(msh)

        if (cell.geometric_dimension() == 1):
            l = ufl.sin(x[0]) * v * dvol
        elif (cell.geometric_dimension() == 2):
            l = ufl.sin(x[0]) * ufl.sin(x[1]) * v * dvol
        else:
            l = ufl.sin(x[0]) * ufl.sin(x[1]) * ufl.sin(x[2]) * v * dvol

    # Calculate global projection
    problem = dfem.petsc.LinearProblem(a, l, bcs=[], petsc_options={
                                       "ksp_type": "preonly", "pc_type": "lu"})
    proj_global = problem.solve()

    # Calculate local projection
    dolfinx_eqlb.cpp.local_solver(
        proj_local._cpp_object, dfem.form(a), dfem.form(l))

    # Compare solutions
    assert np.allclose(proj_global.vector.array, proj_local.vector.array)


if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)
