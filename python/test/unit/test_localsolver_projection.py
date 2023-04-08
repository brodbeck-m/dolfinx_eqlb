from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import pytest

import basix
import basix.ufl_wrapper

import dolfinx.mesh as dmesh
import dolfinx.fem as dfem

import ufl

from dolfinx_eqlb import lsolver


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


@pytest.mark.parametrize("cell", [ufl.triangle, ufl.tetrahedron])
@pytest.mark.parametrize("family_basix",
                         [basix.ElementFamily.P,
                          basix.ElementFamily.RT,
                          basix.ElementFamily.BDM])
@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_elmtbasix_discontinous(cell, family_basix, degree):
    # Create problem
    msh, cell_basix = setup_problem_projection(cell, 2)

    # Create Finite element (discontinous!), FunctionSpace and Function
    V = dfem.FunctionSpace(msh, create_fespace_discontinous(
        family_basix, cell_basix, degree))
    func = dfem.Function(V)

    # Check if overall number of DOFs is correct
    n_elmt = msh.topology.index_map(cell.geometric_dimension()).size_local
    n_dof_dg = V.element.space_dimension * n_elmt * V.dofmap.bs

    assert (V.dofmap.index_map.size_global * V.dofmap.index_map_bs) == n_dof_dg


'''Test projection into discontinous Lagrange elements'''


@pytest.mark.parametrize("cell", [ufl.triangle, ufl.quadrilateral, ufl.tetrahedron, ufl.hexahedron])
@pytest.mark.parametrize("is_vectorvalued", [False, True])
@pytest.mark.parametrize("n_elmt", [2, 3])
@pytest.mark.parametrize("degree", [1, 2, 3])
# @pytest.mark.parametrize("test_func", ["const", "ufl", "function","function_ufl"])
@pytest.mark.parametrize("test_func", ["const", "ufl"])
def test_localprojection_ufl_vector(cell, is_vectorvalued, n_elmt, degree, test_func):
    # Create problem
    msh, cell_basix = setup_problem_projection(cell, n_elmt)

    # Create Function space
    if (cell == ufl.triangle or cell == ufl.tetrahedron):
        if is_vectorvalued:
            elmt = ufl.VectorElement('DG', msh.ufl_cell(), degree)
            if (test_func == 'function'):
                elmt_rhs = ufl.VectorElement('DG', msh.ufl_cell(), degree+1)
            elif (test_func == 'function_ufl'):
                elmt_rhs = ufl.FiniteElement('DG', msh.ufl_cell(), degree+1)
        else:
            elmt = ufl.FiniteElement('DG', msh.ufl_cell(), degree)
            if (test_func == 'function'):
                elmt_rhs = ufl.FiniteElement('DG', msh.ufl_cell(), degree+1)
            elif (test_func == 'function_ufl'):
                elmt_rhs = ufl.VectorElement('DG', msh.ufl_cell(), degree+1)
    else:
        if is_vectorvalued:
            elmt = ufl.VectorElement('DQ', msh.ufl_cell(), degree)
            if (test_func == 'function'):
                elmt_rhs = ufl.VectorElement('DQ', msh.ufl_cell(), degree+1)
            elif (test_func == 'function_ufl'):
                elmt_rhs = ufl.FiniteElement('DQ', msh.ufl_cell(), degree+1)
        else:
            elmt = ufl.FiniteElement('DQ', msh.ufl_cell(), degree)
            if (test_func == 'function'):
                elmt_rhs = ufl.FiniteElement('DQ', msh.ufl_cell(), degree+1)
            elif (test_func == 'function_ufl'):
                elmt_rhs = ufl.VectorElement('DQ', msh.ufl_cell(), degree+1)
    V = dfem.FunctionSpace(msh, elmt)
    proj_local = dfem.Function(V)

    # Linear- and bilineaform
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(u, v)*ufl.dx

    if is_vectorvalued:
        if test_func == "const":
            dvol = ufl.dx
            quad_deg = None
            if (cell.geometric_dimension() == 2):
                f_rhs = dfem.Constant(msh, PETSc.ScalarType(np.array([2, 5])))
            elif (cell.geometric_dimension() == 3):
                f_rhs = dfem.Constant(
                    msh, PETSc.ScalarType(np.array([2, 5, 1])))
            else:
                assert False
        elif test_func == "ufl":
            dvol = ufl.Measure("dx", domain=msh,
                               metadata={"quadrature_degree": 3*degree})
            quad_deg = 3 * degree
            x = ufl.SpatialCoordinate(msh)

            if (cell.geometric_dimension() == 2):
                f_rhs = ufl.as_vector([ufl.sin(x[0]) * ufl.sin(x[1]),
                                      ufl.cos(x[0]) * ufl.cos(x[1])])
            elif (cell.geometric_dimension() == 3):
                f_rhs = ufl.as_vector([ufl.sin(x[0])*ufl.sin(x[1]) *
                                      ufl.sin(x[2]),
                                      ufl.cos(x[0])*ufl.cos(x[1]) *
                                      ufl.cos(x[2]),
                                      ufl.sin(x[0])*ufl.cos(x[1]) *
                                      ufl.sin(x[2])])
            else:
                assert False
        elif test_func == "function":
            dvol = ufl.dx
            quad_deg = None

            # Interpolation function
            def rhs_2D(x):
                u = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
                u[0] = np.sin(x[0])*np.sin(x[1])
                u[1] = np.cos(x[0])*np.cos(x[1])
                return u

            def rhs_3D(x):
                u = np.zeros((3, x.shape[1]), dtype=PETSc.ScalarType)
                u[0] = np.sin(x[0])*np.sin(x[1])*np.sin(x[2])
                u[1] = np.cos(x[0])*np.cos(x[1])*np.cos(x[2])
                u[2] = np.cos(x[0])*np.cos(x[1])*np.sin(x[2])
                return u

            # Create function-representation of rhs
            V_rhs = dfem.FunctionSpace(msh, elmt_rhs)
            f_rhs = dfem.Function(V_rhs)

            if (cell.geometric_dimension() == 2):
                f_rhs.interpolate(rhs_2D)
            elif (cell.geometric_dimension() == 3):
                f_rhs.interpolate(rhs_3D)
            else:
                assert False
        else:
            dvol = ufl.Measure("dx", domain=msh,
                               metadata={"quadrature_degree": 3*degree})
            quad_deg = 3 * degree

            # Interpolation function
            def rhs_2D(x):
                u = np.zeros((1, x.shape[1]), dtype=PETSc.ScalarType)
                u[0] = np.sin(x[0])*np.sin(x[1])
                return u

            def rhs_3D(x):
                u = np.zeros((1, x.shape[1]), dtype=PETSc.ScalarType)
                u[0] = np.sin(x[0])*np.sin(x[1])*np.sin(x[2])
                return u

            # Create function-representation of rhs
            V_rhs = dfem.FunctionSpace(msh, elmt_rhs)
            func_rhs = dfem.Function(V_rhs)

            if (cell.geometric_dimension() == 2):
                func_rhs.interpolate(rhs_2D)
            elif (cell.geometric_dimension() == 3):
                func_rhs.interpolate(rhs_3D)
            else:
                assert False

            # Create source term by taking divergence of vector
            f_rhs = dfem.Constant(
                msh, PETSc.ScalarType(1.5))*ufl.grad(func_rhs)
    else:
        if (test_func == 'const'):
            dvol = ufl.dx
            quad_deg = None
            f_rhs = dfem.Constant(msh, PETSc.ScalarType(2))
        elif (test_func == 'ufl'):
            dvol = ufl.Measure("dx", domain=msh,
                               metadata={"quadrature_degree": 3*degree})
            quad_deg = 3 * degree
            x = ufl.SpatialCoordinate(msh)

            if (cell.geometric_dimension() == 2):
                f_rhs = ufl.sin(x[0])*ufl.sin(x[1])
            elif (cell.geometric_dimension() == 3):
                f_rhs = ufl.sin(x[0])*ufl.sin(x[1])*ufl.sin(x[2])
            else:
                assert False
        elif (test_func == 'function'):
            dvol = ufl.dx
            quad_deg = None

            # Interpolation function
            def rhs_2D(x):
                u = np.zeros((1, x.shape[1]), dtype=PETSc.ScalarType)
                u[0] = np.sin(x[0])*np.sin(x[1])
                return u

            def rhs_3D(x):
                u = np.zeros((1, x.shape[1]), dtype=PETSc.ScalarType)
                u[0] = np.sin(x[0])*np.sin(x[1])*np.sin(x[2])
                return u

            # Create function-representation of rhs
            V_rhs = dfem.FunctionSpace(msh, elmt_rhs)
            f_rhs = dfem.Function(V_rhs)

            if (cell.geometric_dimension() == 2):
                f_rhs.interpolate(rhs_2D)
            elif (cell.geometric_dimension() == 3):
                f_rhs.interpolate(rhs_3D)
            else:
                assert False
        else:
            dvol = ufl.Measure("dx", domain=msh,
                               metadata={"quadrature_degree": 3*degree})
            quad_deg = 3 * degree

            # Interpolation function
            def rhs_2D(x):
                u = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
                u[0] = np.sin(x[0])*np.sin(x[1])
                u[1] = np.cos(x[0])*np.cos(x[1])
                return u

            def rhs_3D(x):
                u = np.zeros((3, x.shape[1]), dtype=PETSc.ScalarType)
                u[0] = np.sin(x[0])*np.sin(x[1])*np.sin(x[2])
                u[1] = np.cos(x[0])*np.cos(x[1])*np.cos(x[2])
                u[2] = np.cos(x[0])*np.cos(x[1])*np.sin(x[2])
                return u

            # Create function-representation of rhs
            V_rhs = dfem.FunctionSpace(msh, elmt_rhs)
            func_rhs = dfem.Function(V_rhs)

            if (cell.geometric_dimension() == 2):
                func_rhs.interpolate(rhs_2D)
            elif (cell.geometric_dimension() == 3):
                func_rhs.interpolate(rhs_3D)
            else:
                assert False

            # Create source term by taking divergence of vector
            f_rhs = dfem.Constant(msh, PETSc.ScalarType(1.5))*ufl.div(func_rhs)

    l = ufl.inner(f_rhs, v) * dvol

    # Calculate global projection
    problem = dfem.petsc.LinearProblem(a, l, bcs=[], petsc_options={"ksp_type": "preonly",
                                                                    "pc_type": "lu"})
    proj_global = problem.solve()

    # Calculate local projection
    # lsolver.local_solver_cholesky([proj_local], dfem.form(a), [dfem.form(l)])
    proj_local = lsolver.local_projector(
        V, [f_rhs], quadrature_degree=quad_deg)

    # Compare solutions
    assert np.allclose(proj_global.vector.array, proj_local[0].vector.array)


'''Test projection into discontinous RT and BDM elements'''


@pytest.mark.parametrize("cell", [ufl.triangle, ufl.tetrahedron])
@pytest.mark.parametrize("family_basix", [basix.ElementFamily.RT,
                                          basix.ElementFamily.BDM])
@pytest.mark.parametrize("n_elmt", [2, 3])
@pytest.mark.parametrize("degree", [1, 2, 3])
# @pytest.mark.parametrize("test_func", ["const", "ufl", "function"])
@pytest.mark.parametrize("test_func", ["const", "ufl"])
def test_localprojection_ufl_Hdiv(cell, family_basix, n_elmt, degree, test_func):
    # Create problem
    msh, cell_basix = setup_problem_projection(cell, n_elmt)

    # Create Function space
    elmt = create_fespace_discontinous(family_basix, cell_basix, degree)
    V = dfem.FunctionSpace(msh, elmt)
    proj_local = dfem.Function(V)

    # Linear- and bilineaform
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(u, v)*ufl.dx

    if test_func == "const":
        dvol = ufl.dx
        quad_deg = None
        if (cell.geometric_dimension() == 1):
            assert False
        elif (cell.geometric_dimension() == 2):
            func = dfem.Constant(msh, PETSc.ScalarType(np.array([2, 5])))

        else:
            func = dfem.Constant(msh, PETSc.ScalarType(np.array([2, 5, 9])))
    elif test_func == "ufl":
        dvol = ufl.Measure("dx", domain=msh,
                           metadata={"quadrature_degree": 3*degree})
        quad_deg = 3 * degree

        x = ufl.SpatialCoordinate(msh)

        if (cell.geometric_dimension() == 1):
            assert False
        elif (cell.geometric_dimension() == 2):
            func = ufl.as_vector([ufl.sin(x[0]) * ufl.sin(x[1]),
                                  ufl.cos(x[0]) * ufl.cos(x[1])])

        else:
            func = ufl.as_vector([ufl.sin(x[0])*ufl.sin(x[1])*ufl.sin(x[2]),
                                  ufl.cos(x[0])*ufl.cos(x[1])*ufl.cos(x[2]),
                                  ufl.sin(x[0])*ufl.cos(x[1])*ufl.sin(x[2])])
    else:
        dvol = ufl.dx
        quad_deg = None

        # Create rhs-function
        elmt_rhs = ufl.VectorElement('CG', msh.ufl_cell(), degree)
        V_rhs = dfem.FunctionSpace(msh, elmt_rhs)
        func = dfem.Function(V_rhs)

        # Interpolation function
        def rhs_2D(x):
            u = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
            u[0] = np.sin(x[0])*np.sin(x[1])
            u[1] = np.cos(x[0])*np.cos(x[1])
            return u

        def rhs_3D(x):
            u = np.zeros((3, x.shape[1]), dtype=PETSc.ScalarType)
            u[0] = np.sin(x[0])*np.sin(x[1])*np.sin(x[2])
            u[1] = np.cos(x[0])*np.cos(x[1])*np.cos(x[2])
            u[2] = np.cos(x[0])*np.cos(x[1])*np.sin(x[2])
            return u

        if (cell.geometric_dimension() == 2):
            func.interpolate(rhs_2D)
        elif (cell.geometric_dimension() == 3):
            func.interpolate(rhs_3D)
        else:
            assert False

    l = ufl.inner(func, v) * dvol

    # Calculate global projection
    problem = dfem.petsc.LinearProblem(a, l, bcs=[], petsc_options={
                                       "ksp_type": "preonly", "pc_type": "lu"})
    proj_global = problem.solve()

    # Calculate local projection
    # lsolver.local_solver_cholesky([proj_local], dfem.form(a), [dfem.form(l)])
    proj_local = lsolver.local_projector(
        V, [func], quadrature_degree=quad_deg)

    # Compare solutions
    assert np.allclose(proj_global.vector.array, proj_local[0].vector.array)


'''Test projection with multiple LHS (same type)'''


@pytest.mark.parametrize("cell", [ufl.triangle, ufl.quadrilateral, ufl.tetrahedron, ufl.hexahedron])
@pytest.mark.parametrize("is_vectorvalued", [False, True])
@pytest.mark.parametrize("degree", [1])
# @pytest.mark.parametrize("test_func", ["const", "ufl", "function","function_ufl"])
@pytest.mark.parametrize("test_func", ["const", "ufl"])
@pytest.mark.parametrize("n_lhs", [2, 3])
def test_localprojection_multi_lhs(cell, is_vectorvalued, degree, test_func, n_lhs):
    # Create problem
    msh, cell_basix = setup_problem_projection(cell, 2)

    # Create Function space
    if (cell == ufl.triangle or cell == ufl.tetrahedron):
        if is_vectorvalued:
            elmt = ufl.VectorElement('DG', msh.ufl_cell(), degree)
            if (test_func == 'function'):
                elmt_rhs = ufl.VectorElement('DG', msh.ufl_cell(), degree+1)
            elif (test_func == 'function_ufl'):
                elmt_rhs = ufl.FiniteElement('DG', msh.ufl_cell(), degree+1)
        else:
            elmt = ufl.FiniteElement('DG', msh.ufl_cell(), degree)
            if (test_func == 'function'):
                elmt_rhs = ufl.FiniteElement('DG', msh.ufl_cell(), degree+1)
            elif (test_func == 'function_ufl'):
                elmt_rhs = ufl.VectorElement('DG', msh.ufl_cell(), degree+1)
    else:
        if is_vectorvalued:
            elmt = ufl.VectorElement('DQ', msh.ufl_cell(), degree)
            if (test_func == 'function'):
                elmt_rhs = ufl.VectorElement('DQ', msh.ufl_cell(), degree+1)
            elif (test_func == 'function_ufl'):
                elmt_rhs = ufl.FiniteElement('DQ', msh.ufl_cell(), degree+1)
        else:
            elmt = ufl.FiniteElement('DQ', msh.ufl_cell(), degree)
            if (test_func == 'function'):
                elmt_rhs = ufl.FiniteElement('DQ', msh.ufl_cell(), degree+1)
            elif (test_func == 'function_ufl'):
                elmt_rhs = ufl.VectorElement('DQ', msh.ufl_cell(), degree+1)
    V = dfem.FunctionSpace(msh, elmt)

    # Linear- and bilineaform
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(u, v)*ufl.dx

    if is_vectorvalued:
        if test_func == "const":
            dvol = ufl.dx
            if (cell.geometric_dimension() == 2):
                f_rhs = dfem.Constant(msh, PETSc.ScalarType(np.array([2, 5])))
            elif (cell.geometric_dimension() == 3):
                f_rhs = dfem.Constant(
                    msh, PETSc.ScalarType(np.array([2, 5, 1])))
            else:
                assert False
        elif test_func == "ufl":
            dvol = ufl.Measure("dx", domain=msh,
                               metadata={"quadrature_degree": 3*degree})
            x = ufl.SpatialCoordinate(msh)

            if (cell.geometric_dimension() == 2):
                f_rhs = ufl.as_vector([ufl.sin(x[0]) * ufl.sin(x[1]),
                                      ufl.cos(x[0]) * ufl.cos(x[1])])
            elif (cell.geometric_dimension() == 3):
                f_rhs = ufl.as_vector([ufl.sin(x[0])*ufl.sin(x[1]) *
                                      ufl.sin(x[2]),
                                      ufl.cos(x[0])*ufl.cos(x[1]) *
                                      ufl.cos(x[2]),
                                      ufl.sin(x[0])*ufl.cos(x[1]) *
                                      ufl.sin(x[2])])
            else:
                assert False
        elif test_func == "function":
            dvol = ufl.dx

            # Interpolation function
            def rhs_2D(x):
                u = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
                u[0] = np.sin(x[0])*np.sin(x[1])
                u[1] = np.cos(x[0])*np.cos(x[1])
                return u

            def rhs_3D(x):
                u = np.zeros((3, x.shape[1]), dtype=PETSc.ScalarType)
                u[0] = np.sin(x[0])*np.sin(x[1])*np.sin(x[2])
                u[1] = np.cos(x[0])*np.cos(x[1])*np.cos(x[2])
                u[2] = np.cos(x[0])*np.cos(x[1])*np.sin(x[2])
                return u

            # Create function-representation of rhs
            V_rhs = dfem.FunctionSpace(msh, elmt_rhs)
            f_rhs = dfem.Function(V_rhs)

            if (cell.geometric_dimension() == 2):
                f_rhs.interpolate(rhs_2D)
            elif (cell.geometric_dimension() == 3):
                f_rhs.interpolate(rhs_3D)
            else:
                assert False
        else:
            dvol = ufl.Measure("dx", domain=msh,
                               metadata={"quadrature_degree": 3*degree})

            # Interpolation function
            def rhs_2D(x):
                u = np.zeros((1, x.shape[1]), dtype=PETSc.ScalarType)
                u[0] = np.sin(x[0])*np.sin(x[1])
                return u

            def rhs_3D(x):
                u = np.zeros((1, x.shape[1]), dtype=PETSc.ScalarType)
                u[0] = np.sin(x[0])*np.sin(x[1])*np.sin(x[2])
                return u

            # Create function-representation of rhs
            V_rhs = dfem.FunctionSpace(msh, elmt_rhs)
            func_rhs = dfem.Function(V_rhs)

            if (cell.geometric_dimension() == 2):
                func_rhs.interpolate(rhs_2D)
            elif (cell.geometric_dimension() == 3):
                func_rhs.interpolate(rhs_3D)
            else:
                assert False

            # Create source term by taking divergence of vector
            f_rhs = dfem.Constant(
                msh, PETSc.ScalarType(1.5))*ufl.grad(func_rhs)
    else:
        if (test_func == 'const'):
            dvol = ufl.dx
            f_rhs = dfem.Constant(msh, PETSc.ScalarType(2))
        elif (test_func == 'ufl'):
            dvol = ufl.Measure("dx", domain=msh,
                               metadata={"quadrature_degree": 3*degree})
            x = ufl.SpatialCoordinate(msh)

            if (cell.geometric_dimension() == 2):
                f_rhs = ufl.sin(x[0])*ufl.sin(x[1])
            elif (cell.geometric_dimension() == 3):
                f_rhs = ufl.sin(x[0])*ufl.sin(x[1])*ufl.sin(x[2])
            else:
                assert False
        elif (test_func == 'function'):
            dvol = ufl.dx

            # Interpolation function
            def rhs_2D(x):
                u = np.zeros((1, x.shape[1]), dtype=PETSc.ScalarType)
                u[0] = np.sin(x[0])*np.sin(x[1])
                return u

            def rhs_3D(x):
                u = np.zeros((1, x.shape[1]), dtype=PETSc.ScalarType)
                u[0] = np.sin(x[0])*np.sin(x[1])*np.sin(x[2])
                return u

            # Create function-representation of rhs
            V_rhs = dfem.FunctionSpace(msh, elmt_rhs)
            f_rhs = dfem.Function(V_rhs)

            if (cell.geometric_dimension() == 2):
                f_rhs.interpolate(rhs_2D)
            elif (cell.geometric_dimension() == 3):
                f_rhs.interpolate(rhs_3D)
            else:
                assert False
        else:
            dvol = ufl.Measure("dx", domain=msh,
                               metadata={"quadrature_degree": 3*degree})

            # Interpolation function
            def rhs_2D(x):
                u = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
                u[0] = np.sin(x[0])*np.sin(x[1])
                u[1] = np.cos(x[0])*np.cos(x[1])
                return u

            def rhs_3D(x):
                u = np.zeros((3, x.shape[1]), dtype=PETSc.ScalarType)
                u[0] = np.sin(x[0])*np.sin(x[1])*np.sin(x[2])
                u[1] = np.cos(x[0])*np.cos(x[1])*np.cos(x[2])
                u[2] = np.cos(x[0])*np.cos(x[1])*np.sin(x[2])
                return u

            # Create function-representation of rhs
            V_rhs = dfem.FunctionSpace(msh, elmt_rhs)
            func_rhs = dfem.Function(V_rhs)

            if (cell.geometric_dimension() == 2):
                func_rhs.interpolate(rhs_2D)
            elif (cell.geometric_dimension() == 3):
                func_rhs.interpolate(rhs_3D)
            else:
                assert False

            # Create source term by taking divergence of vector
            f_rhs = dfem.Constant(msh, PETSc.ScalarType(1.5))*ufl.div(func_rhs)

    list_l = []
    list_sol = []
    list_sol_ref = []
    for i in range(0, n_lhs):
        l = (i+1) * ufl.inner(f_rhs, v) * dvol
        list_l.append(dfem.form(l))
        list_sol.append(dfem.Function(V))

        # Calculate refernce solution
        problem = dfem.petsc.LinearProblem(a, l, bcs=[], petsc_options={"ksp_type": "preonly",
                                                                        "pc_type": "lu"})
        list_sol_ref.append(problem.solve())

    # Calculate local projection
    lsolver.local_solver_cholesky(list_sol, dfem.form(a), list_l)

    # Compare solutions
    for i in range(0, n_lhs):
        assert np.allclose(
            list_sol_ref[i].vector.array, list_sol[i].vector.array)


'''Test different solver for LGS'''


@pytest.mark.parametrize("type_solver", ["lu", "cholesky"])
def test_localprojection_solvers(type_solver):
    # Create problem
    msh, cell_basix = setup_problem_projection(ufl.tetrahedron, 2)

    # Create Function space
    elmt = ufl.FiniteElement('DG', msh.ufl_cell(), 2)
    elmt_rhs = ufl.FiniteElement('DG', msh.ufl_cell(), 3)

    V = dfem.FunctionSpace(msh, elmt)
    proj_local = dfem.Function(V)

    # Linear- and bilineaform
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(u, v)*ufl.dx

    # Set right-hand-side
    dvol = ufl.Measure("dx", domain=msh, metadata={"quadrature_degree": 3*2})
    x = ufl.SpatialCoordinate(msh)
    f_rhs = ufl.sin(x[0])*ufl.sin(x[1])*ufl.sin(x[2])

    l = ufl.inner(f_rhs, v) * dvol

    # Calculate global projection
    problem = dfem.petsc.LinearProblem(a, l, bcs=[], petsc_options={"ksp_type": "preonly",
                                                                    "pc_type": "lu"})
    proj_global = problem.solve()

    # Calculate local projection
    form_a = dfem.form(a)
    form_l = dfem.form(l)

    if type_solver == 'lu':
        lsolver.local_solver_lu([proj_local], form_a, [form_l])
    elif type_solver == 'cholesky':
        lsolver.local_solver_cholesky([proj_local], form_a, [form_l])
    elif type_solver == 'cg':
        lsolver.local_solver_cg([proj_local], form_a, [form_l])
    else:
        assert False

    # Compare solutions
    assert np.allclose(proj_global.vector.array, proj_local.vector.array)


if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)
