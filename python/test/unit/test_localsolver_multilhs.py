from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import pytest

import dolfinx.fem as dfem

import ufl

from dolfinx_eqlb.lsolver import local_solver_cholesky
from test_localsolver_projection import setup_problem_projection

"""Test projection with multiple LHS (same type)"""


@pytest.mark.parametrize(
    "cell", [ufl.triangle, ufl.quadrilateral, ufl.tetrahedron, ufl.hexahedron]
)
@pytest.mark.parametrize("is_vectorvalued", [False, True])
@pytest.mark.parametrize("degree", [1])
@pytest.mark.parametrize("test_func", ["const", "ufl", "function", "function_ufl"])
@pytest.mark.parametrize("n_lhs", [2, 3])
def test_localprojection_multi_lhs(cell, is_vectorvalued, degree, test_func, n_lhs):
    # Create problem
    msh, cell_basix = setup_problem_projection(cell, 2)

    # Create Function space
    if cell == ufl.triangle or cell == ufl.tetrahedron:
        if is_vectorvalued:
            elmt = ufl.VectorElement("DG", msh.ufl_cell(), degree)
            if test_func == "function":
                elmt_rhs = ufl.VectorElement("DG", msh.ufl_cell(), degree + 1)
            elif test_func == "function_ufl":
                elmt_rhs = ufl.FiniteElement("DG", msh.ufl_cell(), degree + 1)
        else:
            elmt = ufl.FiniteElement("DG", msh.ufl_cell(), degree)
            if test_func == "function":
                elmt_rhs = ufl.FiniteElement("DG", msh.ufl_cell(), degree + 1)
            elif test_func == "function_ufl":
                elmt_rhs = ufl.VectorElement("DG", msh.ufl_cell(), degree + 1)
    else:
        if is_vectorvalued:
            elmt = ufl.VectorElement("DQ", msh.ufl_cell(), degree)
            if test_func == "function":
                elmt_rhs = ufl.VectorElement("DQ", msh.ufl_cell(), degree + 1)
            elif test_func == "function_ufl":
                elmt_rhs = ufl.FiniteElement("DQ", msh.ufl_cell(), degree + 1)
        else:
            elmt = ufl.FiniteElement("DQ", msh.ufl_cell(), degree)
            if test_func == "function":
                elmt_rhs = ufl.FiniteElement("DQ", msh.ufl_cell(), degree + 1)
            elif test_func == "function_ufl":
                elmt_rhs = ufl.VectorElement("DQ", msh.ufl_cell(), degree + 1)
    V = dfem.FunctionSpace(msh, elmt)

    # Linear- and bilineaform
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(u, v) * ufl.dx

    if is_vectorvalued:
        if test_func == "const":
            dvol = ufl.dx
            if cell.geometric_dimension() == 2:
                f_rhs = dfem.Constant(msh, PETSc.ScalarType(np.array([2, 5])))
            elif cell.geometric_dimension() == 3:
                f_rhs = dfem.Constant(msh, PETSc.ScalarType(np.array([2, 5, 1])))
            else:
                assert False
        elif test_func == "ufl":
            dvol = ufl.Measure(
                "dx", domain=msh, metadata={"quadrature_degree": 3 * degree}
            )
            x = ufl.SpatialCoordinate(msh)

            if cell.geometric_dimension() == 2:
                f_rhs = ufl.as_vector(
                    [ufl.sin(x[0]) * ufl.sin(x[1]), ufl.cos(x[0]) * ufl.cos(x[1])]
                )
            elif cell.geometric_dimension() == 3:
                f_rhs = ufl.as_vector(
                    [
                        ufl.sin(x[0]) * ufl.sin(x[1]) * ufl.sin(x[2]),
                        ufl.cos(x[0]) * ufl.cos(x[1]) * ufl.cos(x[2]),
                        ufl.sin(x[0]) * ufl.cos(x[1]) * ufl.sin(x[2]),
                    ]
                )
            else:
                assert False
        elif test_func == "function":
            dvol = ufl.dx

            # Interpolation function
            def rhs_2D(x):
                u = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
                u[0] = np.sin(x[0]) * np.sin(x[1])
                u[1] = np.cos(x[0]) * np.cos(x[1])
                return u

            def rhs_3D(x):
                u = np.zeros((3, x.shape[1]), dtype=PETSc.ScalarType)
                u[0] = np.sin(x[0]) * np.sin(x[1]) * np.sin(x[2])
                u[1] = np.cos(x[0]) * np.cos(x[1]) * np.cos(x[2])
                u[2] = np.cos(x[0]) * np.cos(x[1]) * np.sin(x[2])
                return u

            # Create function-representation of rhs
            V_rhs = dfem.FunctionSpace(msh, elmt_rhs)
            f_rhs = dfem.Function(V_rhs)

            if cell.geometric_dimension() == 2:
                f_rhs.interpolate(rhs_2D)
            elif cell.geometric_dimension() == 3:
                f_rhs.interpolate(rhs_3D)
            else:
                assert False
        else:
            dvol = ufl.Measure(
                "dx", domain=msh, metadata={"quadrature_degree": 3 * degree}
            )

            # Interpolation function
            def rhs_2D(x):
                u = np.zeros((1, x.shape[1]), dtype=PETSc.ScalarType)
                u[0] = np.sin(x[0]) * np.sin(x[1])
                return u

            def rhs_3D(x):
                u = np.zeros((1, x.shape[1]), dtype=PETSc.ScalarType)
                u[0] = np.sin(x[0]) * np.sin(x[1]) * np.sin(x[2])
                return u

            # Create function-representation of rhs
            V_rhs = dfem.FunctionSpace(msh, elmt_rhs)
            func_rhs = dfem.Function(V_rhs)

            if cell.geometric_dimension() == 2:
                func_rhs.interpolate(rhs_2D)
            elif cell.geometric_dimension() == 3:
                func_rhs.interpolate(rhs_3D)
            else:
                assert False

            # Create source term by taking divergence of vector
            f_rhs = dfem.Constant(msh, PETSc.ScalarType(1.5)) * ufl.grad(func_rhs)
    else:
        if test_func == "const":
            dvol = ufl.dx
            f_rhs = dfem.Constant(msh, PETSc.ScalarType(2))
        elif test_func == "ufl":
            dvol = ufl.Measure(
                "dx", domain=msh, metadata={"quadrature_degree": 3 * degree}
            )
            x = ufl.SpatialCoordinate(msh)

            if cell.geometric_dimension() == 2:
                f_rhs = ufl.sin(x[0]) * ufl.sin(x[1])
            elif cell.geometric_dimension() == 3:
                f_rhs = ufl.sin(x[0]) * ufl.sin(x[1]) * ufl.sin(x[2])
            else:
                assert False
        elif test_func == "function":
            dvol = ufl.dx

            # Interpolation function
            def rhs_2D(x):
                u = np.zeros((1, x.shape[1]), dtype=PETSc.ScalarType)
                u[0] = np.sin(x[0]) * np.sin(x[1])
                return u

            def rhs_3D(x):
                u = np.zeros((1, x.shape[1]), dtype=PETSc.ScalarType)
                u[0] = np.sin(x[0]) * np.sin(x[1]) * np.sin(x[2])
                return u

            # Create function-representation of rhs
            V_rhs = dfem.FunctionSpace(msh, elmt_rhs)
            f_rhs = dfem.Function(V_rhs)

            if cell.geometric_dimension() == 2:
                f_rhs.interpolate(rhs_2D)
            elif cell.geometric_dimension() == 3:
                f_rhs.interpolate(rhs_3D)
            else:
                assert False
        else:
            dvol = ufl.Measure(
                "dx", domain=msh, metadata={"quadrature_degree": 3 * degree}
            )

            # Interpolation function
            def rhs_2D(x):
                u = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
                u[0] = np.sin(x[0]) * np.sin(x[1])
                u[1] = np.cos(x[0]) * np.cos(x[1])
                return u

            def rhs_3D(x):
                u = np.zeros((3, x.shape[1]), dtype=PETSc.ScalarType)
                u[0] = np.sin(x[0]) * np.sin(x[1]) * np.sin(x[2])
                u[1] = np.cos(x[0]) * np.cos(x[1]) * np.cos(x[2])
                u[2] = np.cos(x[0]) * np.cos(x[1]) * np.sin(x[2])
                return u

            # Create function-representation of rhs
            V_rhs = dfem.FunctionSpace(msh, elmt_rhs)
            func_rhs = dfem.Function(V_rhs)

            if cell.geometric_dimension() == 2:
                func_rhs.interpolate(rhs_2D)
            elif cell.geometric_dimension() == 3:
                func_rhs.interpolate(rhs_3D)
            else:
                assert False

            # Create source term by taking divergence of vector
            f_rhs = dfem.Constant(msh, PETSc.ScalarType(1.5)) * ufl.div(func_rhs)

    list_l = []
    list_sol = []
    list_sol_ref = []
    for i in range(0, n_lhs):
        l = (i + 1) * ufl.inner(f_rhs, v) * dvol
        list_l.append(dfem.form(l))
        list_sol.append(dfem.Function(V))

        # Calculate refernce solution
        problem = dfem.petsc.LinearProblem(
            a, l, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
        )
        list_sol_ref.append(problem.solve())

    # Calculate local projection
    local_solver_cholesky(list_sol, dfem.form(a), list_l)

    # Compare solutions
    for i in range(0, n_lhs):
        assert np.allclose(list_sol_ref[i].vector.array, list_sol[i].vector.array)


"""Test projection with multiple LHS (different type)"""


@pytest.mark.parametrize(
    "cell", [ufl.triangle, ufl.quadrilateral, ufl.tetrahedron, ufl.hexahedron]
)
@pytest.mark.parametrize("is_vectorvalued", [False, True])
@pytest.mark.parametrize("degree", [1])
def test_localprojection_diff_lhs(cell, is_vectorvalued, degree):
    # Create problem
    msh, cell_basix = setup_problem_projection(cell, 1)

    # Create Function space
    if cell == ufl.triangle or cell == ufl.tetrahedron:
        if is_vectorvalued:
            elmt = ufl.VectorElement("DG", msh.ufl_cell(), degree)
            elmt_rhs3 = ufl.VectorElement("DG", msh.ufl_cell(), degree + 1)
            elmt_rhs4 = ufl.FiniteElement("DG", msh.ufl_cell(), degree + 1)
        else:
            elmt = ufl.FiniteElement("DG", msh.ufl_cell(), degree)
            elmt_rhs3 = ufl.FiniteElement("DG", msh.ufl_cell(), degree + 1)
            elmt_rhs4 = ufl.VectorElement("DG", msh.ufl_cell(), degree + 1)
    else:
        if is_vectorvalued:
            elmt = ufl.VectorElement("DQ", msh.ufl_cell(), degree)
            elmt_rhs3 = ufl.VectorElement("DQ", msh.ufl_cell(), degree + 1)
            elmt_rhs4 = ufl.FiniteElement("DQ", msh.ufl_cell(), degree + 1)
        else:
            elmt = ufl.FiniteElement("DQ", msh.ufl_cell(), degree)
            elmt_rhs3 = ufl.FiniteElement("DQ", msh.ufl_cell(), degree + 1)
            elmt_rhs4 = ufl.VectorElement("DQ", msh.ufl_cell(), degree + 1)
    V = dfem.FunctionSpace(msh, elmt)

    # --- Bilinearform
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(u, v) * ufl.dx

    # --- Linearforms
    # Integrator and spacial position
    dvol = ufl.Measure("dx", domain=msh, metadata={"quadrature_degree": 3 * degree})
    x = ufl.SpatialCoordinate(msh)

    # Function definition
    def rhs_2D_scal(x):
        u = np.zeros((1, x.shape[1]), dtype=PETSc.ScalarType)
        u[0] = np.sin(x[0]) * np.sin(x[1])
        return u

    def rhs_2D(x):
        u = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
        u[0] = np.sin(x[0]) * np.sin(x[1])
        u[1] = np.cos(x[0]) * np.cos(x[1])
        return u

    def rhs_3D_scal(x):
        u = np.zeros((1, x.shape[1]), dtype=PETSc.ScalarType)
        u[0] = np.sin(x[0]) * np.sin(x[1]) * np.sin(x[2])
        return u

    def rhs_3D(x):
        u = np.zeros((3, x.shape[1]), dtype=PETSc.ScalarType)
        u[0] = np.sin(x[0]) * np.sin(x[1]) * np.sin(x[2])
        u[1] = np.cos(x[0]) * np.cos(x[1]) * np.cos(x[2])
        u[2] = np.cos(x[0]) * np.cos(x[1]) * np.sin(x[2])
        return u

    # FE-Functions
    V_rhs3 = dfem.FunctionSpace(msh, elmt_rhs3)
    V_rhs4 = dfem.FunctionSpace(msh, elmt_rhs4)
    func_rhs3 = dfem.Function(V_rhs3)
    func_rhs4 = dfem.Function(V_rhs4)
    f_rhs = []

    if is_vectorvalued:
        if cell.geometric_dimension() == 2:
            # Required interpoations
            func_rhs3.interpolate(rhs_2D)
            func_rhs4.interpolate(rhs_2D_scal)

            # Definitions
            f_rhs.append(dfem.Constant(msh, PETSc.ScalarType(np.array([2, 5]))))
            f_rhs.append(
                ufl.as_vector(
                    [ufl.sin(x[0]) * ufl.sin(x[1]), ufl.cos(x[0]) * ufl.cos(x[1])]
                )
            )
            f_rhs.append(dfem.Constant(msh, PETSc.ScalarType(1.5)) * func_rhs3)
            f_rhs.append(ufl.grad(func_rhs4))
        elif cell.geometric_dimension() == 3:
            # Required interpoations
            func_rhs3.interpolate(rhs_3D)
            func_rhs4.interpolate(rhs_3D_scal)

            f_rhs.append(dfem.Constant(msh, PETSc.ScalarType(np.array([2, 5, 1]))))
            f_rhs.append(
                ufl.as_vector(
                    [
                        ufl.sin(x[0]) * ufl.sin(x[1]) * ufl.sin(x[2]),
                        ufl.cos(x[0]) * ufl.cos(x[1]) * ufl.cos(x[2]),
                        ufl.sin(x[0]) * ufl.cos(x[1]) * ufl.sin(x[2]),
                    ]
                )
            )
            f_rhs.append(dfem.Constant(msh, PETSc.ScalarType(1.5)) * func_rhs3)
            f_rhs.append(ufl.grad(func_rhs4))
        else:
            assert False
    else:
        if cell.geometric_dimension() == 2:
            # Required interpoations
            func_rhs3.interpolate(rhs_2D_scal)
            func_rhs4.interpolate(rhs_2D)

            # Definitions
            f_rhs.append(dfem.Constant(msh, PETSc.ScalarType(2)))
            f_rhs.append(ufl.sin(x[0]) * ufl.sin(x[1]))
            f_rhs.append(dfem.Constant(msh, PETSc.ScalarType(1.5)) * func_rhs3)
            f_rhs.append(ufl.div(func_rhs4))
        elif cell.geometric_dimension() == 3:
            # Required interpoations
            func_rhs3.interpolate(rhs_3D_scal)
            func_rhs4.interpolate(rhs_3D)

            f_rhs.append(dfem.Constant(msh, PETSc.ScalarType(2)))
            f_rhs.append(ufl.sin(x[0]) * ufl.sin(x[1]) * ufl.sin(x[2]))
            f_rhs.append(dfem.Constant(msh, PETSc.ScalarType(1.5)) * func_rhs3)
            f_rhs.append(ufl.div(func_rhs4))
        else:
            assert False

    list_l = []
    list_sol = []
    list_sol_ref = []
    for i in range(0, len(f_rhs)):
        l = ufl.inner(f_rhs[i], v) * dvol
        list_l.append(dfem.form(l))
        list_sol.append(dfem.Function(V))

        # Calculate refernce solution
        problem = dfem.petsc.LinearProblem(
            a, l, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
        )
        list_sol_ref.append(problem.solve())

    # Calculate local projection
    local_solver_cholesky(list_sol, dfem.form(a), list_l)

    # Compare solutions
    for i in range(0, len(f_rhs)):
        assert np.allclose(list_sol_ref[i].vector.array, list_sol[i].vector.array)


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
