# Copyright (C) 2024 Maximilian Brodbeck
#
# This file is part of dolfinx_eqlb
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Test local projections into discontinuous FE-spaces with multiple RHS"""

import numpy as np
from petsc4py import PETSc
import pytest

from dolfinx import fem
import ufl

from dolfinx_eqlb.lsolver import local_solver_cholesky
from test_localsolver_projection import setup_problem_projection


@pytest.mark.parametrize(
    "cell", [ufl.triangle, ufl.quadrilateral, ufl.tetrahedron, ufl.hexahedron]
)
@pytest.mark.parametrize("is_vectorvalued", [False, True])
@pytest.mark.parametrize("degree", [1, 2])
def test_localprojection_multiple_rhs(cell, is_vectorvalued, degree):
    """Test local projection with different righ-hand-sides (RHS)

    Args:
        cell:            The cell type
        is_vectorvalued: Flag for vector-valued FE-spaces
        degree:          The degree of the FE-space
    """
    # Create problem
    msh, cell_basix = setup_problem_projection(cell, 3)

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
    V = fem.FunctionSpace(msh, elmt)

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
    V_rhs3 = fem.FunctionSpace(msh, elmt_rhs3)
    V_rhs4 = fem.FunctionSpace(msh, elmt_rhs4)
    func_rhs3 = fem.Function(V_rhs3)
    func_rhs4 = fem.Function(V_rhs4)
    f_rhs = []

    if is_vectorvalued:
        if cell.geometric_dimension() == 2:
            # Required interpoations
            func_rhs3.interpolate(rhs_2D)
            func_rhs4.interpolate(rhs_2D_scal)

            # Definitions
            f_rhs.append(fem.Constant(msh, PETSc.ScalarType(np.array([2, 5]))))
            f_rhs.append(
                ufl.as_vector(
                    [ufl.sin(x[0]) * ufl.sin(x[1]), ufl.cos(x[0]) * ufl.cos(x[1])]
                )
            )
            f_rhs.append(fem.Constant(msh, PETSc.ScalarType(1.5)) * func_rhs3)
            f_rhs.append(ufl.grad(func_rhs4))
        elif cell.geometric_dimension() == 3:
            # Required interpoations
            func_rhs3.interpolate(rhs_3D)
            func_rhs4.interpolate(rhs_3D_scal)

            f_rhs.append(fem.Constant(msh, PETSc.ScalarType(np.array([2, 5, 1]))))
            f_rhs.append(
                ufl.as_vector(
                    [
                        ufl.sin(x[0]) * ufl.sin(x[1]) * ufl.sin(x[2]),
                        ufl.cos(x[0]) * ufl.cos(x[1]) * ufl.cos(x[2]),
                        ufl.sin(x[0]) * ufl.cos(x[1]) * ufl.sin(x[2]),
                    ]
                )
            )
            f_rhs.append(fem.Constant(msh, PETSc.ScalarType(1.5)) * func_rhs3)
            f_rhs.append(ufl.grad(func_rhs4))
        else:
            assert False
    else:
        if cell.geometric_dimension() == 2:
            # Required interpoations
            func_rhs3.interpolate(rhs_2D_scal)
            func_rhs4.interpolate(rhs_2D)

            # Definitions
            f_rhs.append(fem.Constant(msh, PETSc.ScalarType(2)))
            f_rhs.append(ufl.sin(x[0]) * ufl.sin(x[1]))
            f_rhs.append(fem.Constant(msh, PETSc.ScalarType(1.5)) * func_rhs3)
            f_rhs.append(ufl.div(func_rhs4))
        elif cell.geometric_dimension() == 3:
            # Required interpoations
            func_rhs3.interpolate(rhs_3D_scal)
            func_rhs4.interpolate(rhs_3D)

            f_rhs.append(fem.Constant(msh, PETSc.ScalarType(2)))
            f_rhs.append(ufl.sin(x[0]) * ufl.sin(x[1]) * ufl.sin(x[2]))
            f_rhs.append(fem.Constant(msh, PETSc.ScalarType(1.5)) * func_rhs3)
            f_rhs.append(ufl.div(func_rhs4))
        else:
            assert False

    list_l = []
    list_sol = []
    list_sol_ref = []
    for i in range(0, len(f_rhs)):
        l = ufl.inner(f_rhs[i], v) * dvol
        list_l.append(fem.form(l))
        list_sol.append(fem.Function(V))

        # Calculate refernce solution
        problem = fem.petsc.LinearProblem(
            a, l, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
        )
        list_sol_ref.append(problem.solve())

    # Calculate local projection
    local_solver_cholesky(list_sol, fem.form(a), list_l)

    # Compare solutions
    for i in range(0, len(f_rhs)):
        assert np.allclose(list_sol_ref[i].vector.array, list_sol[i].vector.array)


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
