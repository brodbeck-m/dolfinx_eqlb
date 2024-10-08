# Copyright (C) 2024 Maximilian Brodbeck
#
# This file is part of dolfinx_eqlb
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Test local projections into discontinuous FE-spaces"""

from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import pytest

import basix
import basix.ufl_wrapper
from dolfinx import fem, mesh
import ufl

from dolfinx_eqlb.lsolver import (
    local_solver_lu,
    local_solver_cholesky,
    local_solver_cg,
    local_projection,
)


# --- Auxiliary functions ---
def setup_problem_projection(cell, n_elmt):
    # Set cell variables
    if cell == ufl.triangle:
        cell_mesh = mesh.CellType.triangle
        cell_basix = basix.CellType.triangle
    elif cell == ufl.tetrahedron:
        cell_mesh = mesh.CellType.tetrahedron
        cell_basix = basix.CellType.tetrahedron
    elif cell == ufl.quadrilateral:
        cell_mesh = mesh.CellType.quadrilateral
        cell_basix = basix.CellType.quadrilateral
    elif cell == ufl.hexahedron:
        cell_mesh = mesh.CellType.hexahedron
        cell_basix = basix.CellType.hexahedron
    else:
        assert False

    # Create mesh
    if cell.geometric_dimension() == 1:
        assert False
    elif cell.geometric_dimension() == 2:
        msh = mesh.create_unit_square(
            MPI.COMM_WORLD, n_elmt, n_elmt, cell_mesh, mesh.GhostMode.shared_facet
        )
    else:
        msh = mesh.create_unit_cube(
            MPI.COMM_WORLD,
            n_elmt,
            n_elmt,
            n_elmt,
            cell_mesh,
            mesh.GhostMode.shared_facet,
        )

    return msh, cell_basix


# --- The tests ---
@pytest.mark.parametrize(
    "cell", [ufl.triangle, ufl.quadrilateral, ufl.tetrahedron, ufl.hexahedron]
)
@pytest.mark.parametrize("is_vectorvalued", [False, True])
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("test_func", ["const", "ufl", "function", "function_ufl"])
def test_localprojection_lagrange(cell, is_vectorvalued, degree, test_func):
    """Test local projection into a discontinuous Lagrange space

    Args:
        cell:            The cell-type
        is_vectorvalued: Flag for vector-valued FE-spaces
        degree:          The degree of the function-space
        test_func:       The type of test data
    """

    # Create problem
    msh, cell_basix = setup_problem_projection(cell, 3)

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

    V = fem.FunctionSpace(msh, elmt)
    proj_local = fem.Function(V)

    # Linear- and bilineaform
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(u, v) * ufl.dx

    if is_vectorvalued:
        if test_func == "const":
            dvol = ufl.dx
            quad_deg = None
            if cell.geometric_dimension() == 2:
                f_rhs = fem.Constant(msh, PETSc.ScalarType(np.array([2, 5])))
            elif cell.geometric_dimension() == 3:
                f_rhs = fem.Constant(msh, PETSc.ScalarType(np.array([2, 5, 1])))
            else:
                assert False
        elif test_func == "ufl":
            dvol = ufl.Measure(
                "dx", domain=msh, metadata={"quadrature_degree": 3 * degree}
            )
            quad_deg = 3 * degree
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
            quad_deg = None

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
            V_rhs = fem.FunctionSpace(msh, elmt_rhs)
            f_rhs = fem.Function(V_rhs)

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
            quad_deg = 3 * degree

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
            V_rhs = fem.FunctionSpace(msh, elmt_rhs)
            func_rhs = fem.Function(V_rhs)

            if cell.geometric_dimension() == 2:
                func_rhs.interpolate(rhs_2D)
            elif cell.geometric_dimension() == 3:
                func_rhs.interpolate(rhs_3D)
            else:
                assert False

            # Create source term by taking divergence of vector
            f_rhs = fem.Constant(msh, PETSc.ScalarType(1.5)) * ufl.grad(func_rhs)
    else:
        if test_func == "const":
            dvol = ufl.dx
            quad_deg = None
            f_rhs = fem.Constant(msh, PETSc.ScalarType(2))
        elif test_func == "ufl":
            dvol = ufl.Measure(
                "dx", domain=msh, metadata={"quadrature_degree": 3 * degree}
            )
            quad_deg = 3 * degree
            x = ufl.SpatialCoordinate(msh)

            if cell.geometric_dimension() == 2:
                f_rhs = ufl.sin(x[0]) * ufl.sin(x[1])
            elif cell.geometric_dimension() == 3:
                f_rhs = ufl.sin(x[0]) * ufl.sin(x[1]) * ufl.sin(x[2])
            else:
                assert False
        elif test_func == "function":
            dvol = ufl.dx
            quad_deg = None

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
            V_rhs = fem.FunctionSpace(msh, elmt_rhs)
            f_rhs = fem.Function(V_rhs)

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
            quad_deg = 3 * degree

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
            V_rhs = fem.FunctionSpace(msh, elmt_rhs)
            func_rhs = fem.Function(V_rhs)

            if cell.geometric_dimension() == 2:
                func_rhs.interpolate(rhs_2D)
            elif cell.geometric_dimension() == 3:
                func_rhs.interpolate(rhs_3D)
            else:
                assert False

            # Create source term by taking divergence of vector
            f_rhs = fem.Constant(msh, PETSc.ScalarType(1.5)) * ufl.div(func_rhs)

    l = ufl.inner(f_rhs, v) * dvol

    # Calculate global projection
    problem = fem.petsc.LinearProblem(
        a, l, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    proj_global = problem.solve()

    # Calculate local projection
    proj_local = local_projection(V, [f_rhs], quadrature_degree=quad_deg)

    # Compare solutions
    assert np.allclose(proj_global.vector.array, proj_local[0].vector.array)


@pytest.mark.parametrize("cell", [ufl.triangle, ufl.tetrahedron])
@pytest.mark.parametrize(
    "family_basix", [basix.ElementFamily.RT, basix.ElementFamily.BDM]
)
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("test_func", ["const", "ufl", "function"])
def test_localprojection_hdiv(cell, family_basix, degree, test_func):
    """Test projection into discontinuous H(div) spaces

    Args:
        cell:          The cell-type
        family_basix:  The element type
        degree:        The degree of the function-space
        test_func:     The type of test data
    """

    # Create problem
    msh, cell_basix = setup_problem_projection(cell, 3)

    # Create Function space
    elmt_basix = basix.create_element(
        family_basix, cell_basix, degree, basix.LagrangeVariant.equispaced, True
    )
    elmt = basix.ufl_wrapper.BasixElement(elmt_basix)

    V = fem.FunctionSpace(msh, elmt)
    proj_local = fem.Function(V)

    # Linear- and bilineaform
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(u, v) * ufl.dx

    if test_func == "const":
        dvol = ufl.dx
        quad_deg = None
        if cell.geometric_dimension() == 1:
            assert False
        elif cell.geometric_dimension() == 2:
            func = fem.Constant(msh, PETSc.ScalarType(np.array([2, 5])))

        else:
            func = fem.Constant(msh, PETSc.ScalarType(np.array([2, 5, 9])))
    elif test_func == "ufl":
        dvol = ufl.Measure("dx", domain=msh, metadata={"quadrature_degree": 3 * degree})
        quad_deg = 3 * degree

        x = ufl.SpatialCoordinate(msh)

        if cell.geometric_dimension() == 1:
            assert False
        elif cell.geometric_dimension() == 2:
            func = ufl.as_vector(
                [ufl.sin(x[0]) * ufl.sin(x[1]), ufl.cos(x[0]) * ufl.cos(x[1])]
            )

        else:
            func = ufl.as_vector(
                [
                    ufl.sin(x[0]) * ufl.sin(x[1]) * ufl.sin(x[2]),
                    ufl.cos(x[0]) * ufl.cos(x[1]) * ufl.cos(x[2]),
                    ufl.sin(x[0]) * ufl.cos(x[1]) * ufl.sin(x[2]),
                ]
            )
    else:
        dvol = ufl.dx
        quad_deg = None

        # Create rhs-function
        elmt_rhs = ufl.VectorElement("P", msh.ufl_cell(), degree)
        V_rhs = fem.FunctionSpace(msh, elmt_rhs)
        func = fem.Function(V_rhs)

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

        if cell.geometric_dimension() == 2:
            func.interpolate(rhs_2D)
        elif cell.geometric_dimension() == 3:
            func.interpolate(rhs_3D)
        else:
            assert False

    l = ufl.inner(func, v) * dvol

    # Calculate global projection
    problem = fem.petsc.LinearProblem(
        a, l, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    proj_global = problem.solve()

    # Calculate local projection
    proj_local = local_projection(V, [func], quadrature_degree=quad_deg)

    # Compare solutions
    assert np.allclose(proj_global.vector.array, proj_local[0].vector.array)


@pytest.mark.parametrize("type_solver", ["lu", "cholesky", "cg"])
def test_localprojection_solvers(type_solver):
    """Test different solvers for local projection

    Args:
        type_solver: The type of solver
    """
    # Create problem
    msh, cell_basix = setup_problem_projection(ufl.tetrahedron, 2)

    # Create Function space
    elmt = ufl.FiniteElement("DG", msh.ufl_cell(), 2)

    V = fem.FunctionSpace(msh, elmt)
    proj_local = fem.Function(V)

    # Linear- and bilineaform
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(u, v) * ufl.dx

    # Set right-hand-side
    dvol = ufl.Measure("dx", domain=msh, metadata={"quadrature_degree": 3 * 2})
    x = ufl.SpatialCoordinate(msh)
    f_rhs = ufl.sin(x[0]) * ufl.sin(x[1]) * ufl.sin(x[2])

    l = ufl.inner(f_rhs, v) * dvol

    # Calculate global projection
    problem = fem.petsc.LinearProblem(
        a, l, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    proj_global = problem.solve()

    # Calculate local projection
    form_a = fem.form(a)
    form_l = fem.form(l)

    if type_solver == "lu":
        local_solver_lu([proj_local], form_a, [form_l])
    elif type_solver == "cholesky":
        local_solver_cholesky([proj_local], form_a, [form_l])
    elif type_solver == "cg":
        local_solver_cg([proj_local], form_a, [form_l])
    else:
        assert False

    # Compare solutions
    assert np.allclose(proj_global.vector.array, proj_local.vector.array)


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
