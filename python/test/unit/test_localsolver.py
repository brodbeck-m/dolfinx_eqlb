# Copyright (C) 2024 Maximilian Brodbeck
#
# This file is part of dolfinx_eqlb
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Test local projections into discontinuous FE-spaces"""

from enum import Enum
from mpi4py import MPI
import numpy as np
import pytest
import typing

import basix.ufl
import dolfinx
import dolfinx.fem.petsc
from dolfinx import fem, mesh
import ufl

from dolfinx_eqlb.lsolver import local_projection


# --- The test routine ---
class RhsType(Enum):
    pure_ufl = 0
    constants = 1
    coefficients = 2
    mixed = 3


def set_rhs(
    rhs_type: RhsType, msh: mesh.Mesh, dim: int, degree_rhs: typing.Optional[int]
):
    # The spatial coordinates
    x = ufl.SpatialCoordinate(msh)

    # Set basic right-hand side
    if dim == 1:
        f = ufl.sin(x[0]) * ufl.sin(x[1])
    elif dim == 2:
        f = ufl.as_vector(
            [ufl.sin(x[0]) * ufl.sin(x[1]), ufl.cos(x[0]) * ufl.cos(x[1])]
        )
    else:
        f = ufl.as_vector(
            [
                ufl.sin(x[0]) * ufl.sin(x[1]) * ufl.sin(x[2]),
                ufl.cos(x[0]) * ufl.cos(x[1]) * ufl.cos(x[2]),
                ufl.sin(x[0]) * ufl.cos(x[1]) * ufl.sin(x[2]),
            ]
        )

    # Constants
    if (rhs_type == RhsType.constants) or (rhs_type == RhsType.mixed):
        # Multiplicative factor
        f *= fem.Constant(msh, dolfinx.default_scalar_type((1.5)))

        # Additive factor
        if dim == 1:
            f += fem.Constant(msh, dolfinx.default_scalar_type(0.3))
        elif dim == 2:
            f += fem.Constant(msh, dolfinx.default_scalar_type((0.3, 0.2)))
        elif dim == 3:
            f += fem.Constant(msh, dolfinx.default_scalar_type((0.3, 0.2, 0.1)))

    # Coefficients
    if (rhs_type == RhsType.coefficients) or (rhs_type == RhsType.mixed):
        # Multiplicative factor
        V1 = fem.functionspace(msh, ("Lagrange", degree_rhs))
        ndofs1 = V1.dofmap.index_map_bs * V1.dofmap.index_map.size_local

        f1 = fem.Function(V1)
        f1.x.array[:] = 0.5 * (np.random.rand(ndofs1) + 0.1)

        f *= f1

        # Additive factor
        if dim == 1:
            V2 = fem.functionspace(msh, ("Lagrange", degree_rhs + 1))
        else:
            V2 = fem.functionspace(msh, ("Lagrange", degree_rhs + 1, (dim,)))

        ndofs2 = V2.dofmap.index_map_bs * V2.dofmap.index_map.size_local

        f2 = fem.Function(V2)
        f2.x.array[:] = np.random.rand(ndofs2) + 0.2

        f += f2

    return f


def compare_projections(
    cell: mesh.CellType,
    space: str,
    degree: int,
    rhs_type: RhsType,
    with_block_size: typing.Optional[bool] = False,
    with_multiple_rhs: typing.Optional[bool] = False,
) -> None:
    """Implementation of the projection test

    Args:
        cell:            The cell-type
        space:           The target function-space
        degree:          The degree of the target function-space
        rhs:             The right-hand side
        with_block_size: If true, a vector-valued space with dim=gdim is created
    """

    # The mesh
    nelmt = 3

    if cell == mesh.CellType.triangle or cell == mesh.CellType.quadrilateral:
        msh = mesh.create_unit_square(MPI.COMM_WORLD, nelmt, nelmt, cell)
    else:
        msh = mesh.create_unit_cube(MPI.COMM_WORLD, nelmt, nelmt, nelmt, cell)

    # The function-space
    if with_block_size:
        element = basix.ufl.element(
            space,
            msh.basix_cell(),
            degree,
            shape=(msh.geometry.dim,),
            discontinuous=True,
            dtype=dolfinx.default_real_type,
        )
        if space == "Lagrange":
            dim = msh.geometry.dim
        else:
            raise NotImplementedError("Tensor valued spaces not tested")
    else:
        element = basix.ufl.element(
            space,
            msh.basix_cell(),
            degree,
            discontinuous=True,
            dtype=dolfinx.default_real_type,
        )

        if space == "Lagrange":
            dim = 1
        else:
            dim = msh.geometry.dim

    V_target = fem.functionspace(msh, element)

    # The data
    if with_multiple_rhs:
        fs = [set_rhs(rhs_type, msh, dim, degree) for _ in range(msh.geometry.dim)]
    else:
        fs = [set_rhs(rhs_type, msh, dim, degree)]

    # Specify quadrature degree
    qdegree = 3 * degree
    dvol = ufl.Measure("dx", domain=msh, metadata={"quadrature_degree": qdegree})

    # Local projection
    fs_h_local = local_projection(V_target, fs, quadrature_degree=qdegree)

    # Projection using build-in methods
    u = ufl.TrialFunction(V_target)
    v = ufl.TestFunction(V_target)

    a = ufl.inner(u, v) * ufl.dx

    for i, f in enumerate(fs):
        l = ufl.inner(f, v) * dvol

        problem = fem.petsc.LinearProblem(
            a, l, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
        )
        f_h_ref = problem.solve()

        assert np.allclose(fs_h_local[i].x.array, f_h_ref.x.array)


# --- The tests ---
@pytest.mark.parametrize("cell", [mesh.CellType.triangle, mesh.CellType.tetrahedron])
@pytest.mark.parametrize("space", ["Lagrange", "RT", "BDM"])
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize(
    "rhs_type", [RhsType.constants, RhsType.coefficients, RhsType.mixed]
)
@pytest.mark.parametrize("multiple_rhs", [False, True])
def test_target_spaces(cell, space, degree, rhs_type, multiple_rhs):
    """Test local projection into discontinuous target spaces

    Args:
        cell:  The cell-type
        space: The target function-space
        degree: The degree of the target function-space
    """

    if space == "Lagrange":
        compare_projections(cell, space, degree, rhs_type, False, multiple_rhs)
        compare_projections(cell, space, degree, rhs_type, True, multiple_rhs)
    else:
        compare_projections(cell, space, degree, rhs_type, False, multiple_rhs)


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
