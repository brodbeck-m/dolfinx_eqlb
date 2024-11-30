# Copyright (C) 2024 Maximilian Brodbeck
#
# This file is part of dolfinx_eqlb
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Projection into FE spaces with cell wise support without global assembly"""

import typing

from dolfinx import fem
import ufl

from .lsolver import local_solver


def local_projection(
    V: fem.FunctionSpace,
    data: typing.List[typing.Any],
    quadrature_degree: typing.Optional[int] = None,
) -> typing.List[fem.Function]:
    """Projection data into DG space

    Project data sets into arbitrary FE spaces with cell-local support:

            (u, v) = (data_i, v) with u,v in V

    Args:
        V:                 The target function-space
        rhs:               The data to project into the space V
        quadrature_degree: The quadrature Degree

    Returns:
        A List of functions into which the data is projected
    """
    # Number of LHS
    nrhs = len(data)

    # --- Setup variational problem
    # Trial- and testfunctions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Bilinear form
    a = fem.form(ufl.inner(u, v) * ufl.dx)

    # Set volume integrator for LHS
    if quadrature_degree is None:
        dvol = ufl.dx
    else:
        dvol = ufl.Measure(
            "dx",
            domain=V.mesh,
            metadata={"quadrature_degree": quadrature_degree},
        )

    # Linear form/ and solution
    solutions = []
    ls = []

    for i in range(0, nrhs):
        # Linear form
        ls.append(fem.form(ufl.inner(data[i], v) * dvol))

        # Solution function
        solutions.append(fem.Function(V))

    # --- Solve projection locally (Cholesky, as problem is SPD)
    local_solver(solutions, a, ls)

    return solutions
