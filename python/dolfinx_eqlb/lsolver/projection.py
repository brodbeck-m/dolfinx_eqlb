# Copyright (C) 2024 Maximilian Brodbeck
#
# This file is part of dolfinx_eqlb
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Projection into FE spaces with cell wise support without global assembly"""

# --- Imports ---
import typing

import dolfinx.fem as dfem
import ufl

from dolfinx_eqlb.cpp import local_solver_cholesky


def local_projection(
    V: dfem.FunctionSpace,
    data: typing.List[typing.Any],
    quadrature_degree: typing.Optional[int] = None,
) -> typing.List[dfem.Function]:
    """Projection into DG spaces

    Solves

            (u, v) = (rhs, v) with u,v in V

    for multiple RHS and arbitrary FE spaces V, with cell local support.

    Args:
        V:                 The target function-space
        data:              The data to project into the space V
        quadrature_degree: The quadrature Degree

    Returns:
        A List of functions into which the data is projected
    """
    # Number of LHS
    n_lhs = len(data)

    # Initialisation
    list_sol = []
    list_sol_cpp = []
    list_l = []

    # --- Setup variational problem
    # Trial- and testfunctions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Bilinear form
    a = dfem.form(ufl.inner(u, v) * ufl.dx)

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
    for i in range(0, n_lhs):
        # Linear form
        list_l.append(dfem.form(ufl.inner(data[i], v) * dvol))

        # Solution function
        func = dfem.Function(V)
        list_sol.append(func)
        list_sol_cpp.append(func._cpp_object)

    # --- Solve projection locally (Cholesky, as problem is SPD)
    local_solver_cholesky(list_sol_cpp, a, list_l)

    return list_sol
