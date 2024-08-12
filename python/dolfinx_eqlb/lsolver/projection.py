# Copyright (C) 2024 Maximilian Brodbeck
#
# This file is part of dolfinx_eqlb
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

# --- Imports ---
import typing

import dolfinx.fem as dfem
import ufl

from dolfinx_eqlb.cpp import local_solver_cholesky


def local_projection(
    V_target: dfem.function.FunctionSpace,
    list_rhs: typing.List[typing.Any],
    quadrature_degree=None,
):
    # Number of LHS
    n_lhs = len(list_rhs)

    # Initialisation
    list_sol = []
    list_sol_cpp = []
    list_l = []

    # --- Setup variational problem
    # Trial- and testfunctions
    u = ufl.TrialFunction(V_target)
    v = ufl.TestFunction(V_target)

    # Bilinear form
    a = dfem.form(ufl.inner(u, v) * ufl.dx)

    # Set volume integrator for LHS
    if quadrature_degree is None:
        dvol = ufl.dx
    else:
        dvol = ufl.Measure(
            "dx",
            domain=V_target.mesh,
            metadata={"quadrature_degree": quadrature_degree},
        )

    # Linear form/ and solution
    for i in range(0, n_lhs):
        # Linear form
        list_l.append(dfem.form(ufl.inner(list_rhs[i], v) * dvol))

        # Solution function
        func = dfem.Function(V_target)
        list_sol.append(func)
        list_sol_cpp.append(func._cpp_object)

    # --- Solve projection locally (Cholesky, as problem is SPD)
    local_solver_cholesky(list_sol_cpp, a, list_l)

    return list_sol
