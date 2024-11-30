# Copyright (C) 2024 Maximilian Brodbeck
#
# This file is part of dolfinx_eqlb
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Solve cell-wise equation system without global assembly"""

import typing

from dolfinx import cpp as _cpp
from dolfinx import fem

import dolfinx_eqlb.cpp as eqlb_cpp


def local_solver(
    solutions: typing.List[fem.Function], a: fem.Form, ls: typing.List[fem.Form]
):
    """Cell local solver based on the Cholesky decomposition

    A problem

                a(u, v) = l(v) with u,v in V

    is solved for multiple RHS ls sharing a common bilinear form a.

    Args:
        solutions: The solution functions
        a:         The bilinear form
        ls:        The linear forms
    """

    # Check input
    if len(solutions) != len(ls):
        raise RuntimeError("Missmatching inputs!")

    # Prepare input for local solver
    solutions_cpp = [s._cpp_object for s in solutions]
    ls_cpp = [l._cpp_object for l in ls]

    # Perform local solution
    eqlb_cpp.local_solver(solutions_cpp, a._cpp_object, ls_cpp)
