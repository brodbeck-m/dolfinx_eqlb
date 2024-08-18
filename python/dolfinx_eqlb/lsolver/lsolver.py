# Copyright (C) 2024 Maximilian Brodbeck
#
# This file is part of dolfinx_eqlb
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Solve cell-wise equation system without global assembly"""

import typing

import dolfinx.fem as dfem

import dolfinx_eqlb.cpp as eqlb_cpp


def prepare_input(
    list_l: typing.List[typing.Any], list_func: typing.List[dfem.Function]
) -> typing.List[typing.Any]:
    """Prepare input for local solvers"""

    # Determine number of inputs and check input data
    n_lhs = len(list_l)

    if len(list_func) != n_lhs:
        raise RuntimeError("Missmatching inputs!")

    # Create list of cpp-functions
    list_func_cpp = []

    for func in list_func:
        list_func_cpp.append(func._cpp_object)

    return list_func_cpp


def local_solver_lu(
    list_func: typing.List[dfem.Function],
    a: dfem.FormMetaClass,
    list_l: typing.List[dfem.FormMetaClass],
):
    """Cell local solver based on the LU decomposition

    Args:
        list_func: The solution functions
        a:         The bilinear form
        list_l:    The list of linear forms
    """

    # Prepare input for local solver
    list_func_cpp = prepare_input(list_l, list_func)

    # Perform local solution
    eqlb_cpp.local_solver_lu(list_func_cpp, a, list_l)


def local_solver_cholesky(
    list_func: typing.List[dfem.Function],
    a: dfem.FormMetaClass,
    list_l: typing.List[dfem.FormMetaClass],
):
    """Cell local solver based on the Cholesky decomposition

    Args:
        list_func: The solution functions
        a:         The bilinear form
        list_l:    The list of linear forms
    """

    # Prepare input for local solver
    list_func_cpp = prepare_input(list_l, list_func)

    # Perform local solution
    eqlb_cpp.local_solver_cholesky(list_func_cpp, a, list_l)


def local_solver_cg(
    list_func: typing.List[dfem.Function],
    a: dfem.FormMetaClass,
    list_l: typing.List[dfem.FormMetaClass],
):
    """Cell local solver based on a CG solver

    Args:
        list_func: The solution functions
        a:         The bilinear form
        list_l:    The list of linear forms
    """

    # Prepare input for local solver
    list_func_cpp = prepare_input(list_l, list_func)

    # Perform local solution
    eqlb_cpp.local_solver_cg(list_func_cpp, a, list_l)
