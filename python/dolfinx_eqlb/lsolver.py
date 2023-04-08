# --- Imports ---
import numpy as np
import typing


import dolfinx.fem as dfem
import dolfinx.mesh as dmesh
import ufl

import dolfinx_eqlb.cpp


# --- Preparation of input data ---
def prepare_input(list_l: typing.List[typing.Any],
                  list_func: typing.List[dfem.function.Function]):
    # Determine number of inputs and check input data
    n_lhs = len(list_l)

    if (len(list_func) != n_lhs):
        raise RuntimeError('Missmatching inputs!')

    # Create list of cpp-functions
    list_func_cpp = []

    for func in list_func:
        list_func_cpp.append(func._cpp_object)

    return list_func_cpp


# --- Local solvers ---

def local_solver_lu(list_func: typing.List[dfem.function.Function],
                    a: dfem.forms.form_types, list_l: typing.List[typing.Any]):
    # Prepare input for local solver
    list_func_cpp = prepare_input(list_l, list_func)

    # Perform local solution
    dolfinx_eqlb.cpp.local_solver_lu(list_func_cpp, a, list_l)


def local_solver_cholesky(list_func: typing.List[dfem.function.Function],
                          a: dfem.forms.form_types, list_l: typing.List[typing.Any]):
    # Prepare input for local solver
    list_func_cpp = prepare_input(list_l, list_func)

    # Perform local solution
    dolfinx_eqlb.cpp.local_solver_cholesky(list_func_cpp, a, list_l)


def local_solver_cg(list_func: typing.List[dfem.function.Function],
                    a: dfem.forms.form_types, list_l: typing.List[typing.Any]):
    # Prepare input for local solver
    list_func_cpp = prepare_input(list_l, list_func)

    raise RuntimeError('CG-solver currently not supported')
