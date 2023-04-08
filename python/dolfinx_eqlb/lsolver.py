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

# --- Local projection ---


def local_projection(V_target: dfem.function.FunctionSpace,
                     list_rhs: typing.List[typing.Any],
                     quadrature_degree=None):
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
        dvol = ufl.Measure("dx", domain=V_target.mesh, metadata={
                           "quadrature_degree": quadrature_degree})

    # Linear form/ and solution
    for i in range(0, n_lhs):
        # Linear form
        list_l.append(dfem.form(ufl.inner(list_rhs[i], v) * dvol))

        # Solution function
        func = dfem.Function(func)
        list_sol.append(func)
        list_sol_cpp.append(func._cpp_object)

    # --- Solve projection locally (Cholesky, as problem is SPD)
    dolfinx_eqlb.cpp.local_solver_cholesky(list_sol_cpp, a, list_l)

    return list_sol
