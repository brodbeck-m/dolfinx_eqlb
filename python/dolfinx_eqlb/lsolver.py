# --- Imports ---
import numpy as np
from mpi4py import MPI

import dolfinx.fem as dfem

import dolfinx_eqlb.cpp


def local_solver(u: dfem.function.Function, a: dfem.forms.form_types, l: dfem.forms.form_types):

    # Perform local solution
    dolfinx_eqlb.cpp.local_solver(u._cpp_object, a, l)
