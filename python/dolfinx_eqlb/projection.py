# --- Imports ---
import numpy as np
from mpi4py import MPI

import cffi

import dolfinx
import dolfinx.fem as dfem
import dolfinx.cpp


ffi = cffi.FFI()


def local_projection(a: dfem.form_types, u: dfem.function.Function):
    pass
