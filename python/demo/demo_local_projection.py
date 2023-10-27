""" Demo local projection

This demo uses a element-local solver to project two non-polynomial 
functions into an element-wise polynomial space.

"""

# --- Imports ---
from mpi4py import MPI
import numpy as np

import dolfinx.mesh as dmesh
import dolfinx.fem as dfem
import ufl

from dolfinx_eqlb.lsolver import local_projection

# --- Setup problem ---
# Generate mesh
domain = dmesh.create_unit_square(
    MPI.COMM_WORLD, 8, 8, dmesh.CellType.triangle, dmesh.GhostMode.shared_facet
)

# Set function space, into which the solution is projected
V_proj = dfem.FunctionSpace(domain, ("DG", 2))

# Define non-linear functions
x_ufl = ufl.SpatialCoordinate(domain)

nlfunc_1 = ufl.cos(2 * ufl.pi * x_ufl[0]) * ufl.cos(2 * ufl.pi * x_ufl[1])
nlfunc_2 = ufl.sin(2 * ufl.pi * x_ufl[0]) * ufl.cos(3 * ufl.pi * x_ufl[1])

# --- Global projection ---
# Volume integrator with specified order
dvol = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 4})

# Define equation system
u = ufl.TrialFunction(V_proj)
v = ufl.TestFunction(V_proj)

a = ufl.inner(u, v) * ufl.dx
l_1 = ufl.inner(nlfunc_1, v) * dvol
l_2 = ufl.inner(nlfunc_2, v) * dvol

# Solve projection
problem_1 = dfem.petsc.LinearProblem(a, l_1, bcs=[], petsc_options={"ksp_type": "cg"})
uproj_1_global = problem_1.solve()

problem_2 = dfem.petsc.LinearProblem(a, l_2, bcs=[], petsc_options={"ksp_type": "cg"})
uproj_2_global = problem_2.solve()

# --- Local projection ---
results_local_proj = local_projection(V_proj, [nlfunc_1, nlfunc_2], quadrature_degree=4)

if np.allclose(uproj_1_global.x.array, results_local_proj[0].x.array) and np.allclose(
    uproj_2_global.x.array, results_local_proj[1].x.array
):
    print("Local projection and global projection are equal!")
