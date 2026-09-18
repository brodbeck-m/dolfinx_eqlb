from mpi4py import MPI

from dolfinx import cpp, fem, io, mesh
import ufl

import dolfinx_eqlb.cpp as _cpp

from dolfinx_eqlb.lsolver import local_projection

import faulthandler

faulthandler.enable()

# --- Parameters ---
cell_type = mesh.CellType.triangle
degree = 2
# ------------------

# Create mesh
msh = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2, cell_type)

with io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh)

# Compute mesh information
gdim = msh.geometry.dim
fdim = gdim - 1

msh.topology.create_entity_permutations()
msh.topology.create_connectivity(0, fdim)
msh.topology.create_connectivity(0, gdim)
msh.topology.create_connectivity(gdim, 0)
msh.topology.create_connectivity(fdim, gdim)
msh.topology.create_connectivity(gdim, 0)
msh.topology.create_connectivity(gdim, fdim)

# The equilibration problem
V, Q = fem.functionspace(msh, ("RT", degree)), fem.functionspace(msh, ("DG", degree))
V_hat = fem.functionspace(msh, ("P", 1))

u, p = ufl.TrialFunction(V), ufl.TrialFunction(Q)
v, q = ufl.TestFunction(V), ufl.TestFunction(Q)

mass_matrix = fem.form(ufl.inner(u, v) * ufl.dx)
constr_matrix = fem.form(ufl.inner(p, ufl.div(v)) * ufl.dx)
lagr_mltp = fem.form(q * ufl.dx)

ls = [fem.form(5 * q * ufl.dx)]

# l = (
#     -(self.hat_function * ufl.inner(-list_proj_flux[ii], v))
#     + self.hat_function * list_rhs[ii] * q
#     - ufl.inner(ufl.grad(self.hat_function), -list_proj_flux[ii]) * q
# ) * ufl.dx

# self.form_a = fem.form((ufl.inner(sig, v) - r * ufl.div(v) + ufl.div(sig) * q) * ufl.dx)
# self.form_lpen = fem.form(q_pen * ufl.dx)

# The equilibrated flux
fluxes = [fem.Function(V)]

print(fluxes[0].x.array)

# Perform local solution
_cpp.equilibration_ev(
    [f._cpp_object for f in fluxes],
    [mass_matrix._cpp_object, constr_matrix._cpp_object, lagr_mltp._cpp_object],
    [l._cpp_object for l in ls],
)

print(fluxes[0].x.array)
