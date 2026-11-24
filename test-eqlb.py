from mpi4py import MPI

from dolfinx import cpp, fem, io, mesh
import ufl
from basix.ufl import element, mixed_element

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
el_rt = element("RT", msh.basix_cell(), degree)
el_dg = element("DG", msh.basix_cell(), degree - 1)

W = fem.functionspace(msh, mixed_element([el_rt, el_dg]))

u, p = ufl.TrialFunctions(W)
v, q = ufl.TestFunctions(W)

V_hat = fem.functionspace(msh, ("P", 1))
hat_function = fem.Function(V_hat, name="hat")

project_sig = fem.Function(fem.functionspace(msh, ("DG", degree - 1, (2,))))
project_f = fem.Function(fem.functionspace(msh, ("DG", degree - 1)))

# numpy array
project_sig.x.array[:] = 1.0
project_f.x.array[:] = 2.5


mass_matrix = fem.form(ufl.inner(u, v) * ufl.dx)
constr_matrix = fem.form(ufl.inner(p, ufl.div(v)) * ufl.dx)
lagr_mltp = fem.form(q * ufl.dx)

ls = [
    fem.form((hat_function * ufl.inner(project_sig, v)) * ufl.dx),
    fem.form(
        (
            hat_function * project_f * q
            + ufl.inner(ufl.grad(hat_function), project_sig) * q
        )
        * ufl.dx
    ),
]

# The equilibrated flux
fluxes = [fem.Function(W)]

print(fluxes[0].x.array)

# Perform local solution
_cpp.equilibration_ev(
    [f._cpp_object for f in fluxes],
    [mass_matrix._cpp_object, constr_matrix._cpp_object, lagr_mltp._cpp_object],
    [l._cpp_object for l in ls],
)

print(fluxes[0].x.array)

sigma_h, _ = fluxes[0].split()
err_div = fem.assemble_scalar(fem.form((ufl.div(sigma_h) - project_f) ** 2 * ufl.dx))
print("L2 divergence error:", err_div**0.5)
