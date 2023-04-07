from mpi4py import MPI
import numpy as np
import petsc4py
from petsc4py import PETSc
import time

import basix
import basix.ufl_wrapper

import dolfinx
import dolfinx.mesh as dmesh
import dolfinx.fem as dfem

import dolfinx_eqlb.cpp

import ufl

# --- Input parameters
# Mesh type
sdisc_ctype = ufl.tetrahedron

# Mesh resolution
sdisc_nelmt = 50

# Element type ('Lagrange', 'VectorLagrange', 'RT', 'BDM')
elmt_type = 'Lagrange'
elmt_order = 3

# Input projection (const, ufl, func_cg, func_dg, ufl_func_cg)
rhs_type = 'ufl'
rhs_retry = 1

# Timing
timing_nretry = 3

# --- Utility routines ---


def create_fespace_discontinous(family_basix, cell_basix, degree):
    elmt_basix = basix.create_element(family_basix, cell_basix, degree,
                                      basix.LagrangeVariant.equispaced, True)
    return basix.ufl_wrapper.BasixElement(elmt_basix)


def set_rhs(rhs_type, msh, is_vectorial=False):
    # Set function rhs:
    def rhs_x(pkg, x, spacedim):
        if (spacedim == 1):
            return pkg.sin(x[0])
        elif (spacedim == 2):
            return pkg.sin(x[0]) * pkg.sin(x[1])
        else:
            return pkg.sin(x[0]) * pkg.sin(x[1]) * pkg.sin(x[2])

    def rhs_y(pkg, x, spacedim):
        if (spacedim == 1):
            return pkg.cos(x[0])
        elif (spacedim == 2):
            return pkg.cos(x[0]) * pkg.cos(x[1])
        else:
            return pkg.cos(x[0]) * pkg.cos(x[1]) * pkg.cos(x[2])

    def rhs_z(pkg, x, spacedim):
        if (spacedim == 1):
            return pkg.cos(x[0])
        elif (spacedim == 2):
            return pkg.cos(x[0]) * pkg.sin(x[1])
        else:
            return pkg.cos(x[0]) * pkg.sin(x[1]) * pkg.cos(x[2])

    def rhs_pkg(pkg, x, spacedim, is_vectorial):
        if pkg == ufl:
            if is_vectorial:
                if (spacedim == 2):
                    rhs = ufl.as_vector([rhs_x(ufl, x, spacedim),
                                         rhs_y(ufl, x, spacedim)])
                else:
                    rhs = ufl.as_vector([rhs_x(ufl, x, spacedim),
                                         rhs_y(ufl, x, spacedim),
                                         rhs_z(ufl, x, spacedim)])
                return rhs
            else:
                return rhs_x(ufl, x, spacedim)
        else:
            if is_vectorial:
                rhs = np.zeros((3, x.shape[1]), dtype=PETSc.ScalarType)
                rhs[0] = rhs_x(np, x, spacedim)
                rhs[1] = rhs_y(np, x, spacedim)
                rhs[2] = rhs_z(np, x, spacedim)
                return rhs[0:spacedim, :]
            else:
                return rhs_x(np, x, spacedim)

    def rhs_const(msh, spacedim, is_vectorial):
        if is_vectorial:
            if (spacedim == 2):
                rhs = dfem.Constant(msh, PETSc.ScalarType((5, 2)))
            else:
                rhs = dfem.Constant(msh, PETSc.ScalarType((5, 2, 3)))
            return rhs
        else:
            return dfem.Constant(msh, PETSc.ScalarType(5))

    if (rhs_type == 'const'):
        func = rhs_const(msh, msh.geometry.dim, is_vectorial)
    elif (rhs_type == 'ufl'):
        x = ufl.SpatialCoordinate(msh)
        func = rhs_pkg(ufl, x, msh.geometry.dim, is_vectorial)
    elif (rhs_type == 'func_cg'):
        raise NotImplementedError('Projection of CG-Function not implemented!')
    elif (rhs_type == 'func_dg'):
        raise NotImplementedError('Projection of DG-Function not implemented!')
    elif (rhs_type == 'ufl_func_cg'):
        raise NotImplementedError(
            'Projection of the gardient of a CG-Function not implemented!')
    else:
        raise NotImplementedError('Unknown RHS-Type!')

    return func


def setup_problem(cell, n_elmt, elmt_type, elmt_degree, rhs_type):
    # Set cell variables
    if cell == ufl.triangle:
        cell_mesh = dmesh.CellType.triangle
        cell_basix = basix.CellType.triangle
    elif cell == ufl.tetrahedron:
        cell_mesh = dmesh.CellType.tetrahedron
        cell_basix = basix.CellType.tetrahedron
    elif cell == ufl.quadrilateral:
        cell_mesh = dmesh.CellType.quadrilateral
        cell_basix = basix.CellType.quadrilateral
    elif cell == ufl.hexahedron:
        cell_mesh = dmesh.CellType.hexahedron
        cell_basix = basix.CellType.hexahedron
    else:
        raise NotImplementedError('Unsupported cell-type!')

    # Create mesh
    if (cell.geometric_dimension() == 1):
        raise NotImplementedError('Projection in 1D not supported!')
    elif (cell.geometric_dimension() == 2):
        msh = dmesh.create_unit_square(MPI.COMM_WORLD, n_elmt, n_elmt, cell_mesh,
                                       dmesh.GhostMode.shared_facet)
    else:
        msh = dmesh.create_unit_cube(MPI.COMM_WORLD, n_elmt, n_elmt, n_elmt, cell_mesh,
                                     dmesh.GhostMode.shared_facet)

    # Create finite element
    is_vectorial = False

    if (elmt_type == 'Lagrange'):
        if (cell == ufl.triangle or cell == ufl.tetrahedron):
            elmt = ufl.FiniteElement('DG', msh.ufl_cell(), elmt_degree)
        else:
            elmt = ufl.FiniteElement('DQ', msh.ufl_cell(), elmt_degree)
    elif (elmt_type == 'VectorLagrange'):
        is_vectorial = True
        if (cell == ufl.triangle or cell == ufl.tetrahedron):
            elmt = ufl.VectorElement('DG', msh.ufl_cell(), elmt_degree)
        else:
            elmt = ufl.VectorElement('DQ', msh.ufl_cell(), elmt_degree)
    elif elmt_type == 'RT':
        is_vectorial = True
        elmt = create_fespace_discontinous(
            basix.ElementFamily.RT, cell_basix, elmt_degree)
    elif elmt_type == 'BDM':
        is_vectorial = True
        elmt = create_fespace_discontinous(
            basix.ElementFamily.BDM, cell_basix, elmt_degree)
    else:
        raise NotImplementedError(
            'Projection into {elmt_type}-space not supported!')

    # Create function space
    V = dfem.FunctionSpace(msh, elmt)

    # Set linear- and bilinear form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(u, v)*ufl.dx

    if (rhs_type == 'ufl'):
        dvol = ufl.Measure("dx", domain=msh, metadata={"quadrature_degree":
                                                       3*elmt_degree})
    else:
        dvol = ufl.dx

    l = ufl.inner(set_rhs(rhs_type, msh, is_vectorial=is_vectorial), v)*dvol

    return V, a, l


# --- Projection routines ---
def global_projection(V, a, l, rhs_retry=1, solver_settings=None):
    # Solution Function
    u_proj = dfem.Function(V)

    # Assemble global equation system
    A = dfem.petsc.assemble_matrix(dfem.form(a))
    A.assemble()
    form_l = dfem.form(l)
    L = dfem.petsc.create_vector(form_l)

    # Set solver
    args = "-ksp_type cg -pc_type hypre -pc_hypre_type euclid -ksp_converged_reason"
    petsc4py.init(args)
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setOperators(A)

    # Solve projection repeatedly
    for ii in range(0, rhs_retry):
        # Recalculate LHS
        with L.localForm() as loc_L:
            loc_L.set(0)
        dfem.petsc.assemble_vector(L, form_l)

        # Solve equation system
        solver(L, u_proj.vector)

    return u_proj


def local_projection(V, a, l, rhs_retry=1):
    # Solution Function
    u_proj = dfem.Function(V)

    # Assemble global equation system
    form_a = dfem.form(a)
    form_l = dfem.form(l)

    # Solve projection repeatedly
    for ii in range(0, rhs_retry):
        # Solve equation system
        dolfinx_eqlb.cpp.local_solver(
            u_proj._cpp_object, form_a, form_l)

    return u_proj


# --- Local vs. global projection ---
# Initialize timing
time_proj_global = np.zeros(timing_nretry)
time_proj_local = np.zeros(timing_nretry)

# Time projections
for n in range(0, timing_nretry):
    # Create mesh
    V_proj, a, l = setup_problem(sdisc_ctype, sdisc_nelmt,
                                 elmt_type, elmt_order, rhs_type)

    # Global projection
    time_proj_global[n] -= time.perf_counter()
    u_global = global_projection(V_proj, a, l, rhs_retry=rhs_retry)
    time_proj_global[n] += time.perf_counter()

    time_proj_local[n] -= time.perf_counter()
    u_local = local_projection(V_proj, a, l, rhs_retry=rhs_retry)
    time_proj_local[n] += time.perf_counter()


# Output results
if (np.allclose(u_global.vector.array, u_local.vector.array)):
    print('Local and global approach have same results!')
else:
    raise ValueError('Projected results does not match!')

print("Global projection: {}, Local projection: {}".format(
    min(time_proj_global), min(time_proj_local)))
