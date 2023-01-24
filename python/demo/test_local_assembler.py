import cffi
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import time

import dolfinx
import dolfinx.fem as dfem
import dolfinx.mesh as dmesh
import ufl

import dolfinx_eqlb.cpp

ffi = cffi.FFI()

# --- Test pybind ---
dolfinx_eqlb.cpp.test_pybind()


# --- Test local solver ---
inp_nelmt = 10
inp_degree = 1
inp_nrets = 1


def create_problem(n_elmt, degree):
    # Create mesh
    mesh = dmesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([1, 1])],
                                  [n_elmt, n_elmt], cell_type=dmesh.CellType.triangle,
                                  diagonal=dmesh.DiagonalType.left)

    outfile = dolfinx.io.XDMFFile(
        MPI.COMM_WORLD, "Test_PartialAssembly.xdmf", "w")
    outfile.write_mesh(mesh)
    outfile.close()

    # Creat function space
    element_v = ufl.FiniteElement('CG', mesh.ufl_cell(), degree)
    V = dfem.FunctionSpace(mesh, element_v)

    # Set linear- and bilinear form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    # l = ufl.inner(dfem.Constant(mesh, PETSc.ScalarType(1.5)), v) * ufl.dx
    l = v * ufl.dx

    return V, a, l


def stiffness_elmt_calc(V, a):
    # Initialize timings
    t_form = 0
    t_assemble = 0
    t_extract_mat = 0

    # Set mesh
    mesh = V.mesh

    t_form -= time.perf_counter()

    # Compile forms
    a_form, _, _ = dolfinx.jit.ffcx_jit(mesh.comm, a)

    # Get stiffness kernel
    a_kernel_cell = a_form.integrals(dolfinx.fem.IntegralType.cell)[
        0].tabulate_tensor_float64

    t_form += time.perf_counter()

    # Initialize storage for local matrizies
    dof_per_elmt = V.dofmap.dof_layout.block_size * V.dofmap.dof_layout.num_dofs
    np.zeros((dof_per_elmt, dof_per_elmt), dtype=PETSc.ScalarType)

    # Extract local matrizies
    t_extract_mat -= time.perf_counter()

    # Extract mesh data
    x = mesh.geometry.x
    x_dofs = mesh.geometry.dofmap

    # Input cell integral
    geometry = np.zeros((1, 3, 3))

    A_local = np.zeros((dof_per_elmt, dof_per_elmt), dtype=PETSc.ScalarType)

    for i in range(0, mesh.topology.index_map(0).size_local):
        # Calculate element stiffness
        c = x_dofs.links(i)

        # Pack geometry
        for j in range(3):
            for k in range(3):
                geometry[0, j, k] = x[c[j], k]

        # No coefficients, no constants
        A_local.fill(0.0)
        a_kernel_cell(ffi.cast("double *", ffi.from_buffer(A_local)), ffi.NULL,
                      ffi.NULL,
                      ffi.cast("double *", ffi.from_buffer(geometry)), ffi.NULL,
                      ffi.NULL)

        if i == 1:
            print(A_local)

    t_extract_mat += time.perf_counter()

    return np.array([t_form, t_extract_mat], dtype=np.double)


for n in range(0, inp_nrets):
    # Setup problem
    V, a, l = create_problem(inp_nelmt, inp_degree)
    func = dfem.Function(V)

    # Call local_solve
    timing_cpp = 0

    timing_cpp -= time.perf_counter()
    dolfinx_eqlb.cpp.local_solver(func._cpp_object, dfem.form(a), dfem.form(l))
    timing_cpp += time.perf_counter()

    # Test against built-in assembler
    timing_python = stiffness_elmt_calc(V, a)

outfile = dolfinx.io.XDMFFile(
    MPI.COMM_WORLD, "Test_cpp.xdmf", "w")
outfile.write_mesh(V.mesh)
outfile.write_function(func)
outfile.close()

PETSc.Sys.Print("Local:  Form: {}, Local stiffness: {}".format(
    timing_python[0]/inp_nrets, timing_python[1]/inp_nrets))
PETSc.Sys.Print("Time c++: {}".format(timing_cpp/inp_nrets))
