import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import pytest

import basix
from dolfinx.mesh import CellType, DiagonalType
import dolfinx.mesh as dmesh
import dolfinx.fem as dfem
import ufl

from dolfinx_eqlb.elmtlib import create_hierarchic_rt
from dolfinx_eqlb.lsolver import local_projection
from dolfinx_eqlb.eqlb import fluxbc, boundarydata

from utils import (
    MeshType,
    create_unitsquare_builtin,
    create_unitsquare_gmsh,
    points_boundary_unitsquare,
    initialise_evaluate_function,
)

""" Test calculation of boundary conditions for
    a.) The polynomial case p<=deg(RT_k)-1 (with and without projection)
    b.) The non-polynomial case only with projection
"""


@pytest.mark.parametrize("mesh_type", [MeshType.builtin, MeshType.gmsh])
@pytest.mark.parametrize("degree", [1, 2, 3, 4])
@pytest.mark.parametrize("rt_space", ["basix", "custom", "subspace"])
@pytest.mark.parametrize("use_projection", [False, True])
def test_boundary_data_polynomial(mesh_type, degree, rt_space, use_projection):
    # Create mesh
    n_cells = 5

    if mesh_type == MeshType.builtin:
        geometry = create_unitsquare_builtin(
            n_cells, dmesh.CellType.triangle, dmesh.DiagonalType.crossed
        )
    elif mesh_type == MeshType.gmsh:
        geometry = create_unitsquare_gmsh(0.5)
    else:
        raise ValueError("Unknown mesh type")

    # Initialise connectivity
    geometry.mesh.topology.create_connectivity(1, 2)
    geometry.mesh.topology.create_connectivity(2, 1)

    # Initialise flux space
    if rt_space == "basix":
        V_flux = dfem.FunctionSpace(geometry.mesh, ("RT", degree))
        custom_rt = False

        boundary_function = dfem.Function(V_flux)
    elif rt_space == "custom":
        elmt_flux = basix.ufl_wrapper.BasixElement(
            create_hierarchic_rt(basix.CellType.triangle, degree, True)
        )
        V_flux = dfem.FunctionSpace(geometry.mesh, elmt_flux)
        custom_rt = True

        boundary_function = dfem.Function(V_flux)
    elif rt_space == "subspace":
        elmt_flux = ufl.FiniteElement("RT", geometry.mesh.ufl_cell(), degree)
        elmt_dg = ufl.FiniteElement("DG", geometry.mesh.ufl_cell(), degree - 1)
        V = dfem.FunctionSpace(geometry.mesh, ufl.MixedElement(elmt_flux, elmt_dg))
        V_flux = dfem.FunctionSpace(geometry.mesh, elmt_flux)
        custom_rt = False

        boundary_function = dfem.Function(V)

    # Initialise reference flux space
    V_ref = dfem.VectorFunctionSpace(geometry.mesh, ("DG", degree - 1))

    # Initialise boundary facets and test-points
    list_boundary_ids = [1, 4]
    points_eval = points_boundary_unitsquare(geometry, list_boundary_ids, degree + 1)
    plist_eval, clist_eval = initialise_evaluate_function(geometry.mesh, points_eval)

    npoints_eval = int(points_eval.shape[0] / 2)

    # Set boundary degree
    for deg in range(0, degree):
        # Data boundary conditions
        if deg == 0:
            V_vec = dfem.VectorFunctionSpace(geometry.mesh, ("DG", deg))
            func_1 = dfem.Function(V_vec)
            func_1.x.array[:] = 0

            V_scal = dfem.FunctionSpace(geometry.mesh, ("DG", deg))
            func_2 = dfem.Function(V_scal)
            func_2.x.array[:] = 0
        else:
            V_vec = dfem.VectorFunctionSpace(geometry.mesh, ("P", deg))
            func_1 = dfem.Function(V_vec)
            func_1.x.array[:] = 2 * (
                np.random.rand(V_vec.dofmap.bs * V_vec.dofmap.index_map.size_local)
                + 0.1
            )

            V_scal = dfem.FunctionSpace(geometry.mesh, ("P", deg))
            func_2 = dfem.Function(V_scal)
            func_2.x.array[:] = 3 * (
                np.random.rand(V_scal.dofmap.index_map.size_local) + 0.3
            )

        c_1 = dfem.Constant(geometry.mesh, PETSc.ScalarType((1.35, 0.25)))
        c_2 = dfem.Constant(geometry.mesh, PETSc.ScalarType(0.75))

        x_ufl = ufl.SpatialCoordinate(geometry.mesh)

        # Create ufl-repr. of boundary condition
        ntrace_ufl = (
            ufl.inner(func_1, c_1)
            + ((x_ufl[0] ** deg) + (x_ufl[1] ** deg))
            + func_2 * c_2
        )

        # Create boundary conditions
        list_bcs = []

        for id in list_boundary_ids:
            # Get boundary facets
            bfcts = geometry.facet_function.indices[
                geometry.facet_function.values == id
            ]

            # Create instance of FluxBC
            list_bcs.append(fluxbc(ntrace_ufl, bfcts, V_flux, use_projection))

        # Initialise boundary data
        if rt_space == "subspace":
            boundary_data = boundarydata(
                [list_bcs],
                [boundary_function],
                V.sub(0),
                custom_rt,
                [[]],
                True,
            )
        else:
            boundary_data = boundarydata(
                [list_bcs],
                [boundary_function],
                V_flux,
                custom_rt,
                [[]],
                True,
            )

        # Interpolate BC into test-space
        rhs_ref = ufl.as_vector([-ntrace_ufl, ntrace_ufl])
        refsol = local_projection(V_ref, [rhs_ref], quadrature_degree=2 * degree)[0]

        # Evaluate functions at comparison points
        if rt_space == "subspace":
            bfunc_flux = boundary_function.sub(0).collapse()
            val_bfunc = bfunc_flux.eval(plist_eval, clist_eval)
        else:
            val_bfunc = boundary_function.eval(plist_eval, clist_eval)

        val_ref = refsol.eval(plist_eval, clist_eval)

        assert np.allclose(val_bfunc[:npoints_eval, 0], val_ref[:npoints_eval, 0])
        assert np.allclose(val_bfunc[npoints_eval:, 1], val_ref[npoints_eval:, 1])


@pytest.mark.parametrize("mesh_type", [MeshType.builtin, MeshType.gmsh])
@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_boundary_data_general(mesh_type, degree):
    # --- Calculate boundary conditions (2D)
    # Create mesh
    n_cells = 5

    if mesh_type == MeshType.builtin:
        geometry = create_unitsquare_builtin(
            n_cells, dmesh.CellType.triangle, dmesh.DiagonalType.crossed
        )
    elif mesh_type == MeshType.gmsh:
        geometry = create_unitsquare_gmsh(1 / n_cells)
    else:
        raise ValueError("Unknown mesh type")

    # Initialise connectivity
    geometry.mesh.topology.create_connectivity(1, 2)
    geometry.mesh.topology.create_connectivity(2, 1)

    # Initialise flux space
    elmt_flux = basix.ufl_wrapper.BasixElement(
        create_hierarchic_rt(basix.CellType.triangle, degree, True)
    )
    V_flux = dfem.FunctionSpace(geometry.mesh, elmt_flux)

    boundary_function = dfem.Function(V_flux)

    # Initialise test-points/ function evaluation
    points_eval = points_boundary_unitsquare(geometry, [1, 4], degree)
    plist_eval, clist_eval = initialise_evaluate_function(geometry.mesh, points_eval)

    npoints_eval = int(points_eval.shape[0] / 2)

    # set ufl-repr. of normal-trace on boundary
    x_ufl = ufl.SpatialCoordinate(geometry.mesh)

    ntrace_ufl_1 = ufl.sin(4 * ufl.pi * x_ufl[1]) * ufl.exp(-x_ufl[1])
    ntrace_ufl_4 = ufl.cos(6 * ufl.pi * x_ufl[0]) * ufl.exp(-x_ufl[0])

    # Create boundary conditions
    list_bcs = []

    bfcts_1 = geometry.facet_function.indices[geometry.facet_function.values == 1]
    list_bcs.append(
        fluxbc(ntrace_ufl_1, bfcts_1, V_flux, True, quadrature_degree=3 * degree)
    )

    bfcts_4 = geometry.facet_function.indices[geometry.facet_function.values == 4]
    list_bcs.append(
        fluxbc(ntrace_ufl_4, bfcts_4, V_flux, True, quadrature_degree=3 * degree)
    )

    # Initialise boundary data
    boundary_data = boundarydata(
        [list_bcs],
        [boundary_function],
        V_flux,
        True,
        [[]],
        True,
        quadrature_degree=3 * degree,
    )

    # Evaluate BCs on control points
    val_bfunc = boundary_function.eval(plist_eval, clist_eval)

    # --- Calculate reference solution (1D)
    # Create mesh
    domain_1d = dmesh.create_unit_interval(MPI.COMM_WORLD, n_cells)

    # Initialise reference space
    V_ref = dfem.FunctionSpace(domain_1d, ("DG", degree - 1))

    # Initialise test-points/ function evaluation
    points_eval_1D = np.zeros((npoints_eval, 3))
    points_eval_1D[:, 0] = points_eval[0:npoints_eval, 1]
    plist_eval, clist_eval = initialise_evaluate_function(domain_1d, points_eval_1D)

    # Reference function 1D
    ntrace_ufl_1d = []
    s_ufl = ufl.SpatialCoordinate(domain_1d)[0]

    ntrace_ufl_1d.append(-ufl.sin(4 * ufl.pi * s_ufl) * ufl.exp(-s_ufl))
    ntrace_ufl_1d.append(ufl.cos(6 * ufl.pi * s_ufl) * ufl.exp(-s_ufl))

    # Projection into reference space
    u = ufl.TrialFunction(V_ref)
    v = ufl.TestFunction(V_ref)

    dvol = ufl.Measure(
        "dx", domain=domain_1d, metadata={"quadrature_degree": 3 * degree}
    )

    a = ufl.inner(u, v) * dvol

    for i in range(0, len(ntrace_ufl_1d)):
        # Solve 1D projection
        l = ufl.inner(ntrace_ufl_1d[i], v) * dvol

        problem = dfem.petsc.LinearProblem(
            a, l, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
        )
        refsol = problem.solve()

        # Evaluate reference solution
        val_ref = refsol.eval(plist_eval, clist_eval)

        # Compare boundary-condition and reference solution
        if i == 0:
            assert np.allclose(val_bfunc[:npoints_eval, 0], val_ref[:, 0])
        else:
            assert np.allclose(val_bfunc[npoints_eval:, 1], val_ref[:, 0])


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
