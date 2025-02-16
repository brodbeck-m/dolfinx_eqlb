# Copyright (C) 2024 Maximilian Brodbeck
#
# This file is part of dolfinx_eqlb
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Test boundary conditions for flux-equilibration"""

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import pytest
import random
import typing

import basix
from dolfinx import default_scalar_type, default_real_type, fem, mesh
import dolfinx.fem.petsc
import ufl

from dolfinx_eqlb.base import create_hierarchic_rt
from dolfinx_eqlb.lsolver import local_projection

from dolfinx_eqlb.eqlb import homogenous_fluxbc, fluxbc, boundarydata
from dolfinx_eqlb.cpp import TimeType

from utils import (
    MeshType,
    Domain,
    create_unitsquare_builtin,
    create_unitsquare_gmsh,
    points_boundary_unitsquare,
    initialise_evaluate_function,
)


# --- Auxiliaries ---
def setup_tests(
    mesh_type: MeshType, degree: int, rt_space: str
) -> typing.Tuple[
    Domain, typing.Tuple[fem.FunctionSpace, fem.FunctionSpace], fem.Function, bool
]:
    # Create mesh
    n_cells = 5

    if mesh_type == MeshType.builtin:
        domain = create_unitsquare_builtin(
            n_cells, mesh.CellType.triangle, mesh.DiagonalType.crossed
        )
    elif mesh_type == MeshType.gmsh:
        domain = create_unitsquare_gmsh(1 / n_cells)
    else:
        raise ValueError("Unknown mesh type")

    # Initialise connectivity
    domain.mesh.topology.create_connectivity(1, 2)
    domain.mesh.topology.create_connectivity(2, 1)

    # Initialise flux space
    if rt_space == "basix":
        V = None
        V_flux = fem.functionspace(domain.mesh, ("RT", degree))
        custom_rt = False

        boundary_function = fem.Function(V_flux)
    elif rt_space == "custom":
        V = None
        V_flux = fem.functionspace(
            domain.mesh, create_hierarchic_rt(basix.CellType.triangle, degree, True)
        )
        custom_rt = True

        boundary_function = fem.Function(V_flux)
    elif rt_space == "subspace":
        elmt_flux = basix.ufl.element(
            "RT", domain.mesh.basix_cell(), degree, dtype=default_real_type
        )
        elmt_dg = basix.ufl.element(
            "DG", domain.mesh.basix_cell(), degree - 1, dtype=default_real_type
        )

        V = fem.functionspace(
            domain.mesh, basix.ufl.mixed_element([elmt_flux, elmt_dg])
        )
        V_flux = fem.functionspace(domain.mesh, elmt_flux)

        custom_rt = False

        boundary_function = fem.Function(V)
    else:
        raise ValueError("Unknown RT-space")

    return domain, (V, V_flux), boundary_function, custom_rt


# --- Test cases ---
@pytest.mark.parametrize("mesh_type", [MeshType.builtin, MeshType.gmsh])
@pytest.mark.parametrize("degree", [1, 2, 3, 4])
@pytest.mark.parametrize("rt_space", ["basix", "custom", "subspace"])
def test_boundary_data_homogenous(mesh_type: MeshType, degree: int, rt_space: str):
    """Test the homogenous boundary conditions

    Args:
        mesh_type:      The mesh type
        degree:         The degree of the RT space, onto the BCs are applied
        rt_space:       Type of RT-space
        use_projection: If True, RT DOFs are gained by projection from boundary data
    """

    # Test setup
    domain, (V, V_flux), boundary_function, custom_rt = setup_tests(
        mesh_type, degree, rt_space
    )

    # Initialise boundary facets and test-points
    list_boundary_ids = [1, 4]
    list_bcs = []

    for id in list_boundary_ids:
        # Get boundary facets
        bfcts = domain.facet_function.indices[domain.facet_function.values == id]

        # Create instance of FluxBC
        list_bcs.append(homogenous_fluxbc(bfcts))

    # Initialise boundary data
    if rt_space == "subspace":
        boundary_data = boundarydata(
            [list_bcs], [boundary_function], V.sub(0), custom_rt, [[]], True
        )
    else:
        boundary_data = boundarydata(
            [list_bcs], [boundary_function], V_flux, custom_rt, [[]], True
        )

    assert np.allclose(boundary_function.x.array, 0.0)


@pytest.mark.parametrize("mesh_type", [MeshType.builtin, MeshType.gmsh])
@pytest.mark.parametrize("degree", [1, 2, 3, 4])
@pytest.mark.parametrize("rt_space", ["basix", "custom", "subspace"])
@pytest.mark.parametrize("use_projection", [False, True])
def test_boundary_data_polynomial(
    mesh_type: MeshType, degree: int, rt_space: str, use_projection: bool
):
    """Test boundary conditions from data with know polynomial degree

    Args:
        mesh_type:      The mesh type
        degree:         The degree of the RT space, onto the BCs are applied
        rt_space:       Type of RT-space
        use_projection: If True, RT DOFs are gained by projection from boundary data
    """

    # Test setup
    domain, (V, V_flux), boundary_function, custom_rt = setup_tests(
        mesh_type, degree, rt_space
    )

    # The spatial dimension
    gdim = domain.mesh.geometry.dim

    # Initialise reference flux space
    V_ref = fem.functionspace(domain.mesh, ("DG", degree - 1, (gdim,)))

    # Initialise boundary facets and test-points
    list_boundary_ids = [1, 4]
    points_eval = points_boundary_unitsquare(domain, list_boundary_ids, degree + 1)
    plist_eval, clist_eval = initialise_evaluate_function(domain.mesh, points_eval)

    npoints_eval = int(points_eval.shape[0] / 2)

    # Set boundary degree
    for deg in range(0, degree):
        # Data boundary conditions
        etype = "DG" if deg == 0 else "Lagrange"

        func_1 = fem.Function(fem.functionspace(domain.mesh, (etype, deg, (gdim,))))
        func_1.x.array[:] = 2 * (
            np.random.rand(
                func_1.function_space.dofmap.bs
                * func_1.function_space.dofmap.index_map.size_local
            )
            + 0.1
        )
        func_2 = fem.Function(fem.functionspace(domain.mesh, (etype, deg)))
        func_2.x.array[:] = 3 * (
            np.random.rand(func_2.function_space.dofmap.index_map.size_local) + 0.3
        )

        c_1 = fem.Constant(domain.mesh, PETSc.ScalarType((1.35, 0.25)))
        c_2 = fem.Constant(domain.mesh, PETSc.ScalarType(0.75))

        x_ufl = ufl.SpatialCoordinate(domain.mesh)

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
            bfcts = domain.facet_function.indices[domain.facet_function.values == id]

            # Create instance of FluxBC
            list_bcs.append(
                fluxbc(ntrace_ufl, bfcts, V_flux, requires_projection=use_projection)
            )

        # Initialise boundary data
        if rt_space == "subspace":
            boundary_data = boundarydata(
                [list_bcs], [boundary_function], V.sub(0), custom_rt, [[]], True
            )
        else:
            boundary_data = boundarydata(
                [list_bcs], [boundary_function], V_flux, custom_rt, [[]], True
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
def test_boundary_data_general(mesh_type: MeshType, degree: int):
    """Test boundary conditions from non-polynomial data

    The boundary values are projected into the RT space. The values on
    the boundary are compared to a projection on a 1D reference space.

    Args:
        mesh_type: The mesh type
        degree:    The degree of the RT space, onto the BCs are applied
    """

    # Test setup
    domain, (_, V_flux), boundary_function, _ = setup_tests(mesh_type, degree, "basix")

    # Initialise test-points/ function evaluation
    points_eval = points_boundary_unitsquare(domain, [1, 4], degree)
    plist_eval, clist_eval = initialise_evaluate_function(domain.mesh, points_eval)

    npoints_eval = int(points_eval.shape[0] / 2)

    # set ufl-repr. of normal-trace on boundary
    x_ufl = ufl.SpatialCoordinate(domain.mesh)

    ntrace_ufl_1 = ufl.sin(4 * ufl.pi * x_ufl[1]) * ufl.exp(-x_ufl[1])
    ntrace_ufl_4 = ufl.cos(6 * ufl.pi * x_ufl[0]) * ufl.exp(-x_ufl[0])

    # Create boundary conditions
    list_bcs = []

    bfcts_1 = domain.facet_function.indices[domain.facet_function.values == 1]
    list_bcs.append(
        fluxbc(ntrace_ufl_1, bfcts_1, V_flux, True, quadrature_degree=3 * degree)
    )

    bfcts_4 = domain.facet_function.indices[domain.facet_function.values == 4]
    list_bcs.append(
        fluxbc(ntrace_ufl_4, bfcts_4, V_flux, True, quadrature_degree=3 * degree)
    )

    # Initialise boundary data
    boundary_data = boundarydata(
        [list_bcs], [boundary_function], V_flux, True, [[]], True
    )

    # Evaluate BCs on control points
    val_bfunc = boundary_function.eval(plist_eval, clist_eval)

    # --- Calculate reference solution (1D)
    # Create mesh
    domain_1d = mesh.create_unit_interval(MPI.COMM_WORLD, 5)

    # Initialise reference space
    V_ref = fem.functionspace(domain_1d, ("DG", degree - 1))

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

        problem = fem.petsc.LinearProblem(
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


@pytest.mark.parametrize("mesh_type", [MeshType.builtin, MeshType.gmsh])
@pytest.mark.parametrize("degree", [1, 2, 3, 4])
@pytest.mark.parametrize("rt_space", ["basix", "custom", "subspace"])
@pytest.mark.parametrize("use_projection", [False, True])
def test_update_boundary_data(
    mesh_type: MeshType, degree: int, rt_space: str, use_projection: bool
):
    """Test update boundary conditions from data with know polynomial degree

    Args:
        mesh_type:      The mesh type
        degree:         The degree of the RT space, onto the BCs are applied
        rt_space:       Type of RT-space
        use_projection: If True, RT DOFs are gained by projection from boundary data
    """

    # Test setup
    domain, (V, V_flux), boundary_function, custom_rt = setup_tests(
        mesh_type, degree, rt_space
    )

    # The spatial dimension
    gdim = domain.mesh.geometry.dim

    # Initialise reference flux space
    V_ref = fem.functionspace(domain.mesh, ("DG", degree - 1, (gdim,)))

    # Initialise boundary facets and test-points
    list_boundary_ids = [1, 4]
    points_eval = points_boundary_unitsquare(domain, list_boundary_ids, degree + 1)
    plist_eval, clist_eval = initialise_evaluate_function(domain.mesh, points_eval)

    npoints_eval = int(points_eval.shape[0] / 2)

    # Set boundary degree
    for deg in range(0, degree):
        # Data boundary conditions
        etype = "DG" if deg == 0 else "Lagrange"

        func_1 = fem.Function(fem.functionspace(domain.mesh, (etype, deg, (gdim,))))
        func_1.x.array[:] = 2 * (
            np.random.rand(
                func_1.function_space.dofmap.bs
                * func_1.function_space.dofmap.index_map.size_local
            )
            + 0.1
        )
        func_2 = fem.Function(fem.functionspace(domain.mesh, (etype, deg)))
        func_2.x.array[:] = 3 * (
            np.random.rand(func_2.function_space.dofmap.index_map.size_local) + 0.3
        )

        c_1 = fem.Constant(domain.mesh, PETSc.ScalarType((1.35, 0.25)))
        c_2 = fem.Constant(domain.mesh, PETSc.ScalarType(0.75))

        x_ufl = ufl.SpatialCoordinate(domain.mesh)

        # Data for the transient behavior
        time = fem.Constant(domain.mesh, default_scalar_type(0.0))
        ct_1 = random.uniform(0.5, 5.5)
        ct_2 = random.uniform(0.1, 2.1)

        # Create ufl-repr. of boundary condition
        tfunc_ufl = ct_1 * time + ct_2

        ntrace_ufl = (
            ufl.inner(func_1, c_1)
            + ((x_ufl[0] ** deg) + (x_ufl[1] ** deg))
            + func_2 * c_2
        )

        # Create boundary conditions
        list_bcs = []

        # BC with TimeType.timedependent
        bfcts = domain.facet_function.indices[domain.facet_function.values == 1]
        list_bcs.append(
            fluxbc(
                tfunc_ufl * ntrace_ufl,
                bfcts,
                V_flux,
                requires_projection=use_projection,
                transient_behavior=TimeType.timedependent,
            )
        )

        # BC with TimeType.timefunction
        bfcts = domain.facet_function.indices[domain.facet_function.values == 4]
        list_bcs.append(
            fluxbc(
                ntrace_ufl,
                bfcts,
                V_flux,
                requires_projection=use_projection,
                transient_behavior=TimeType.timefunction,
            )
        )

        # Initialise boundary data
        if rt_space == "subspace":
            boundary_data = boundarydata(
                [list_bcs], [boundary_function], V.sub(0), custom_rt, [[]], True
            )
        else:
            boundary_data = boundarydata(
                [list_bcs], [boundary_function], V_flux, custom_rt, [[]], True
            )

        # Update boundary data
        time.value = default_scalar_type(0.35)
        tfuncs_cnst = fem.Constant(
            domain.mesh, default_scalar_type([1.0, ct_1 * 0.35 + ct_2])
        )

        boundary_data.update([tfuncs_cnst._cpp_object])

        # Interpolate BC into test-space
        rhs_ref = ufl.as_vector([-tfunc_ufl * ntrace_ufl, tfunc_ufl * ntrace_ufl])
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


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
