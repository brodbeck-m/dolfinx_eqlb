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

from dolfinx_eqlb import eqlb, lsolver

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
    mesh_type: MeshType,
    degree: int,
    rt_space: str,
    quadrature_degree: typing.Optional[int] = None,
) -> typing.Tuple[
    Domain,
    typing.Type[eqlb.basics.EquilibratorMetaClass],
    typing.Tuple[fem.FunctionSpace, fem.FunctionSpace],
    fem.Function,
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

    # Initialise Equilibrator
    if rt_space == "custom":
        raise ValueError("Custom RT space currently not supported.")
    else:
        from dolfinx_eqlb.eqlb.constrained_minimisation import Equilibrator

        equilibrator = Equilibrator(
            domain.mesh.ufl_domain(),
            eqlb.ProblemType.flux,
            degree,
            quadrature_degree,
        )

    # Initialise FunctionSpace
    # TODO - Avoid recreation of flux space (Equilibartor should hold mesh-independent definition)

    if rt_space == "subspace":
        elmt_dg = basix.ufl.element(
            "DG", domain.mesh.basix_cell(), degree - 1, dtype=default_real_type
        )

        V = fem.functionspace(
            domain.mesh,
            basix.ufl.mixed_element([equilibrator.element_flux(), elmt_dg]),
        )
        V_flux = fem.functionspace(domain.mesh, equilibrator.element_flux())

        boundary_function = fem.Function(V)
    else:
        V = None
        V_flux = fem.functionspace(domain.mesh, equilibrator.element_flux())

        boundary_function = fem.Function(V_flux)

    return (
        domain,
        equilibrator,
        (V, V_flux),
        boundary_function,
    )


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
    domain, equilibrator, (V, V_flux), boundary_function = setup_tests(
        mesh_type, degree, rt_space
    )

    # Initialise boundary facets and test-points
    list_boundary_ids = [1, 4]
    list_bcs = []

    for id in list_boundary_ids:
        # Get boundary facets
        bfcts = domain.facet_function.indices[domain.facet_function.values == id]

        # Create instance of FluxBC
        list_bcs.append(eqlb.homogenous_fluxbc(bfcts))

    # Initialise boundary data
    V_bc = V.sub(0) if (rt_space == "subspace") else V_flux
    boundary_data = eqlb.boundarydata(
        [list_bcs],
        [boundary_function],
        V_bc,
        [[]],
        equilibrator.kernel_data_boundary_conditions(),
        equilibrator.problem_type(),
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

    # The used quadrature degree
    qdegree = 2 * degree - 2 if use_projection else None

    # Test setup
    domain, equilibrator, (V, V_flux), boundary_function = setup_tests(
        mesh_type, degree, rt_space, qdegree
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
                eqlb.fluxbc(ntrace_ufl, bfcts, V_flux, use_projection, qdegree)
            )

        # Initialise boundary data
        V_bc = V.sub(0) if (rt_space == "subspace") else V_flux
        boundary_data = eqlb.boundarydata(
            [list_bcs],
            [boundary_function],
            V_bc,
            [[]],
            equilibrator.kernel_data_boundary_conditions(),
            equilibrator.problem_type(),
        )

        # Interpolate BC into test-space
        rhs_ref = ufl.as_vector([-ntrace_ufl, ntrace_ufl])
        refsol = lsolver.local_projection(
            V_ref, [rhs_ref], quadrature_degree=2 * degree
        )[0]

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

    # The used quadrature degree
    qdegree = 3 * degree

    # Test setup
    domain, equilibrator, (V, V_flux), boundary_function = setup_tests(
        mesh_type, degree, "basix", qdegree
    )

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
    list_bcs.append(eqlb.fluxbc(ntrace_ufl_1, bfcts_1, V_flux, True, qdegree))

    bfcts_4 = domain.facet_function.indices[domain.facet_function.values == 4]
    list_bcs.append(eqlb.fluxbc(ntrace_ufl_4, bfcts_4, V_flux, True, qdegree))

    # Initialise boundary data
    boundary_data = eqlb.boundarydata(
        [list_bcs],
        [boundary_function],
        V_flux,
        [[]],
        equilibrator.kernel_data_boundary_conditions(),
        equilibrator.problem_type(),
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

    # The used quadrature degree
    qdegree = 2 * degree - 2 if (use_projection) else None

    # Test setup
    domain, equilibrator, (V, V_flux), boundary_function = setup_tests(
        mesh_type, degree, rt_space, qdegree
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
            eqlb.fluxbc(
                tfunc_ufl * ntrace_ufl,
                bfcts,
                V_flux,
                use_projection,
                qdegree,
                eqlb.TimeType.timedependent,
            )
        )

        # BC with TimeType.timefunction
        bfcts = domain.facet_function.indices[domain.facet_function.values == 4]
        list_bcs.append(
            eqlb.fluxbc(
                ntrace_ufl,
                bfcts,
                V_flux,
                use_projection,
                qdegree,
                eqlb.TimeType.timefunction,
            )
        )

        # Initialise boundary data
        V_bc = V.sub(0) if (rt_space == "subspace") else V_flux
        boundary_data = eqlb.boundarydata(
            [list_bcs],
            [boundary_function],
            V_bc,
            [[]],
            equilibrator.kernel_data_boundary_conditions(),
            equilibrator.problem_type(),
        )

        # Update boundary data
        time.value = default_scalar_type(0.35)
        tfuncs_cnst = fem.Constant(
            domain.mesh, default_scalar_type([1.0, ct_1 * 0.35 + ct_2])
        )

        boundary_data.update([tfuncs_cnst._cpp_object])

        # Interpolate BC into test-space
        rhs_ref = ufl.as_vector([-tfunc_ufl * ntrace_ufl, tfunc_ufl * ntrace_ufl])
        refsol = lsolver.local_projection(
            V_ref, [rhs_ref], quadrature_degree=2 * degree
        )[0]

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
