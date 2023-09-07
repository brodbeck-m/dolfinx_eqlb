import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import pytest

import basix
import dolfinx
import dolfinx.mesh as dmesh
import dolfinx.fem as dfem
import dolfinx.geometry as dgeom
import ufl

from dolfinx_eqlb.cpp import BoundaryData

from dolfinx_eqlb.elmtlib import create_hierarchic_rt
from dolfinx_eqlb.lsolver import local_projection
from dolfinx_eqlb.eqlb import fluxbc

from utils import Geometry, create_unitsquare_builtin, create_unitsquare_gmsh

""" Unitility routines """


def initialise_fct_list(geometry: Geometry, npoints_per_fct: int):
    # Extract boundary facets
    fcts_1 = geometry.facet_function.indices[geometry.facet_function.values == 1]
    fcts_4 = geometry.facet_function.indices[geometry.facet_function.values == 4]

    n_fcts = fcts_1.size

    # Initialise test-nodes
    n_cpoints = n_fcts * (npoints_per_fct)
    s_points = np.zeros(n_cpoints)

    for i in range(0, n_fcts):
        start_point = 0 + i * (1 / n_fcts)
        end_point = start_point + (1 / n_fcts)

        s_points[i * npoints_per_fct : (i + 1) * npoints_per_fct] = np.linspace(
            start_point, end_point, npoints_per_fct + 1, endpoint=False
        )[1:]

    points = np.zeros((2 * n_cpoints, 3))
    points[:n_cpoints, 1] = s_points
    points[n_cpoints:, 0] = s_points
    points[n_cpoints:, 1] = 1.0

    return [fcts_1, fcts_4], points


def initialise_eval_fe_function(domain: dmesh.Mesh, points: np.ndarray):
    bb_tree = dgeom.BoundingBoxTree(domain, domain.topology.dim)

    cells = []
    points_on_proc = []
    # Find cells whose bounding-box collide with the the points
    cell_candidates = dgeom.compute_collisions(bb_tree, points)
    # Choose one of the cells that contains the point
    colliding_cells = dgeom.compute_colliding_cells(domain, cell_candidates, points)
    for i, point in enumerate(points):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])

    points_on_proc = np.array(points_on_proc, dtype=np.float64)

    return points_on_proc, cells


@pytest.mark.parametrize("mesh_type", ["builtin"])
@pytest.mark.parametrize("degree", [1, 2, 3, 4])
@pytest.mark.parametrize("rt_space", ["basix", "custom"])
def test_creation_bounddata(mesh_type, degree, rt_space):
    # Create mesh
    n_cells = 5

    if mesh_type == "builtin":
        geometry = create_unitsquare_builtin(
            n_cells, dmesh.CellType.triangle, dmesh.DiagonalType.crossed
        )
    elif mesh_type == "gmsh":
        raise NotImplementedError("GMSH mesh not implemented yet")
    else:
        raise ValueError("Unknown mesh type")

    # Initialise connectivity
    geometry.mesh.topology.create_connectivity(1, 2)
    geometry.mesh.topology.create_connectivity(2, 1)

    # Initialise flux space
    if rt_space == "basix":
        V_flux = dfem.FunctionSpace(geometry.mesh, ("RT", degree))
        custom_rt = False
    elif rt_space == "custom":
        elmt_flux = basix.ufl_wrapper.BasixElement(
            create_hierarchic_rt(basix.CellType.triangle, degree, True)
        )
        V_flux = dfem.FunctionSpace(geometry.mesh, elmt_flux)
        custom_rt = True

    boundary_function = dfem.Function(V_flux)

    # Initialise reference flux space
    V_ref = dfem.VectorFunctionSpace(geometry.mesh, ("DG", degree - 1))

    # Initialise boundary facets and test-points
    list_bfcts, points_eval = initialise_fct_list(geometry, degree)
    npoints_eval = int(points_eval.shape[0] / 2)
    plist_eval, clist_eval = initialise_eval_fe_function(geometry.mesh, points_eval)

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
            V_vec = dfem.VectorFunctionSpace(geometry.mesh, ("CG", deg))
            func_1 = dfem.Function(V_vec)
            func_1.x.array[:] = 2 * (
                np.random.rand(V_vec.dofmap.bs * V_vec.dofmap.index_map.size_local)
                + 0.1
            )

            V_scal = dfem.FunctionSpace(geometry.mesh, ("CG", deg))
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
        list_bcs.append(fluxbc(ntrace_ufl, list_bfcts[0], V_flux, False))
        list_bcs.append(fluxbc(ntrace_ufl, list_bfcts[1], V_flux, False))

        # Initialise boundary data
        boundary_data = BoundaryData(
            [list_bcs],
            [boundary_function._cpp_object],
            V_flux._cpp_object,
            custom_rt,
            [[]],
        )

        # Interpolate BC into testspace
        rhs_ref = ufl.as_vector([-ntrace_ufl, ntrace_ufl])
        refsol = local_projection(V_ref, [rhs_ref], quadrature_degree=2 * degree)[0]

        # Evaluate functions at comparison points
        val_bfunc = boundary_function.eval(plist_eval, clist_eval)
        val_ref = refsol.eval(plist_eval, clist_eval)

        assert np.allclose(val_bfunc[:npoints_eval, 0], val_ref[:npoints_eval, 0])
        assert np.allclose(val_bfunc[npoints_eval:, 1], val_ref[npoints_eval:, 1])


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
