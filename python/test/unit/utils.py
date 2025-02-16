# Copyright (C) 2024 Maximilian Brodbeck
#
# This file is part of dolfinx_eqlb
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Utility routines for unit tests"""

from enum import Enum
import gmsh
import numpy as np
from mpi4py import MPI
import typing

from dolfinx import geometry, io, mesh

import ufl

from dolfinx_eqlb.base.mesh import prepare_mesh_for_equilibration
from dolfinx_eqlb.eqlb import check_eqlb_conditions


# --- Mesh generation ---
class MeshType(Enum):
    builtin = 0
    gmsh = 1


class Domain:
    def __init__(
        self,
        mesh: mesh.Mesh,
        facet_fkt: mesh.MeshTags,
        ds: ufl.Measure,
        dv: typing.Optional[ufl.Measure] = None,
    ):
        # Mesh
        self.mesh = mesh

        # Facet functions
        self.facet_function = facet_fkt

        # Integrators
        self.ds = ds

        if dv is None:
            self.dv = ufl.dx
        else:
            self.dv = dv


def create_unitsquare_builtin(
    n_elmt: int, cell: mesh.CellType, diagonal_type: mesh.DiagonalType
) -> Domain:
    # --- Set default options
    if cell is None:
        cell = mesh.CellType.triangle

    if diagonal_type is None:
        diagonal_type = mesh.DiagonalType.crossed

    # --- Create mesh
    msh = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0, 0]), np.array([1, 1])],
        [n_elmt, n_elmt],
        cell_type=cell,
        diagonal=diagonal_type,
    )

    boundaries = [
        (1, lambda x: np.isclose(x[0], 0)),
        (2, lambda x: np.isclose(x[1], 0)),
        (3, lambda x: np.isclose(x[0], 1)),
        (4, lambda x: np.isclose(x[1], 1)),
    ]

    facet_indices, facet_markers = [], []
    for marker, locator in boundaries:
        facets = mesh.locate_entities(msh, 1, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full(len(facets), marker))

    facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
    facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_function = mesh.meshtags(
        msh, 1, facet_indices[sorted_facets], facet_markers[sorted_facets]
    )
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_function)

    return Domain(msh, facet_function, ds)


def create_unitsquare_gmsh(hmin: float) -> Domain:
    # --- Build model
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.Verbosity", 2)

    # Name of the geometry
    gmsh.model.add("LShape")

    # Points
    list_pnts = [[0, 0], [0, 1], [1, 1], [1, 0]]

    pnts = [gmsh.model.occ.add_point(pnt[0], pnt[1], 0.0) for pnt in list_pnts]

    # Bounding curves and 2D surface
    bfcts = [
        gmsh.model.occ.add_line(pnts[0], pnts[1]),
        gmsh.model.occ.add_line(pnts[1], pnts[2]),
        gmsh.model.occ.add_line(pnts[2], pnts[3]),
        gmsh.model.occ.add_line(pnts[3], pnts[0]),
    ]

    boundary = gmsh.model.occ.add_curve_loop(bfcts)
    surface = gmsh.model.occ.add_plane_surface([boundary])
    gmsh.model.occ.synchronize()

    # Set tag on boundaries and surface
    for i, bfct in enumerate(bfcts):
        gmsh.model.addPhysicalGroup(1, [bfct], i + 1)

    gmsh.model.addPhysicalGroup(2, [surface], 1)

    # --- Generate mesh
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", hmin)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", hmin)
    gmsh.model.mesh.generate(2)

    msh = prepare_mesh_for_equilibration(
        io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)[0]
    )

    if not check_eqlb_conditions.mesh_has_reversed_edges(msh):
        raise ValueError("Mesh does not contain reversed edges")

    # --- Mark facets
    boundaries = [
        (1, lambda x: np.isclose(x[0], 0)),
        (2, lambda x: np.isclose(x[1], 0)),
        (3, lambda x: np.isclose(x[0], 1)),
        (4, lambda x: np.isclose(x[1], 1)),
    ]

    facet_indices, facet_markers = [], []
    for marker, locator in boundaries:
        facets = mesh.locate_entities(msh, 1, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full(len(facets), marker))

    facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
    facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_function = mesh.meshtags(
        msh, 1, facet_indices[sorted_facets], facet_markers[sorted_facets]
    )
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_function)

    return Domain(msh, facet_function, ds)


# --- Point-evaluation of FE functions ---
def points_boundary_unitsquare(
    domain: Domain, boundary_id: typing.List[int], npoints_per_fct: int
) -> np.ndarray:
    """Creates points on boundary
    Evaluation points are cerated per call-facet on boundary, while the mesh-nodes itself are excluded.

    Args:
        domain:          The domain
        boundary_id:     List of boundary ids on which the evaluation points are created
        npoints_per_fct: Number of evaluation points per facet

    Returns:
        The list of evaluation points per boundary id
    """

    # The number of boundary facets
    n_bfcts = int(domain.facet_function.indices.size / 4)

    # Initialise output
    n_points = len(boundary_id) * n_bfcts * npoints_per_fct
    n_points_per_boundary = n_bfcts * npoints_per_fct
    points = np.zeros((n_points, 3), dtype=np.float64)

    # 1D mesh coordinates
    s_points = np.zeros(n_points_per_boundary)

    for i in range(0, n_bfcts):
        start_point = 0 + i * (1 / n_bfcts)
        end_point = start_point + (1 / n_bfcts)

        s_points[i * npoints_per_fct : (i + 1) * npoints_per_fct] = np.linspace(
            start_point, end_point, npoints_per_fct + 1, endpoint=False
        )[1:]

    # Push 1D coordinates to 3D
    for i in range(0, len(boundary_id)):
        begin = i * n_points_per_boundary
        end = (i + 1) * n_points_per_boundary
        if boundary_id[i] == 1:
            points[begin:end, 0] = 0
            points[begin:end, 1] = s_points
        elif boundary_id[i] == 2:
            points[begin:end, 0] = s_points
            points[begin:end, 1] = 0
        elif boundary_id[i] == 3:
            points[begin:end, 0] = 1
            points[begin:end, 1] = s_points
        else:
            points[begin:end, 0] = s_points
            points[begin:end, 1] = 1

    return points


def initialise_evaluate_function(
    msh: mesh.Mesh, points: np.ndarray
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Prepare evaluation of finite-element function

    Function evaluation requires list of points and the adjacent cells per
    processor.

    Args:
        msh:    The Mesh
        points: The points, at which the function has to be evaluated

    Returns:
        Evaluation points on each processor,
        Cell within which the points are located
    """

    # The search tree
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)

    # Initialise output
    cells = []
    points_on_proc = []

    # Find cells whose bounding-box collide with the the points
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)

    # Choose one of the cells that contains the point
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points)
    for i, point in enumerate(points):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])

    points_on_proc = np.array(points_on_proc, dtype=np.float64)

    return points_on_proc, cells
