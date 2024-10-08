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

import dolfinx
import dolfinx.fem as dfem
import dolfinx.geometry as dgeom
from dolfinx.io import gmshio
import dolfinx.mesh as dmesh

import ufl

from dolfinx_eqlb.eqlb import check_eqlb_conditions


# --- Mesh generation ---
class MeshType(Enum):
    builtin = 0
    gmsh = 1


class Geometry:
    def __init__(
        self,
        mesh: dmesh.Mesh,
        facet_fkt: dmesh.MeshTagsMetaClass,
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
    n_elmt: int, cell: dolfinx.mesh.CellType, diagonal_type: dolfinx.mesh.DiagonalType
) -> Geometry:
    # --- Set default options
    if cell is None:
        cell = dmesh.CellType.triangle

    if diagonal_type is None:
        diagonal_type = dmesh.DiagonalType.crossed

    # --- Create mesh
    domain = dmesh.create_rectangle(
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
        facets = dolfinx.mesh.locate_entities(domain, 1, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full(len(facets), marker))

    facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
    facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_function = dolfinx.mesh.meshtags(
        domain, 1, facet_indices[sorted_facets], facet_markers[sorted_facets]
    )
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_function)

    return Geometry(domain, facet_function, ds)


def create_unitsquare_gmsh(hmin: float) -> Geometry:
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

    domain_init, _, _ = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    reversed_edges = check_eqlb_conditions.mesh_has_reversed_edges(domain_init)

    if not reversed_edges:
        raise ValueError("Mesh does not contain reversed edges")

    # --- Test if boundary patches contain at least 2 cells
    # List of refined cells
    refined_cells = []

    # Required connectivity's
    domain_init.topology.create_connectivity(0, 2)
    domain_init.topology.create_connectivity(1, 2)
    pnt_to_cell = domain_init.topology.connectivity(0, 2)

    # The boundary facets
    bfcts = dmesh.exterior_facet_indices(domain_init.topology)

    # Get boundary nodes
    V = dfem.FunctionSpace(domain_init, ("Lagrange", 1))
    bpnts = dfem.locate_dofs_topological(V, 1, bfcts)

    # Check if point is linked with only on cell
    for pnt in bpnts:
        cells = pnt_to_cell.links(pnt)

        if len(cells) == 1:
            refined_cells.append(cells[0])

    # Refine mesh
    list_ref_cells = list(set(refined_cells))

    if len(list_ref_cells) > 0:
        domain = dmesh.refine(
            domain_init,
            np.setdiff1d(
                dmesh.compute_incident_entities(domain_init, list_ref_cells, 2, 1),
                bfcts,
            ),
        )
    else:
        domain = domain_init

    # --- Mark facets
    boundaries = [
        (1, lambda x: np.isclose(x[0], 0)),
        (2, lambda x: np.isclose(x[1], 0)),
        (3, lambda x: np.isclose(x[0], 1)),
        (4, lambda x: np.isclose(x[1], 1)),
    ]

    facet_indices, facet_markers = [], []
    for marker, locator in boundaries:
        facets = dolfinx.mesh.locate_entities(domain, 1, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full(len(facets), marker))

    facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
    facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_function = dolfinx.mesh.meshtags(
        domain, 1, facet_indices[sorted_facets], facet_markers[sorted_facets]
    )
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_function)

    return Geometry(domain, facet_function, ds)


# --- Point-evaluation of FE functions ---
def points_boundary_unitsquare(
    geometry: Geometry, boundary_id: typing.List[int], npoints_per_fct: int
) -> np.ndarray:
    """Creates points on boundary
    Evaluation points are cerated per call-facet on boundary, while the mesh-nodes itself are excluded.

    Args:
        geometry:        The Geometry of the domain
        boundary_id:     List of boundary ids on which the evaluation points are created
        npoints_per_fct: Number of evaluation points per facet

    Returns:
        points:         List of evaluation points per boundary id
    """

    # The number of boundary facets
    n_bfcts = int(geometry.facet_function.indices.size / 4)

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
    domain: dmesh.Mesh, points: np.ndarray
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Prepare evaluation of dfem.Function
    Function evaluation requires list of points and the adjacent cells per
    processor.

    Args:
        domain: The Mesh
        points: The points, at which the function has to be evaluated

    Returns:
        Evaluation points on each processor,
        Cell within which the points are located
    """
    # The search tree
    bb_tree = dgeom.BoundingBoxTree(domain, domain.topology.dim)

    # Initialise output
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
