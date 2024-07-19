# --- Includes ---
import gmsh
import numpy as np
from mpi4py import MPI
from typing import Any, Callable, List

import dolfinx
import dolfinx.fem as dfem
import dolfinx.geometry as dgeom
from dolfinx.io import gmshio
import dolfinx.mesh as dmesh

import ufl

from dolfinx_eqlb.eqlb import check_eqlb_conditions


"""
Mesh generation
"""


class Geometry:
    def __init__(self, mesh, facet_fkt, ds, dv=None):
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

    domain, _, _ = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    reversed_edges = check_eqlb_conditions.mesh_has_reversed_edges(domain)

    if not reversed_edges:
        raise ValueError("Mesh does not contain reversed edges")

    # --- Test if boundary patches contain at least 2 cells

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


def create_quatercircle_gmsh():
    raise NotImplementedError("Not implemented yet")


"""
Point evaluation of fe-functions
"""


def points_boundary_unitsquare(
    geometry: Geometry, boundary_id: List, npoints_per_fct: int
) -> np.array:
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


def initialise_evaluate_function(domain: dmesh.Mesh, points: np.ndarray):
    """Prepare evaluation of dfem.Function
    Function evaluation requires list of points and the adjacent cells per
    processor.

    Args:
        domain:          The Mesh
        points:          The points, at which the function
                         has to be evaluated


    Returns:
        points_on_proc: Evaluation points on each processor
        cells:          Cell within which the point is located
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


"""
Interpolation of ufl- into fe-function 
"""


def interpolate_ufl_to_function(f_ufl: Any, f_fe: dfem.Function):
    # Create expression
    expr = dfem.Expression(f_ufl, f_fe.function_space.element.interpolation_points())

    # Perform interpolation
    f_fe.interpolate(expr)


"""
Calculate Convergence rates
"""


def error_L2(diff_u_uh, qorder=None):
    if qorder is None:
        dvol = ufl.dx
    else:
        dvol = ufl.dx(degree=qorder)
    return dfem.form(ufl.inner(diff_u_uh, diff_u_uh) * dvol)


def error_h1(diff_u_uh, qorder=None):
    if qorder is None:
        dvol = ufl.dx
    else:
        dvol = ufl.dx(degree=qorder)
    return dfem.form(ufl.inner(ufl.grad(diff_u_uh), ufl.grad(diff_u_uh)) * dvol)


def error_hdiv0(diff_u_uh, qorder=None):
    if qorder is None:
        dvol = ufl.dx
    else:
        dvol = ufl.dx(degree=qorder)
    return dfem.form(ufl.inner(ufl.div(diff_u_uh), ufl.div(diff_u_uh)) * dvol)


def flux_error(
    uh: Any, u_ex: Any, form_error: Callable, degree_raise=2, uex_is_ufl=False
):
    """Calculate convergence rate
    Assumption: uh is constructed from a FE-space with block-size 1 and
    the FE-space ca be interpolated by dolfinX!

    Args:
        uh (Any):              Approximate solution
                               (DOLFINx-function (if u_ex is callable) or ufl-expression)
        u_ex (Any):            Exact solution
                               (callable function for interpolation or ufl expr.)
        form_error (Callable): Generates form for error calculation

    Returns:
        error: The global error measured in the given norm

    """
    # Initialise quadrature degree
    qdegree = None

    if not uex_is_ufl:
        # Get mesh
        mesh = uh.function_space.mesh

        # Create higher order function space
        degree = uh.function_space.ufl_element().degree() + degree_raise
        family = uh.function_space.ufl_element().family()
        mesh = uh.function_space.mesh

        elmt = ufl.FiniteElement(family, mesh.ufl_cell(), degree)

        W = dfem.FunctionSpace(mesh, elmt)

        # Interpolate approximate solution
        u_W = dfem.Function(W)
        u_W.interpolate(uh)

        # Interpolate exact solution, special handling if exact solution
        # is a ufl expression or a python lambda function
        u_ex_W = dfem.Function(W)
        u_ex_W.interpolate(u_ex)

        # Compute the error in the higher order function space
        e_W = dfem.Function(W)
        e_W.x.array[:] = u_W.x.array - u_ex_W.x.array
    else:
        # Get mesh
        try:
            mesh = uh.function_space.mesh
        except:
            mesh = uh.ufl_operands[0].function_space.mesh

        # Set quadrature degree
        qdegree = 10

        # Get spacial coordinate and set error functional
        e_W = u_ex - uh

    # Integrate the error
    error_local = dfem.assemble_scalar(form_error(e_W, qorder=qdegree))
    error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)
