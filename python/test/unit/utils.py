# --- Includes ---
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from typing import Any, Callable, List

import basix

import dolfinx
import dolfinx.fem as dfem
import dolfinx.mesh as dmesh

import ufl


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


def create_unitsquare_gmsh(
    n_elmt: int, cell: dolfinx.mesh.CellType, diagonal_type: dolfinx.mesh.DiagonalType
) -> Geometry:
    raise NotImplementedError("Not implemented yet")


def create_quatercircle_gmsh():
    raise NotImplementedError("Not implemented yet")


"""
Calculate Convergence rates
"""
