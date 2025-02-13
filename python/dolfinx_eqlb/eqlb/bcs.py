# Copyright (C) 2024 Maximilian Brodbeck
#
# This file is part of dolfinx_eqlb
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Boundary conditions for flux equilibration"""

import numpy as np
from numpy.typing import NDArray
import typing

import cffi

import basix
import dolfinx
from dolfinx import fem, mesh
import ufl

from dolfinx_eqlb.cpp import FluxBC, BoundaryData

ffi = cffi.FFI()


def fluxbc(
    value: typing.Any,
    facets: NDArray,
    V: fem.FunctionSpace,
    is_time_depended: typing.Optional[bool] = False,
    has_time_function: typing.Optional[bool] = False,
    requires_projection: typing.Optional[bool] = False,
    quadrature_degree: typing.Optional[int] = None,
) -> FluxBC:
    """Essential boundary condition for one flux on a set of facets

    Args:
        value:               Boundary values (flux x normal) as ufl expression
        facets:              The boundary facets
        V:                   The function space of the reconstructed flux
        is_time_depended:    True, if the boundary data are time dependent
        has_time_function:   True, if the time-dependency can be expressed by a
                             multiplicative factor
        requires_projection: Perform projection for non matching (non-polynomial or
                             higher-order polynomial) boundary data
        quadrature_degree:   Degree of quadrature rule for projection

    Returns:
        The essential flux BC on a group of facets
    """

    # --- Extract required data
    # The mesh
    msh = V.mesh

    # Degree of flux element
    flux_degree = V.element.basix_element.degree

    # --- Compile boundary function
    if requires_projection:
        qdegree = (
            2 * flux_degree - 2 if quadrature_degree is None else quadrature_degree
        )
    else:
        qdegree = 2 * flux_degree - 1

    if msh.topology.cell_type == mesh.CellType.triangle:
        pnts_eval, _ = basix.make_quadrature(basix.CellType.interval, qdegree)
    elif msh.topology.cell_type == mesh.CellType.tetrahedron:
        pnts_eval, _ = basix.make_quadrature(basix.CellType.triangle, qdegree)
        raise NotImplementedError("3D meshes currently not supported")
    else:
        raise NotImplementedError("Unsupported cell type")

    expr = fem.Expression(value, pnts_eval, dtype=np.float64)

    return FluxBC(expr._cpp_object, facets, V._cpp_object)


def boundarydata(
    flux_conditions: typing.List[typing.List[FluxBC]],
    boundary_data: typing.List[fem.Function],
    V: fem.FunctionSpace,
    custom_rt: bool,
    dirichlet_facets: typing.List[NDArray],
    equilibrate_stress: bool,
) -> BoundaryData:
    """The collected essential boundary conditions for set of reconstructed fluxes

    Collects handles for essential boundary conditions alongside with
    the actual storage of the boundary values and the markers for the
    essential boundary facets of the primal problem.

    Args:
        flux_conditions:  List of essential flux BCs each flux
        boundary_data:    List functions holding the boundary values
        V:                The function space of the reconstructed flux
        custom_rt:        Identifier if custom RT element is used
        dirichlet_facets: Identifier if stresses are equilibrated

    Returns:
        The collection of essential BC of a reconstructed flux
    """

    # Check input
    n_rhs = len(flux_conditions)

    if (n_rhs != len(boundary_data)) or (n_rhs != len(dirichlet_facets)):
        raise RuntimeError("Size of input data does not match!")

    # Set (default) quadrature degree
    degree_flux = V.element.basix_element.degree
    qdegree = 2 * (degree_flux - 1)

    for bcs in flux_conditions:
        for bc in bcs:
            qdegree = max(qdegree, bc.quadrature_degree)

    # Extract cpp-objects from boundary data
    boundary_data_cpp = [f._cpp_object for f in boundary_data]

    return BoundaryData(
        flux_conditions,
        boundary_data_cpp,
        V._cpp_object,
        custom_rt,
        qdegree,
        dirichlet_facets,
        equilibrate_stress,
    )
