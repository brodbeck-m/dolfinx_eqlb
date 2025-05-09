# Copyright (C) 2024 Maximilian Brodbeck
#
# This file is part of dolfinx_eqlb
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Boundary conditions for flux equilibration"""

from enum import Enum
import numpy as np
from numpy.typing import NDArray
import typing

import cffi

import basix
from dolfinx import fem, mesh

import dolfinx_eqlb.cpp as _cpp

ffi = cffi.FFI()


# --- Time dependency of a boundary condition ---
class TimeType(Enum):
    stationary = 0
    timefunction = 1
    timedependent = 2


def time_type_to_cpp(time_type: TimeType) -> _cpp.TimeType:
    """Python enum TimeType to its c++ counterpart

    Args:
        time_type: The TimeType of the equilibration as python enum

    Returns:
        The TimeType as c++ enum
    """

    if time_type == TimeType.stationary:
        return _cpp.TimeType.stationary
    elif time_type == TimeType.timefunction:
        return _cpp.TimeType.timefunction
    elif time_type == TimeType.timedependent:
        return _cpp.TimeType.timedependent
    else:
        raise ValueError("Invalid time type.")


def cpp_to_time_type(time_type: _cpp.TimeType) -> TimeType:
    """c++ enum TimeType to its python counterpart

    Args:
        time_type: The TimeType of the equilibration as c++ enum

    Returns:
        The TimeType as python enum
    """

    if time_type == _cpp.TimeType.stationary:
        return TimeType.stationary
    elif time_type == _cpp.TimeType.timefunction:
        return TimeType.timefunction
    elif time_type == _cpp.TimeType.timedependent:
        return TimeType.timedependent
    else:
        raise ValueError("Invalid time type.")


# --- Set one essential boundary condition ---
def homogenous_fluxbc(facets: NDArray) -> _cpp.FluxBC:
    """Essential, homogenous boundary condition for one flux on a set of facets

    Args:
        facets:              The boundary facets

    Returns:
        The essential flux BC on a group of facets
    """

    return _cpp.FluxBC(facets)


def fluxbc(
    value: typing.Any,
    facets: NDArray,
    V: fem.FunctionSpace,
    requires_projection: typing.Optional[bool] = False,
    quadrature_degree: typing.Optional[int] = None,
    transient_behavior: typing.Optional[TimeType] = TimeType.stationary,
) -> _cpp.FluxBC:
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
        quadrature_degree = qdegree
    else:
        qdegree = 2 * flux_degree - 1
        quadrature_degree = -1

    if msh.topology.cell_type == mesh.CellType.triangle:
        pnts_eval, _ = basix.make_quadrature(basix.CellType.interval, qdegree)
    elif msh.topology.cell_type == mesh.CellType.tetrahedron:
        pnts_eval, _ = basix.make_quadrature(basix.CellType.triangle, qdegree)
        raise NotImplementedError("3D meshes currently not supported")
    else:
        raise NotImplementedError("Unsupported cell type")

    expr = fem.Expression(value, pnts_eval, dtype=np.float64)

    return _cpp.FluxBC(
        expr._cpp_object,
        facets,
        V._cpp_object,
        quadrature_degree,
        time_type_to_cpp(transient_behavior),
    )


# --- Collected boundary data for an equilibration problem ---
def boundarydata(
    flux_conditions: typing.List[typing.List[_cpp.FluxBC]],
    boundary_data: typing.List[fem.Function],
    V: fem.FunctionSpace,
    dirichlet_facets: typing.List[NDArray],
    kernel_data: typing.Any,
    problem_type: _cpp.ProblemType,
) -> _cpp.BoundaryData:
    """The collected essential boundary conditions for set of reconstructed fluxes

    Collects handles for essential boundary conditions alongside with
    the actual storage of the boundary values and the markers for the
    essential boundary facets of the primal problem.

    Args:
        flux_conditions:  List of essential flux BCs each flux
        boundary_data:    List functions holding the boundary values
        V:                The function space of the reconstructed flux
        dirichlet_facets: Identifier if stresses are equilibrated

    Returns:
        The collection of essential BC of a reconstructed flux
    """

    # Check input
    n_rhs = len(flux_conditions)

    if (n_rhs != len(boundary_data)) or (n_rhs != len(dirichlet_facets)):
        raise RuntimeError("Size of input data does not match!")

    # Extract cpp-objects from boundary data
    boundary_data_cpp = [f._cpp_object for f in boundary_data]

    return _cpp.BoundaryData(
        flux_conditions,
        boundary_data_cpp,
        V._cpp_object,
        dirichlet_facets,
        kernel_data,
        problem_type,
    )
