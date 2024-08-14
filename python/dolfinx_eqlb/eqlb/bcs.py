# Copyright (C) 2024 Maximilian Brodbeck
#
# This file is part of dolfinx_eqlb
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

# --- Imports ---
import numpy as np
import typing

import cffi

import basix
import dolfinx
import dolfinx.fem as dfem
import dolfinx.mesh as dmesh
import ufl

from dolfinx_eqlb.cpp import FluxBC, BoundaryData

ffi = cffi.FFI()


def fluxbc(
    value: typing.Any,
    facets: np.ndarray,
    V: dfem.FunctionSpace,
    requires_projection: typing.Optional[bool] = True,
    quadrature_degree: typing.Optional[int] = None,
) -> FluxBC:
    """Essential boundary condition for one flux on a set of facets

    Args:
        value:               Boundary values (flux x normal) as ufl-function
        facets:              The boundary facets
        V:                   The function space of the reconstructed flux
        requires_projection: Specifies if boundary values have to be projected into appropriate P space
        quadrature_degree:   Degree of quadrature rule for projection

    Returns:
        The essential flux BCs on a group of facets
    """

    # --- Extract required data
    # The mesh
    domain = V.mesh

    # Number of facets per cell
    if domain.topology.cell_type == dmesh.CellType.triangle:
        fct_type = basix.CellType.interval
        nfcts_per_cell = 3
    elif domain.topology.cell_type == dmesh.CellType.tetrahedron:
        fct_type = basix.CellType.triangle
        nfcts_per_cell = 4
        raise NotImplementedError("3D meshes currently not supported")
    else:
        raise NotImplementedError("Unsupported cell type")

    # Degree of flux element
    flux_degree = V.element.basix_element.degree

    # --- Compile boundary function
    # Evaluation points of boundary function
    if requires_projection:
        # Quadrature degree
        if quadrature_degree is None:
            qdegree = 2 * (flux_degree - 1)
        else:
            qdegree = quadrature_degree

        # Create appropriate quadrature rule
        qpnts, _ = basix.make_quadrature(fct_type, qdegree)

        # Number of evaluation points per facet
        neval_per_fct = qpnts.shape[0]

        # Map points to reference cell
        pnts_eval = np.zeros(
            (nfcts_per_cell * neval_per_fct, domain.topology.dim), dtype=np.float64
        )

        if domain.topology.cell_type == dmesh.CellType.triangle:
            # Initialisations
            id_fct1 = neval_per_fct
            id_fct2 = 2 * neval_per_fct

            # Map 1D points to 2D facet
            for i in range(0, neval_per_fct):
                # Point on facet 0
                pnts_eval[i, 0] = 1 - qpnts[i]
                pnts_eval[i, 1] = qpnts[0]

                # Point on facet 1
                pnts_eval[id_fct1, 0] = 0
                pnts_eval[id_fct1, 1] = qpnts[i]
                id_fct1 += 1

                # Point on facet 2
                pnts_eval[id_fct2, 0] = qpnts[i]
                pnts_eval[id_fct2, 1] = 0
                id_fct2 += 1
        else:
            raise NotImplementedError("3D meshes currently not supported")
    else:
        # Points required for interpolation into element
        pnts = V.element.basix_element.points

        # Number of evaluation points per facet
        # (Check if point is on facet 0 --> x != 0)
        x_pnt = pnts[0, 0]
        neval_per_fct = 0

        while x_pnt > 0:
            neval_per_fct += 1
            x_pnt = pnts[neval_per_fct, 0]

        # Extract points
        pnts_eval = pnts[: neval_per_fct * nfcts_per_cell, :]

    # Precompile ufl-function
    ufcx_form, _, _ = dolfinx.jit.ffcx_jit(domain.comm, (value, pnts_eval))

    # Extract constants
    constants = ufl.algorithms.analysis.extract_constants(value)

    constants_cpp = [c._cpp_object for c in constants]

    # Extract coefficients
    coefficients = ufl.algorithms.analysis.extract_coefficients(value)
    n_positions = ufcx_form.num_coefficients
    c_positions = ufcx_form.original_coefficient_positions

    coefficients_cpp = []
    positions = []
    for i in range(0, n_positions):
        coefficients_cpp.append(coefficients[i]._cpp_object)
        positions.append(c_positions[i])

    return FluxBC(
        V._cpp_object,
        facets,
        ffi.cast("uintptr_t", ffi.addressof(ufcx_form)),
        int(pnts_eval.shape[0] / nfcts_per_cell),
        requires_projection,
        coefficients_cpp,
        positions,
        constants_cpp,
    )


def boundarydata(
    flux_conditions: typing.List[typing.List[FluxBC]],
    boundary_data: typing.List[dfem.Function],
    V: dfem.FunctionSpace,
    custom_rt: bool,
    dirichlet_facets: typing.List[np.ndarray],
    equilibrate_stress: bool,
    quadrature_degree: typing.Optional[int] = None,
) -> BoundaryData:
    """The collected essential boundary conditions for set of reconstructed fluxes

    Collects handles for essential boundary conditions alongside with
    the actual storage of the boundary values and the markers for the
    essential boundary facets of the primal problem.

    Args:
        flux_conditions:   List of essential flux BCs each flux
        boundary_data:     List functions holding the boundary values
        V:                 The function space of the reconstructed flux
        custom_rt:         Identifier if custom RT element is used
        dirichlet_facets:  Identifier if stresses are equilibrated
        quadrature_degree: Degree of quadrature rule for projection

    Returns:
        The collection of essential BC of a reconstructed flux
    """

    # Check input
    n_rhs = len(flux_conditions)

    if (n_rhs != len(boundary_data)) or (n_rhs != len(dirichlet_facets)):
        raise RuntimeError("Size of input data does not match!")

    # Set (default) quadrature degree
    degree_flux = V.element.basix_element.degree
    if quadrature_degree is None:
        qdegree = 2 * (degree_flux - 1)
    else:
        if quadrature_degree < 2 * (degree_flux - 1):
            raise RuntimeError("Quadrature has to be at least 2*k!")
        else:
            qdegree = quadrature_degree

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
