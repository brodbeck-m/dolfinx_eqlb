# --- Imports ---
import numpy as np
import typing

import cffi

import basix
import dolfinx
import dolfinx.fem as dfem
import dolfinx.mesh as dmesh
import ufl

from dolfinx_eqlb.cpp import FluxBC

ffi = cffi.FFI()


def fluxbc(
    value: typing.Any,
    facets: np.ndarray,
    V: dfem.FunctionSpace,
    requires_projection: typing.Optional[bool] = True,
) -> FluxBC:
    """Create a representation of Dirichlet boundary for reconstructed flux spaces

    Function holds dirichlet facets, alongside with an ufl-function containing the exact boundary

    Args:
        value:               Boundary values (flux x normal) as ufl-function
        facets:              The boundary facets
        V:                   The function space of the reconstructed flux
        requires_projection: Specifies if boundary values have to be projected into appropriate P space
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
        qdegree = 2 * flux_degree

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
                pnts_eval[i, 0] = 1 - qpnts[0]
                pnts_eval[i, 1] = qpnts[0]

                # Point on facet 1
                id_fct1 += 1
                pnts_eval[id_fct1, 0] = 0
                pnts_eval[id_fct1, 1] = qpnts[0]

                # Point on facet 2
                id_fct1 += 1
                pnts_eval[id_fct2, 0] = qpnts[0]
                pnts_eval[id_fct2, 1] = 0
        else:
            raise NotImplementedError("3D meshes currently not supported")
    else:
        # The basix element
        belmt = V.element.basix_element.points

        # Points required for interpolation into element
        pnts = belmt.points

        # Number of evaluation points per facet
        # (Check if point is on facet 0 --> x != 0)
        x_pnt = pnts[0, 0] == 0
        neval_per_fct = 0

        while x_pnt > 0:
            neval_per_fct += 1
            x_pnt = pnts[: nfcts_per_cell * neval_per_fct, 0]

        # Extract points
        pnts_eval = pnts[:neval_per_fct, :]

    # Evaluation points (Debug)
    V_rb = dfem.FunctionSpace(domain, ("DG", 1))

    basix_elmt = V_rb.element.basix_element
    pnts_eval = basix_elmt.points

    # --- Create flux-bc
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
        neval_per_fct,
        requires_projection,
        coefficients_cpp,
        positions,
        constants_cpp,
    )
