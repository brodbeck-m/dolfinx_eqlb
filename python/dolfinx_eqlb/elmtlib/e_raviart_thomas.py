# Copyright (C) 2024 Maximilian Brodbeck
#
# This file is part of dolfinx_eqlb
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Hierarchic Raviart-Thomas spaces"""

import numpy as np

import basix


def create_hierarchic_rt(cell: basix.CellType, degree: int, discontinuous: bool):
    """Create hierarchic RT element

    Generates a basix element of the RT space following [1].

    Warning: The classical function.interpolate(...) method is not working,
    as therefore derivatives would be required.

    [1] Bertrand, F. et al.: https://doi.org/10.1007/s00211-023-01366-8, 2023

    Args:
        cell:          The basix cell type
        degree:        The degree of the element
        discontinuous: Id if the created element is discontinuous

    Returns:
        The basix element
    """

    # Check degree
    if degree < 1:
        raise ValueError("Degree must be at least 1")

    # Check cell type
    if cell != basix.CellType.triangle:
        raise ValueError("Only triangular cells supported")

    # Geometry information
    if cell == basix.CellType.triangle:
        fct = basix.CellType.interval
        tdim = 2
        nnodes_cell = 3
    else:
        fct = basix.CellType.triangle
        tdim = 3
        nnodes_cell = 4

    # Representation of the basis forming monomials
    rt = basix.create_element(
        basix.ElementFamily.RT, cell, degree, basix.LagrangeVariant.equispaced
    )
    wcoeffs = rt.wcoeffs

    # --- Creating the interpolation operator ---
    # Initialisation
    x = [[], [], [], []]
    M = [[], [], [], []]

    # Check if derivatives are required for definition of basis
    if degree > 1:
        n_derivatives = 1
        qdegree = 2 * degree
    else:
        n_derivatives = 0
        qdegree = degree

    # Quadrature points on reference fact
    pnt, wts = basix.make_quadrature(fct, qdegree)

    # Set-up the interpolation
    if tdim == 2:
        # --- Facet contribution
        # Quadrature points on cell facets
        x[1].append(np.array([[1 - p[0], p[0]] for p in pnt]))  # q. points on edge 0
        x[1].append(np.array([[0, p[0]] for p in pnt]))  # q. points on edge 1
        x[1].append(np.array([[p[0], 0] for p in pnt]))  # q. points on edge 2

        # Set weight factors according to facet integrals
        for normal in [[-1, -1], [-1, 0], [0, 1]]:
            mat = np.zeros((degree, 2, len(wts), 1 + n_derivatives * tdim))

            for j in range(0, degree):
                # Lambda = int_E f * n * s^j ds
                mat[j, 0, :, 0] = normal[0] * (pnt[:, 0] ** j) * wts[:]
                mat[j, 1, :, 0] = normal[1] * (pnt[:, 0] ** j) * wts[:]

            M[1].append(mat)

        # --- Cell contribution
        if degree > 1:
            # quadrature points on cell
            pnt, wts = basix.make_quadrature(cell, qdegree)
            x[2].append(pnt)

            # initialisation of interpolation matrix
            mat = np.zeros(
                ((degree**2) - degree, 2, len(wts), 1 + n_derivatives * tdim)
            )
            n = 0

            # cell integrals of divergence
            for l in range(0, degree):
                for m in range(0, degree - l):
                    if l + m >= 1:
                        # lambda = int_T div v * x^l * y^m dx
                        mat[n, 0, :, 1] = (pnt[:, 0] ** l) * (pnt[:, 1] ** m) * wts[:]
                        mat[n, 1, :, 2] = (pnt[:, 0] ** l) * (pnt[:, 1] ** m) * wts[:]

                        n = n + 1

            # remaining cell contributions
            if degree > 2:
                for l in range(1, degree - 1):
                    for m in range(0, degree - 1 - l):
                        # lambda = int_T e_2 * x^l * y^m dx
                        mat[n, 0, :, 0] = 0
                        mat[n, 1, :, 0] = (pnt[:, 0] ** l) * (pnt[:, 1] ** m) * wts[:]

                        n = n + 1

            M[2].append(mat)
        else:
            x[2].append(np.zeros((0, 2)))
            M[2].append(np.zeros((0, 2, 0, 1 + n_derivatives * tdim)))

        # No nodal contributions to shape functions
        for _ in range(nnodes_cell):
            x[0].append(np.zeros((0, 2)))
            M[0].append(np.zeros((0, 2, 0, 1 + n_derivatives * tdim)))
    else:
        raise NotImplementedError("Raviart-Thomas on tetrahedra not implemented")

    # Make element discontinous
    if discontinuous:
        # Number of DOFs/ qpoint facet
        nfct_cell = len(x[tdim - 1])
        ndofs_fct = degree
        nqpoints_fct = x[tdim - 1][0].shape[0]

        ndofs_cell = (degree**2) - degree
        nqpoints_cell = x[tdim][0].shape[0]

        # Reinitialise data on cell
        nqpoints = nfct_cell * nqpoints_fct + nqpoints_cell
        ndofs = nfct_cell * ndofs_fct + ndofs_cell
        points_cell = np.zeros((nqpoints, tdim))
        mat = np.zeros((ndofs, tdim, nqpoints, 1 + n_derivatives * tdim))

        # Move data from facets
        for i in range(0, nfct_cell):
            # move quadrature points
            points_cell[i * nqpoints_fct : (i + 1) * nqpoints_fct, :] = x[tdim - 1][i][
                :, :
            ]
            x[tdim - 1][i] = np.zeros((0, tdim))

            # Move weight factors
            mat[
                i * ndofs_fct : (i + 1) * ndofs_fct,
                :,
                i * nqpoints_fct : (i + 1) * nqpoints_fct,
                :,
            ] = M[tdim - 1][i][:, :, :, :]
            M[tdim - 1][i] = np.zeros((0, tdim, 0, 1 + n_derivatives * tdim))

        # Copy data from cell
        points_cell[nfct_cell * nqpoints_fct :, :] = x[tdim][0][:, :]
        mat[nfct_cell * ndofs_fct :, :, nfct_cell * nqpoints_fct :, :] = M[tdim][0][
            :, :, :, :
        ]

        # Reset interpolation operator
        x[tdim][0] = points_cell
        M[tdim][0] = mat

        # Set Sobolev space
        space = basix.SobolevSpace.L2
    else:
        space = basix.SobolevSpace.HDiv

    return basix.create_custom_element(
        cell,
        [tdim],
        wcoeffs,
        x,
        M,
        n_derivatives,
        basix.MapType.contravariantPiola,
        space,
        discontinuous,
        degree - 1,
        degree,
    )
