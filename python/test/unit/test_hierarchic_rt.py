# Copyright (C) 2024 Maximilian Brodbeck
#
# This file is part of dolfinx_eqlb
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Test custom implementation of a hierarchic Raviart-Thomas element"""

from mpi4py import MPI
import numpy as np
import pytest

import basix
import basix.ufl
from dolfinx import fem, mesh

from dolfinx_eqlb.elmtlib import create_hierarchic_rt


# --- Auxiliary functions ---
def evaluate_fe_functions(shp_fkt, dofs):
    # initialisation
    values = np.zeros((shp_fkt.shape[0], shp_fkt.shape[2]))

    # loop over all points
    for p in range(0, shp_fkt.shape[0]):
        # loop over all basis functions
        for i in range(0, shp_fkt.shape[1]):
            # loop over all dimensions
            for d in range(0, shp_fkt.shape[2]):
                values[p, d] += shp_fkt[p, i, d] * dofs[i]

    return values


def evaluate_dofs_hierarchic_rt(tdim, degree, rt_basix, dofs_basix):
    if tdim == 2:
        # --- Initialisation
        dofs_custom = np.zeros(dofs_basix.shape[0])

        # --- Facet contribution
        # evaluate 1D quadrature rule
        pnt, wts = basix.make_quadrature(basix.CellType.interval, 2 * degree)

        for ifct in range(0, 3):
            # map quadrature points of facet of reference cell
            if ifct == 0:
                normal = [-1, -1]
                pnt_fct = np.array([[1 - p[0], p[0]] for p in pnt])
            elif ifct == 1:
                normal = [-1, 0]
                pnt_fct = np.array([[0, p[0]] for p in pnt])
            else:
                normal = [0, 1]
                pnt_fct = np.array([[p[0], 0] for p in pnt])

            # evaluate reference function
            shp_fkt = rt_basix.basix_element.tabulate(0, pnt_fct)

            values_fct = evaluate_fe_functions(shp_fkt[0, :, :, :], dofs_basix)
            normal_moment = values_fct[:, 0] * normal[0] + values_fct[:, 1] * normal[1]

            # evaluate DOFs on fct_i
            for i in range(0, degree):
                # c_TE = int_E f * n * s^i ds
                dofs_custom[degree * ifct + i] = np.dot(
                    normal_moment * (pnt[:, 0] ** i), wts
                )

        # --- Cell contribution
        if degree > 1:
            # evaluate 2D quadrature rule
            pnt, wts = basix.make_quadrature(basix.CellType.triangle, 2 * degree)

            # evaluate reference function
            shp_fkt = rt_basix.basix_element.tabulate(1, pnt)

            values_cell = evaluate_fe_functions(shp_fkt[0, :, :, :], dofs_basix)
            values_cell_dx = evaluate_fe_functions(shp_fkt[1, :, :, :], dofs_basix)
            values_cell_dy = evaluate_fe_functions(shp_fkt[2, :, :, :], dofs_basix)

            divergence = values_cell_dx[:, 0] + values_cell_dy[:, 1]

            # initialise counter
            n = 3 * degree

            # cell integrals of divergence
            for l in range(0, degree):
                for m in range(0, degree - l):
                    if l + m >= 1:
                        # c_Tdiv = int_T div(f) * x^l * y^m dx
                        dofs_custom[n] = np.dot(
                            divergence * pnt[:, 0] ** l * pnt[:, 1] ** m, wts
                        )

                        n = n + 1

            # remaining cell contributions
            if degree > 2:
                for l in range(1, degree - 1):
                    for m in range(0, degree - 1 - l):
                        # c_T2 = int_T f * e_2 * x^l * y^m dx
                        dofs_custom[n] = np.dot(
                            values_cell[:, 1] * pnt[:, 0] ** l * pnt[:, 1] ** m, wts
                        )

                        n = n + 1
    else:
        raise NotImplementedError("Test only implemented for 2D")

    return dofs_custom


# --- The tests ---
@pytest.mark.parametrize("cell", [basix.CellType.triangle])
@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("discontinuous", [False, True])
def test_element_reference(cell: basix.CellType, degree: int, discontinuous: bool):
    """Test the interpolation on a reference cell

    Interpolate a function into the basix RT space and compare the result
    with the custom implementation.

    Args:
        cell:          The cell type
        degree:        The degree of the RT element
        discontinuous: Flag for discontinuous elements
    """

    # get topological dimension
    if cell == basix.CellType.triangle:
        tdim = 2
    else:
        tdim = 3

    # setup test function
    rt_basix = basix.ufl.element(
        basix.ElementFamily.RT,
        cell,
        degree,
        discontinuous=discontinuous,
    )
    dofs_basix = 2 * (np.random.rand(rt_basix.dim) + 0.1)

    # create custom element
    rt_custom = create_hierarchic_rt(cell, degree, discontinuous)
    dofs_custom = evaluate_dofs_hierarchic_rt(tdim, degree, rt_basix, dofs_basix)

    # set test points on reference cell
    points = basix.create_lattice(cell, degree + 2, basix.LatticeType.equispaced, True)

    # compare functions at test-points
    pvalues_basix = evaluate_fe_functions(
        rt_basix.basix_element.tabulate(0, points)[0, :, :, :], dofs_basix
    )
    pvalues_custom = evaluate_fe_functions(
        rt_custom.basix_element.tabulate(0, points)[0, :, :, :], dofs_custom
    )

    assert np.allclose(pvalues_basix, pvalues_custom)


@pytest.mark.parametrize("cell", [basix.CellType.triangle])
@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("discontinuous", [True])
def test_interpolation(cell: basix.CellType, degree: int, discontinuous: bool):
    """Test the interpolation on a real mesh

    Create an arbitrary function based on the custom RT element, interpolate it into
    the basix-based RT function and compare the resulting values on each mesh cell.

    Args:
        cell:          The cell type
        degree:        The degree of the RT element
        discontinuous: Flag for discontinuous elements
    """

    # generate mesh
    if cell == basix.CellType.triangle:
        domain = mesh.create_rectangle(
            MPI.COMM_WORLD,
            [np.array([0, 0]), np.array([1, 1])],
            [5, 5],
            cell_type=mesh.CellType.triangle,
        )
    else:
        raise NotImplementedError("Test only implemented for triangles")

    dofmap_geom = domain.geometry.dofmap

    # create function spaces
    P_rt_custom = create_hierarchic_rt(cell, degree, discontinuous)
    P_rt_basix = basix.ufl.element(
        basix.ElementFamily.RT,
        cell,
        degree,
        discontinuous=discontinuous,
    )

    V_rt_custom = fem.functionspace(domain, P_rt_custom)
    V_rt_basix = fem.functionspace(domain, P_rt_basix)

    # create random function
    f_rt_custom = fem.Function(V_rt_custom)
    f_rt_basix = fem.Function(V_rt_basix)

    f_rt_custom.x.array[:] = 0.7 * (
        np.random.rand(V_rt_basix.dofmap.index_map.size_local) + 0.25
    )
    f_rt_basix.interpolate(f_rt_custom)

    # --- Check cell values
    # create point grid on reference cell
    points = basix.create_lattice(cell, degree + 2, basix.LatticeType.equispaced, True)

    # tabulate geometry element
    c_element = basix.create_element(
        basix.ElementFamily.P, cell, 1, basix.LagrangeVariant.gll_warped
    )
    shpfkt_geom = c_element.tabulate(1, np.array([[0, 0]]))

    dphi_geom = shpfkt_geom[1 : 2 + 1, 0, :, 0].copy()
    ndof_cell_geom = c_element.dim

    # create storage for geometry data
    geometry = np.zeros((ndof_cell_geom, 2), dtype=np.float64)

    # tabulate shape functions on reference cell
    shpfkt_custom_ref = P_rt_custom.basix_element.tabulate(0, points)
    shpfkt_basix_ref = P_rt_basix.basix_element.tabulate(0, points)

    # loop over cells
    ncells = domain.topology.index_map(domain.topology.dim).size_local

    for c in range(0, ncells):
        # mapping of shape functions
        geometry[:] = domain.geometry.x[dofmap_geom[c, :], :2]

        J_q = np.dot(geometry.T, dphi_geom.T)
        K_q = np.linalg.inv(J_q)
        detj = np.linalg.det(J_q)

        shpfkt_custom_cur = P_rt_custom.basix_element.push_forward(
            shpfkt_custom_ref[0],
            np.array([J_q for p in points]),
            np.array([detj for p in points]),
            np.array([K_q for p in points]),
        )
        shpfkt_basix_cur = P_rt_basix.basix_element.push_forward(
            shpfkt_basix_ref[0],
            np.array([J_q for p in points]),
            np.array([detj for p in points]),
            np.array([K_q for p in points]),
        )

        # extract cell DOFs
        dofs_custom = f_rt_custom.x.array[V_rt_custom.dofmap.list[c, :]]
        dofs_basix = f_rt_basix.x.array[V_rt_basix.dofmap.list[c, :]]

        # evaluate functions at test-points
        pvalues_custom = evaluate_fe_functions(shpfkt_custom_cur[:, :, :], dofs_custom)
        pvalues_basix = evaluate_fe_functions(shpfkt_basix_cur[:, :, :], dofs_basix)

        assert np.allclose(pvalues_basix, pvalues_custom)


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
