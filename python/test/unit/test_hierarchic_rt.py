import numpy as np
import pytest

import basix
from basix import CellType

from dolfinx_eqlb.e_raviart_thomas import create_hierarchic_rt


'''Utility routines'''
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
        pnt, wts = basix.make_quadrature(CellType.interval, 2 * degree)

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
            shp_fkt = rt_basix.tabulate(0, pnt_fct)

            values_fct = evaluate_fe_functions(shp_fkt[0,:,:,:], dofs_basix)
            normal_moment = values_fct[:, 0] * normal[0] + values_fct[:, 1] * normal[1]

            # evaluate DOFs on fct_i
            for i in range(0, degree):
                # c_TE = int_E f * n * s^i ds
                dofs_custom[degree * ifct + i] = np.dot(normal_moment * pnt[:, 0]**i, wts)

        # --- Cell contribution
        if degree > 1:
            # evaluate 2D quadrature rule
            pnt, wts = basix.make_quadrature(CellType.triangle, 2 * degree)

            # evaluate reference function
            shp_fkt = rt_basix.tabulate(1, pnt)

            values_cell = evaluate_fe_functions(shp_fkt[0, :, :, :], dofs_basix)
            values_cell_dx = evaluate_fe_functions(shp_fkt[1, :, :, :], dofs_basix)
            values_cell_dy = evaluate_fe_functions(shp_fkt[2, :, :, :], dofs_basix)

            divergence = values_cell_dx[:, 0] + values_cell_dy[:, 1]

            # initialise counter
            n = 3 * degree

            # cell integrals of divergence
            for l in range(0, degree):
                for m in range(0, degree - l):
                    if (l + m >= 1):
                        # c_Tdiv = int_T div(f) * x^l * y^m dx
                        dofs_custom[n] = np.dot(divergence * pnt[:, 0]**l * pnt[:, 1]**m, wts)

                        n = n + 1

            # remaining cell contributions
            if degree > 2:
                for l in range(1, degree - 1):
                    for m in range(0, degree - 1 - l):
                        # c_T2 = int_T f * e_2 * x^l * y^m dx
                        dofs_custom[n] = np.dot(values_cell[:, 1] * pnt[:, 0]**l * pnt[:, 1]**m, wts)

                        n = n + 1
    else:
        raise NotImplementedError("Test only implemented for 2D")
        
    return dofs_custom


'''Test interpolation of (reference) cell'''

@pytest.mark.parametrize("cell", [CellType.triangle])
@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("in_ref_cell", [True])
def test_element_reference(cell, degree, in_ref_cell):
    # get topological dimension
    if cell == CellType.triangle:
        tdim = 2
    else:
        tdim = 3

    # setup test function
    rt_basix = basix.create_element(basix.ElementFamily.RT, CellType.triangle, 
                                    degree, basix.LagrangeVariant.equispaced)
    dofs_basix = 2 * (np.random.rand(rt_basix.dim) + 0.1)

    # create custom element
    rt_custom = create_hierarchic_rt(cell, degree, False)

    if in_ref_cell:
        dofs_custom = evaluate_dofs_hierarchic_rt(tdim, degree, rt_basix, dofs_basix)
    else:
        raise NotImplementedError("Test only implemented for reference cell")

    # set test points on reference cell
    points = basix.create_lattice(cell, degree + 2, basix.LatticeType.equispaced, True)

    # compare functions at test-points
    pvalues_basix = evaluate_fe_functions(rt_basix.tabulate(0, points)[0,:,:,:], dofs_basix)
    pvalues_custom = evaluate_fe_functions(rt_custom.tabulate(0, points)[0,:,:,:], dofs_custom)

    assert np.allclose(pvalues_basix, pvalues_custom)


if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)