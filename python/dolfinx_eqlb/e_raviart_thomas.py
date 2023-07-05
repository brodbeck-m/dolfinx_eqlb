import numpy as np

import basix
from basix import CellType, MapType, LagrangeVariant, SobolevSpace


def create_hierarchic_rt(cell: CellType, degree: int, discontinuous: bool):
    """Create hierarchic RT element

    The ansatz-functions of Raviart-Thomas elements with degree > 1 are not uniquely
    defined. Beside the definition implemented in basix, Bertrand et al. [1] propose
    an alternative representation, where the facet moments rely on the divergence of
    the function itself. This offers advantages in context of minimisation problems 
    on divergence free spaces, as some of the element DOFs can directly be eliminated
    from te resulting equation system.

    This routine generates a basix element, based in [1], which can than be used with-
    in DOLFINx.

    [1] Bertrand et al.: Stabilization-free HHO a posteriori error control, 2022

    Args:
        cell (CellType):      the basix cell type
        degree (int):         the degree of the element
        discontinuous (bool): create discontinuous version of the element

    Returns:
        the basix element
    """

    # check degree
    if degree < 1:
        raise ValueError("Degree must be at least 1")

    # check cell type
    if cell != CellType.triangle:
        raise ValueError("Only triangular cells supported")
    
    # geometrical information
    if cell == CellType.triangle:
        fct = CellType.interval
        tdim = 2
        nnodes_cell = 3
    else:
        fct = CellType.triangle
        tdim = 3
        nnodes_cell = 4

    # representation of the basis forming monomials
    rt = basix.create_element(basix.ElementFamily.RT, cell, degree, LagrangeVariant.equispaced)
    wcoeffs = rt.wcoeffs

    # --- Creating the interpolation operator ---
    # initialisation
    x = [[], [], [], []]
    M = [[], [], [], []]

    # check if derivatives are required for definition of basis
    if degree > 1:
        n_derivatives = 1
    else:   
        n_derivatives = 0

    # quadrature points on fact
    pnt, wts = basix.make_quadrature(fct, degree)

    # set-up the interpolation
    if tdim == 2:
        # --- Facet contribution
        # quadrature points on facets
        x[1].append(np.array([[1 - p[0], p[0]] for p in pnt])) # q. points on edge 0
        x[1].append(np.array([[0, p[0]] for p in pnt]))        # q. points on edge 1
        x[1].append(np.array([[p[0], 0] for p in pnt]))        # q. points on edge 2

        # set weight factors according to facet integrals
        for normal in [[-1, -1], [-1, 0], [0, 1]]:
            mat = np.zeros((degree, 2, len(wts), 1 + n_derivatives * tdim))

            for j in range(0, degree):
                # lambda = int_E f * n * s^j ds
                mat[j, 0, :, 0] = normal[0] * (pnt[:, 0]**j) * wts[:]
                mat[j, 1, :, 0] = normal[1] * (pnt[:, 0]**j) * wts[:]

            M[1].append(mat)

        # --- Cell contribution
        if degree > 1:
            # quadrature points on cell
            pnt, wts = basix.make_quadrature(cell, 2 * degree)
            x[2].append(pnt)

            # initialisation of interpolation matrix
            mat = np.zeros(((degree**2) - degree, 2, len(wts), 1 + n_derivatives * tdim))
            n = 0

            # cell integrals of divergence
            for l in range(0, degree):
                for m in range(0, degree - l):
                    if (l + m >= 1):
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


        # no nodal contributions to shape functions
        for _ in range(nnodes_cell):
            x[0].append(np.zeros((0, 2)))
            M[0].append(np.zeros((0, 2, 0, 1 + n_derivatives * tdim)))
    else:
        raise NotImplementedError("Raviart-Thomas on tetrahedra not implemented")
    
    return basix.create_custom_element(cell, [tdim], wcoeffs, x, M, n_derivatives, 
                                       MapType.contravariantPiola, SobolevSpace.HDiv,
                                       discontinuous, degree - 1, degree)