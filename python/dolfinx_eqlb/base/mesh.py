from mpi4py import MPI
import numpy as np

import basix
from dolfinx import default_real_type, fem, mesh
import ufl


def prepare_mesh_for_equilibration(msh: mesh.Mesh) -> mesh.Mesh:
    """Prepare a mesh for flux equilibration

    The current implementation of the flux equilibration requires at least two cells
    linked to each boundary node. This routines modifies meshes, such that this requi-
    rement is met.

    Args:
        msh: The mesh

    Returns:
        The (optionally adjusted) mesh
    """

    # The spatial dimension
    gdim = msh.geometry.dim
    fdim = gdim - 1

    # List of refined cells
    refined_cells = []

    # Required connectivity's
    msh.topology.create_connectivity(0, gdim)
    msh.topology.create_connectivity(fdim, gdim)
    pnt_to_cell = msh.topology.connectivity(0, gdim)

    # The boundary facets
    bfcts = mesh.exterior_facet_indices(msh.topology)

    # Get boundary nodes
    V = fem.functionspace(msh, ("Lagrange", 1))
    bpnts = fem.locate_dofs_topological(V, 1, bfcts)

    # Check if point is linked with only on cell
    for pnt in bpnts:
        cells = pnt_to_cell.links(pnt)

        if len(cells) == 1:
            refined_cells.append(cells[0])

    list_ref_cells = list(set(refined_cells))  # remove duplicates

    if len(list_ref_cells) > 0:
        # Add central node into refined cells
        x_new = np.copy(msh.geometry.x[:, 0:2])  # remove third component
        cells_new = np.copy(msh.geometry.dofmap)
        cells_add = np.zeros((2, 3), dtype=np.int32)

        list_ref_cells.sort()
        for i, c_init in enumerate(list_ref_cells):
            # The cell
            c = c_init + 2 * i

            # Nodes on cell
            cnodes = cells_new[c, :]
            x_cnodes = x_new[cnodes]

            # Coordinate of central node
            node_central = (1 / 3) * np.sum(x_cnodes, axis=0)

            # New node coordinates
            id_new = max(cnodes) + 1
            x_new = np.insert(x_new, id_new, node_central, axis=0)

            # Adjust definition of existing cells
            cells_new[cells_new >= id_new] += 1

            # Add new cells
            cells_add[0, :] = [cells_new[c, 1], cells_new[c, 2], id_new]
            cells_add[1, :] = [cells_new[c, 2], cells_new[c, 0], id_new]
            cells_new = np.insert(cells_new, c + 1, cells_add, axis=0)

            # Correct definition of cell c
            cells_new[c, 2] = id_new

        # Update mesh
        etype = "triangle" if (gdim == 2) else "tetrahedron"
        gelmt = basix.ufl.element("P", etype, 1, shape=(gdim,), dtype=default_real_type)

        return mesh.create_mesh(MPI.COMM_WORLD, cells_new, x_new, ufl.Mesh(gelmt))
    else:
        return msh
