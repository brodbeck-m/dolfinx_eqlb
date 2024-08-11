# --- Imports ---
from mpi4py import MPI
import numpy as np
import typing

import basix
import dolfinx.fem as dfem
import dolfinx.fem.petsc as dfem_petsc
import dolfinx.mesh as dmesh
import ufl

from dolfinx_eqlb.lsolver import local_projection


# --- Check the mesh ---


def mesh_has_reversed_edges(
    domain: dmesh.Mesh,
    print_debug_information: typing.Optional[bool] = False,
) -> bool:
    """Check mech for facets with reversed definition

    Args:
        domain: The mesh
    """

    # --- Extract geometry data
    # number of cells/ factes in mesh
    n_fcts = domain.topology.index_map(1).size_local
    n_cells = domain.topology.index_map(2).size_local

    # facet/cell connectivity
    domain.topology.create_connectivity(1, 2)
    fct_to_cell = domain.topology.connectivity(1, 2)
    domain.topology.create_connectivity(2, 1)
    cell_to_fct = domain.topology.connectivity(2, 1)

    # initialise facet orientations
    domain.topology.create_entity_permutations()
    fct_permutations = domain.topology.get_facet_permutations().reshape((n_cells, 3))

    # --- Check for reversed edges
    reversed_fcts = np.zeros((0, 3))

    for f in range(0, n_fcts):
        # get cells adjacent to f
        cells = fct_to_cell.links(f)

        if len(cells) > 1:
            # local facet id of cell
            if_plus = np.where(cell_to_fct.links(cells[0]) == f)[0][0]
            if_minus = np.where(cell_to_fct.links(cells[1]) == f)[0][0]

            # Get facet permutation
            perm_plus = fct_permutations[cells[0], if_plus]
            perm_minus = fct_permutations[cells[1], if_minus]

            if perm_plus != perm_minus:
                reversed_fcts = np.append(
                    reversed_fcts, [[f, cells[0], cells[1]]], axis=0
                )
        else:
            # local facet id of cell
            if_plus = np.where(cell_to_fct.links(cells[0]) == f)[0][0]

            # Get facet permutation
            perm_plus = fct_permutations[cells[0], if_plus]

            if perm_plus != 0:
                reversed_fcts = np.append(
                    reversed_fcts, [[f, cells[0], cells[0]]], axis=0
                )

    if reversed_fcts.shape[0] == 0:
        return False
    else:
        if print_debug_information:
            print("Facets with reversed edges:")
            print(reversed_fcts)

        return True


# --- Check Boundary Conditions ---


def check_boundary_conditions(
    sigma_eq: dfem.Function,
    sigma_proj: dfem.Function,
    boundary_function: dfem.Function,
    facet_function: typing.Any,
    boundary_facets: typing.List[int],
) -> bool:
    """Check boundary conditions

    Function checks if projected flux boundary conditions are satisfied after
    the equilibration.

    Args:
        sigma_eq:           The equilibrated flux
        sigma_proj:         The projected flux
        boundary_function:  The boundary function
        facet_function:     The facet function
        boundary_facets:    The boundary facets
    """

    # Check if flux-space is discontinuous
    flux_is_dg = sigma_eq.function_space.element.basix_element.discontinuous

    # Initialise storage of test data
    boundary_dofs_eflux = np.array([], dtype=np.int32)
    boundary_dofs_data = np.array([], dtype=np.int32)

    # Provide equilibrated flux and boundary function in basix spaces
    if flux_is_dg:
        # The mesh
        domain = sigma_eq.function_space.mesh

        # Create basix RT-space
        degree = sigma_eq.function_space.element.basix_element.degree

        P_rt = ufl.FiniteElement("RT", domain.ufl_cell(), degree)
        V_rt = dfem.FunctionSpace(domain, P_rt)

        # Interpolate eqlb. flux into basix RT-space
        sigma = dfem.Function(V_rt)
        x_sigma = np.zeros(sigma.x.array.shape, dtype=np.float64)

        sigma.interpolate(sigma_eq)
        x_sigma += sigma.x.array

        sigma.x.array[:] = 0.0
        sigma.interpolate(sigma_proj)
        x_sigma += sigma.x.array

        sigma.x.array[:] = x_sigma[:]

        # Interpolate boundary conditions into basix RT-space
        sigma_bc = dfem.Function(V_rt)
        sigma_bc.interpolate(boundary_function)
    else:
        sigma = sigma_eq
        sigma_bc = boundary_function

    # Extract boundary DOFs
    for id in boundary_facets:
        # Get facets
        fcts = facet_function.indices[facet_function.values == id]

        # Get DOFs (equilibrated flux)
        boundary_dofs_eflux = np.append(
            boundary_dofs_eflux,
            dfem.locate_dofs_topological(sigma.function_space, 1, fcts),
        )

        # Get DOFs (boundary function)
        if not flux_is_dg:
            boundary_dofs_data = np.append(
                boundary_dofs_data,
                dfem.locate_dofs_topological(sigma_bc.function_space.sub(0), 1, fcts),
            )

    if flux_is_dg:
        boundary_dofs_data = boundary_dofs_eflux

    # Check boundary DOFs
    if np.allclose(
        sigma.x.array[boundary_dofs_eflux],
        sigma_bc.x.array[boundary_dofs_data],
    ):
        return True
    else:
        return False


# --- Check the equilibration ---


def check_divergence_condition(
    sigma_eq: typing.Union[dfem.Function, typing.Any],
    sigma_proj: typing.Union[dfem.Function, typing.Any],
    rhs_proj: dfem.Function,
    mesh: typing.Optional[dmesh.Mesh] = None,
    degree: typing.Optional[int] = None,
    flux_is_dg: typing.Optional[bool] = None,
    print_debug_information: typing.Optional[bool] = False,
) -> bool:
    """Check the divergence condition

    Let sigma_eq be the equilibrated flux, then

                        div(sigma_eq) = rhs_proj

    must hold. To check the condition the divergence of the flux within the RT space
    is calculated and compared to the RHS. Therefore point-evaluations on a test-set
    are performed.

    Available debug information:
        - List of cells, where norm is greater than tolerance

    Args:
        sigma_eq (Function or ufl-Argument):      The equilibrated flux
        sigma_proj (Function or ufl-Argument):    The projected flux
        rhs_proj (Function):                      The projected right-hand side
        mesh (optional, dmesh.Mesh):              The mesh (optional, only for ufl flux required)
        degree (optional, int):                   The flux degree (optional, only for ufl flux required)
        flux_is_dg (optional, bool):              Identifier id flux is in DRT space (optional, only for ufl flux required)
        print_debug_information (optional, bool): Print debug information (optional)
    """
    # --- Extract solution data
    if type(sigma_eq) is dfem.Function:
        # the mesh
        mesh = sigma_eq.function_space.mesh

        # degree of the flux space
        degree = sigma_eq.function_space.element.basix_element.degree

        # check if flux space is discontinuous
        flux_is_dg = sigma_eq.function_space.element.basix_element.discontinuous
    else:
        # Check input
        if mesh is None:
            raise ValueError("Mesh must be provided")

        if degree is None:
            raise ValueError("Flux degree must be provided")

        if flux_is_dg is None:
            raise ValueError("Flux type must be provided")

    # the geometry DOFmap
    gdmap = mesh.geometry.dofmap

    # number of cells in mesh
    n_cells = mesh.topology.index_map(2).size_local

    # --- Calculate divergence of the equilibrated flux
    if rhs_proj.function_space.dofmap.index_map_bs == 1:
        V_div = dfem.FunctionSpace(mesh, ("DG", degree - 1))
    else:
        V_div = dfem.VectorFunctionSpace(mesh, ("DG", degree - 1))

    if flux_is_dg:
        div_sigeq = local_projection(V_div, [ufl.div(sigma_eq + sigma_proj)])[0]
    else:
        div_sigeq = local_projection(V_div, [ufl.div(sigma_eq)])[0]

    # points for checking divergence condition
    n_points = int(0.5 * (degree + 3) * (degree + 4))
    points_3d = np.zeros((n_points, 3))

    # loop over cells
    error_cells = []

    for c in range(0, n_cells):
        # points on current element
        x = np.sort(np.random.rand(2, n_points), axis=0)
        points = (
            np.column_stack([x[0], x[1] - x[0], 1.0 - x[1]])
            @ mesh.geometry.x[gdmap.links(c), :2]
        )
        points_3d[:, :2] = points

        # evaluate div(sig_eqlb)
        val_div_sigeq = div_sigeq.eval(points_3d, n_points * [c])

        # evaluate RHS
        val_rhs = rhs_proj.eval(points_3d, n_points * [c])

        if not np.allclose(val_div_sigeq, val_rhs):
            error_cells.append(c)

    if len(error_cells) == 0:
        return True
    else:
        # Print cells with divergence error
        if print_debug_information:
            print("Cells with divergence error:")
            print(error_cells)

        return False


def check_jump_condition(
    sigma_eq: dfem.Function,
    sigma_proj: dfem.Function,
    print_debug_information: typing.Optional[bool] = False,
) -> bool:
    """Check the jump condition

    For the semi-explicit equilibration procedure the flux within the H(div)
    conforming RT-space is constructed from the projected flux as well as a
    reconstruction within the element-wise RT space. This routine checks if
    the normal component of sigma_proj + sigma_eq is continuous across all
    internal facets by comparing the H(div) norm of an H(div) interpolant
    of sigma_proj + sigma_eq with the function itself.

    Available debug information:
        - List of cells, where norm is greater than tolerance

    Args:
        sigma_eq:                                 The equilibrated flux
        sigma_proj:                               The projected flux
        print_debug_information (optional, bool): Print debug information (optional)
    """

    # --- Extract data
    # The mesh
    domain = sigma_eq.function_space.mesh

    # The flux degree
    degree = sigma_eq.function_space.element.basix_element.degree

    # The test function
    V_test = dfem.FunctionSpace(domain, ("RT", degree))
    f_test = dfem.Function(V_test)

    # The marker space
    V_marker = dfem.FunctionSpace(domain, ("DG", 0))
    v_m = ufl.TestFunction(V_marker)

    # Project equilibrated flux into DG_k (RT_k in DG_k)
    V_flux = dfem.VectorFunctionSpace(domain, ("DG", degree))
    sigma_eq_dg = local_projection(V_flux, [sigma_eq + sigma_proj])[0]

    # Interpolate sigma_R into f_test
    f_test.interpolate(sigma_eq_dg)

    # Check if sigma_eq is in H(div)
    err = f_test - sigma_eq_dg
    form_error = dfem.form(
        (ufl.inner(err, err) + ufl.inner(ufl.div(err), ufl.div(err))) * v_m * ufl.dx
    )

    L_error = dfem_petsc.create_vector(form_error)
    dfem_petsc.assemble_vector(L_error, form_error)

    error = np.sum(L_error.array)

    # Print debug-information
    if print_debug_information:
        if not np.isclose(error, 0.0, atol=1.0e-12):
            print("Cells with non-zero ||f - I_RT(f)||_H(div):")
            print(np.where(~np.isclose(L_error.array, 0.0, atol=1.0e-12))[0])

    return np.isclose(error, 0.0, atol=1.0e-12)


def check_jump_condition_per_facet(
    sigma_eq: dfem.Function,
    sigma_proj: dfem.Function,
    print_debug_information: typing.Optional[bool] = False,
) -> bool:
    """Check the jump condition

    For the semi-explicit equilibration procedure the flux within the H(div)
    conforming RT-space is constructed from the projected flux as well as a
    reconstruction within the element-wise RT space. This routine checks if
    the normal component of sigma_proj + sigma_eq is continuous across all
    internal facets.

    Available debug information:
        [[cell+, cell-, reversed orientation]]

    Args:
        sigma_eq:                                 The equilibrated flux
        sigma_proj:                               The projected flux
        print_debug_information (optional, bool): Print debug information (optional)
    """
    # --- Extract geometry data
    # the mesh
    domain = sigma_eq.function_space.mesh

    # number of cells/ factes in mesh
    n_fcts = domain.topology.index_map(1).size_local
    n_cells = domain.topology.index_map(2).size_local

    # facet/cell connectivity
    fct_to_pnt = domain.topology.connectivity(1, 0)
    fct_to_cell = domain.topology.connectivity(1, 2)
    cell_to_fct = domain.topology.connectivity(2, 1)

    # initialise facet orientations
    domain.topology.create_entity_permutations()
    fct_permutations = domain.topology.get_facet_permutations().reshape((n_cells, 3))

    # --- Express the equilibrated flux in DRT
    # the function space
    V_drt = sigma_eq.function_space

    degree_sigrt = V_drt.element.basix_element.degree

    # project sigma_proj into DRT
    sigma_rt = local_projection(V_drt, [sigma_proj])[0]

    # calculate reconstructed flux
    sigma_rt.x.array[:] += sigma_eq.x.array[:]

    # --- Check jump condition on all facets
    error_cells = np.zeros((0, 4))

    for f in range(0, n_fcts):
        # get cells adjacent to f
        cells = fct_to_cell.links(f)

        if len(cells) > 1:
            # get points on facet
            pnts = fct_to_pnt.links(f)

            # the local facet ids
            if_plus = np.where(cell_to_fct.links(cells[0]) == f)[0][0]
            if_minus = np.where(cell_to_fct.links(cells[1]) == f)[0][0]

            # the permutation information
            perm_plus = fct_permutations[cells[0], if_plus]
            perm_minus = fct_permutations[cells[1], if_minus]

            # Create checkpoints
            dx = domain.geometry.x[pnts[1], :] - domain.geometry.x[pnts[0], :]
            pnts_3d = (
                domain.geometry.x[pnts[0], :]
                + dx * np.linspace(0, 1, degree_sigrt + 4)[1:-1, np.newaxis]
            )

            # determine facet values
            val_plus = sigma_rt.eval(pnts_3d, pnts_3d.shape[0] * [cells[0]])
            val_minus = sigma_rt.eval(pnts_3d, pnts_3d.shape[0] * [cells[1]])

            # the normal components
            normal = (1 / np.sqrt(dx[0] ** 2 + dx[1] ** 2)) * np.array([-dx[1], dx[0]])

            if not np.allclose(np.dot(val_plus, normal), np.dot(val_minus, normal)):
                if perm_plus == perm_minus:
                    error_cells = np.append(
                        error_cells, [[f, cells[0], cells[1], 1]], axis=0
                    )
                else:
                    error_cells = np.append(
                        error_cells, [[f, cells[0], cells[1], 0]], axis=0
                    )

    if error_cells.shape[0] == 0:
        return True
    else:
        if print_debug_information:
            print("Cells with jump error:")
            for i in range(0, error_cells.shape[0]):
                tf = "true" if np.isclose(error_cells[i, 3], 1) else "false"
                print(
                    "facet: {0:.0f} - cells: {1:.0f}, {2:.0f} - fct aligned: {3:s}".format(
                        error_cells[i, 0], error_cells[i, 1], error_cells[i, 2], tf
                    )
                )

        return False


def check_weak_symmetry_condition(sigma_eq: dfem.Function):
    """Check the weak symmetry condition

    Let sigma_eq be the equilibrated flux, then

                (sig_eq, J(z)) = 0

    must hold, for all z in P1. This routine checks if this condition holds true.

    Args:
        sigma_eq (Function):   The equilibrated flux
    """
    # --- Extract solution data
    # The mesh
    mesh = sigma_eq[0].function_space.mesh

    # --- Assemble weak symmetry condition
    #  The (continuous) test space
    V_test = dfem.FunctionSpace(mesh, ("P", 1))

    # The test functional
    if mesh.topology.dim == 2:
        # The test function
        v = ufl.TestFunction(V_test)

        # The linear form
        l_weaksym = dfem.form(ufl.inner(sigma_eq[0][1] - sigma_eq[1][0], v) * ufl.dx)
    else:
        raise ValueError("Test on weak symmetry condition not implemented for 3D")

    # Assemble linear form
    L = dfem_petsc.create_vector(l_weaksym)

    with L.localForm() as loc:
        loc.set(0)

    dfem_petsc.assemble_vector(L, l_weaksym)

    # --- Test weak-symmetry condition
    if np.allclose(L.array, 0):
        return True
    else:
        return False
