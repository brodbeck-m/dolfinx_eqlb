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
    fct_to_cell = domain.topology.connectivity(1, 2)
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
        print("Cells with non-zero ||f - I_RT(f)||_H(div):")
        print(np.where(np.isclose(L_error.array, 0.0, atol=1.0e-12))[0])

    return np.isclose(error, 0.0, atol=1.0e-12)


def check_jump_condition_per_facet(
    sigma_eq: dfem.Function,
    sig_proj: dfem.Function,
    print_debug_information: typing.Optional[bool] = False,
) -> bool:
    """Check the jump condition

    For the semi-explicit equilibration procedure the flux within the H(div)
    conforming RT-space is constructed from the projected flux as well as a
    reconstruction within the element-wise RT space. This routine checks if
    the normal component of sigma_proj + sigma_eq is continuous across all
    internal facets.

    Available debug information:
        [[cell+, cell-, reversed orientation, DOF, relative error]]

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
    fct_to_cell = domain.topology.connectivity(1, 2)
    cell_to_fct = domain.topology.connectivity(2, 1)

    # initialise facet orientations
    domain.topology.create_entity_permutations()
    fct_permutations = domain.topology.get_facet_permutations().reshape((n_cells, 3))

    # --- Interpolate proj./equilibr. flux into Baisx RT space
    # the flux degree
    degree = sigma_eq.function_space.element.basix_element.degree

    # create discontinuous RT space
    P_drt = basix.create_element(
        basix.ElementFamily.RT,
        basix.CellType.triangle,
        degree,
        basix.LagrangeVariant.equispaced,
        True,
    )

    V_drt = dfem.FunctionSpace(domain, basix.ufl_wrapper.BasixElement(P_drt))
    dofmap_sigrt = V_drt.dofmap.list

    # interpolate functions into space
    sig_eq_rt = dfem.Function(V_drt)
    sig_eq_rt.interpolate(sigma_eq)

    sig_proj_rt = dfem.Function(V_drt)
    sig_proj_rt.interpolate(sig_proj)

    # calculate reconstructed flux (use default RT-space)
    x_sig_rt = sig_proj_rt.x.array[:] + sig_eq_rt.x.array[:]

    # --- Determine sign of detj per cell
    # tabulate shape functions of geometry element
    c_element = basix.create_element(
        basix.ElementFamily.P,
        basix.CellType.triangle,
        1,
        basix.LagrangeVariant.gll_warped,
    )

    dphi_geom = c_element.tabulate(1, np.array([[0, 0]]))[1 : 2 + 1, 0, :, 0]

    # determine sign of detj per cell
    sign_detj = np.zeros(n_cells, dtype=np.float64)
    gdofs = np.zeros((3, 2), dtype=np.float64)

    for c in range(0, n_cells):
        # evaluate detj
        gdofs[:] = domain.geometry.x[domain.geometry.dofmap.links(c), :2]

        J_q = np.dot(gdofs.T, dphi_geom.T)
        detj = np.linalg.det(J_q)

        # determine sign of detj
        sign_detj[c] = np.sign(detj)

    # --- Check jump condition on all facets
    error_cells = np.zeros((0, 5))

    for f in range(0, n_fcts):
        # get cells adjacent to f
        cells = fct_to_cell.links(f)

        if len(cells) > 1:
            # signs of cell jacobians
            sign_plus = sign_detj[cells[0]]
            sign_minus = sign_detj[cells[1]]

            # local facet id of cell
            if_plus = np.where(cell_to_fct.links(cells[0]) == f)[0][0]
            if_minus = np.where(cell_to_fct.links(cells[1]) == f)[0][0]

            for i in range(0, degree):
                # local dof id of facet-normal flux
                dof_plus = dofmap_sigrt.links(cells[0])[if_plus * degree + i]
                dof_minus = dofmap_sigrt.links(cells[1])[if_minus * degree + i]

                # calculate outward flux
                if if_plus == 1:
                    flux_plus = x_sig_rt[dof_plus] * sign_plus
                else:
                    flux_plus = x_sig_rt[dof_plus] * (-sign_plus)

                if if_minus == 1:
                    flux_minus = x_sig_rt[dof_minus] * sign_minus
                else:
                    flux_minus = x_sig_rt[dof_minus] * (-sign_minus)

                # check continuity of facet-normal flux
                if not np.isclose(flux_plus + flux_minus, 0):
                    # Get facet permutation
                    perm_plus = fct_permutations[cells[0], if_plus]
                    perm_minus = fct_permutations[cells[1], if_minus]

                    # Relative error
                    error = (flux_plus + flux_minus) / flux_plus

                    # Set error information
                    if perm_plus == perm_minus:
                        error_cells = np.append(
                            error_cells, [[cells[0], cells[1], 1, i, error]], axis=0
                        )
                    else:
                        error_cells = np.append(
                            error_cells, [[cells[0], cells[1], 0, i, error]], axis=0
                        )

        if error_cells.shape[0] == 0:
            return True
        else:
            if print_debug_information:
                print("Cells with jump error:")
                print(error_cells)

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
