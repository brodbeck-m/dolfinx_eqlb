# Copyright (C) 2024 Maximilian Brodbeck
#
# This file is part of dolfinx_eqlb
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Test boundary conditions for flux-equilibration"""

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import pytest

import basix
from dolfinx import default_scalar_type, default_real_type, fem, mesh
import dolfinx.fem.petsc
import ufl

from dolfinx_eqlb.elmtlib import create_hierarchic_rt
from dolfinx_eqlb.lsolver import local_projection

# from dolfinx_eqlb.eqlb import fluxbc, boundarydata
from dolfinx_eqlb.eqlb import BoundaryFunction, FluxBCs

from utils import (
    MeshType,
    create_unitsquare_builtin,
    create_unitsquare_gmsh,
    points_boundary_unitsquare,
    initialise_evaluate_function,
)


@pytest.mark.parametrize("bc_type", ["constants", "constants_and_coefficients"])
@pytest.mark.parametrize("mixed_primal_problem", [False, True])
@pytest.mark.parametrize("tensor_valued_flux", [False, True])
def test_FluxBCs_linearform(
    bc_type: str, mixed_primal_problem: bool, tensor_valued_flux: bool
):
    class BfuncConstsX(BoundaryFunction):
        def constants(self, domain, time):
            constant_map = {}

            # Append by time
            constant_map.update(time)

            # Append by own constants
            constant_map.update(
                {
                    self.cnsts[0]: fem.Constant(domain, default_scalar_type(0.75)),
                    self.cnsts[1]: fem.Constant(domain, default_scalar_type(1.5)),
                }
            )

            return constant_map

        def value(self, domain, time):
            # Initialise constants
            if not self.has_constants:
                self.cnsts = [ufl.Constant(domain), ufl.Constant(domain)]
                self.has_constants = True

            # The spatial coordinates
            x = ufl.SpatialCoordinate(domain)

            return self.cnsts[0] * x[0] + self.cnsts[1] * time

    class BfuncConstsY(BoundaryFunction):
        def constants(self, domain, time):
            constant_map = {}

            # Append by time
            constant_map.update(time)

            # Append by own constants
            constant_map.update(
                {
                    self.cnsts[0]: fem.Constant(domain, default_scalar_type(1.5)),
                    self.cnsts[1]: fem.Constant(domain, default_scalar_type(3.0)),
                }
            )

            return constant_map

        def value(self, domain, time):
            # Initialise constants
            if not self.has_constants:
                self.cnsts = [ufl.Constant(domain), ufl.Constant(domain)]
                self.has_constants = True

            # The spatial coordinates
            x = ufl.SpatialCoordinate(domain)

            return self.cnsts[0] * x[1] + self.cnsts[1] * time

    class BfuncCoeffsX(BoundaryFunction):
        def constants(self, domain, time):
            constant_map = {}

            # Append by time
            constant_map.update(time)

            # Append by own constants
            constant_map.update(
                {
                    self.cnsts[0]: fem.Constant(domain, default_scalar_type(0.75)),
                    self.cnsts[1]: fem.Constant(domain, default_scalar_type(1.5)),
                }
            )

            return constant_map

        def coefficients(self, domain, time):
            # The function space
            Vh = fem.functionspace(domain, self.coffs[0].ufl_element())
            f = fem.Function(Vh)
            f.interpolate(lambda x: x[0])

            return {self.coffs[0]: f}

        def value(self, domain, time):
            # Initialise constants
            if not self.has_constants:
                self.cnsts = [ufl.Constant(domain), ufl.Constant(domain)]
                self.has_constants = True

            # Initialise coefficients
            if not self.has_coefficients:
                el = basix.ufl.element("Lagrange", domain.ufl_cell()._cellname, 1)
                V = ufl.FunctionSpace(domain, el)
                self.coffs = [ufl.Coefficient(V)]
                self.has_coefficients = True

            return self.cnsts[0] * self.coffs[0] + self.cnsts[1] * time

    class BfuncCoeffsY(BoundaryFunction):
        def constants(self, domain, time):
            constant_map = {}

            # Append by time
            constant_map.update(time)

            # Append by own constants
            constant_map.update(
                {
                    self.cnsts[0]: fem.Constant(domain, default_scalar_type(1.5)),
                    self.cnsts[1]: fem.Constant(domain, default_scalar_type(3.0)),
                }
            )

            return constant_map

        def coefficients(self, domain, time):
            # The function space
            Vh = fem.functionspace(domain, self.coffs[0].ufl_element())
            f = fem.Function(Vh)
            f.interpolate(lambda x: x[1])

            return {self.coffs[0]: f}

        def value(self, domain, time):
            # Initialise constants
            if not self.has_constants:
                self.cnsts = [ufl.Constant(domain), ufl.Constant(domain)]
                self.has_constants = True

            # Initialise coefficients
            if not self.has_coefficients:
                el = basix.ufl.element("Lagrange", domain.ufl_cell()._cellname, 2)
                V = ufl.FunctionSpace(domain, el)
                self.coffs = [ufl.Coefficient(V)]
                self.has_coefficients = True

            return self.cnsts[0] * self.coffs[0] + self.cnsts[1] * time

    if bc_type == "constants":
        bfx = BfuncConstsX
        bfy = BfuncConstsY
    else:
        bfx = BfuncCoeffsX
        bfy = BfuncCoeffsY

    if mixed_primal_problem:
        id_subspaces = [0, 1]
    else:
        id_subspaces = [None]

    # --- Mesh-independent definitions
    # The abstracted mesh
    c_el = basix.ufl.element(
        "Lagrange", "triangle", 1, shape=(2,), dtype=default_real_type
    )
    domain = ufl.Mesh(c_el)

    if mixed_primal_problem:
        # Initialise collection of boundary functions (FluxBCs)
        flux_bcs = FluxBCs(2, [tensor_valued_flux, not tensor_valued_flux])

        # The abstract function space of the primal problem
        if flux_bcs.flux_is_tensorvalued[0]:
            element_u = basix.ufl.element(
                "Lagrange", "triangle", 2, shape=(2,), dtype=default_real_type
            )
            element_p = basix.ufl.element(
                "Lagrange", "triangle", 1, dtype=default_real_type
            )
        else:
            element_u = basix.ufl.element(
                "Lagrange", "triangle", 2, dtype=default_real_type
            )
            element_p = basix.ufl.element(
                "Lagrange", "triangle", 1, shape=(2,), dtype=default_real_type
            )

        V = ufl.FunctionSpace(domain, basix.ufl.mixed_element([element_u, element_p]))

        # Trail functions
        v1, v2 = ufl.TrialFunctions(V)
        trial_functions = [v1, v2]
    else:
        flux_bcs = FluxBCs(1, tensor_valued_flux)

        if tensor_valued_flux:
            element_u = basix.ufl.element(
                "Lagrange", "triangle", 2, shape=(2,), dtype=default_real_type
            )
        else:
            element_u = basix.ufl.element(
                "Lagrange", "triangle", 2, dtype=default_real_type
            )

        V = ufl.FunctionSpace(domain, element_u)

        # Trail functions
        trial_functions = ufl.TrialFunction(V)

    # Add boundary functions and precompile forms
    tfunc = lambda t: 1 - t
    for id, tvalued in zip(id_subspaces, flux_bcs.flux_is_tensorvalued):
        if tvalued:
            flux_bcs.set_constant_boundary_function([[1, 4], [2, 3]], 2.5, id)
            flux_bcs.set_boundary_function(bfx([[2], [4]], id, None, True, tfunc))
            flux_bcs.set_boundary_function(bfy([[3], [1]], id, None, True, tfunc))
        else:
            flux_bcs.set_constant_boundary_function([1, 4], 2.5, id)
            flux_bcs.set_boundary_function(bfx([2], id, None, True, tfunc))
            flux_bcs.set_boundary_function(bfy([3], id, None, True, tfunc))

    # Pre-compile linear form (primal problem)
    l = fem.compile_form(
        MPI.COMM_WORLD, flux_bcs.linear_form_primal(domain, trial_functions)
    )

    # Evaluate the linear form on a series of meshes
    def create_and_integrate(n, time_value):
        def bvalues_cnst(ds, id, v):
            return 2.5 * v * ds(id)

        def bvalues_x(msh, ds, id, time, v):
            x = ufl.SpatialCoordinate(msh)
            return (1 - time) * (0.75 * x[0] + 1.5 * time) * v * ds(id)

        def bvalues_y(msh, ds, id, time, v):
            x = ufl.SpatialCoordinate(msh)
            return (1 - time) * (1.5 * x[1] + 3.0 * time) * v * ds(id)

        def l_scalar(msh, ds, time, v):
            return (
                bvalues_cnst(ds, 1, v)
                + bvalues_cnst(ds, 4, v)
                + bvalues_x(msh, ds, 2, time, v)
                + bvalues_y(msh, ds, 3, time, v)
            )

        def l_vector(msh, ds, time, v):
            return (
                bvalues_cnst(ds, 1, v[0])
                + bvalues_cnst(ds, 4, v[0])
                + bvalues_x(msh, ds, 2, time, v[0])
                + bvalues_y(msh, ds, 3, time, v[0])
                + bvalues_cnst(ds, 2, v[1])
                + bvalues_cnst(ds, 3, v[1])
                + bvalues_x(msh, ds, 4, time, v[1])
                + bvalues_y(msh, ds, 1, time, v[1])
            )

        # The mesh
        domain_h = create_unitsquare_builtin(
            n, mesh.CellType.triangle, mesh.DiagonalType.left
        )

        # Subdomains
        domain_h.mesh.topology.create_connectivity(1, 2)

        boundaries = [
            (1, lambda x: np.isclose(x[0], 0)),
            (2, lambda x: np.isclose(x[1], 0)),
            (3, lambda x: np.isclose(x[0], 1)),
            (4, lambda x: np.isclose(x[1], 1)),
        ]

        list_bfcts = []

        for marker, locator in boundaries:
            fcts = mesh.locate_entities(domain_h.mesh, 1, locator)
            integrator = fem.compute_integration_domains(
                fem.IntegralType.exterior_facet, domain_h.mesh.topology, fcts, 1
            )
            list_bfcts.append((marker, integrator))

        subdomains = {fem.IntegralType.exterior_facet: list_bfcts}

        # The function space
        Vh = fem.functionspace(domain_h.mesh, V.ufl_element())

        # Required constants
        time = fem.Constant(domain_h.mesh, default_scalar_type(0.0))
        tfunc = fem.Constant(
            domain_h.mesh, default_scalar_type([1.0 for _ in range(flux_bcs.size)])
        )

        # Compile form
        cnsts_map, coeff_map = flux_bcs.constants_and_coefficients_map(
            domain_h.mesh, time, tfunc
        )
        l_h = fem.create_form(l, [Vh], domain_h.mesh, subdomains, coeff_map, cnsts_map)

        # The reference form
        if mixed_primal_problem:
            # The trial function
            v_u_h, v_p_h = ufl.TrialFunctions(Vh)

            # The reference linear form
            if tensor_valued_flux:
                l_h_ref = fem.form(
                    l_vector(domain_h.mesh, domain_h.ds, time, v_u_h)
                    + l_scalar(domain_h.mesh, domain_h.ds, time, v_p_h)
                )
            else:
                l_h_ref = fem.form(
                    l_scalar(domain_h.mesh, domain_h.ds, time, v_u_h)
                    + l_vector(domain_h.mesh, domain_h.ds, time, v_p_h)
                )
        else:
            # The trial function
            v_h = ufl.TrialFunction(Vh)

            # The reference linear form
            if tensor_valued_flux:
                l_h_ref = fem.form(l_vector(domain_h.mesh, domain_h.ds, time, v_h))
            else:
                l_h_ref = fem.form(l_scalar(domain_h.mesh, domain_h.ds, time, v_h))

        # Assemble the forms
        L = fem.assemble_vector(l_h)
        L_ref = fem.assemble_vector(l_h_ref)

        assert np.allclose(L.array, L_ref.array)

        # Change time
        time.value = time_value
        flux_bcs.update_time(time, tfunc)

        with L.petsc_vec.localForm() as loc:
            loc.set(0)
        fem.petsc.assemble_vector(L.petsc_vec, l_h)

        with L_ref.petsc_vec.localForm() as loc:
            loc.set(0)
        fem.petsc.assemble_vector(L_ref.petsc_vec, l_h_ref)

        assert np.allclose(L.array, L_ref.array)

    for n in range(1, 4):
        create_and_integrate(n, 1.5)


# @pytest.mark.parametrize("mesh_type", [MeshType.builtin, MeshType.gmsh])
# @pytest.mark.parametrize("degree", [1, 2, 3, 4])
# @pytest.mark.parametrize("rt_space", ["basix", "custom", "subspace"])
# @pytest.mark.parametrize("use_projection", [False, True])
# def test_boundary_data_polynomial(
#     mesh_type: MeshType, degree: int, rt_space: str, use_projection: bool
# ):
#     """Test boundary conditions from data with know polynomial degree

#     Args:
#         mesh_type:      The mesh type
#         degree:         The degree of the RT space, onto the BCs are applied
#         rt_space:       Type of RT-space
#         use_projection: If True, RT DOFs are gained by projection from boundary data
#     """

#     # Create mesh
#     n_cells = 5

#     if mesh_type == MeshType.builtin:
#         geometry = create_unitsquare_builtin(
#             n_cells, mesh.CellType.triangle, mesh.DiagonalType.crossed
#         )
#     elif mesh_type == MeshType.gmsh:
#         geometry = create_unitsquare_gmsh(0.5)
#     else:
#         raise ValueError("Unknown mesh type")

#     # Initialise connectivity
#     geometry.mesh.topology.create_connectivity(1, 2)
#     geometry.mesh.topology.create_connectivity(2, 1)

#     # Initialise flux space
#     if rt_space == "basix":
#         V_flux = fem.FunctionSpace(geometry.mesh, ("RT", degree))
#         custom_rt = False

#         boundary_function = fem.Function(V_flux)
#     elif rt_space == "custom":
#         elmt_flux = basix.ufl_wrapper.BasixElement(
#             create_hierarchic_rt(basix.CellType.triangle, degree, True)
#         )
#         V_flux = fem.FunctionSpace(geometry.mesh, elmt_flux)
#         custom_rt = True

#         boundary_function = fem.Function(V_flux)
#     elif rt_space == "subspace":
#         elmt_flux = ufl.FiniteElement("RT", geometry.mesh.ufl_cell(), degree)
#         elmt_dg = ufl.FiniteElement("DG", geometry.mesh.ufl_cell(), degree - 1)
#         V = fem.FunctionSpace(geometry.mesh, ufl.MixedElement(elmt_flux, elmt_dg))
#         V_flux = fem.FunctionSpace(geometry.mesh, elmt_flux)
#         custom_rt = False

#         boundary_function = fem.Function(V)

#     # Initialise reference flux space
#     V_ref = fem.VectorFunctionSpace(geometry.mesh, ("DG", degree - 1))

#     # Initialise boundary facets and test-points
#     list_boundary_ids = [1, 4]
#     points_eval = points_boundary_unitsquare(geometry, list_boundary_ids, degree + 1)
#     plist_eval, clist_eval = initialise_evaluate_function(geometry.mesh, points_eval)

#     npoints_eval = int(points_eval.shape[0] / 2)

#     # Set boundary degree
#     for deg in range(0, degree):
#         # Data boundary conditions
#         if deg == 0:
#             V_vec = fem.VectorFunctionSpace(geometry.mesh, ("DG", deg))
#             func_1 = fem.Function(V_vec)
#             func_1.x.array[:] = 0

#             V_scal = fem.FunctionSpace(geometry.mesh, ("DG", deg))
#             func_2 = fem.Function(V_scal)
#             func_2.x.array[:] = 0
#         else:
#             V_vec = fem.VectorFunctionSpace(geometry.mesh, ("P", deg))
#             func_1 = fem.Function(V_vec)
#             func_1.x.array[:] = 2 * (
#                 np.random.rand(V_vec.dofmap.bs * V_vec.dofmap.index_map.size_local)
#                 + 0.1
#             )

#             V_scal = fem.FunctionSpace(geometry.mesh, ("P", deg))
#             func_2 = fem.Function(V_scal)
#             func_2.x.array[:] = 3 * (
#                 np.random.rand(V_scal.dofmap.index_map.size_local) + 0.3
#             )

#         c_1 = fem.Constant(geometry.mesh, PETSc.ScalarType((1.35, 0.25)))
#         c_2 = fem.Constant(geometry.mesh, PETSc.ScalarType(0.75))

#         x_ufl = ufl.SpatialCoordinate(geometry.mesh)

#         # Create ufl-repr. of boundary condition
#         ntrace_ufl = (
#             ufl.inner(func_1, c_1)
#             + ((x_ufl[0] ** deg) + (x_ufl[1] ** deg))
#             + func_2 * c_2
#         )

#         # Create boundary conditions
#         list_bcs = []

#         for id in list_boundary_ids:
#             # Get boundary facets
#             bfcts = geometry.facet_function.indices[
#                 geometry.facet_function.values == id
#             ]

#             # Create instance of FluxBC
#             list_bcs.append(fluxbc(ntrace_ufl, bfcts, V_flux, use_projection))

#         # Initialise boundary data
#         if rt_space == "subspace":
#             boundary_data = boundarydata(
#                 [list_bcs], [boundary_function], V.sub(0), custom_rt, [[]], True
#             )
#         else:
#             boundary_data = boundarydata(
#                 [list_bcs], [boundary_function], V_flux, custom_rt, [[]], True
#             )

#         # Interpolate BC into test-space
#         rhs_ref = ufl.as_vector([-ntrace_ufl, ntrace_ufl])
#         refsol = local_projection(V_ref, [rhs_ref], quadrature_degree=2 * degree)[0]

#         # Evaluate functions at comparison points
#         if rt_space == "subspace":
#             bfunc_flux = boundary_function.sub(0).collapse()
#             val_bfunc = bfunc_flux.eval(plist_eval, clist_eval)
#         else:
#             val_bfunc = boundary_function.eval(plist_eval, clist_eval)

#         val_ref = refsol.eval(plist_eval, clist_eval)

#         assert np.allclose(val_bfunc[:npoints_eval, 0], val_ref[:npoints_eval, 0])
#         assert np.allclose(val_bfunc[npoints_eval:, 1], val_ref[npoints_eval:, 1])


# @pytest.mark.parametrize("mesh_type", [MeshType.builtin, MeshType.gmsh])
# @pytest.mark.parametrize("degree", [1, 2, 3, 4])
# def test_boundary_data_general(mesh_type: MeshType, degree: int):
#     """Test boundary conditions from non-polynomial data

#     The boundary values are projected into the RT space. The values on
#     the boundary are compared to a projection on a 1D reference space.

#     Args:
#         mesh_type: The mesh type
#         degree:    The degree of the RT space, onto the BCs are applied
#     """

#     # --- Calculate boundary conditions (2D)
#     # Create mesh
#     n_cells = 5

#     if mesh_type == MeshType.builtin:
#         geometry = create_unitsquare_builtin(
#             n_cells, mesh.CellType.triangle, mesh.DiagonalType.crossed
#         )
#     elif mesh_type == MeshType.gmsh:
#         geometry = create_unitsquare_gmsh(1 / n_cells)
#     else:
#         raise ValueError("Unknown mesh type")

#     # Initialise connectivity
#     geometry.mesh.topology.create_connectivity(1, 2)
#     geometry.mesh.topology.create_connectivity(2, 1)

#     # Initialise flux space
#     elmt_flux = basix.ufl_wrapper.BasixElement(
#         create_hierarchic_rt(basix.CellType.triangle, degree, True)
#     )
#     V_flux = fem.FunctionSpace(geometry.mesh, elmt_flux)

#     boundary_function = fem.Function(V_flux)

#     # Initialise test-points/ function evaluation
#     points_eval = points_boundary_unitsquare(geometry, [1, 4], degree)
#     plist_eval, clist_eval = initialise_evaluate_function(geometry.mesh, points_eval)

#     npoints_eval = int(points_eval.shape[0] / 2)

#     # set ufl-repr. of normal-trace on boundary
#     x_ufl = ufl.SpatialCoordinate(geometry.mesh)

#     ntrace_ufl_1 = ufl.sin(4 * ufl.pi * x_ufl[1]) * ufl.exp(-x_ufl[1])
#     ntrace_ufl_4 = ufl.cos(6 * ufl.pi * x_ufl[0]) * ufl.exp(-x_ufl[0])

#     # Create boundary conditions
#     list_bcs = []

#     bfcts_1 = geometry.facet_function.indices[geometry.facet_function.values == 1]
#     list_bcs.append(
#         fluxbc(ntrace_ufl_1, bfcts_1, V_flux, True, quadrature_degree=3 * degree)
#     )

#     bfcts_4 = geometry.facet_function.indices[geometry.facet_function.values == 4]
#     list_bcs.append(
#         fluxbc(ntrace_ufl_4, bfcts_4, V_flux, True, quadrature_degree=3 * degree)
#     )

#     # Initialise boundary data
#     boundary_data = boundarydata(
#         [list_bcs], [boundary_function], V_flux, True, [[]], True
#     )

#     # Evaluate BCs on control points
#     val_bfunc = boundary_function.eval(plist_eval, clist_eval)

#     # --- Calculate reference solution (1D)
#     # Create mesh
#     domain_1d = mesh.create_unit_interval(MPI.COMM_WORLD, n_cells)

#     # Initialise reference space
#     V_ref = fem.FunctionSpace(domain_1d, ("DG", degree - 1))

#     # Initialise test-points/ function evaluation
#     points_eval_1D = np.zeros((npoints_eval, 3))
#     points_eval_1D[:, 0] = points_eval[0:npoints_eval, 1]
#     plist_eval, clist_eval = initialise_evaluate_function(domain_1d, points_eval_1D)

#     # Reference function 1D
#     ntrace_ufl_1d = []
#     s_ufl = ufl.SpatialCoordinate(domain_1d)[0]

#     ntrace_ufl_1d.append(-ufl.sin(4 * ufl.pi * s_ufl) * ufl.exp(-s_ufl))
#     ntrace_ufl_1d.append(ufl.cos(6 * ufl.pi * s_ufl) * ufl.exp(-s_ufl))

#     # Projection into reference space
#     u = ufl.TrialFunction(V_ref)
#     v = ufl.TestFunction(V_ref)

#     dvol = ufl.Measure(
#         "dx", domain=domain_1d, metadata={"quadrature_degree": 3 * degree}
#     )

#     a = ufl.inner(u, v) * dvol

#     for i in range(0, len(ntrace_ufl_1d)):
#         # Solve 1D projection
#         l = ufl.inner(ntrace_ufl_1d[i], v) * dvol

#         problem = fem.petsc.LinearProblem(
#             a, l, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
#         )
#         refsol = problem.solve()

#         # Evaluate reference solution
#         val_ref = refsol.eval(plist_eval, clist_eval)

#         # Compare boundary-condition and reference solution
#         if i == 0:
#             assert np.allclose(val_bfunc[:npoints_eval, 0], val_ref[:, 0])
#         else:
#             assert np.allclose(val_bfunc[npoints_eval:, 1], val_ref[:, 0])


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
