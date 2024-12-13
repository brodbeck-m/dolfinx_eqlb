# Copyright (C) 2024 Maximilian Brodbeck
#
# This file is part of dolfinx_eqlb
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import cffi
from dataclasses import dataclass
from mpi4py import MPI
import numpy as np
import numpy.typing as npt
import typing

import basix.ufl
import basix.cell
from dolfinx import default_scalar_type, fem, mesh
import ufl

from dolfinx_eqlb.base import (
    CompiledExpression,
    compile_expression,
    extract_constants_and_coefficients,
)

from dolfinx_eqlb.cpp import FluxBC

from .BoundaryFunction import BoundaryFunction, ConstantBoundaryFunction

ffi = cffi.FFI()


@dataclass
class CompiledForm:
    """Compiled UFL form without associated DOLFINx data."""

    ufl_form: ufl.Form  # The original ufl form
    ufcx_form: typing.Any  # The compiled form
    module: typing.Any  #  The module
    code: str  # The source code
    dtype: npt.DTypeLike  # data type used for the `ufcx_form`
    evaluation_points: int  # The evaluation points


class FluxBCs:
    """The collection of all Neumann boundary conditions of a problem"""

    def __init__(
        self, dim: int, flux_is_tensorvalued: typing.Union[bool, typing.List[bool]]
    ) -> None:
        """Initialise the collection of Neumann boundary conditions

        Args:
            dim:                  The spatial dimension
            flux_is_tensorvalued: A list of flags, indicating if a fluxes is tensor-valued
        """

        # Problem specification
        if type(flux_is_tensorvalued) == bool:
            self.flux_is_tensorvalued = [flux_is_tensorvalued]
            self.nfluxes = dim if (flux_is_tensorvalued) else 1
        else:
            if len(flux_is_tensorvalued) == 1:
                raise ValueError("flux_is_tensorvalued has to be a boolean!")

            self.flux_is_tensorvalued = flux_is_tensorvalued
            self.nfluxes = dim if (flux_is_tensorvalued[0]) else 1

            for i in range(1, len(flux_is_tensorvalued)):
                if flux_is_tensorvalued[i]:
                    self.nfluxes += dim
                else:
                    self.nfluxes += 1

        # General
        self.size: int = 0
        self.boundary_functions_checked: bool = False
        self.boundary_functions_complied: bool = False

        # Time dependency
        self.timedependent: bool = False

        # Evaluation of boundary data
        self.projections_for_eqlb: bool = False
        self.quadrature_degree: typing.Optional[int] = None

        # Boundary functions
        self.boundary_functions: typing.List[typing.Type[BoundaryFunction]] = []
        self.boundary_expressions: typing.List[CompiledExpression] = []

        # Required constants (for compilation)
        self.cnsts: typing.Dict[str, ufl.Constant] = {}

    def check_boundary_functions(self, domain: ufl.Mesh) -> None:
        """Check the boundary functions for consistency"""

        # Check specification of required constants
        if not self.cnsts:
            self.cnsts.update({"time": ufl.Constant(domain)})
            self.cnsts.update({"tfunc": ufl.Constant(domain, shape=(self.size,))})

        # Check the boundary functions
        if not self.boundary_functions_checked:
            for f in self.boundary_functions:
                # Flag, if the flux is tensor-valued
                if f.id_subspace is None:
                    tvalued_flux = self.flux_is_tensorvalued[0]
                else:
                    tvalued_flux = self.flux_is_tensorvalued[f.id_subspace]

                tvalued_flux = (
                    self.flux_is_tensorvalued[0]
                    if (f.id_subspace is None)
                    else self.flux_is_tensorvalued[f.id_subspace]
                )

                # Check specified boundary markers
                if tvalued_flux:
                    if not all(isinstance(el, list) for el in f.boundary_markers):
                        raise ValueError("Invalid definition of boundary markers")
                else:
                    if not all(isinstance(el, int) for el in f.boundary_markers):
                        raise ValueError("Invalid definition of boundary markers")

                # Set time-dependency marker
                if f.is_timedependent:
                    self.timedependent = True

        # Set flag performed checks
        self.boundary_functions_checked = True

    # --- Setter methods
    def set_boundary_function(
        self, boundary_function: typing.Type[BoundaryFunction]
    ) -> None:
        """Add a boundary function

        Args:
            boundary_function: The boundary function
        """
        # Append list of boundary functions
        self.boundary_functions.append(boundary_function)
        self.boundary_functions[-1].tag = self.size

        # Check quadrature degree
        if boundary_function.projected_for_eqlb:
            if self.projections_for_eqlb:
                if boundary_function.quadrature_degree != self.quadrature_degree:
                    raise ValueError(
                        "All flux BCs for the equilibrated flux must rely on the same quadrature degree"
                    )
            else:
                self.quadrature_degree = boundary_function.quadrature_degree
                self.projections_required = True

        # Increase counter
        self.size += 1

    def set_constant_boundary_function(
        self,
        boundary_markers: typing.Union[typing.List[int], typing.List[typing.List[int]]],
        value: float,
        id_subspace: typing.Optional[int] = None,
    ):
        self.set_boundary_function(
            ConstantBoundaryFunction(boundary_markers, value, id_subspace)
        )

    # --- Getter methods
    def is_empty(self) -> bool:
        return not bool(self.boundary_functions)

    def constants_and_coefficients_map(
        self,
        domain: mesh.Mesh,
        time: typing.Optional[fem.Constant] = None,
        tfunc: typing.Optional[fem.Constant] = None,
    ) -> typing.Tuple[
        typing.Dict[ufl.Constant, fem.Constant],
        typing.Dict[ufl.Coefficient, fem.Function],
    ]:
        """Get the overall constants and coefficients map

        This maps link the abstract definitions of constants and coefficients with
        their mesh-dependent counterparts with actual values.

        Args:
            domain: The mesh
            time:   The physical time
            tfunc:  The time-dependent scaling factors for each boundary function

        Returns:
            The constants map,
            The coefficients map
        """

        if time is None:
            # Set time to zero
            time = fem.Constant(domain, default_scalar_type(0))

            # Initialise scaling with 1
            if tfunc is None:
                tfunc = fem.Constant(
                    domain, default_scalar_type(tuple([1.0 for _ in range(self.size)]))
                )
            else:
                raise ValueError(
                    "No time-dependent scaling factors in stationary problem"
                )
        else:
            if tfunc is None:
                raise ValueError("Time-dependent scaling factors have to be provided")

        # Initialise the dicts
        cnsts_map = {self.cnsts["tfunc"]: tfunc}
        coeff_map = {}

        # Loop over all boundary functions
        val_time = time.value

        for i, f in enumerate(self.boundary_functions):
            # Update time-functions
            if f.is_timedependent and f.has_time_function:
                tfunc.value[i] = f.time_function(val_time)
            else:
                tfunc.value[i] = 1.0

            # Extend constants map
            if f.has_constants:
                if f.is_timedependent:
                    cnsts_map.update(f.constants(domain, {self.cnsts["time"]: time}))
                else:
                    cnsts_map.update(f.constants(domain))

            # Extend coefficients map
            if f.has_coefficients:
                coeff_map.update(f.coefficients(domain, val_time))

        return cnsts_map, coeff_map

    def essential_bcs_dual(
        self,
        V: fem.FunctionSpace,
        facet_functions: mesh.MeshTags,
        time: fem.Constant,
        tfunc: fem.Constant,
    ) -> typing.List[FluxBC]:
        """The essential boundary conditions for the equilibration problem

        Args:
            V:               The function space
            facet_functions: The boundary markers
            time:            The physical time
            tfunc:           The time-dependent scaling factors for each boundary function
        """

        # Check compilation status
        if not self.boundary_functions_complied:
            raise RuntimeError("Essential flux BCs have not been pre-complied yet!")

        # The mesh
        domain = V.mesh

        # The geometric dimension
        gdim = V.mesh.geometry.dim

        # Create data structure with flux BCs
        bcs = [[] for _ in range(self.nfluxes)]

        for f, cf in zip(self.boundary_functions, self.boundary_expressions):
            # Constant and coefficient maps
            cnsts_map = f.constants(domain, time)
            cnsts_map.update({self.cnsts["tfunc"]: tfunc})
            coeff_map = f.coefficients(domain, time)

            # Data of pre-complied expression
            cnsts, coeff = extract_constants_and_coefficients(
                cf.ufl_expression, cf.ufcx_expression, cnsts_map, coeff_map
            )

            # The FluxBC
            offs = 0
            for j in range(0, f.id_subspace):
                if self.flux_is_tensorvalued[j]:
                    offs += gdim
                else:
                    offs += 1

            for k, bmarkers in enumerate(f.boundary_markers):
                # The boundary facets
                fcts = facet_functions.indices[
                    np.isin(facet_functions.values, bmarkers)
                ]

                # The flux BC
                bcs[offs + k].append(
                    FluxBC(
                        fcts,
                        cnsts,
                        coeff,
                        f.is_zero,
                        f.is_timedependent,
                        f.has_time_function,
                        f.quadrature_degree,
                    )
                )

    # --- Time update
    def update_time(self, time: fem.Constant, tfunc: fem.Constant) -> None:
        if self.timedependent:
            # The value of the physical time
            val_time = time.value

            # Update the time-dependent boundary functions
            for i, f in enumerate(self.boundary_functions):
                # Update time-functions
                if f.is_timedependent and f.has_time_function:
                    tfunc.value[i] = f.time_function(val_time)

    # --- Mesh-independent definitions
    def linear_form_primal(
        self,
        domain: ufl.Mesh,
        test_functions: typing.Union[typing.Any, typing.List[typing.Any]],
    ) -> typing.Any:
        """Combine all boundary terms considered in the primal problem

        Args:
            domain:         The mesh
            test_functions: The test function(s)

        Returns:
            The boundary terms for the linear form of the primal problem
        """

        def linear_form(
            value: typing.Any,
            v: typing.Any,
            boundary_markers: typing.Union[
                typing.List[int], typing.List[typing.List[int]]
            ],
            tensor_valued: bool,
        ) -> typing.Any:
            """Provides the resulting linear form for a given boundary function

            Args:
                l:        The linear form
                value:    The boundary
                v:        The test function
                bfct_ids: The boundary tags

            Returns:
                The linear form
            """

            def surface_integrator(
                bfct_ids: typing.List[int],
            ) -> typing.Tuple[bool, typing.Any]:
                """The surface integrator

                This function creates a surface integrator from a list of
                given boundary markers.

                Args:
                    bfct_ids: The boundary markers

                Returns:
                    True, if the surface integrator is not None
                    The surface integrator
                """
                # The number of surfaces
                nsurf = len(bfct_ids)

                # The surface integrator
                if nsurf > 0:
                    dsurf = ufl.ds(bfct_ids[0])

                    if nsurf > 1:
                        for i in range(1, nsurf):
                            dsurf += ufl.ds(bfct_ids[i])

                    return True, dsurf
                else:
                    return False, None

            if tensor_valued:
                # Initialise this boundary contribution
                l_i = 0

                for j, bmarkers in enumerate(boundary_markers):
                    # The surface integrator
                    is_not_none, ds = surface_integrator(bmarkers)

                    # Extend the linear form
                    if is_not_none:
                        l_i += value * v[j] * ds

                return l_i
            else:
                # The surface integrator
                is_not_none, ds = surface_integrator(boundary_markers)

                return value * v * ds

        # Check boundary functions (if not already done)
        self.check_boundary_functions(domain)

        # Has non-zero BCs
        has_boundary_contributions = False

        # Initialise the linear form
        l = 0

        for i, f in enumerate(self.boundary_functions):
            # The id of the sub-space
            id = f.id_subspace

            # Frag, if the flux is tensor-valued
            tvalued_flux = (
                self.flux_is_tensorvalued[0]
                if (id is None)
                else self.flux_is_tensorvalued[id]
            )

            if not f.is_zero:
                # Set flag
                has_boundary_contributions = True

                # The (scaled) boundary value
                value = self.cnsts["tfunc"][i] * f.value(domain, self.cnsts["time"])

                # The test function
                v = test_functions if (id is None) else test_functions[id]

                # The boundary contributions in the linear form
                l_i = linear_form(value, v, f.boundary_markers, tvalued_flux)

                if l_i is not None:
                    l += l_i

        if has_boundary_contributions:
            return l
        else:
            return None

    def compile_boundary_functions(
        self,
        domain: ufl.Mesh,
        element: typing.Any,
        form_compiler_options: typing.Optional[dict] = {
            "scalar_type": default_scalar_type
        },
        jit_options: typing.Optional[dict] = None,
    ) -> None:
        """Mesh-independent compilation of the boundary functions

        Args:
            domain:  The (abstract) mesh definition
            element: The finite element of the flux

        Returns:
            The boundary contributions to the linear form of the primal problem
        """

        # Check boundary functions (if not already done)
        self.check_boundary_functions(domain)

        # The geometric dimension
        gdim = domain.geometric_dimension()

        # The mesh cell
        cell_type = domain._ufl_coordinate_element.cell_type

        facet_types = basix.cell.subentity_types(cell_type)[gdim - 1]
        facet_type = facet_types[0]
        nfcts = len(facet_types)

        # Pre-compile the boundary functions for the equilibration problem
        try:
            # Evaluation points (interpolation of a projected function)
            if self.projections_for_eqlb:
                # Quadrature points on the reference facet
                qpnts_fct, _ = basix.make_quadrature(facet_type, self.quadrature_degree)
                nqpnts_fct = qpnts_fct.shape[0]

                # Quadrature points on all facets of a reference cell
                qpnts = np.zeros((nfcts * nqpnts_fct, gdim), dtype=np.float64)

                if cell_type == basix.CellType.triangle:
                    # Initialisations
                    id_fct1 = nqpnts_fct
                    id_fct2 = 2 * nqpnts_fct

                    # Map 1D points to 2D facet
                    for i in range(0, nqpnts_fct):
                        # Point on facet 0
                        qpnts[i, 0] = 1 - qpnts_fct[i]
                        qpnts[i, 1] = qpnts_fct[i]

                        # Point on facet 1
                        qpnts[id_fct1, 0] = 0
                        qpnts[id_fct1, 1] = qpnts_fct[i]
                        id_fct1 += 1

                        # Point on facet 2
                        qpnts[id_fct2, 0] = qpnts_fct[i]
                        qpnts[id_fct2, 1] = 0
                        id_fct2 += 1
                else:
                    raise NotImplementedError("3D meshes currently not supported")

            # Evaluation points (pure interpolation)
            nipnts_fct = np.sum(np.isclose(element._element.points[:, 0], 0))
            ipnts = element._element.points[: nipnts_fct * nfcts, :]

            for i, f in enumerate(self.boundary_functions):
                # The boundary value
                value = f.value(domain, self.cnsts["time"])

                # Compile the value for later usage in c++
                pnts = qpnts if (f.projected_for_eqlb) else ipnts

                self.boundary_expressions.append(
                    compile_expression(
                        MPI.COMM_WORLD, value, pnts, form_compiler_options, jit_options
                    )
                )
        except:
            raise RuntimeError(
                "Error in compiling the BCs for the equilibration problem"
            )

        # Set compilation flag
        self.boundary_functions_complied = True
