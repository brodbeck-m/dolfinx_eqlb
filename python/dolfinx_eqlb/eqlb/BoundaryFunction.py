# Copyright (C) 2024 Maximilian Brodbeck
#
# This file is part of dolfinx_eqlb
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np
import typing

from dolfinx import fem, mesh
import ufl


class BoundaryFunction:
    """A normal-trace of a flux on the boundary"""

    def __init__(
        self,
        boundary_markers: typing.Union[typing.List[int], typing.List[typing.List[int]]],
        id_subspace: typing.Optional[int] = None,
        quadrature_degree: typing.Optional[int] = None,
        is_timedependent: typing.Optional[bool] = False,
        time_function: typing.Optional[typing.Callable] = None,
    ) -> None:
        """Initialise a function on the boundary

        Applies a normal trace - a scalar value - on all specified facets. Time dependency and
        a quadrature degree for the equilibration process can be specified.

        For vector-valued fluxes

                    boundary_markers[:]

        specifies the boundary ids, where the normal trace is applied. For tensor-valued
        fluxes the normal trace can be applied to individual rows, with

                    boundary_markers[i][:]

        specifying the boundary ids, where the normal trace is applied on the i-th row. Pass an
        empty list, if there is no condition on an individual row.

        If the function space of the underlying primal problem is a mixed space, the id of the subspace
        which is affected by the boundary condition has to be specified.

        Args:
            boundary_markers:  The boundary markers where the normal trace is applied
            id_subspace:       Id of the subspace V.sub(i) affected by the BC
            quadrature_degree: The quadrature degree used for projection into an discrete
                               H(div) space for flux equilibration
            is_timedependent:  True, if the function is time-dependent
            time_function:     A time dependent function, scaling the boundary values over time
        """

        self.is_initialised: bool = False
        self.is_zero: bool = False

        # Name and Tag of the boundary function
        self.name: typing.Optional[str] = None
        self.tag: typing.Optional[int] = None

        # Boundary facets
        self.boundary_markers = boundary_markers

        # Subspace
        self.id_subspace = id_subspace

        # Time dependency
        self.is_timedependent = is_timedependent
        self.has_time_function = True if (time_function is not None) else False
        self.time_function = time_function

        # Required quadrature for equilibration
        self.projected_for_eqlb = False if (quadrature_degree is None) else True
        self.quadrature_degree = quadrature_degree

        # Required constants and coefficients
        self.has_constants: bool = False
        self.has_coefficients: bool = False

        self.cnsts: typing.List[ufl.Constant] = []
        self.coffs: typing.List[ufl.Coefficient] = []

    # --- Initialisation
    def initialise_ufl(
        self,
        constants: typing.Optional[typing.List[ufl.Constant]] = None,
        coefficients: typing.Optional[typing.List[ufl.Coefficient]] = None,
    ) -> None:
        """Initialise the abstract constants and coefficients

        Args:
            constants:    The constants
            coefficients: The coefficients
        """
        if constants is not None:
            self.has_constants = True
            self.cnsts = constants

        if coefficients is not None:
            self.has_coefficients = True
            self.coffs = coefficients

    # --- Problem specific definitions
    def constants(
        self,
        domain: mesh.Mesh,
        time: typing.Optional[typing.Dict[ufl.Constant, fem.Constant]],
    ) -> typing.Dict[ufl.Constant, fem.Constant]:
        """Get mesh-dependent constants

        Set the mesh dependent values of the required constants and return them
        as a dict. Add the time dict if the value of this boundary function depends
        on the constant 'time'.

        Args:
            domain: The mesh
            time:   The dict constant for the physical time
                    (add only when required)

        Returns:
            The constants
        """
        raise {}

    def coefficients(
        self, domain: mesh.Mesh, time: typing.Optional[float]
    ) -> typing.Dict[ufl.Coefficient, fem.Function]:
        """Get mesh-dependent coefficients

        Set the mesh dependent functions and return them as a list.

        Args:
            domain: The mesh

        Returns:
            The constants
        """
        raise {}

    def value(self, domain: ufl.Mesh, time: ufl.Constant) -> typing.Any:
        """Value of the function

        This is the mesh independent definition of the normal trace of a flux.
        Use abstract definitions of constants and FEM functions here, which have
        to be initialised by the "initialise_ufl"  method. The mesh dependent values
        of used constants and coefficients on a specific mesh can be created using
        the methods "constants" resp. "coefficients".

        Args:
            domain: The mesh
            time:   The physical time

        Returns:
            The boundary value
        """
        raise NotImplementedError


class ConstantBoundaryFunction(BoundaryFunction):
    """A constant normal-trace of a flux on the boundary"""

    def __init__(
        self,
        boundary_markers: typing.List[typing.List[int]],
        value: float,
        id_subspace: typing.Optional[typing.Tuple[int]] = None,
    ) -> None:
        """Initialise a constant function on the boundary

        Applies a constant normal trace - a constant, scalar value - on all specified facets. Time
        dependency and a quadrature degree for the equilibration process can be specified.

        For vector-valued fluxes

                    boundary_markers[:]

        specifies the boundary ids, where the normal trace is applied. For tensor-valued
        fluxes the normal trace can be applied to individual rows, with

                    boundary_markers[i][:]

        specifying the boundary ids, where the normal trace is applied on the i-th row. Pass an
        empty list, if there is no condition on an individual row.

        If the function space of the underlying primal problem is a mixed space, the id of the subspace
        which is affected by the boundary condition has to be specified.

        Args:
            boundary_markers: The boundary markers where the normal trace is applied
            id_subspace:      Id of the subspace V.sub(i) affected by the BC
            value:            The (constant) value of the normal trace
        """

        # Constructor of the base class
        super().__init__(boundary_markers, id_subspace, None, False, None)

        # Constant boundary value
        if np.isclose(value, 0.0):
            self.is_zero = True
            self.constant_value = 0.0
        else:
            self.constant_value = value

    # --- Setter methods
    def value(self, domain: ufl.Mesh, time: ufl.Constant) -> float:
        """Value of the function

        Args:
            domain: The mesh
            time:   The physical time

        Returns:
            The boundary values
        """
        # Initialisations
        self.has_constants = False
        self.has_coefficients = False

        return self.constant_value
