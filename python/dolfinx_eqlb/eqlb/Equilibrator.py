# Copyright (C) 2024 Maximilian Brodbeck
#
# This file is part of dolfinx_eqlb
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Mesh independent definition of an equilibration problem"""

import numpy as np
import typing

import basix
from dolfinx import default_real_type
import ufl

import dolfinx_eqlb.cpp as _cpp
from dolfinx_eqlb.base import create_hierarchic_rt

from .equilibration import (
    ProblemType,
    EqlbStrategy,
    problemtype_to_cpp,
    strategy_to_cpp,
)


class Equilibrator:
    """The basic, mesh-independent definition of an equilibration problem"""

    def __init__(
        self,
        msh: ufl.Mesh,
        problem_type: ProblemType,
        strategy: EqlbStrategy,
        degree: int,
        quadrature_degree: typing.Optional[int] = None,
    ):
        """The constructor

        Args:
            msh:          The generalised mesh
            problem_type: The type of the equilibration problem
            strategy:     The equilibration strategy
            degree:       The degree of the flux space

        """

        # The type of the equilibration problem
        self._problem_type = problem_type

        # The equilibration strategy
        self._strategy = strategy

        # The degree of equilibrated flux
        self._degree = degree

        # --- Required finite element spaces
        cell = msh._ufl_coordinate_element._cellname
        dim = msh.geometric_dimension()

        # The approximation of the geometry
        self._element_geom = msh.ufl_coordinate_element()

        # The hat function
        self._element_hat = basix.ufl.element(
            "P", cell, 1, shape=(dim,), dtype=np.float64
        )

        # The flux space
        if strategy == EqlbStrategy.semi_explicit:
            self._element_flux = create_hierarchic_rt(
                self._element_geom.cell_type, degree, True
            )
        else:
            self._element_flux = basix.ufl.element(
                "RT", cell, degree, dtype=default_real_type
            )

        # The cpp backend
        qdegree = 2 * degree - 2 if (quadrature_degree is None) else quadrature_degree

        self._cpp_object = _cpp.Equilibrator(
            problemtype_to_cpp(problem_type),
            strategy_to_cpp(strategy),
            self._element_geom.basix_element._e,
            self._element_hat.basix_element._e,
            self._element_flux.basix_element._e,
            qdegree,
        )

    # --- Getter methods ---
    def degree(self) -> int:
        """The degree of the flux space"""
        return self._degree

    def problem_type(self) -> _cpp.ProblemType:
        """The type of the equilibration problem"""
        return self._cpp_object.problem_type

    def strategy(self) -> _cpp.EqlbStrategy:
        """The equilibration strategy"""
        return self._cpp_object.strategy

    def element_geom(self) -> basix.ufl._ElementBase:
        """The finite element for the geometry"""
        return self._element_geom

    def element_hat(self) -> basix.ufl._ElementBase:
        """The finite element for the hat function"""
        return self._element_hat

    def element_flux(self) -> basix.ufl._ElementBase:
        """The finite element for the flux space"""
        return self._element_flux

    def kernel_data_boundary_conditions(self) -> _cpp.KernelDataBC:
        """The kernel data for evaluating the BoundaryData"""
        return self._cpp_object.kernel_data_bcs
