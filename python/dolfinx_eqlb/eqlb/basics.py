# Copyright (C) 2024 Maximilian Brodbeck
#
# This file is part of dolfinx_eqlb
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from enum import Enum
import numpy as np
import typing

import basix
from dolfinx import default_real_type
import ufl

import dolfinx_eqlb.cpp as _cpp
from dolfinx_eqlb.base import create_hierarchic_rt


# --- Type type of an equilibration problem ---
class ProblemType(Enum):
    """The type of an equilibration problem"""

    flux = 0
    stress = 1
    stress_and_flux = 2


def problemtype_to_cpp(problem_type: ProblemType) -> _cpp.ProblemType:
    """Python enum ProblemType to its c++ counterpart

    Args:
        problem_type: The ProblemType of the equilibration as python enum

    Returns:
        The ProblemType as c++ enum
    """

    if problem_type == ProblemType.flux:
        return _cpp.ProblemType.flux
    elif problem_type == ProblemType.stress:
        return _cpp.ProblemType.stress
    elif problem_type == ProblemType.stress_and_flux:
        return _cpp.ProblemType.stress_and_flux
    else:
        raise ValueError("Invalid problem type.")


def cpp_to_problemtype(problem_type: _cpp.ProblemType) -> ProblemType:
    """c++ enum ProblemType to its python counterpart

    Args:
        problem_type: The ProblemType of the equilibration as c++ enum

    Returns:
        The ProblemType as python enum
    """

    if problem_type == _cpp.ProblemType.flux:
        return ProblemType.flux
    elif problem_type == _cpp.ProblemType.stress:
        return ProblemType.stress
    elif problem_type == _cpp.ProblemType.stress_and_flux:
        return ProblemType.stress_and_flux
    else:
        raise ValueError("Invalid problem type.")


# --- The equilibration strategy ---
class EqlbStrategy(Enum):
    """The equilibration strategy"""

    semi_explicit = 0
    constrained_minimisation = 1


def strategy_to_cpp(strategy: EqlbStrategy) -> _cpp.EqlbStrategy:
    """Python enum Strategy to its c++ counterpart

    Args:
        strategy: The Strategy of the equilibration as python enum

    Returns:
        The EqlbStrategy as c++ enum
    """

    if strategy == EqlbStrategy.semi_explicit:
        return _cpp.EqlbStrategy.semi_explicit
    elif strategy == EqlbStrategy.constrained_minimisation:
        return _cpp.EqlbStrategy.constrained_minimisation
    else:
        raise ValueError("Invalid equilibration strategy.")


def cpp_to_strategy(strategy: _cpp.EqlbStrategy) -> EqlbStrategy:
    """c++ enum Strategy to its python counterpart

    Args:
        strategy: The Strategy of the equilibration as c++ enum

    Returns:
        The Strategy as python enum
    """

    if strategy == _cpp.EqlbStrategy.semi_explicit:
        return EqlbStrategy.semi_explicit
    elif strategy == _cpp.EqlbStrategy.constrained_minimisation:
        return EqlbStrategy.constrained_minimisation
    else:
        raise ValueError("Invalid equilibration strategy.")


# --- Meta class for mesh-independent definition of an equilibration problem ---
class EquilibratorMetaClass:
    """The basic, mesh-independent definition of an equilibration problem"""

    def __init__(
        self,
        msh: ufl.Mesh,
        problem_type: ProblemType,
        strategy: EqlbStrategy,
        degree: int,
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

        # The approximation of the geometry
        self._element_geom = msh.ufl_coordinate_element()

        # The hat function
        self._element_hat = basix.ufl.element("P", cell, 1, dtype=np.float64)

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
        self._cpp_object = None

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
