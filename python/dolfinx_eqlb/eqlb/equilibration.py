# Copyright (C) 2024 Maximilian Brodbeck
#
# This file is part of dolfinx_eqlb
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from enum import Enum

import dolfinx_eqlb.cpp as _cpp


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
