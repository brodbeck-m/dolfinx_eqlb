# Copyright (C) 2024 Maximilian Brodbeck
#
# This file is part of dolfinx_eqlb
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

# from .FluxEqlbEV import FluxEqlbEV
# from .FluxEqlbSE import FluxEqlbSE

from .equilibration import ProblemType, EqlbStrategy
from .Equilibrator import Equilibrator
from .bcs import homogenous_fluxbc, fluxbc, boundarydata

__all__ = [
    "ProblemType",
    "Strategy",
    "Equilibrator",
    # "FluxEqlbEV",
    # "FluxEqlbSE",
    "homogenous_fluxbc",
    "fluxbc",
    "boundarydata",
]
