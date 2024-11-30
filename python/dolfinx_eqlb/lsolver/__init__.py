# Copyright (C) 2024 Maximilian Brodbeck
#
# This file is part of dolfinx_eqlb
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from .lsolver import local_solver
from .projection import local_projection

__all__ = ["local_solver", "local_projection"]
