# Copyright (C) 2024 Maximilian Brodbeck
#
# This file is part of dolfinx_eqlb
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Flux equilibration based a semi-explicit strategy"""

import numpy as np
import typing

import basix
import dolfinx.fem as dfem
import dolfinx.mesh as dmesh

from dolfinx_eqlb.cpp import (
    FluxBC,
    reconstruct_fluxes_semiexplt,
    reconstruct_fluxes_semiexplt_with_kornconst,
)
from dolfinx_eqlb.elmtlib import create_hierarchic_rt
from .FluxEquilibrator import FluxEquilibrator
from .bcs import boundarydata


class FluxEqlbSE(FluxEquilibrator):
    """Equilibrate fluxes/stresses in a semi-explicit manner

    The algorithm follows a three-step procedure:
        1.) Find any function in H(div) that fulfills the divergence condition [1]
        2.) Minimise the flux on patch-wise H(div=0) spaces [1]
        3.) If stress: Incorporate weak symmetry condition by an additional
            constrained minimisation on patch-wise H(div=0) spaces [2]

    [1] Bertrand, F. et al.: https://doi.org/10.1007/s00211-023-01366-8, 2023
    [2] Bertrand, F. et al.: https://doi.org/10.1002/num.22741, 2021
    """

    def __init__(
        self,
        degree_flux: int,
        msh: dmesh.Mesh,
        list_rhs: typing.List[typing.Any],
        list_proj_flux: typing.List[dfem.function.Function],
        equilibrate_stress: typing.Optional[bool] = False,
        estimate_korn_constant: typing.Optional[bool] = False,
    ):
        """Initialise semi-explicit flux equilibrator

        Args:
            degree_flux:            The degree of the H(div) conforming fluxes
            msh:                    The mesh
            list_rhs:               The projected right-hand sides
            list_proj_flux:         The projected fluxes
            equilibrate_stress:     Identifier if the first gdim fluxes are treated as stresses
            estimate_korn_constant: Identifier if the cells Korn constants should be estimated
        """

        # Constructor of base class
        super().__init__(degree_flux, len(list_rhs), equilibrate_stress)

        # Store list of projected fluxes and RHS
        self.list_proj_flux = list_proj_flux
        self.list_rhs = list_rhs

        self.list_proj_flux_cpp = []
        self.list_rhs_cpp = []

        # Store cells Korn constants
        self.estimate_korn_constant = estimate_korn_constant
        self.korn_constants = None

        # Problem setup
        if len(list_proj_flux) != self.n_fluxes:
            raise RuntimeError("Mismatching inputs!")

        self.setup_patch_problem(msh, list_rhs, list_proj_flux)

    def setup_patch_problem(
        self,
        msh: dmesh.Mesh,
        list_rhs: typing.List[dfem.Function],
        list_proj_flux: typing.List[dfem.Function],
    ):
        """Setup the patch problems

        Args:
            msh:            The mesh
            list_rhs:       The projected right-hand sides
            list_proj_flux: The projected fluxes
        """

        # Initialize connectivities
        super().initialise_mesh_info(msh)

        # --- Create solution function
        # Definition of finite elements
        P_flux = create_hierarchic_rt(basix.CellType.triangle, self.degree_flux, True)

        # Function spaces
        self.V_flux = dfem.FunctionSpace(msh, basix.ufl_wrapper.BasixElement(P_flux))

        # Initialise list of fluxes and RHS
        for ii in range(0, self.n_fluxes):
            # Create flux-functions
            flux = dfem.Function(self.V_flux)
            self.list_flux.append(flux)

            # Add cpp objects to separated list
            self.list_flux_cpp.append(flux._cpp_object)
            self.list_proj_flux_cpp.append(list_proj_flux[ii]._cpp_object)
            self.list_rhs_cpp.append(list_rhs[ii]._cpp_object)

        # Initialise cells korn constants
        if self.estimate_korn_constant:
            self.korn_constants = dfem.Function(dfem.FunctionSpace(msh, ("DG", 0)))

    def set_boundary_conditions(
        self,
        list_bfct_prime: typing.List[np.ndarray],
        list_bcs_flux: typing.List[typing.List[FluxBC]],
    ):
        """Set boundary conditions

        Args:
            list_bfct_prime:   The facets with essential BCs of the primal problem
            list_bcs_flux:     The list of boundary conditions
        """

        # Check input data
        if self.n_fluxes != len(list_bfct_prime) | self.n_fluxes != len(list_bcs_flux):
            raise RuntimeError("Mismatching inputs!")

        # Initialise boundary data
        self.list_bfunctions = [
            dfem.Function(self.V_flux) for i in range(self.n_fluxes)
        ]

        self.boundary_data = boundarydata(
            list_bcs_flux,
            self.list_bfunctions,
            self.V_flux,
            True,
            list_bfct_prime,
            self.equilibrate_stresses,
        )

        for i in range(0, self.n_fluxes):
            self.list_bfunctions[i].x.scatter_forward()

    def equilibrate_fluxes(self):
        """Equilibrate the fluxes"""

        if self.estimate_korn_constant:
            # Equilibrate fluxes
            reconstruct_fluxes_semiexplt_with_kornconst(
                self.list_flux_cpp,
                self.list_proj_flux_cpp,
                self.list_rhs_cpp,
                self.boundary_data,
                self.equilibrate_stresses,
                self.korn_constants._cpp_object,
            )

            # Post-Process Korn constants
            # TODO - Add values in Ghost cells to parent cells!
            self.korn_constants.x.array[:] = np.sqrt(self.korn_constants.x.array[:])
            self.korn_constants.x.scatter_forward()
        else:
            reconstruct_fluxes_semiexplt(
                self.list_flux_cpp,
                self.list_proj_flux_cpp,
                self.list_rhs_cpp,
                self.boundary_data,
                self.equilibrate_stresses,
            )

    def get_recontructed_fluxes(self, subproblem: int) -> typing.Any:
        """Get the reconstructed fluxes

        Args:
            subproblem: Id of the flux

        Returns:
            The reconstructed flux (sum of projected flux and equilibrated corrector)
        """

        return self.list_flux[subproblem] + self.list_proj_flux[subproblem]

    def get_korn_constants(self) -> dfem.Function:
        """Get the Korn constants

        Returns:
            The Korn constants in a DG0 function
        """

        if self.estimate_korn_constant:
            return self.korn_constants
        else:
            raise RuntimeError("Korn constants are not estimated!")
