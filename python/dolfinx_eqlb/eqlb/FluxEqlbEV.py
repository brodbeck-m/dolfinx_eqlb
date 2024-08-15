# Copyright (C) 2024 Maximilian Brodbeck
#
# This file is part of dolfinx_eqlb
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

# --- Imports ---
import numpy as np
import typing

import dolfinx.fem as dfem
import dolfinx.mesh as dmesh
import ufl

from dolfinx_eqlb.cpp import FluxBC, reconstruct_fluxes_minimisation
from .FluxEquilibrator import FluxEquilibrator
from .bcs import boundarydata


class FluxEqlbEV(FluxEquilibrator):
    """Equilibrate fluxes by a series of constrained minimisation problems[1]

    [1] Ern, A. and Vohralík, M.: https://doi.org/10.1137/130950100, 2015
    """

    def __init__(
        self,
        degree_flux: int,
        msh: dmesh.Mesh,
        list_rhs: typing.List[dfem.Function],
        list_proj_flux: typing.List[dfem.Function],
    ):
        """Initialise constrained minimisation based flux equilibrator

        Args:
            degree_flux:            The degree of the H(div) conforming fluxes
            msh:                    The mesh
            list_rhs:               The projected right-hand sides
            list_proj_flux:         The projected fluxes
        """

        # Constructor of base class
        super().__init__(degree_flux, len(list_rhs), False)

        # --- Storage of variational problem (equilibration) ---
        # --- Function spaces
        # Mixed problem
        self.V = None

        # Hat functions
        self.V_hat = None

        # --- Functions
        # Hat function
        self.hat_function = None

        # --- Equation system
        # Bilinear form
        self.form_a = None

        # Penalisation
        self.form_lpen = None

        # Linear form
        self.list_form_l = []

        # --- Problem setup ---
        # Check input
        if len(list_proj_flux) != self.n_fluxes:
            raise RuntimeError("Missmatching inputs!")

        # Call setup routine
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

        # --- Create function-spaces
        # Definition of finite elements
        P_hat = ufl.FiniteElement("P", msh.ufl_cell(), 1)
        P_flux = ufl.FiniteElement("RT", msh.ufl_cell(), self.degree_flux)
        P_DG = ufl.FiniteElement("DG", msh.ufl_cell(), self.degree_flux - 1)

        # Function spaces
        self.V = dfem.FunctionSpace(msh, ufl.MixedElement(P_flux, P_DG))
        self.V_flux = dfem.FunctionSpace(msh, P_flux)
        self.V_hat = dfem.FunctionSpace(msh, P_hat)

        # --- Setup variational problems
        # Set hat-function
        self.hat_function = dfem.Function(self.V_hat)

        # Set identifire
        self.hat_function.name = "hat"

        # Create trial and test functions
        sig, r = ufl.TrialFunctions(self.V)
        v, q = ufl.TestFunctions(self.V)
        q_pen = ufl.TestFunction(dfem.FunctionSpace(msh, P_DG))

        # Set bilinear-form
        self.form_a = dfem.form(
            (ufl.inner(sig, v) - r * ufl.div(v) + ufl.div(sig) * q) * ufl.dx
        )
        self.form_lpen = dfem.form(q_pen * ufl.dx)

        # Create variational problems and H(div) conforming fluxes
        for ii in range(0, self.n_fluxes):
            # Create flux-functions
            flux = dfem.Function(self.V_flux)
            self.list_flux.append(flux)
            self.list_flux_cpp.append(flux._cpp_object)

            # Set linear-form
            l = (
                -(self.hat_function * ufl.inner(-list_proj_flux[ii], v))
                + self.hat_function * list_rhs[ii] * q
                - ufl.inner(ufl.grad(self.hat_function), -list_proj_flux[ii]) * q
            ) * ufl.dx
            self.list_form_l.append(dfem.form(l))

    def set_boundary_conditions(
        self,
        list_bfct_prime: typing.List[np.ndarray],
        list_bcs_flux: typing.List[typing.List[FluxBC]],
        quadrature_degree: typing.Optional[int] = None,
    ):
        """Set boundary conditions

        Args:
            list_bfct_prime:   The facets with essential BCs of the primal problem
            list_bcs_flux:     The list of boundary conditions
            quadrature_degree: The quadrature degree (for projecting the BCs)
        """

        # Check input data
        if self.n_fluxes != len(list_bfct_prime) | self.n_fluxes != len(list_bcs_flux):
            raise RuntimeError("Mismatching inputs!")

        # Initialise boundary data
        self.list_bfunctions = [dfem.Function(self.V) for i in range(self.n_fluxes)]

        self.boundary_data = boundarydata(
            list_bcs_flux,
            self.list_bfunctions,
            self.V.sub(0),
            False,
            list_bfct_prime,
            self.equilibrate_stresses,
            quadrature_degree,
        )

        for i in range(0, self.n_fluxes):
            self.list_bfunctions[i].x.scatter_forward()

    def equilibrate_fluxes(self):
        """Equilibrate the fluxes"""

        reconstruct_fluxes_minimisation(
            self.form_a,
            self.form_lpen,
            self.list_form_l,
            self.list_flux_cpp,
            self.boundary_data,
        )

    def get_recontructed_fluxe(self, subproblem: int):
        """Get the reconstructed fluxes

        Args:
            subproblem: Id of the flux

        Returns:
            The reconstructed flux (result of the const. minim. problems)
        """

        return self.list_flux[subproblem]
