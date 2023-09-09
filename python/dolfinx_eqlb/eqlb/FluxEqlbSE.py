# --- Imports ---
import numpy as np
import typing

import basix
import dolfinx.fem as dfem
import dolfinx.mesh as dmesh

from dolfinx_eqlb.cpp import FluxBC, BoundaryData, reconstruct_fluxes_semiexplt
from dolfinx_eqlb.elmtlib import create_hierarchic_rt
from .FluxEquilibrator import FluxEquilibrator


class FluxEqlbSE(FluxEquilibrator):
    def __init__(
        self,
        degree_flux: int,
        msh: dmesh.Mesh,
        list_rhs: typing.List[typing.Any],
        list_proj_flux: typing.List[dfem.function.Function],
    ):
        # Constructor of base class
        super().__init__(degree_flux, len(list_rhs))

        # Store list of projected fluxes and RHS
        self.list_proj_flux = list_proj_flux
        self.list_rhs = list_rhs

        self.list_proj_flux_cpp = []
        self.list_rhs_cpp = []

        # Problem setup
        if len(list_proj_flux) != self.n_fluxes:
            raise RuntimeError("Missmatching inputs!")

        self.setup_patch_problem(msh, list_rhs, list_proj_flux)

    def setup_patch_problem(
        self,
        msh: dmesh.Mesh,
        list_rhs: typing.List[typing.Any],
        list_proj_flux: typing.List[dfem.function.Function],
    ):
        # Initialize connectivities
        super().set_mesh_connectivities(msh)

        # --- Create solution function
        # Definition of finite elements
        P_flux = create_hierarchic_rt(basix.CellType.triangle, self.degree_flux, True)

        # Function spaces
        self.V_flux = dfem.FunctionSpace(msh, basix.ufl_wrapper.BasixElement(P_flux))

        # Create variational problems and H(div) conforming fluxes
        for ii in range(0, self.n_fluxes):
            # Create flux-functions
            flux = dfem.Function(self.V_flux)
            self.list_flux.append(flux)

            # Add cpp objects to separated list
            self.list_flux_cpp.append(flux._cpp_object)
            self.list_proj_flux_cpp.append(list_proj_flux[ii]._cpp_object)
            self.list_rhs_cpp.append(list_rhs[ii]._cpp_object)

    def set_boundary_conditions(
        self,
        list_bfct_prime: typing.List[np.ndarray],
        list_bcs_flux: typing.List[typing.List[FluxBC]],
    ):
        # Check input data
        if self.n_fluxes != len(list_bfct_prime) | self.n_fluxes != len(list_bcs_flux):
            raise RuntimeError("Mismatching inputs!")

        # Initialise boundary data
        list_bfunctions_cpp = []

        for i in range(0, self.n_fluxes):
            self.list_bfunctions.append(dfem.Function(self.V_flux))
            list_bfunctions_cpp.append(self.list_bfunctions[i]._cpp_object)

        self.boundary_data = BoundaryData(
            list_bcs_flux,
            list_bfunctions_cpp,
            self.V_flux._cpp_object,
            True,
            list_bfct_prime,
        )

        for i in range(0, self.n_fluxes):
            self.list_bfunctions[i].x.scatter_forward()

    def equilibrate_fluxes(self):
        reconstruct_fluxes_semiexplt(
            self.list_flux_cpp,
            self.list_proj_flux_cpp,
            self.list_rhs_cpp,
            self.boundary_data,
        )

    def get_recontructed_fluxe(self, subproblem: int):
        return self.list_flux[subproblem] + self.list_proj_flux[subproblem]
