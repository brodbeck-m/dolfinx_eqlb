# --- Imports ---
import numpy as np
import typing

import dolfinx.fem as dfem
import dolfinx.mesh as dmesh

from dolfinx_eqlb.cpp import FluxBC, BoundaryData


# --- Equilibration of fluxes ---


class FluxEquilibrator:
    def __init__(self, degree_flux: int, n_eqlbs: int, equilibrate_stress: bool):
        # Order of reconstructed flux
        self.degree_flux = degree_flux

        # Number of reconstructed fluxes
        self.n_fluxes = n_eqlbs

        # Equilibrate stresses
        self.equilibrate_stresses = equilibrate_stress

        # --- The flux ---
        # Function space
        self.V_flux = None

        # Function
        self.list_flux = []
        self.list_flux_cpp = []

        # --- The BoundaryData ---
        # List of functions with projected boundary data
        self.list_bfunctions = []

        # BoundaryData
        self.boundary_data = None

    def set_mesh_connectivities(self, msh: dmesh.Mesh):
        msh.topology.create_connectivity(0, 1)
        msh.topology.create_connectivity(0, 2)
        msh.topology.create_connectivity(1, 0)
        msh.topology.create_connectivity(1, 2)
        msh.topology.create_connectivity(2, 0)
        msh.topology.create_connectivity(2, 1)

    def setup_patch_problem(
        self,
        msh: dmesh.Mesh,
        list_rhs: typing.List[typing.Any],
        list_proj_flux: typing.List[dfem.function.Function],
    ):
        raise NotImplementedError

    def update_patch_problem(
        self,
        msh: dmesh.Mesh,
        list_rhs: typing.List[typing.Any],
        list_proj_flux: typing.List[dfem.function.Function],
    ):
        raise NotImplementedError

    def set_boundary_conditions(
        self,
        list_bfct_prime: typing.List[np.ndarray],
        list_bcs_flux: typing.List[typing.List[FluxBC]],
        quadrature_degree: typing.Optional[int] = None,
    ):
        raise NotImplementedError

    def equilibrate_fluxes(self):
        raise NotImplementedError

    def get_recontructed_fluxe(self, subproblem: int):
        raise NotImplementedError
