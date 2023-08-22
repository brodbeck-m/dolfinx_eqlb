# --- Imports ---
import numpy as np
import typing

import dolfinx.fem as dfem
import dolfinx.mesh as dmesh
from dolfinx.fem.bcs import DirichletBCMetaClass

# --- Equilibration of fluxes ---


class FluxEquilibrator:
    def __init__(self, degree_flux: int, n_eqlbs: int):
        # Order of reconstructed flux
        self.degree_flux = degree_flux

        # Number of reconstructed fluxes
        self.n_fluxes = n_eqlbs

        # Identification boundary-facets
        self.list_bfct_prime = None
        self.list_bfct_flux = None

        # --- Storage of reconstructed fluxes ---
        # Function space
        self.V_flux = None

        # Function
        self.list_flux = []
        self.list_flux_cpp = []

        # Boundary conditions
        self.list_bcs_flux = None

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
        list_bfct_prime: typing.List[np.ndarray] = [],
        list_bfct_flux: typing.List[np.ndarray] = [],
        list_bcs_flux: typing.List[typing.List[DirichletBCMetaClass]] = [[]],
    ):
        # Check input data
        if (
            self.n_fluxes
            != len(list_bfct_prime) | self.n_fluxes
            != len(list_bfct_flux) | self.n_fluxes
            != len(list_bcs_flux)
        ):
            raise RuntimeError("Missmatching inputs!")

        # Set inputs
        self.list_bfct_prime = list_bfct_prime
        self.list_bfct_flux = list_bfct_flux

        self.list_bcs_flux = list_bcs_flux

    def equilibrate_fluxes(self):
        raise NotImplementedError

    def get_recontructed_fluxe(self, subproblem: int):
        raise NotImplementedError
