# --- Imports ---
import numpy as np
import typing

import basix
import dolfinx.fem as dfem
import dolfinx.mesh as dmesh
from dolfinx.fem.bcs import DirichletBCMetaClass
import ufl

from dolfinx_eqlb.cpp import reconstruct_fluxes_minimisation, reconstruct_fluxes_semiexplt

# --- Equilibration of fluxes ---

class FluxEquilibrator():
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

    def setup_patch_problem(self, msh: dmesh.Mesh, list_rhs: typing.List[typing.Any],
                            list_proj_flux: typing.List[dfem.function.Function]):
        raise NotImplementedError

    def update_patch_problem(self, msh: dmesh.Mesh, list_rhs: typing.List[typing.Any],
                             list_proj_flux: typing.List[dfem.function.Function]):
        raise NotImplementedError

    def set_boundary_conditions(self, list_bfct_prime: typing.List[np.ndarray] = [],
                                list_bfct_flux: typing.List[np.ndarray] = [],
                                list_bcs_flux: typing.List[typing.List[DirichletBCMetaClass]] = [[]]):
        # Check input data
        if (self.n_fluxes != len(list_bfct_prime) |
            self.n_fluxes != len(list_bfct_flux) |
                self.n_fluxes != len(list_bcs_flux)):
            raise RuntimeError('Missmatching inputs!')

        # Set inputs
        self.list_bfct_prime = list_bfct_prime
        self.list_bfct_flux = list_bfct_flux

        self.list_bcs_flux = list_bcs_flux

    def equilibrate_fluxes(self):
        raise NotImplementedError

    def get_recontructed_fluxe(self, subproblem: int):
        raise NotImplementedError

class EquilibratorEV(FluxEquilibrator):
    def __init__(self, degree_flux: int, msh: dmesh.Mesh,
                 list_rhs: typing.List[typing.Any],
                 list_proj_flux: typing.List[dfem.function.Function]):
        # Constructor of base class
        super().__init__(degree_flux, len(list_rhs))

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
        if (len(list_proj_flux) != self.n_fluxes):
            raise RuntimeError('Missmatching inputs!')

        # Call setup routine
        self.setup_patch_problem(msh, list_rhs, list_proj_flux)

    def setup_patch_problem(self, msh: dmesh.Mesh, list_rhs: typing.List[typing.Any],
                            list_proj_flux: typing.List[dfem.function.Function]):
        # Initialize connectivities
        super().set_mesh_connectivities(msh)

        # --- Create function-spaces
        # Definition of finite elements
        P_hat = ufl.FiniteElement('CG', msh.ufl_cell(), 1)
        P_flux = ufl.FiniteElement('RT', msh.ufl_cell(), self.degree_flux)
        P_DG = ufl.FiniteElement('DG', msh.ufl_cell(), self.degree_flux-1)

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
            (ufl.inner(sig, v) - r*ufl.div(v) + ufl.div(sig)*q)*ufl.dx)
        self.form_lpen = dfem.form(q_pen*ufl.dx)

        # Create variational problems and H(div) conforming fluxes
        for ii in range(0, self.n_fluxes):
            # Create flux-functions
            flux = dfem.Function(self.V_flux)
            self.list_flux.append(flux)
            self.list_flux_cpp.append(flux._cpp_object)

            # Set linear-form
            l = (-(self.hat_function * ufl.inner(-list_proj_flux[ii], v)) + self.hat_function * list_rhs[ii] * q -
                 ufl.inner(ufl.grad(self.hat_function), -list_proj_flux[ii]) * q) * ufl.dx
            self.list_form_l.append(dfem.form(l))

    def equilibrate_fluxes(self):
        reconstruct_fluxes_minimisation(self.form_a, self.form_lpen, self.list_form_l, self.list_bfct_prime,
                                        self.list_bfct_flux, self.list_bcs_flux, self.list_flux_cpp)

    def get_recontructed_fluxe(self, subproblem: int):
        return self.list_flux[subproblem]


class EquilibratorSemiExplt(FluxEquilibrator):
    def __init__(self, degree_flux: int, msh: dmesh.Mesh,
                 list_rhs: typing.List[typing.Any],
                 list_proj_flux: typing.List[dfem.function.Function]):
        # Constructor of base class
        super().__init__(degree_flux, len(list_rhs))

        # Store list of projected fluxes and RHS
        self.list_proj_flux = list_proj_flux
        self.list_rhs = list_rhs

        self.list_proj_flux_cpp = []
        self.list_rhs_cpp = []

        # Problem setup
        if (len(list_proj_flux) != self.n_fluxes):
            raise RuntimeError('Missmatching inputs!')

        self.setup_patch_problem(msh, list_rhs, list_proj_flux)

    def setup_patch_problem(self, msh: dmesh.Mesh, list_rhs: typing.List[typing.Any],
                            list_proj_flux: typing.List[dfem.function.Function]):
        # Initialize connectivities
        super().set_mesh_connectivities(msh)

        # --- Create solution function
        # Definition of finite elements
        P_flux = basix.create_element(basix.ElementFamily.RT, basix.CellType.triangle, 
                                     self.degree_flux, basix.LagrangeVariant.equispaced, True)

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

    def equilibrate_fluxes(self):
        reconstruct_fluxes_semiexplt(self.list_flux_cpp, self.list_proj_flux_cpp, 
                                     self.list_rhs_cpp, self.list_bfct_prime,
                                     self.list_bfct_flux, self.list_bcs_flux)

    def get_recontructed_fluxe(self, subproblem: int):
        return self.list_flux[subproblem] + self.list_proj_flux[subproblem]


