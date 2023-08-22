# --- Imports ---
import typing

import dolfinx.fem as dfem
import dolfinx.mesh as dmesh
import ufl

from dolfinx_eqlb.cpp import reconstruct_fluxes_minimisation
from .FluxEquilibrator import FluxEquilibrator


class FluxEqlbEV(FluxEquilibrator):
    def __init__(
        self,
        degree_flux: int,
        msh: dmesh.Mesh,
        list_rhs: typing.List[typing.Any],
        list_proj_flux: typing.List[dfem.function.Function],
    ):
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
        if len(list_proj_flux) != self.n_fluxes:
            raise RuntimeError("Missmatching inputs!")

        # Call setup routine
        self.setup_patch_problem(msh, list_rhs, list_proj_flux)

    def setup_patch_problem(
        self,
        msh: dmesh.Mesh,
        list_rhs: typing.List[typing.Any],
        list_proj_flux: typing.List[dfem.function.Function],
    ):
        # Initialize connectivities
        super().set_mesh_connectivities(msh)

        # --- Create function-spaces
        # Definition of finite elements
        P_hat = ufl.FiniteElement("CG", msh.ufl_cell(), 1)
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

    def equilibrate_fluxes(self):
        reconstruct_fluxes_minimisation(
            self.form_a,
            self.form_lpen,
            self.list_form_l,
            self.list_bfct_prime,
            self.list_bfct_flux,
            self.list_bcs_flux,
            self.list_flux_cpp,
        )

    def get_recontructed_fluxe(self, subproblem: int):
        return self.list_flux[subproblem]
