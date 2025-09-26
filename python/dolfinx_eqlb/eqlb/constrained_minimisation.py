# Copyright (C) 2024 Maximilian Brodbeck
#
# This file is part of dolfinx_eqlb
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import basix.ufl
import numpy as np
from numpy.typing import NDArray
import typing

import basix
from dolfinx import default_real_type, fem, mesh
import ufl

import dolfinx_eqlb.cpp as _cpp

from .basics import (
    ProblemType,
    EqlbStrategy,
    problemtype_to_cpp,
    strategy_to_cpp,
    EquilibratorMetaClass,
)
from .bcs import boundarydata


# --- Equilibrator ---
class Equilibrator(EquilibratorMetaClass):
    """Mesh-independent definition of an equilibration problem

    The equilibration is based on the solution of constrained minimisation problems
    on patches based on [1].

    [1] Ern, A. and Vohralík, M.: https://doi.org/10.1137/130950100, 2015
    """

    def __init__(
        self,
        msh: ufl.Mesh,
        problem_type: ProblemType,
        degree: int,
        quadrature_degree: typing.Optional[int] = None,
    ):
        """The constructor

        Args:
            msh:               The generalised mesh
            problem_type:      The type of the equilibration problem
            degree:            The degree of the flux space
            quadrature_degree: The degree of the quadrature rule
        """

        # Constructor base class
        super().__init__(
            msh, problem_type, EqlbStrategy.constrained_minimisation, degree
        )

        # --- Required finite element spaces
        cell = msh._ufl_coordinate_element._cellname

        # The constrained space
        self._element_cnstr = basix.ufl.element(
            "DG", cell, degree - 1, dtype=default_real_type
        )

        # --- The cpp backend
        qdegree = 2 * degree - 2 if (quadrature_degree is None) else quadrature_degree

        self._cpp_object = _cpp.Equilibrator(
            problemtype_to_cpp(problem_type),
            strategy_to_cpp(self._strategy),
            self._element_geom.basix_element._e,
            self._element_hat.basix_element._e,
            self._element_flux.basix_element._e,
            qdegree,
        )

    # --- Getter methods ---
    def element_cnstr(self) -> basix.ufl._ElementBase:
        """The finite element for the constrained space"""
        return self._element_cnstr

    # --- Setter method ---
    def compile_forms(self):
        raise NotImplementedError("Mesh-Independent form compilation not implemented.")


# --- EquilibrationProblem ---
class EquilibrationProblem:
    """Mesh-dependent definition of an equilibration problem

    The equilibration follows [1] utilizing constrained minimisation
    problems on each patch.

    [1] Ern, A. and Vohralík, M.: https://doi.org/10.1137/130950100, 2015
    """

    def __init__(
        self,
        equilibrator: Equilibrator,
        V: fem.FunctionSpace,
        a: fem.Form,
        ls: typing.List[fem.Form],
        lp: fem.Form,
    ):
        """The constructor

        Args:
            equilibrator: The mesh-independent definition
            V:            The function space of the constrained minimisation problem
            a:            The bilinear form of the constrained minimisation problem
            ls:           The linear forms of the constrained minimisation problem
            lp:           The Lagrange multiplier
        """

        # The mesh-independent definition
        self._equilibrator = equilibrator

        # The constrained minimisation problem
        self._V_mixed = V

        # The equilibrated fluxes
        self._V, _ = V.sub(0).collapse()
        self._fluxes = [fem.Function(self._V) for _ in range(len(ls))]

        # The mesh-dependent forms
        self._as = a
        self._lp = lp
        self._ls = ls

        # The boundary data
        self._boundary_values = [fem.Function(self._V_mixed) for _ in range(len(ls))]
        self._boundary_data = None

        # Prepare mesh for equilibration
        msh, cdim, fdim, pdim = V.mesh, V.mesh.geometry.dim, V.mesh.geometry.dim - 1, 0

        msh.topology.create_entity_permutations()

        msh.topology.create_connectivity(pdim, fdim)
        msh.topology.create_connectivity(pdim, cdim)
        msh.topology.create_connectivity(fdim, pdim)
        msh.topology.create_connectivity(fdim, cdim)
        msh.topology.create_connectivity(cdim, pdim)
        msh.topology.create_connectivity(cdim, fdim)

    def create_boundary_data(
        self,
        bfct_prime: typing.List[NDArray],
        bcs: typing.List[typing.List[_cpp.FluxBC]],
    ):
        """Create boundary data

        Args:
            bfct_prime: Factes, where essential BCs are enforced on the primal
            bcs:        The essential BCs of the dual problem
        """

        self._boundary_data = boundarydata(
            bcs,
            self._boundary_values,
            self._V,
            bfct_prime,
            self._equilibrator.kernel_data_boundary_conditions(),
            self._equilibrator.problem_type(),
        )

    def solve(self):
        if self._boundary_data is None:
            raise ValueError("Boundary conditions for the dual problem required.")

        _cpp.reconstruct_fluxes_minimisation(
            self._as._cpp_object,
            self._lp._cpp_object,
            [l._cpp_object for l in self._ls],
            [f._cpp_object for f in self._fluxes],
            self._boundary_data,
        )


def equilibrationproblem(
    equilibrator: Equilibrator,
    msh: mesh.Mesh,
    proj_rhs: typing.List[fem.Function],
    proj_flux: typing.List[fem.Function],
) -> EquilibrationProblem:
    """Create an EquilibrationProblem

    Args:
        equilibrator: The mesh-independent problem definition
        msh:          The mesh
        proj_rhs:     The projected RHS
        proj_flux:    The projected fluxes

    Returns:
        An EquilibrationProblem
    """

    if len(proj_rhs) != len(proj_flux):
        raise ValueError("The number of projected RHS and fluxes must match.")

    # The function space of the mixed problem
    V = fem.functionspace(
        msh,
        basix.ufl.mixed_element(
            [equilibrator.element_flux(), equilibrator.element_cnstr()]
        ),
    )

    # The hat function
    hat = fem.Function(fem.functionspace(msh, equilibrator.element_hat()))
    hat.name = "hat"

    # Trial- and test functions
    sig, r = ufl.TrialFunctions(V)
    v, q = ufl.TestFunctions(V)
    q_pen = ufl.TestFunction(fem.functionspace(msh, equilibrator.element_cnstr()))

    # Bilinear form
    a = fem.form((ufl.inner(sig, v) - r * ufl.div(v) + ufl.div(sig) * q) * ufl.dx)

    # Lagrange multiplier
    lp = fem.form(q_pen * ufl.dx)

    # Linear forms
    ls = []

    for prhs, pflux in zip(proj_rhs, proj_flux):
        ls.append(
            fem.form(
                (
                    -(hat * ufl.inner(-pflux, v))
                    + hat * prhs * q
                    - ufl.inner(ufl.grad(hat), -pflux) * q
                )
                * ufl.dx
            )
        )

    return EquilibrationProblem(equilibrator, V, a, ls, lp)
