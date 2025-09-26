// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <basix/e-lagrange.h>
#include <basix/finite-element.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx_eqlb/base/QuadratureRule.hpp>
#include <dolfinx_eqlb/base/equilibration.h>

#include "KernelData.hpp"

#include <memory>
// #include <span>
// #include <vector>

using namespace dolfinx;

namespace dolfinx_eqlb::se
{

template <dolfinx::scalar T, std::floating_point U>
class Equilibrator : public base::Equilibrator
{
  Equilibrator(const base::ProblemType problem_type,
               const base::EqStrategy strategy,
               const basix::FiniteElement<U>& element_flux,
               const basix::FiniteElement<U>& element_proj)
      : base::Equilibrator(problem_type, strategy),
        _element_flux(std::make_unique<basix::FiniteElement<U>>(element_flux)),
        _element_proj(std::make_unique<basix::FiniteElement<U>>(element_proj)),
        _element_proj_cnst(basix::element::create_lagrange<U>(
            _element_proj->cell_type(), _element_proj->degree(),
            _element_proj->lagrange_variant(), false)),
        _element_hat(basix::element::create_lagrange<U>(
            _element_proj->cell_type(), 1, _element_proj->lagrange_variant(),
            false)),
        _quadrature_rule(
            (_element_flux->degree() == 1)
                ? base::QuadratureRule(mesh->topology().cell_type(), 2, 2)
                : base::QuadratureRule(mesh->topology().cell_type(),
                                       2 * _element_flux->degree() + 1, 2)),
        _kernel_data(KernelData<T>(
            mesh, std::make_shared<base::QuadratureRule>(_quadrature_rule),
            _element_flux, _element_proj, _element_hat))
  {
  }

protected:
  /* Variable definitions */
  // The Basix element for the equilibrated fluxes
  std::unique_ptr<basix::FiniteElement<U>> _element_flux;

  // The Basix element for RHS and approximated flux
  basix::FiniteElement<U> _element_proj_cnst;
  std::unique_ptr<basix::FiniteElement<U>> _element_proj;

  // The Basix element for the hat-function
  basix::FiniteElement<U> _element_hat;

  // The KernelData
  base::QuadratureRule<U> _quadrature_rule;
  KernelData<T> _kernel_data;
};
} // namespace dolfinx_eqlb::se