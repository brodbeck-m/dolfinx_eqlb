// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "BoundaryData.hpp"
#include "KernelDataBC.hpp"
#include "equilibration.hpp"

#include <basix/cell.h>
#include <basix/finite-element.h>

#include <memory>
#include <span>
#include <vector>

using namespace dolfinx;

namespace dolfinx_eqlb::base
{

template <dolfinx::scalar T, std::floating_point U>
class Equilibrator
{
public:
  Equilibrator(const ProblemType problem_type, const EqStrategy strategy,
               const basix::FiniteElement<U>& element_geom,
               const basix::FiniteElement<U>& element_hat,
               const basix::FiniteElement<U>& element_flux,
               const int quadrature_degree_bcs)
      : _problem_type(problem_type), _strategy(strategy),
        _gdim(basix::cell::topological_dimension(element_geom.cell_type())),
        _element_geom(std::make_shared<basix::FiniteElement<U>>(element_geom)),
        _element_hat(std::make_unique<basix::FiniteElement<U>>(element_hat)),
        _element_flux(std::make_unique<basix::FiniteElement<U>>(element_flux)),
        _kernel_data_bcs(KernelDataBC<T, U>(
            element_geom, std::make_tuple(quadrature_degree_bcs, _gdim - 1),
            element_hat, element_flux, strategy))
  {
  }

  /* Getter methods */
  /// Get the problem type
  /// @return The problem type
  ProblemType problem_type() const { return _problem_type; }

  /// Get the equilibration strategy
  /// @return The equilibration strategy
  EqStrategy strategy() const { return _strategy; }

  /// Return the Baisx element of the hat-function
  /// @return The Basix element
  const basix::FiniteElement<U>& basix_element_hat() const
  {
    return *_element_hat;
  }

  /// Return the Baisx element of the equilibrated flux
  /// @return The Basix element
  const basix::FiniteElement<U>& basix_element_flux() const
  {
    return *_element_flux;
  }

  /// Return KernelData for the evaluation of BCs
  /// @return The KernelDataBC
  const KernelDataBC<T, U>& kernel_data_bcs() const { return _kernel_data_bcs; }

protected:
  /* Variable definitions */

  const ProblemType _problem_type;
  const EqStrategy _strategy;

  // The spatial dimension
  const int _gdim;

  // The geometry element
  std::shared_ptr<basix::FiniteElement<U>> _element_geom;

  // The hat-function element
  std::unique_ptr<basix::FiniteElement<U>> _element_hat;

  // The flux element
  std::unique_ptr<basix::FiniteElement<U>> _element_flux;

  // KernelData for the evaluation of BCs
  KernelDataBC<T, U> _kernel_data_bcs;
};

} // namespace dolfinx_eqlb::base