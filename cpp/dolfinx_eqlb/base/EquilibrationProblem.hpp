// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "BoundaryData.hpp"

#include <dolfinx/fem/Function.h>

#include <memory>
#include <span>
#include <vector>

using namespace dolfinx;

namespace dolfinx_eqlb::base
{

template <dolfinx::scalar T, std::floating_point U>
class EquilibrationProblem
{
  EquilibrationProblem(
      std::shared_ptr<BoundaryData<T, U>> bcs,
      std::vector<std::shared_ptr<fem::Function<T, U>>> fluxes,
      const std::vector<bool> discontinous_diffusion_coefficient)
      : _bcs(bcs), _fluxes(_fluxes), _nfluxes(_fluxes.size()),
        _gdim(_fluxes[0]->function_space()->mesh()->geometry().dim()),
        _reassemble_patch_problem(discontinous_diffusion_coefficient)
  {
  }

  EquilibrationProblem(std::shared_ptr<BoundaryData<T, U>> bcs,
                       std::vector<std::shared_ptr<fem::Function<T, U>>> stress,
                       std::shared_ptr<fem::Function<T, U>> korns_constants
                       = nullptr)
      : _bcs(bcs), _fluxes(stress), _nfluxes(_fluxes.size()),
        _gdim(_fluxes[0]->function_space()->mesh()->geometry().dim()),
        _korncnsts(korns_constants),
        _wsym_stress((korns_constants == nullptr) ? false : true),
        _eval_korncnsts((_wsym_stress) ? true : false)
  {
    // The spatial dimension
    const int gdim = stress[0]->function_space()->mesh()->geometry().dim();

    if (stress.size() != gdim)
    {
      throw std::runtime_error("Stress tensor with wrong number of rows!");
    }

    // Prevent re-evaluation of equation systems on patch
    for (int i = 0; i < gdim; ++i)
    {
      _reassemble_patch_problem.push_back(false);
    }
  }

  EquilibrationProblem(
      std::shared_ptr<BoundaryData<T, U>> bcs,
      std::vector<std::shared_ptr<fem::Function<T, U>>> stress,
      std::vector<std::shared_ptr<fem::Function<T, U>>> fluxes,
      const std::vector<bool> discontinous_diffusion_coefficient,
      std::shared_ptr<fem::Function<T, U>> korns_constants)
      : _bcs(bcs), _nfluxes(stress.size() + fluxes.size()),
        _gdim(stress[0]->function_space()->mesh()->geometry().dim()),
        _korncnsts(korns_constants),
        _wsym_stress((korns_constants == nullptr) ? false : true),
        _eval_korncnsts((_wsym_stress) ? true : false)
  {
    // The spatial dimension
    const int gdim = stress[0]->function_space()->mesh()->geometry().dim();

    if (stress.size() != gdim)
    {
      throw std::runtime_error("Stress tensor with wrong number of rows!");
    }

    if (fluxes.size() != discontinous_diffusion_coefficient.size())
    {
      throw std::runtime_error("Inconsistent input of fluxes!");
    }

    // Prepare list of equilibrated fluxes (row of stress is one flux)
    for (auto row_i : stress)
    {
      _fluxes.push_back(row_i);
      _reassemble_patch_problem.push_back(false);
    }

    for (int i = 0; i < fluxes.size(); ++i)
    {
      _fluxes.push_back(fluxes[i]);
      _reassemble_patch_problem.push_back(
          discontinous_diffusion_coefficient[i]);
    }
  }

  /* Setter methods */

  /* Getter methods */
  bool evaluate_korn_constants()
  {
    if (_eval_korncnsts)
    {
      // Unset marker (as re-evaluation on same mesh is not required)
      _eval_korncnsts = false;

      return true;
    }
    else
    {
      return false;
    }
  }

  std::shared_ptr<BoundaryData<T, U>> bcs() { return _bcs; }

  std::shared_ptr<fem::Function<T, U>> korn_cornstants() { return _korncnsts; }

  std::span<std::shared_ptr<fem::Function<T, U>>> stress()
  {
    return std::span<std::shared_ptr<fem::Function<T, U>>>(_fluxes.data(),
                                                           _gdim);
  }

  std::span<std::shared_ptr<fem::Function<T, U>>> fluxes()
  {
    return std::span<std::shared_ptr<fem::Function<T, U>>>(_fluxes.data(),
                                                           _nfluxes);
  }

protected:
  /* Variable definitions */
  // The spatial dimension
  const int _gdim;

  // Markers for the equilibration process
  bool _wsym_stress, _eval_korncnsts;
  std::vector<bool> _reassemble_patch_problem;

  // The boundary conditions
  std::shared_ptr<BoundaryData<T, U>> _bcs;

  // The equilibrated fluxes
  const int _nfluxes;
  std::vector<std::shared_ptr<fem::Function<T, U>>> _fluxes;

  // The problems Korn constants
  std::shared_ptr<fem::Function<T, U>> _korncnsts;
};

} // namespace dolfinx_eqlb::base