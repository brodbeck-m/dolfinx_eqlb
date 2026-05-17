// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Equilibrator.hpp"

using namespace dolfinx;
using namespace dolfinx_eqlb::base;

template <dolfinx::scalar T, std::floating_point U>
void Equilibrator<T, U>::print_info() const
{
  // Implementation of the print_info function
  std::cout << "Equilibrator Information:" << std::endl;
  std::cout << "Problem Type: " << static_cast<int>(_problem_type) << std::endl;
  std::cout << "Equilibration Strategy: " << static_cast<int>(_strategy)
            << std::endl;
  std::cout << "Geometric Dimension: " << _gdim << std::endl;
}

// ------------------------------------------------------------------------------
template class Equilibrator<double, double>;
// ------------------------------------------------------------------------------
