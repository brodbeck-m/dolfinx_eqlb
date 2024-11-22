// Copyright (C) 2024 Maximilian Brodbeck
//
// This file is part of dolfinx_eqlb
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.hpp"

using namespace dolfinx;

namespace dolfinx_eqlb
{
std::array<std::size_t, 4> interpolation_data_facet_rt(
    const basix::FiniteElement& basix_element, const bool flux_is_custom,
    const std::size_t gdim, const std::size_t nfcts_per_cell,
    std::vector<double>& ipoints_fct, std::vector<double>& data_M_fct)
{
  // Extract interpolation points
  auto [X, Xshape] = basix_element.points();
  const auto [Mdata, Mshape] = basix_element.interpolation_matrix();
  mdspan_t<const double, 2> M(Mdata.data(), Mshape);

  // Determine number of pointe per facet
  std::size_t nipoints_per_fct = 0;

  double x_fctpoint = X[0];

  while (x_fctpoint > 0.0)
  {
    // Increment number of points per facet
    nipoints_per_fct++;

    // Get next x-coordinate
    x_fctpoint = X[nipoints_per_fct * gdim];
  }

  std::size_t nipoints_fct = nipoints_per_fct * nfcts_per_cell;

  // Resize storage of interpolation data
  const int degree = basix_element.degree();
  std::size_t ndofs_fct
      = (gdim == 2) ? degree : (degree + 1) * (degree + 2) / 2;

  ipoints_fct.resize(nipoints_fct * gdim, 0);
  data_M_fct.resize(ndofs_fct * nipoints_fct * gdim, 0);

  // Cast Interpolation matrix into mdspan
  std::array<std::size_t, 4> M_fct_shape
      = {nfcts_per_cell, ndofs_fct, gdim, nipoints_per_fct};
  mdspan_t<double, 4> M_fct(data_M_fct.data(), M_fct_shape);

  // Copy interpolation points (on facets)
  std::copy_n(X.begin(), nipoints_fct * gdim, ipoints_fct.begin());

  // Copy interpolation matrix (on facets)
  int offs_drvt;

  if (flux_is_custom)
  {
    offs_drvt = (degree > 1) ? gdim + 1 : 1;
  }
  else
  {
    offs_drvt = 1;
  }

  int id_dof = 0;
  int offs_pnt = 0;

  for (std::size_t i = 0; i < gdim; ++i)
  {
    for (std::size_t j = 0; j < nfcts_per_cell; ++j)
    {
      for (std::size_t k = 0; k < ndofs_fct; ++k)
      {
        // Determine cell-local DOF id
        id_dof = j * ndofs_fct + k;
        offs_pnt = (i * Xshape[0] + j * nipoints_per_fct) * offs_drvt;

        // Copy interpolation coefficients
        for (std::size_t l = 0; l < nipoints_per_fct; ++l)
        {
          M_fct(j, k, i, l) = M(id_dof, offs_pnt + offs_drvt * l);
        }
      }
    }
  }

  return std::move(M_fct_shape);
}
} // namespace dolfinx_eqlb