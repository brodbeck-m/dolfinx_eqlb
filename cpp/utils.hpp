#pragma once

#include <basix/finite-element.h>
#include <basix/mdspan.hpp>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/CoordinateElement.h>

#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

using namespace dolfinx;

namespace dolfinx_eqlb
{
// Define mdspan types
namespace stdex = std::experimental;
using mdspan2_t = stdex::mdspan<double, stdex::dextents<std::size_t, 2>>;
using cmdspan2_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
using mdspan3_t = stdex::mdspan<double, stdex::dextents<std::size_t, 3>>;
using cmdspan3_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 3>>;
using mdspan4_t = stdex::mdspan<double, stdex::dextents<std::size_t, 4>>;
using cmdspan4_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 4>>;
using s_mdspan1_t = stdex::mdspan<double, stdex::dextents<std::size_t, 1>,
                                  stdex::layout_stride>;
using s_cmdspan1_t
    = stdex::mdspan<const double, stdex::dextents<std::size_t, 1>,
                    stdex::layout_stride>;
using s_mdspan2_t = stdex::mdspan<double, stdex::dextents<std::size_t, 2>,
                                  stdex::layout_stride>;
using s_cmdspan2_t
    = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>,
                    stdex::layout_stride>;
using s_cmdspan3_t
    = stdex::mdspan<const double, stdex::dextents<std::size_t, 3>,
                    stdex::layout_stride>;
using s_cmdspan4_t
    = stdex::mdspan<const double, stdex::dextents<std::size_t, 4>,
                    stdex::layout_stride>;

template <typename T, std::size_t d>
using mdspan_t = stdex::mdspan<T, stdex::dextents<std::size_t, d>>;

template <typename T, std::size_t d>
using smdspan_t
    = stdex::mdspan<T, stdex::dextents<std::size_t, d>, stdex::layout_stride>;

// ------------------------------------------------------------------------------
/// Compute size of coefficient data
/// @tparam T Scalar value type
/// @param constants Vector with fem::Constant objects
/// @return Size of flattened storage vector
template <typename T>
std::int32_t size_constants_data(
    const std::vector<std::shared_ptr<const fem::Constant<T>>>& constants)
{
  std::int32_t size = std::accumulate(constants.cbegin(), constants.cend(), 0,
                                      [](std::int32_t sum, auto& c)
                                      { return sum + c->value.size(); });
  return size;
}

/// Extract coefficients into flattened storage
/// @tparam T Scalar value type
/// @param constants      Vector with fem::Constant objects
/// @param data_constants Flattened storage for coefficients
template <typename T>
void extract_constants_data(
    const std::vector<std::shared_ptr<const fem::Constant<T>>>& constants,
    std::span<T> data_constants)
{
  std::int32_t offset = 0;

  for (auto& cnst : constants)
  {
    const std::vector<T>& value = cnst->value;
    std::copy(value.begin(), value.end(),
              std::next(data_constants.begin(), offset));
    offset += value.size();
  }
}
// ------------------------------------------------------------------------------

// ------------------------------------------------------------------------------
/// Extract interpolation data of an RT-space on facets
/// @param[in] basix_element   The basix element (has to be RT!)
/// @param[in] gdim            The geometric dimension of the problem
/// @param[in] nfcts_per_cell  The number of facets per cell
/// @param[in,out] ipoints_fct Storage for interpolation points
/// @param[in,out] data_M_fct  Storage for interpolation matrix
/// @param[out] shape          The Shape M for creation of an mdspan
std::array<std::size_t, 4>
interpolation_data_facet_rt(const basix::FiniteElement& basix_element,
                            std::size_t gdim, std::size_t nfcts_per_cell,
                            std::vector<double>& ipoints_fct,
                            std::vector<double>& data_M_fct)
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
  const int offs_drvt = (degree > 1) ? gdim + 1 : 1;

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

/// Get shape of facet interpolation data
/// @param[in] shape Shape, used for creation of the mdspan
/// @param[out] std::tuple(nipoints_per_fct, nipoints_all_fcts)
std::pair<std::size_t, std::size_t>
size_interpolation_data_facet_rt(std::array<std::size_t, 4> shape)
{
  return {shape[3], shape[0] * shape[3]};
}
// ------------------------------------------------------------------------------

} // namespace dolfinx_eqlb