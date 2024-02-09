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
// ------------------------------------------------------------------------------

/* Definition of spans */

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

/* Extract coefficients of a form */

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

/* Extract interpolation data of RT elements */

/// Extract interpolation data of an RT-space on facets
/// @param[in] basix_element   The Basix element (has to be RT!)
/// @param[in] flux_is_custom  Flag, if custom flux space is used
/// @param[in] gdim            The geometric dimension of the problem
/// @param[in] nfcts_per_cell  The number of facets per cell
/// @param[in,out] ipoints_fct Storage for interpolation points
/// @param[in,out] data_M_fct  Storage for interpolation matrix
/// @return The Shape M for creation of an mdspan
std::array<std::size_t, 4> interpolation_data_facet_rt(
    const basix::FiniteElement& basix_element, const bool flux_is_custom,
    const std::size_t gdim, const std::size_t nfcts_per_cell,
    std::vector<double>& ipoints_fct, std::vector<double>& data_M_fct);

/// Get shape of facet interpolation data
/// @param[in] shape Shape, used for creation of the mdspan
/// @return std::tuple(nipoints_per_fct, nipoints_all_fcts)
static inline std::pair<std::size_t, std::size_t>
size_interpolation_data_facet_rt(std::array<std::size_t, 4> shape)
{
  return {shape[3], shape[0] * shape[3]};
}
// ------------------------------------------------------------------------------

// ------------------------------------------------------------------------------

/* Copy data into flatted storage */

/// Copy cell data from global storage into flattened array (per cell)
/// @param data_global The global data storage
/// @param dofs_cell   The DOFs on current cell
/// @param data_cell   The flattened storage of current cell
/// @param bs_data     Block size of data
template <typename T, int _bs_data = 1>
void copy_cell_data(std::span<const T> data_global,
                    std::span<const std::int32_t> dofs_cell,
                    std::span<T> data_cell, const int bs_data)
{
  for (std::size_t j = 0; j < dofs_cell.size(); ++j)
  {
    if constexpr (_bs_data == 1)
    {
      std::copy_n(std::next(data_global.begin(), dofs_cell[j]), 1,
                  std::next(data_cell.begin(), j));
    }
    else if constexpr (_bs_data == 2)
    {
      std::copy_n(std::next(data_global.begin(), 2 * dofs_cell[j]), 2,
                  std::next(data_cell.begin(), 2 * j));
    }
    else if constexpr (_bs_data == 3)
    {
      std::copy_n(std::next(data_global.begin(), 3 * dofs_cell[j]), 3,
                  std::next(data_cell.begin(), 3 * j));
    }
    else
    {
      std::copy_n(std::next(data_global.begin(), bs_data * dofs_cell[j]),
                  bs_data, std::next(data_cell.begin(), bs_data * j));
    }
  }
}

/// Copy cell data from global storage into flattened array (per patch)
/// @param cells        List of cells on patch
/// @param dofmap_data  DOFmap of data
/// @param data_global  The global data storage
/// @param data_cell    The flattened storage of current patch
/// @param cstride_data Number of data-points per cell
/// @param bs_data      Block size of data
template <typename T, int _bs_data = 4>
void copy_cell_data(std::span<const std::int32_t> cells,
                    const graph::AdjacencyList<std::int32_t>& dofmap_data,
                    std::span<const T> data_global, std::vector<T>& data_cell,
                    const int cstride_data, const int bs_data)
{
  for (std::size_t index = 0; index < cells.size(); ++index)
  {
    // Extract cell
    std::int32_t c = cells[index];

    // DOFs on current cell
    std::span<const std::int32_t> data_dofs = dofmap_data.links(c);

    // Copy DOFs into flattend storage
    std::span<T> data_dofs_e(data_cell.data() + index * cstride_data,
                             cstride_data);
    if constexpr (_bs_data == 1)
    {
      copy_cell_data<T, 1>(data_global, data_dofs, data_dofs_e, bs_data);
    }
    else if constexpr (_bs_data == 2)
    {
      copy_cell_data<T, 2>(data_global, data_dofs, data_dofs_e, bs_data);
    }
    else if constexpr (_bs_data == 3)
    {
      copy_cell_data<T, 3>(data_global, data_dofs, data_dofs_e, bs_data);
    }
    else
    {
      copy_cell_data<T>(data_global, data_dofs, data_dofs_e, bs_data);
    }
  }
}

// ------------------------------------------------------------------------------

} // namespace dolfinx_eqlb