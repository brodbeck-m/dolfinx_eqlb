#pragma once

#include <basix/mdspan.hpp>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/CoordinateElement.h>

#include <algorithm>
#include <memory>
#include <span>
#include <vector>

using namespace dolfinx;

namespace dolfinx_adaptivity
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

} // namespace dolfinx_adaptivity