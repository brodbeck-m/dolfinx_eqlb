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

/* Reverse patch-wise DOFmaps */
template <typename T>
void reverse_blocked_data(std::vector<T>& dataset_1, const int data_size,
                          const int block_size, const int block_index,
                          const int block_offset_pre,
                          const int block_offset_post)
{
  // Initialise temporary storage
  std::int32_t temp;

  // Copy data-blocks
  const int size = block_size + block_offset_pre + block_offset_post;

  const int offs_base_front = block_index * size + block_offset_pre;
  const int offs_base_back
      = data_size - (block_index + 1) * size + block_offset_pre;

  for (std::size_t i = 0; i < block_size; ++i)
  {
    // Calculate offsets
    int offs_front = offs_base_front + i;
    int offs_back = offs_base_back + i;

    // --- Handle data-set 1
    // Copy data to temporary storage
    temp = dataset_1[offs_front];

    // Exchange data of blocks
    dataset_1[offs_front] = dataset_1[offs_back];
    dataset_1[offs_back] = temp;
  }
}

template <typename T>
void reverse_blocked_data(std::vector<T>& dataset_1, std::vector<T>& dataset_2,
                          const int data_size, const int block_size,
                          const int block_index, const int block_offset_pre,
                          const int block_offset_post)
{
  // Initialise temporary storage
  std::int32_t temp;

  // Copy data-blocks
  const int size = block_size + block_offset_pre + block_offset_post;

  const int offs_base_front = block_index * size + block_offset_pre;
  const int offs_base_back
      = data_size - (block_index + 1) * size + block_offset_pre;

  for (std::size_t i = 0; i < block_size; ++i)
  {
    // Calculate offsets
    int offs_front = offs_base_front + i;
    int offs_back = offs_base_back + i;

    // --- Handle data-set 1
    // Copy data to temporary storage
    temp = dataset_1[offs_front];

    // Exchange data of blocks
    dataset_1[offs_front] = dataset_1[offs_back];
    dataset_1[offs_back] = temp;

    // --- Handle data-set 2
    // Copy data to temporary storage
    temp = dataset_2[offs_front];

    // Exchange data of blocks
    dataset_2[offs_front] = dataset_2[offs_back];
    dataset_2[offs_back] = temp;
  }
}
// ------------------------------------------------------------------------------

} // namespace dolfinx_eqlb