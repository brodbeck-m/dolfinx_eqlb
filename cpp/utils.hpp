#pragma once

#include <basix/mdspan.hpp>
#include <dolfinx/fem/CoordinateElement.h>

using namespace dolfinx;

namespace dolfinx_adaptivity
{
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

} // namespace dolfinx_adaptivity