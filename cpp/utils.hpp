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
using s_cmdspan2_t
    = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>,
                    stdex::layout_stride>;
using s_cmdspan3_t
    = stdex::mdspan<const double, stdex::dextents<std::size_t, 3>,
                    stdex::layout_stride>;
using s_cmdspan4_t
    = stdex::mdspan<const double, stdex::dextents<std::size_t, 4>,
                    stdex::layout_stride>;

double compute_jacobian(mdspan2_t J, mdspan2_t K,
                        std::span<double> detJ_scratch, s_cmdspan2_t dphi,
                        cmdspan2_t coords)
{
  std::size_t gdim = J.extent(0);
  auto coordinate_dofs
      = stdex::submdspan(coords, stdex::full_extent, std::pair{0, gdim});
  for (std::size_t i = 0; i < J.extent(0); ++i)
  {
    for (std::size_t j = 0; j < J.extent(1); ++j)
    {
      J(i, j) = 0;
      K(i, j) = 0;
    }
  }

  fem::CoordinateElement::compute_jacobian(dphi, coordinate_dofs, J);
  fem::CoordinateElement::compute_jacobian_inverse(J, K);

  return std::fabs(
      fem::CoordinateElement::compute_jacobian_determinant(J, detJ_scratch));
}

double compute_jacobian(mdspan2_t J, mdspan2_t K,
                        std::span<double> detJ_scratch,
                        std::array<std::size_t, 4> dphi_shape,
                        std::vector<double> dphi_values, cmdspan2_t coords)
{
  // Spacial dimension
  int dim = J.extent(0);

  // Reshape shape functions
  cmdspan4_t full_basis(dphi_values.data(), dphi_shape);
  s_cmdspan2_t dphi = stdex::submdspan(
      full_basis, std::pair{1, (std::size_t)dim + 1}, 0, stdex::full_extent, 0);

  return compute_jacobian(J, K, detJ_scratch, dphi, coords);
}

} // namespace dolfinx_adaptivity