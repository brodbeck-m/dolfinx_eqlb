#pragma once

#include "KernelData.hpp"
#include "PatchFluxCstm.hpp"
#include "utils.hpp"

#include <array>
#include <span>
#include <stdexcept>

using namespace dolfinx;

namespace dolfinx_adaptivity::equilibration
{
template <typename T, int id_flux_order = -1, bool asmbl_systmtrx = true>
void kernel_flux_minimisation(std::span<T> Te, int cstride_Te,
                              KernelData& kernel_data,
                              std::array<double, 2> prefactors_dof, double detJ,
                              std::span<T> coefficients,
                              std::span<const std::int32_t> fct_dofs,
                              std::span<const std::int32_t> cell_dofs)
{
  /* Extract shape functions and quadrature data */
  dolfinx_adaptivity::s_cmdspan3_t phi = kernel_data.shapefunctions_flux();

  const std::vector<double>& quadrature_weights
      = kernel_data.quadrature_weights_cell();

  /* Initialisation */
  const int ndofs_fct = fct_dofs.size() / 2;

  std::array<T, 2> sigtilde_q;

  /* Perform quadrature */
  for (std::size_t iq = 0; iq < quadrature_weights.size(); ++iq)
  {
    // Interpolate sigma_tilde
    if constexpr (id_flux_order == 1)
    {
      sigtilde_q[0] = coefficients[0] * phi(iq, 0, 0)
                      + coefficients[1] * phi(iq, 1, 0)
                      + coefficients[2] * phi(iq, 2, 0);
      sigtilde_q[1] = coefficients[0] * phi(iq, 0, 1)
                      + coefficients[1] * phi(iq, 1, 1)
                      + coefficients[2] * phi(iq, 2, 1);
    }
    else
    {
      throw std::runtime_error("Minimization only implemented for RT0");
    }

    // Correct normal orientation
    double p_Eam1 = prefactors_dof[0], p_Ea = prefactors_dof[1];

    // Coefficient d_0: Linear form
    std::int32_t dofl_Eam1 = fct_dofs[0];
    std::int32_t dofl_Ea = fct_dofs[ndofs_fct];

    std::array<T, 2> diff_phi;
    diff_phi[0] = p_Eam1 * phi(iq, dofl_Eam1, 0) - p_Ea * phi(iq, dofl_Ea, 0);
    diff_phi[1] = p_Eam1 * phi(iq, dofl_Eam1, 1) - p_Ea * phi(iq, dofl_Ea, 1);

    Te[0] = -(diff_phi[0] * sigtilde_q[0] - diff_phi[1] * sigtilde_q[1])
            * std::fabs(detJ);

    // Coefficient d_0: Bilinear form
    if constexpr (asmbl_systmtrx)
    {
      Te[cstride_Te] = (diff_phi[0] * diff_phi[0] + diff_phi[1] * diff_phi[1])
                       * std::fabs(detJ);
    }

    // Coefficients d^l_Eam1 and d^l_Ea
    // Interactions with d_0 are also calculated here
    if constexpr (id_flux_order > 1)
    {
      throw std::runtime_error("Minimisation only implemented for RT0");
    }

    // Coefficients d^r_Ta
    // Interactions with d_0 and d^l_E are also calculated here
    if constexpr (id_flux_order > 2)
    {
      throw std::runtime_error("Minimisation only implemented for RT0");
    }
  }
}
} // namespace dolfinx_adaptivity::equilibration