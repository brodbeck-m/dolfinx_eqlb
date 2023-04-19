#pragma once

#include "ProblemData.hpp"
#include "ProblemDataFluxEV.hpp"

#include <algorithm>
#include <cmath>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/utils.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

using namespace dolfinx;

namespace dolfinx_adaptivity::equilibration
{
template <typename T>
class ProblemDataFluxCstm : public ProblemDataFluxEV<T>
{
public:
  /// Initialize storage of data for equilibration of (multiple) fluxes
  ///
  /// Initializes storage of the boundary-DOF lookup tables, the boundary values
  /// as well as the functions for the reconstructed flux and projected flues
  /// and RHS for the entire set of problems.
  ///
  /// @param fluxes    List of list of flux functions (H(div))
  /// @param fluxed_dg List of list of flux functions (DG)
  /// @param rhs_dg    List of list of projected right-hand-sides
  /// @param bcs_flux  List of list of BCs for each equilibarted flux
  ProblemDataFluxCstm(
      std::vector<std::shared_ptr<fem::Function<T>>>& fluxes,
      std::vector<std::shared_ptr<fem::Function<T>>>& fluxes_dg,
      std::vector<std::shared_ptr<fem::Function<T>>>& rhs_dg,
      const std::vector<
          std::vector<std::shared_ptr<const fem::DirichletBC<T>>>>& bcs_flux)
      : ProblemDataFluxEV<T>(fluxes, bcs_flux), _flux_dg(fluxes_dg),
        _rhs_dg(rhs_dg), _begin_rhsdg(fluxes.size(), 0)
  {
  }

  void initialize_coefficients(const std::vector<std::int32_t>& cells)
  {
    // FIXME: Use fem::IntegralType and region id as input and determine list of
    // cells here number of cells
    const int n_cells = cells.size();

    /* Determine size of coefficient storage */
    std::int32_t size_coef = 0;

    for (std::size_t i = 0; i < this->_nlhs; ++i)
    {
      // Determine DOF number per element
      int ndofs_fluxdg
          = _flux_dg[i]->function_space()->element()->space_dimension();
      int ndofs_rhsdg
          = _rhs_dg[i]->function_space()->element()->space_dimension();

      int cstride_i = ndofs_fluxdg + ndofs_rhsdg;

      // Increment overall number of coefficients
      size_coef += cstride_i * n_cells;

      // Set offsets and cstride
      this->_offset_coef[i + 1] = size_coef;
      this->_cstride[i] = cstride_i;
    }

    // Resize storage for coefficients
    this->_data_coef.resize(size_coef);

    /* Set coefficient data */
    // TODO - Add copy of function values into coefficient array
    throw std::runtime_error('Initialization of coefficients not implemented');
  }

  /* Setter functions*/
  void set_form(const std::vector<std::shared_ptr<const fem::Form<T>>>& forms)
  {
    // Check input data
    if (forms.size() != this->_nlhs)
    {
      throw std::runtime_error(
          "Equilibration: Input sizes of RHS does not match");
    }

    // Call setter from parent
    this->set_rhs(forms);
  }

  /* Getter functions*/
  /// Extract mesh
  /// @return The mesh
  std::shared_ptr<const mesh::Mesh> mesh()
  {
    return this->_solfunc[0]->function_space()->mesh();
  }

  /// Extract FunctionSpace of H(div) flux
  /// @return The FunctionSpace
  std::shared_ptr<const fem::FunctionSpace> fspace_flux_hdiv() const
  {
    return this->_solfunc[0]->function_space();
  }

  /// Extract FunctionSpace of projected flux
  /// @return The FunctionSpace
  std::shared_ptr<const fem::FunctionSpace> fspace_flux_dg() const
  {
    return _flux_dg[0]->function_space();
  }

  /// Extract FunctionSpace of projected RHS
  /// @return The FunctionSpace
  std::shared_ptr<const fem::FunctionSpace> fspace_rhs_dg() const
  {
    return _rhs_dg[0]->function_space();
  }

protected:
  /* Variables */
  // Fe functions of projected flux and RHS
  std::vector<std::shared_ptr<fem::Function<T>>>&_flux_dg, _rhs_dg;

  // Begin different fields in cell-wise coefficient vector
  std::vector<int> _begin_rhsdg;
};
} // namespace dolfinx_adaptivity::equilibration