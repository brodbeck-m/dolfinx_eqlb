#pragma once

#include "eigen3/Eigen/Dense"

#include <span>
#include <vector>

using namespace dolfinx;

namespace dolfinx_eqlb
{
template <typename T, int id_flux_order, bool constr_minms>
class PatchDataCstm
{
public:
  PatchDataCstm(PatchFluxCstm<T, id_flux_order, constr_minms>& patch,
                const int niponts_per_fct)
      : _gdim(patch.dim()), _ndofs_flux(patch.ndofs_fluxdg_cell()),
        _ncells_max(patch.ncells_max()), _size_j(_gdim * _gdim)
  {
    // The patch
    const int ncells_max = patch.ncells_max();
    const int nfcts_per_cell = patch.fcts_per_cell();

    // Counter flux DOFs
    const int degree_flux_rt = patch.degree_raviart_thomas();

    const int ndofs_projflux = patch.ndofs_fluxdg_cell();
    const int ndofs_flux_fct = patch.ndofs_flux_fct();

    // --- Initialise storage
    // Piola mapping
    _data_J.resize(_size_j * ncells_max, 0);
    _data_K.resize(_size_j * ncells_max, 0);
    _data_detJ.resize(ncells_max, 0);

    // // Facet prefactors on cells
    // _data_fctprefactors_cell.resize(gdim * ncells_max);

    // // Mapped interpolation matrix
    // _data_M_mapped.resize(niponts_per_fct * ncells_max * ndofs_flux_fct, 0);

    // // Coefficients
    // _coefficients_rhs.resize(ncells_max * patch.ndofs_rhs_cell(), 0);
    // _coefficients_G_Tap1.resize(ncells_max * ndofs_projflux, 0);
    // _coefficients_G_Ta.resize(ncells_max * ndofs_projflux, 0);
    // _coefficients_flux.resize(ncells_max * patch.nrhs() * _ndofs_flux, 0);

    // // Jumps of the projected flux
    // _data_jumpG_Eam1.resize(_gdim * niponts_per_fct, 0);

    // // Higher order DOFs (explicit solution step)
    // _c_ta_div.resize(patch.ndofs_flux_cell_div(), 0);
    // _cj_ta_ea.resize(ndofs_flux_fct - 1, 0);

    // // --- Initialise equation system
    // int ndofs_hdivz, ndofs_hdivz_constr;

    // if (_gdim == 2)
    // {
    //   ndofs_hdivz = 1 + degree_flux_rt * (ncells_max + 1)
    //                 + 0.5 * degree_flux_rt * (degree_flux_rt - 1) *
    //                 ncells_max;
    //   ndofs_hdivz_constr = 2 * ndofs_hdivz + ncells_max + 2;
    // }
    // else
    // {
    //   throw std::runtime_error("3D not implemented");
    // }

    // // DOFmap and boundary markers
    // if constexpr (constr_minms)
    // {
    //   //   // DOFmap
    //   //   const int size_per_cell = (_gdim == 2)
    //   //                                 ? ndofs_hdivz_constr +
    //   nfcts_per_cell
    //   //                                 : ndofs_hdivz_constr + 3 *
    //   //                                 nfcts_per_cell;
    //   //   _shape_dofmap = {4, ncells_max, size_per_cell};
    //   //   _data_dofmap.resize(4 * ncells_max * size_per_cell, 0);

    //   // Boundary markers
    //   _boundary_markers.resize(ndofs_hdivz_constr, false);
    // }
    // else
    // {
    //   // Boundary markers
    //   _boundary_markers.resize(ndofs_hdivz, false);
    // }

    // // System matrix minimisation
    // _A.resize(ndofs_hdivz, ndofs_hdivz);
    // _L.resize(ndofs_hdivz);
    // _u.resize(ndofs_hdivz);

    // if constexpr (constr_minms)
    // {
    //   _A_constr.resize(ndofs_hdivz_constr, ndofs_hdivz_constr);
    //   _L_constr.resize(ndofs_hdivz_constr);
    //   _u_constr.resize(ndofs_hdivz_constr);
    // }
  }

  // --- Setter methods ---
  void set_patch_length(const int ncells) { _ncells = ncells; }

  void store_piola_mapping(const int cell_id, const double detJ,
                           mdspan_t<const double, 2> J)
  {
    // The offset
    const int offset = _size_j * cell_id;

    // Storage
    _data_detJ[cell_id] = detJ;

    store_piola_matrix(std::span<double>(_data_J.data() + offset, _size_j), J);
  }

  void store_piola_mapping(const int cell_id, const double detJ,
                           mdspan_t<const double, 2> J,
                           mdspan_t<const double, 2> K)
  {
    // The offset
    const int offset = _size_j * cell_id;

    // Storage
    _data_detJ[cell_id] = detJ;

    store_piola_matrix(std::span<double>(_data_J.data() + offset, _size_j), J);
    store_piola_matrix(std::span<double>(_data_K.data() + offset, _size_j), K);
  }

  // --- Getter methods ---
  int ncells() const { return _ncells; }

  /* Piola mapping */
  mdspan_t<const double, 2> jacobian(const int cell_id) const
  {
    // Set offset
    return mdspan_t<const double, 2>(_data_J.data() + _size_j * cell_id, _gdim,
                                     _gdim);
  }

  mdspan_t<const double, 2> inverse_jacobian(const int cell_id) const
  {
    // Set offset
    return mdspan_t<const double, 2>(_data_K.data() + _size_j * cell_id, _gdim,
                                     _gdim);
  }

  double jacobi_determinant(const int cell_id) const
  {
    return _data_detJ[cell_id];
  }

  std::span<const double> jacobi_determinants(const int ncells) const
  {
    return std::span<const double>(_data_detJ.data(), ncells);
  }

  /* Interpolation */

protected:
  void store_piola_matrix(std::span<double> storage,
                          mdspan_t<const double, 2> matrix)
  {
    if (_gdim == 2)
    {
      storage[0] = matrix(0, 0);
      storage[1] = matrix(0, 1);
      storage[2] = matrix(1, 0);
      storage[3] = matrix(1, 1);
    }
    else
    {
      storage[0] = matrix(0, 0);
      storage[1] = matrix(0, 1);
      storage[2] = matrix(0, 2);
      storage[3] = matrix(1, 0);
      storage[4] = matrix(1, 1);
      storage[5] = matrix(1, 2);
      storage[6] = matrix(2, 0);
      storage[7] = matrix(2, 1);
      storage[8] = matrix(2, 2);
    }
  }

  /* Variables */
  // --- General information
  // The spatial dimension
  const std::size_t _gdim;

  // Counter flux DOFs
  const int _ndofs_flux;

  // The length of the patch
  const int _ncells_max;
  int _ncells;

  // --- Patch-wise data (pre-calculated)
  // Storage of data for Piola mapping
  const int _size_j;
  std::vector<double> _data_J, _data_K, _data_detJ;

  // Pre-factors cell facets
  std::vector<double> _data_fctprefactors_cell;

  // The mapped interpolation matrix
  std::vector<double> _data_M_mapped;

  // --- Intermediate storage
  // Coefficients (RHS, projected flux, equilibrated flux)
  std::vector<T> _coefficients_rhs, _coefficients_G_Tap1, _coefficients_G_Ta,
      _coefficients_flux;

  // Jumps of the projected flux
  std::vector<T> _data_jumpG_Eam1;

  // Cell-wise solutions (explicit setp)
  std::vector<T> _c_ta_div, _cj_ta_ea;

  // --- The equation system
  // DOFmap
  std::array<std::size_t, 3> _shape_dofmap;
  std::vector<std::int32_t> _data_dofmap;

  // Boundary markers
  std::vector<std::int8_t> _boundary_markers;

  // Equation system
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> _A, _A_constr;
  Eigen::Matrix<T, Eigen::Dynamic, 1> _L, _L_constr, _u, _u_constr;

  // Solver
  Eigen::LLT<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> _solver;
  Eigen::PartialPivLU<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
      _solver_constr;
};
} // namespace dolfinx_eqlb