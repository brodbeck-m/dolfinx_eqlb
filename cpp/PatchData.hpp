#pragma once

#include "eigen3/Eigen/Dense"

#include "utils.hpp"

#include <algorithm>
#include <span>
#include <vector>

using namespace dolfinx;

namespace dolfinx_eqlb
{
template <typename T, int id_flux_order>
class PatchDataCstm
{
public:
  /// Initialisation
  ///
  /// Holds temporary storage, required within semi-explicit flux equilibration
  /// in order to avoid repeated reallocation of memory.
  ///
  /// @param patch              The patch
  /// @param niponts_per_fct    The number of integration points per facet
  /// @param symconstr_required Flag for constrained minimisation
  PatchDataCstm(PatchFluxCstm<T, id_flux_order>& patch,
                const int niponts_per_fct, const bool symconstr_required)
      : _symconstr_required(symconstr_required), _gdim(patch.dim()),
        _degree_flux_rt(patch.degree_raviart_thomas()),
        _ndofs_flux(patch.ndofs_flux()), _ncells_max(patch.ncells_max()),
        _size_j(_gdim * _gdim)
  {
    // The patch
    const int ncells_max = patch.ncells_max();
    const int nfcts_per_cell = patch.fcts_per_cell();

    // Counter flux DOFs
    const int ndofs_projflux = patch.ndofs_fluxdg_cell();
    const int ndofs_flux_fct = patch.ndofs_flux_fct();

    // --- Initialise storage
    // Piola mapping
    _data_J.resize(_size_j * ncells_max, 0);
    _data_K.resize(_size_j * ncells_max, 0);
    _data_detJ.resize(ncells_max, 0);

    // Facet prefactors on cells
    _data_fctprefactors_cell.resize(_gdim * ncells_max);

    // Mapped interpolation matrix
    _shape_Mm = {_ncells_max, ndofs_flux_fct, _gdim, niponts_per_fct};
    _data_Mm.resize(_shape_Mm[0] * _shape_Mm[1] * _shape_Mm[2] * _shape_Mm[3],
                    0);

    // Coefficients
    _coefficients_rhs.resize(ncells_max * patch.ndofs_rhs_cell(), 0);
    _coefficients_G_Tap1.resize(ncells_max * ndofs_projflux, 0);
    _coefficients_G_Ta.resize(ncells_max * ndofs_projflux, 0);

    _shape_coeffsflux = {patch.nrhs(), _ncells_max, _ndofs_flux};
    _coefficients_flux.resize(
        _shape_coeffsflux[0] * _shape_coeffsflux[1] * _shape_coeffsflux[2], 0);

    // Jumps of the projected flux
    _shape_jGEam1 = {niponts_per_fct, _gdim};
    _data_jumpG_Eam1.resize(_shape_jGEam1[0] * _shape_jGEam1[1], 0);

    // Higher order DOFs (explicit solution step)
    _c_ta_div.resize(patch.ndofs_flux_cell_div(), 0);
    _cj_ta_ea.resize(ndofs_flux_fct - 1, 0);

    // --- Initialise equation system
    // FIXME - ndofs_hdivz_per_cell wrong for 2D quads + 3D
    const int nfcts_max = ncells_max + 1;

    const std::size_t ndofs_hdivz
        = dimension_uconstrained_minspace(ncells_max, nfcts_max);
    const int ndofs_hdivz_per_cell
        = 2 * ndofs_flux_fct + patch.ndofs_flux_cell_add() - 1;

    // Equation system (unconstrained minimisation)
    _A.resize(ndofs_hdivz, ndofs_hdivz);
    _L.resize(ndofs_hdivz);
    _u.resize(ndofs_hdivz);

    if (symconstr_required)
    {
      // Dimension constraint space
      const int npnts_max = nfcts_max + 1;
      const std::size_t ndofs_hdivz_constr
          = dimension_constrained_minspace(ndofs_hdivz, npnts_max);

      // DOFs per cell
      // FIXME -- Incorrect for 3D or non triangular cells
      const int ndofs_constrhdivz_per_cell = 2 * ndofs_hdivz_per_cell + 3;

      // DOFmap
      _shape_dofmap = {4, _ncells_max + 2, ndofs_constrhdivz_per_cell + 2};
      _data_dofmap.resize(
          _shape_dofmap[0] * _shape_dofmap[1] * _shape_dofmap[2], 0);

      // Boundary markers
      _boundary_markers.resize(ndofs_hdivz_constr, false);

      // Intermediate storage element contribution
      _shape_Te = {ndofs_hdivz_per_cell + 1, ndofs_hdivz_per_cell};
      _shape_Te_constr
          = {ndofs_constrhdivz_per_cell + 1, ndofs_constrhdivz_per_cell};

      std::cout << "Shape Te: " << ndofs_constrhdivz_per_cell + 1 << ", "
                << ndofs_constrhdivz_per_cell << std::endl;

      _data_Te.resize(_shape_Te_constr[0] * _shape_Te_constr[1], 0);

      // Intermediate storage of the stress coefficients
      _coefficients_stress.resize(ncells_max * _gdim * _ndofs_flux, 0);

      // Equation system (constrained minimisation)
      _A_constr.resize(ndofs_hdivz_constr, ndofs_hdivz_constr);
      _L_constr.resize(ndofs_hdivz_constr);
      _u_constr.resize(ndofs_hdivz_constr);
    }
    else
    {
      // Boundary markers
      _boundary_markers.resize(ndofs_hdivz, false);

      // Intermediate storage element contribution
      _shape_Te = {ndofs_hdivz_per_cell + 1, ndofs_hdivz_per_cell};
      _data_Te.resize(_shape_Te[0] * _shape_Te[1], 0);
    }
  }

  // --- Setter methods ---
  /// Reinitialise patch data
  ///
  /// Set current lengths of storage on patch and set array to zero (if
  /// required).
  ///
  /// @param ncells The number of cells
  /// @param nfcts  The number of facets
  /// @param npnts  The number of points
  void reinitialisation(const int ncells, const int nfcts, const int npnts)
  {
    // Set current patch length
    _ncells = ncells;

    // Set dimension of minimisation spaces
    dimension_minspaces(ncells, nfcts, npnts);

    // --- Update length of mdspans
    _shape_Mm[0] = ncells;
    _shape_coeffsflux[1] = ncells;
    _shape_dofmap[1] = ncells + 2;

    // --- Re-initialise storage
    // Coefficients of the flux
    const int length_cflux
        = _shape_coeffsflux[0] * ncells * _shape_coeffsflux[2];
    std::fill_n(_coefficients_flux.begin(), length_cflux, 0.0);

    // Storage jump of the projected flux
    reinitialise_jumpG_Eam1();
  }

  /// Set storage of facet jump to zero
  void reinitialise_jumpG_Eam1()
  {
    std::fill(_data_jumpG_Eam1.begin(), _data_jumpG_Eam1.end(), 0.0);
  }

  /// Set storage element contribution (minimisation) to zero
  void reinitialise_Te(const bool constrained_system)
  {
    if (constrained_system)
    {
      std::fill(_data_Te.begin(), _data_Te.end(), 0.0);
    }
    else
    {
      std::fill_n(_data_Te.begin(), _shape_Te[0] * _shape_Te[1], 0.0);
    }
  }

  /* Piola mapping */

  /// Store data for Piola mapping
  /// @param cell_id The cell id (starting at 0)
  /// @param detJ    The determinant of the Jacobian
  /// @param J       The Jacobian
  void store_piola_mapping(const int cell_id, const double detJ,
                           mdspan_t<const double, 2> J)
  {
    // The offset
    const int offset = _size_j * cell_id;

    // Storage
    _data_detJ[cell_id] = detJ;

    store_piola_matrix(std::span<double>(_data_J.data() + offset, _size_j), J);
  }

  /// Store data for Piola mapping
  /// @param cell_id The cell id (starting at 0)
  /// @param detJ    The determinant of the Jacobian
  /// @param J       The Jacobian
  /// @param K       The inverse Jacobian
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

  /// The spatial dimension
  /// @return The spatial dimension
  int gdim() const { return _gdim; }

  /// Number of cells on current patch
  /// @return The cell number
  int ncells() const { return _ncells; }

  /* Piola mapping */

  /// Extract the Jacobian of the i-th cell
  /// @param cell_id The cell id (starting at 0)
  /// @return mdspan of the Jacobian
  mdspan_t<const double, 2> jacobian(const int cell_id) const
  {
    // Set offset
    return mdspan_t<const double, 2>(_data_J.data() + _size_j * cell_id, _gdim,
                                     _gdim);
  }

  /// Extract the inverse Jacobian of the i-th cell
  /// @param cell_id The cell id (starting at 0)
  /// @return mdspan of the inverse Jacobian
  mdspan_t<const double, 2> inverse_jacobian(const int cell_id) const
  {
    // Set offset
    return mdspan_t<const double, 2>(_data_K.data() + _size_j * cell_id, _gdim,
                                     _gdim);
  }

  /// Extract the determinant of the Jacobian of the i-th cell
  /// @param cell_id The cell id (starting at 0)
  /// @return The determinant of the Jacobian
  double jacobi_determinant(const int cell_id) const
  {
    return _data_detJ[cell_id];
  }

  /// Extract the determinants of the Jacobian on patch
  /// @return span of the determinants on entire patch
  std::span<const double> jacobi_determinant() const
  {
    return std::span<const double>(_data_detJ.data(), _ncells);
  }

  /* Interpolation */

  /// Prefactors (facet orientation) on patch cells
  /// @return mdspan (cells x dim) of the prefactors
  mdspan_t<T, 2> prefactors_facet_per_cell()
  {
    // Set offset
    return mdspan_t<T, 2>(_data_fctprefactors_cell.data(), _ncells, _gdim);
  }

  /// Mapped interpolation matrix
  ///
  /// Structure mdspan: cells x dofs x dim x points
  /// Structure DOFs: Zero order on Eam1, higher order on Ea
  ///
  /// @return mdspan of the interpolation matrix
  mdspan_t<double, 4> mapped_interpolation_matrix()
  {
    return mdspan_t<double, 4>(_data_Mm.data(), _shape_Mm);
  }

  /* Coefficients */

  /// Coefficients of the flux
  /// @return mdspan (rhs x cells x dofs) of the coefficients
  mdspan_t<T, 3> coefficients_flux()
  {
    return mdspan_t<T, 3>(_coefficients_flux.data(), _shape_coeffsflux);
  }

  /// Coefficients of the i-th flux
  /// @param i The flux id (starting at 0)
  /// @return mdspan (cells x dofs) of the coefficients (i-th flux)
  mdspan_t<T, 2> coefficients_flux(const int i)
  {
    const int offset = i * _ncells * _shape_coeffsflux[2];
    return mdspan_t<T, 2>(_coefficients_flux.data() + offset,
                          _shape_coeffsflux[1], _shape_coeffsflux[2]);
  }

  /// Coefficients of the i-th flux on cell a
  /// @param i The flux id (starting at 0)
  /// @param a The cell id (starting at 1)
  /// @return span of the coefficients
  std::span<const T> coefficients_flux(const int i, const int a) const
  {
    const int offset
        = (i * _shape_coeffsflux[1] + a - 1) * _shape_coeffsflux[2];
    return std::span<const T>(_coefficients_flux.data() + offset,
                              _shape_coeffsflux[2]);
  }

  /// Coefficients of the stress tensor
  ///
  /// Structure [[dofs_T1_r0, ..., dofs_T1_rn], ...,
  ///            [dofs_Tm_r0, ...,dofs_Tm_rn]]
  ///
  /// @return span of the coefficients
  std::span<T> coefficients_stress()
  {
    const int size = _ncells * _gdim * _ndofs_flux;
    return std::span<T>(_coefficients_stress.data(), size);
  }

  /// Coefficients of the stress tensor on cell a
  /// @param a The cell id (starting at 1)
  /// @return span of the coefficients
  std::span<const T> coefficients_stress(const int a) const
  {
    const int size = _gdim * _ndofs_flux;
    const int offset = (a - 1) * size;
    return std::span<const T>(_coefficients_stress.data() + offset, size);
  }

  /// Coefficients of the projected flux (cell Ta)
  /// @return span of the coefficients
  std::span<T> coefficients_projflux_Ta()
  {
    return std::span<T>(_coefficients_G_Ta.data(), _coefficients_G_Ta.size());
  }

  /// Coefficients of the projected flux (cell Tap1)
  /// @return span of the coefficients
  std::span<T> coefficients_projflux_Tap1()
  {
    return std::span<T>(_coefficients_G_Tap1.data(),
                        _coefficients_G_Tap1.size());
  }

  /// Coefficients of the projected right-hand side
  /// @return span of the coefficients
  std::span<T> coefficients_rhs()
  {
    return std::span<T>(_coefficients_rhs.data(), _coefficients_rhs.size());
  }

  /* Intermediate storage */

  /// Jump of the projected flux on facet Eam1
  /// @return mdspan (ipoints x dim) of the jump
  mdspan_t<T, 2> jumpG_Eam1()
  {
    return mdspan_t<T, 2>(_data_jumpG_Eam1.data(), _shape_jGEam1);
  }

  /// Explicite solution: Divergence cell moments
  /// @return span of the solution coefficients
  std::span<T> c_ta_div()
  {
    return std::span<T>(_c_ta_div.data(), _c_ta_div.size());
  }

  /// Explicite solution: Higher order facet moments
  /// @return span of the solution coefficients
  std::span<T> cj_ta_ea()
  {
    return std::span<T>(_cj_ta_ea.data(), _cj_ta_ea.size());
  }

  /* The equation system */

  /// The DOFmap of the constrained minimisation space
  ///
  /// id within the first index:
  ///   id = 0: local DOFs
  ///   id = 1: global DOFs
  ///   id = 2: patch-local DOFs
  ///   id = 3: prefactor for construction of H(div=0) space
  ///
  /// @return mdspan (id x ncells x ndofs_per_cell) of the DOFmap
  mdspan_t<std::int32_t, 3> assembly_info_constained_minimisation()
  {
    return mdspan_t<std::int32_t, 3>(_data_dofmap.data(), _shape_dofmap);
  }

  /// The boundary markers (const version)
  /// @param constrained_system Id for constrained minimisation
  /// @return span of the boundary markers
  std::span<const std::int8_t> boundary_markers(bool constrained_system) const
  {
    if (constrained_system)
    {
      return std::span<const std::int8_t>(_boundary_markers.data(),
                                          _dim_hdivz_constr);
    }
    else
    {
      return std::span<const std::int8_t>(_boundary_markers.data(), _dim_hdivz);
    }
  }

  /// The equation system matrix
  /// @param constrained_system Id for constrained minimisation
  /// @return span of the matrix
  std::span<std::int8_t> boundary_markers(bool constrained_system)
  {
    if (constrained_system)
    {
      return std::span<std::int8_t>(_boundary_markers.data(),
                                    _dim_hdivz_constr);
    }
    else
    {
      return std::span<std::int8_t>(_boundary_markers.data(), _dim_hdivz);
    }
  }

  /// Storage cell-contribution minimisation system
  /// @param constrained_system Id for constrained minimisation
  /// @return mdspan (ndofs_per_cell + 1, ndofs_per_cell) of the storage
  mdspan_t<T, 2> Te(bool constrained_system)
  {
    if (constrained_system)
    {
      return mdspan_t<T, 2>(_data_Te.data(), _shape_Te_constr);
    }
    else
    {
      return mdspan_t<T, 2>(_data_Te.data(), _shape_Te);
    }
  }

  /// System matrix for minimisation problem
  /// @param constrained_system Id for constrained minimisation
  /// @return Eigen matrix of the minimisation system
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>&
  A_patch(bool constrained_system)
  {
    if (constrained_system)
    {
      return _A_constr;
    }
    else
    {
      return _A;
    }
  }

  /// System vector for minimisation problem
  /// @param constrained_system Id for constrained minimisation
  /// @return Eigen vector of the minimisation system
  Eigen::Matrix<T, Eigen::Dynamic, 1>& L_patch(bool constrained_system)
  {
    if (constrained_system)
    {
      return _L_constr;
    }
    else
    {
      return _L;
    }
  }

  /// Solution vector for minimisation problem
  /// @param constrained_system Id for constrained minimisation
  /// @return Eigen vector of the solution of minimisation problem
  Eigen::Matrix<T, Eigen::Dynamic, 1>& u_patch(bool constrained_system)
  {
    if (constrained_system)
    {
      return _u_constr;
    }
    else
    {
      return _u;
    }
  }

  /// Factorise the system matrix
  void factorise_system(bool constrained_system)
  {
    if (constrained_system)
    {
      _solver_constr.compute(
          _A_constr.topLeftCorner(_dim_hdivz_constr, _dim_hdivz_constr));
      std::cout << "Size system: " << _dim_hdivz_constr << std::endl;
    }
    else
    {
      if constexpr (id_flux_order > 1)
      {
        _solver.compute(_A.topLeftCorner(_dim_hdivz, _dim_hdivz));
      }
    }
  }

  /// Solve the minimisation problem
  void solve_system(bool constrained_system)
  {
    if (constrained_system)
    {
      _u_constr.head(_dim_hdivz_constr)
          = _solver_constr.solve(_L_constr.head(_dim_hdivz_constr));
    }
    else
    {
      if constexpr (id_flux_order == 1)
      {
        _u(0) = _L(0) / _A(0, 0);
      }
      else
      {
        _u.head(_dim_hdivz) = _solver.solve(_L.head(_dim_hdivz));
      }
    }
  }

protected:
  /// Store a Piola matrix in flattended storage
  /// @param storage The storage
  /// @param matrix  The matrix
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

  /// Dimension (patch-wise) of the unconstrained minimisation space
  /// @param ncells The number of cells on patch
  /// @param nfcts  The number of facets on patch
  /// @return       The dimension
  std::size_t dimension_uconstrained_minspace(const int ncells, const int nfcts)
  {
    return 1 + _degree_flux_rt * nfcts
           + 0.5 * _degree_flux_rt * (_degree_flux_rt - 1) * ncells;
  }

  /// Dimension (patch-wise) of the constrained minimisation space
  /// @param dim_hdivz The dimension of the unconstrained minimisation space
  /// @param npnt      The number of points on patch
  /// @return          The dimension
  std::size_t dimension_constrained_minspace(const int dim_hdivz,
                                             const int npnt)
  {
    if (_gdim == 2)
    {
      return 2 * dim_hdivz + npnt;
    }
    else
    {
      return 3 * (dim_hdivz + npnt);
    }
  }

  /// Set dimensions of minimisation spaces
  /// @param ncells The number of cells on patch
  /// @param nfcts  The number of factes on patch
  /// @param npnt   The number of points on patch
  void dimension_minspaces(const int ncells, const int nfcts, const int npnt)
  {
    // Dimension of the unconstrained minimisation space
    _dim_hdivz = dimension_uconstrained_minspace(ncells, nfcts);

    // Dimension of the constrained minimisation space
    _dim_hdivz_constr = dimension_constrained_minspace(_dim_hdivz, npnt);
  }

  /* Variables */
  const bool _symconstr_required;

  // --- General information
  // The spatial dimension
  const std::size_t _gdim;

  // Counter reconstructed flux
  const int _degree_flux_rt, _ndofs_flux;

  // Dimension H(div=0) space
  std::size_t _dim_hdivz, _dim_hdivz_constr;

  // The length of the patch
  const int _ncells_max;
  int _ncells;

  // --- Patch-wise data (pre-calculated)
  // Storage of data for Piola mapping
  const int _size_j;
  std::vector<double> _data_J, _data_K, _data_detJ;

  // Pre-factors cell facets
  std::vector<T> _data_fctprefactors_cell;

  // The mapped interpolation matrix
  std::array<std::size_t, 4> _shape_Mm;
  std::vector<double> _data_Mm;

  // --- Intermediate storage
  // Coefficients (RHS, projected flux, equilibrated flux)
  std::array<std::size_t, 3> _shape_coeffsflux;
  std::vector<T> _coefficients_rhs, _coefficients_G_Tap1, _coefficients_G_Ta,
      _coefficients_flux, _coefficients_stress;

  // Jumps of the projected flux
  std::array<std::size_t, 2> _shape_jGEam1;
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
  std::array<std::size_t, 2> _shape_Te, _shape_Te_constr;
  std::vector<T> _data_Te;

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> _A, _A_constr;
  Eigen::Matrix<T, Eigen::Dynamic, 1> _L, _L_constr, _u, _u_constr;

  // Solver
  Eigen::LLT<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> _solver;
  Eigen::PartialPivLU<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
      _solver_constr;
};
} // namespace dolfinx_eqlb