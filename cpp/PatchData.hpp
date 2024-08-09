#pragma once

#include "eigen3/Eigen/Dense"

#include "Patch.hpp"
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
        _ndofs_flux(patch.ndofs_flux()),
        _ndofs_flux_fct(patch.ndofs_flux_fct()),
        _dim_hdivz_per_cell(_gdim * _ndofs_flux_fct
                            + patch.ndofs_flux_cell_add() - 1),
        _size_J(_gdim * _gdim)
  {
    // The patch
    const int ncells_max = patch.ncells_max();
    const int groupsize_max = patch.groupsize_max();
    const int nfcts_per_cell = patch.fcts_per_cell();

    // Counter flux DOFs
    const int ndofs_projflux = patch.ndofs_fluxdg_cell();

    // --- Initialise storage
    // Piola mapping
    _data_J.resize(_size_J * ncells_max, 0);
    _data_K.resize(_size_J * ncells_max, 0);
    _data_detJ.resize(ncells_max, 0);

    // Facet prefactors on cells
    _data_fctprefactors_cell.resize(_gdim * ncells_max);

    // Markers for reversed edges
    _data_reversedfct_cell.resize(_gdim * ncells_max);

    // Mapped interpolation matrix
    _shape_Mm = {ncells_max, _ndofs_flux_fct + 1, _gdim, niponts_per_fct};
    _data_Mm.resize(_shape_Mm[0] * _shape_Mm[1] * _shape_Mm[2] * _shape_Mm[3],
                    0);

    // Coefficients
    _coefficients_rhs.resize(ncells_max * patch.ndofs_rhs_cell(), 0);
    _coefficients_G_Tap1.resize(ncells_max * ndofs_projflux, 0);
    _coefficients_G_Ta.resize(ncells_max * ndofs_projflux, 0);

    _shape_coeffsflux = {patch.nrhs(), ncells_max, _ndofs_flux};
    _coefficients_flux.resize(
        _shape_coeffsflux[0] * _shape_coeffsflux[1] * _shape_coeffsflux[2], 0);

    // Jumps of the projected flux
    _shape_jGEam1 = {niponts_per_fct, 2, _gdim};
    _data_jumpG_Eam1.resize(
        _shape_jGEam1[0] * _shape_jGEam1[1] * _shape_jGEam1[2], 0);

    // Higher order DOFs (explicit solution step)
    _c_ta_div.resize(patch.ndofs_flux_cell_div(), 0);
    _cj_ta_ea.resize(_ndofs_flux_fct - 1, 0);
    _cj_ta_ea_interm.resize(_ndofs_flux_fct - 1, 0);

    // --- Initialise equation system
    // FIXME - ndofs_hdivz_per_cell wrong for 2D quads + 3D
    const int nfcts_max = ncells_max + 1;
    const std::size_t ndofs_hdivz_max
        = (groupsize_max == 1)
              ? dimension_uconstrained_minspace(ncells_max, nfcts_max)
              : dimension_uconstrained_minspace(ncells_max, nfcts_max)
                    + groupsize_max * (_ndofs_flux_fct - 1);

    // Identifier for mean-value zero condition
    _meanvalue_condition_required = false;

    // Equation system (unconstrained minimisation)
    _A.resize(ndofs_hdivz_max, ndofs_hdivz_max);

    // Intermediate storage element contribution
    const std::size_t dim_hdivz_per_cell_max
        = _dim_hdivz_per_cell + _ndofs_flux_fct - 1;
    _shape_Te = {dim_hdivz_per_cell_max + 1, dim_hdivz_per_cell_max};
    _data_Te.resize(_shape_Te[0] * _shape_Te[1], 0);

    if (symconstr_required)
    {
      // Dimension constraint space
      // (including lagrangian multiplier for mean-value zero constraint)
      const int npnts_max = nfcts_max + 1;

      const std::size_t ndofs_constr = (_gdim == 2) ? npnts_max : 3 * npnts_max;
      const int ndofs_constr_per_cell
          = (_gdim == 2) ? nfcts_per_cell : 3 * nfcts_per_cell;

      // Boundary markers
      _boundary_markers.resize(_gdim * ndofs_hdivz_max, false);

      // Intermediate storage element contribution
      _shape_Be = {dim_hdivz_per_cell_max, _gdim * ndofs_constr_per_cell};

      _data_Be.resize(_shape_Be[0] * _shape_Be[1], 0);
      _data_Ce.resize(ndofs_constr_per_cell, 0);
      _data_Le.resize(_gdim * dim_hdivz_per_cell_max + ndofs_constr_per_cell,
                      0);

      // Intermediate storage of the stress coefficients
      _coefficients_stress.resize(ncells_max * _gdim * _ndofs_flux, 0);

      // Equation system (unconstrained minimisation)
      _L.resize(_gdim * ndofs_hdivz_max + ndofs_constr + 1);
      _Ainv_t_fu.resize(ndofs_hdivz_max);
      _u_sigma.resize(_gdim * ndofs_hdivz_max);

      // Equation system (constrained minimisation)
      _A_rec.resize(ndofs_hdivz_max, ndofs_hdivz_max);
      _B.resize(ndofs_hdivz_max, _gdim * ndofs_constr);
      _Ainv_t_B.resize(ndofs_hdivz_max, ndofs_constr);
      _C.resize(ndofs_constr + 1, ndofs_constr + 1);
      _u_c.resize(ndofs_constr + 1);
    }
    else
    {
      // Boundary markers
      _boundary_markers.resize(ndofs_hdivz_max, false);

      // Solution vector for flux/stress
      _L.resize(ndofs_hdivz_max);
      _u_sigma.resize(ndofs_hdivz_max);
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
  void reinitialisation(std::span<const PatchType> type_patch, int ncells)
  {
    // --- Data patch
    // Set current patch length
    _ncells = ncells;

    // Set dimension of minimisation spaces
    if (type_patch[0] == PatchType::internal)
    {
      // Calculate dimension of minimisation spaces
      dimension_minspaces(ncells, ncells, ncells + 1);

      // Consider lagrangian multiplier
      _meanvalue_condition_required = true;
    }
    else
    {
      // Calculate dimension of minimisation spaces
      dimension_minspaces(ncells, ncells + 1, ncells + 2);

      // Check if lagrangian multiplier is required
      _meanvalue_condition_required = true;
      int condition_count = 0;

      for (std::size_t i = 0; i < _gdim; ++i)
      {
        if ((type_patch[i] == PatchType::bound_essnt_primal)
            || (type_patch[i] == PatchType::bound_mixed))
        {
          condition_count += 1;
        }
      }

      if (condition_count == _gdim)
      {
        _meanvalue_condition_required = false;
      }
    }

    // Identitier for reversed facets
    std::fill_n(_data_reversedfct_cell.begin(), _gdim * ncells, false);

    // --- Update length of mdspans
    _shape_Mm[0] = ncells;
    _shape_coeffsflux[1] = ncells;

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
  void reinitialise_Te() { std::fill(_data_Te.begin(), _data_Te.end(), 0.0); }

  void reinitialise_Ae() { reinitialise_Te(); }

  void reinitialise_Be() { std::fill(_data_Be.begin(), _data_Be.end(), 0.0); }

  void reinitialise_Ce() { std::fill(_data_Ce.begin(), _data_Ce.end(), 0.0); }

  void reinitialise_Le() { std::fill(_data_Le.begin(), _data_Le.end(), 0.0); }

  /* Piola mapping */

  /// Store data for Piola mapping
  /// @param cell_id The cell id (starting at 0)
  /// @param detJ    The determinant of the Jacobian
  /// @param J       The Jacobian
  void store_piola_mapping(const int cell_id, const double detJ,
                           mdspan_t<const double, 2> J)
  {
    // The offset
    const int offset = _size_J * cell_id;

    // Storage
    _data_detJ[cell_id] = detJ;

    store_piola_matrix(std::span<double>(_data_J.data() + offset, _size_J), J);
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
    const int offset = _size_J * cell_id;

    // Storage
    _data_detJ[cell_id] = detJ;

    store_piola_matrix(std::span<double>(_data_J.data() + offset, _size_J), J);
    store_piola_matrix(std::span<double>(_data_K.data() + offset, _size_J), K);
  }

  // --- Getter methods ---

  /// The spatial dimension
  /// @return The spatial dimension
  int gdim() const { return _gdim; }

  /// Number of cells on current patch
  /// @return The cell number
  int ncells() const { return _ncells; }

  /// Dimension of of patch-wise H(div=0) space per cell
  /// @return The dimension
  int ndofs_flux_hdivz_per_cell() const { return _dim_hdivz_per_cell; }

  /// Dimension of of patch-wise H(div=0) space
  /// @return The dimension
  int ndofs_flux_hdivz() const { return _dim_hdivz; }

  /// Dimension of the constrained space (weak symmetry condition)
  /// @return The dimension
  int ndofs_constraint() const { return _dim_constr; }

  /* Piola mapping */

  /// Extract the Jacobian of the i-th cell
  /// @param cell_id The cell id (starting at 0)
  /// @return mdspan of the Jacobian
  mdspan_t<const double, 2> jacobian(const int cell_id) const
  {
    // Set offset
    return mdspan_t<const double, 2>(_data_J.data() + _size_J * cell_id, _gdim,
                                     _gdim);
  }

  /// Extract the inverse Jacobian of the i-th cell
  /// @param cell_id The cell id (starting at 0)
  /// @return mdspan of the inverse Jacobian
  mdspan_t<const double, 2> inverse_jacobian(const int cell_id) const
  {
    // Set offset
    return mdspan_t<const double, 2>(_data_K.data() + _size_J * cell_id, _gdim,
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
    return mdspan_t<T, 2>(_data_fctprefactors_cell.data(), _ncells, _gdim);
  }

  /// Marker for reversed facets on patch cells
  /// @return mdspan (cells x dim) of the markers
  mdspan_t<std::uint8_t, 2> reversed_facets_per_cell()
  {
    return mdspan_t<uint8_t, 2>(_data_reversedfct_cell.data(), _ncells, _gdim);
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

  /// Coefficients of the stress tensor on cell a
  /// @param a The cell id (starting at 1)
  /// @return span of the coefficients
  std::span<const T> coefficients_stress(const int a) const
  {
    const int size = _gdim * _ndofs_flux;
    const int offset = (a - 1) * size;
    return std::span<const T>(_coefficients_stress.data() + offset, size);
  }

  /// Coefficients of the stress tensor (row i) on cell a
  /// @param i The id of the stress row (starting at 0)
  /// @param a The cell id (starting at 1)
  /// @return span of the coefficients
  std::span<T> coefficients_stress(const int i, const int a)
  {
    const int offset = ((a - 1) * _gdim + i) * _ndofs_flux;
    return std::span<T>(_coefficients_stress.data() + offset, _ndofs_flux);
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
  ///
  /// For reversed facets: Data is stored seperatly for the two adjacet cell
  /// (first index 0: Tam1, first index 1: Tap1)
  ///
  /// @return mdspan (ipoints x 2 x dim) of jump-related data
  mdspan_t<T, 3> jumpG_Eam1()
  {
    return mdspan_t<T, 3>(_data_jumpG_Eam1.data(), _shape_jGEam1);
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

  /// Explicite solution: Intermediate staorge for higher order facet moments
  /// @return span of the solution coefficients
  std::span<T> cj_intermediate()
  {
    return std::span<T>(_cj_ta_ea_interm.data(), _cj_ta_ea_interm.size());
  }

  /* The equation system */

  /// Check wether constrained minimisation requires lagrangian multiplier
  /// @return True if lagrangian multiplier is required
  bool meanvalue_zero_condition_required() const
  {
    return _meanvalue_condition_required;
  }

  /// The boundary markers (const version)
  /// @return span of the boundary markers
  std::span<const std::int8_t> boundary_markers(bool constrained_system) const
  {
    if (constrained_system)
    {
      return std::span<const std::int8_t>(_boundary_markers.data(),
                                          _gdim * _dim_hdivz);
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
                                    _gdim * _dim_hdivz);
    }
    else
    {
      return std::span<std::int8_t>(_boundary_markers.data(), _dim_hdivz);
    }
  }

  /// Storage cell-contribution unconstrained minimisation
  /// @return mdspan (ndofs_per_cell + 1, ndofs_per_cell) of the storage
  mdspan_t<T, 2> Te() { return mdspan_t<T, 2>(_data_Te.data(), _shape_Te); }

  /// Storage cell-contribution sub-matrix A
  /// @return mdspan (ndofs_per_cell + 1, ndofs_per_cell) of the storage
  mdspan_t<T, 2> Ae() { return Te(); }

  /// Storage cell-contribution sub-matrix B1 and B2
  /// @return mdspan (ndofs_per_cell, nnodes_per_ecll) of the storage
  mdspan_t<T, 2> Be() { return mdspan_t<T, 2>(_data_Be.data(), _shape_Be); }

  /// Storage cell-contribution sub-matrix C
  /// @return mdspan (nnodes_per_cell + 1, nnodes_per_cell + 1) of the storage
  std::span<T> Ce() { return std::span<T>(_data_Ce.data(), _data_Ce.size()); }

  /// Storage cell-contribution load vector
  /// @return mdspan (nnodes_per_cell + 1, nnodes_per_cell + 1) of the storage
  std::span<T> Le() { return std::span<T>(_data_Le.data(), _data_Le.size()); }

  /// The sub-matrix A
  /// @return Eigen representation of A
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& matrix_A() { return _A; }

  /// The sub-matrix A without boundary conditions
  /// @return Eigen representation of A
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& matrix_A_without_bc()
  {
    return _A_rec;
  }

  /// The sub-matrces B
  ///
  /// Cumulated storage of B_i: [B_1, ..., B_n]
  ///
  /// @return Eigen representation of commulated B_i
  ///         (ndofs_hdivz x gdim * ndofs_constr)
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& matrix_B() { return _B; }

  /// The sub-matrix C
  /// @return Eigen representation of C
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& matrix_C() { return _C; }

  /// The right-hand side L
  ///
  /// Comulated storage of RHS: [Lu_r1, ..., Lu_rn, L_c, 0]
  ///
  /// @return Eigen representation of the load vector
  Eigen::Matrix<T, Eigen::Dynamic, 1>& vector_L() { return _L; }

  /// Solution vector for minimisation problem
  /// @param constrained_system Id for constrained minimisation
  /// @return Eigen vector of the solution of minimisation problem
  Eigen::Matrix<T, Eigen::Dynamic, 1>& vector_u_sigma() { return _u_sigma; }

  /// Factorise matrix A
  void factorise_matrix_A()
  {
    if constexpr (id_flux_order > 1)
    {
      _solver_A.compute(_A.topLeftCorner(_dim_hdivz, _dim_hdivz));
    }
  }

  /// Solve unconstrained minimisation problem
  void solve_unconstrained_minimisation()
  {
    if constexpr (id_flux_order == 1)
    {
      _u_sigma(0) = _L(0) / _A(0, 0);
    }
    else
    {
      _u_sigma.head(_dim_hdivz) = _solver_A.solve(_L.head(_dim_hdivz));
    }
  }

  /// Solve constrained minimisation problem
  void solve_constrained_minimisation(const bool requires_flux_bc)
  {
    // Offset for vector L_c
    const int offset_Lc = _gdim * _dim_hdivz;

    // Calculate Schur complement
    for (int k = 0; k < _gdim; ++k)
    {
      // Offsets
      int offset_Bk = k * _dim_constr;
      int offset_uk = k * _dim_hdivz;

      // Apply boundary conditions on A
      if (requires_flux_bc)
      {
        // Modify A and f_uk
        apply_bcs_on_A(k);

        // Factorise A
        factorise_matrix_A();

        std::cout << "Matrix A k=" << k << ": " << std::endl;
        for (std::size_t m1 = 0; m1 < _dim_hdivz; ++m1)
        {
          for (std::size_t m2 = 0; m2 < _dim_hdivz; ++m2)
          {
            std::cout << _A(m1, m2) << " ";
          }
          std::cout << "\n";
        }
      }

      // Compute invers(A) * B_k
      _Ainv_t_B.topLeftCorner(_dim_hdivz, _dim_constr)
          = _solver_A.solve(_B.block(0, offset_Bk, _dim_hdivz, _dim_constr));

      // Contribution to Schur complement
      _C.topLeftCorner(_dim_constr, _dim_constr).noalias()
          -= _B.block(0, offset_Bk, _dim_hdivz, _dim_constr).transpose()
             * _Ainv_t_B.topLeftCorner(_dim_hdivz, _dim_constr);
    }

    // Factorise schur complement
    const int dim_c
        = (_meanvalue_condition_required) ? _dim_constr + 1 : _dim_constr;

    _solver_C.compute(_C.topLeftCorner(dim_c, dim_c));

    // Solve for constraints
    _u_c.head(dim_c) = _solver_C.solve(_L.segment(offset_Lc, dim_c));

    // Solve for u_k
    _u_sigma.setZero();

    for (int k = _gdim - 1; k > -1; --k)
    {
      // Offsets
      int offset_Bk = k * _dim_constr;
      int offset_uk = k * _dim_hdivz;

      // Refactorise A (with correct boundary conditions)
      if (requires_flux_bc & (k != (_gdim - 1)))
      {
        // Modify A and f_uk
        apply_bcs_on_A(k);

        // Factorise A
        factorise_matrix_A();
      }

      // Calculate u_k
      _u_sigma.segment(offset_uk, _dim_hdivz)
          = _solver_A.solve(-_B.block(0, offset_Bk, _dim_hdivz, _dim_constr)
                            * _u_c.head(_dim_constr));

      std::cout << "u_k=" << k << ": " << std::endl;
      for (std::size_t m = 0; m < _dim_hdivz; ++m)
      {
        std::cout << _u_sigma(offset_uk + m) << " ";
      }
      std::cout << "\n";
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

    // Number of constraints
    _dim_constr = (_gdim == 2) ? npnt : 3 * npnt;
  }

  /// Apply boundary conditions on matrix A
  /// @param subspace_k Id of the row of the stress tensor
  void apply_bcs_on_A(int subspace_k)
  {
    _A.setZero();
    const int offset_uk = subspace_k * _dim_hdivz;

    for (std::size_t i = 0; i < _dim_hdivz; ++i)
    {
      // Boundary marker dof_i
      std::int8_t bmarker_i = _boundary_markers[offset_uk + i];

      if (bmarker_i)
      {
        // Let RHS to zero
        _L(offset_uk + i) = 0.0;

        // Set 1 one main diagonal of A
        _A(i, i) = 1.0;
      }
      else
      {
        for (std::size_t j = 0; j < _dim_hdivz; ++j)
        {
          if (_boundary_markers[offset_uk + j])
          {
            _A(i, j) = 0.0;
          }
          else
          {
            _A(i, j) = _A_rec(i, j);
          }
        }
      }
    }
  }

  /* Variables */
  const bool _symconstr_required;

  // --- General information
  // The spatial dimension
  const std::size_t _gdim;

  // Counter reconstructed flux
  const int _degree_flux_rt, _ndofs_flux, _ndofs_flux_fct;

  // Dimension H(div=0) space
  const std::size_t _dim_hdivz_per_cell;
  std::size_t _dim_hdivz, _dim_constr;

  // The length of the patch
  int _ncells;

  // --- Patch-wise data (pre-calculated)
  // Storage of data for Piola mapping
  const int _size_J;
  std::vector<double> _data_J, _data_K, _data_detJ;

  // Pre-factors cell facets
  std::vector<T> _data_fctprefactors_cell;

  // Marker for reversed edges
  std::vector<std::uint8_t> _data_reversedfct_cell;

  // The mapped interpolation matrix
  std::array<std::size_t, 4> _shape_Mm;
  std::vector<double> _data_Mm;

  // --- Intermediate storage
  // Coefficients (RHS, projected flux, equilibrated flux)
  std::array<std::size_t, 3> _shape_coeffsflux;
  std::vector<T> _coefficients_rhs, _coefficients_G_Tap1, _coefficients_G_Ta,
      _coefficients_flux, _coefficients_stress;

  // Jumps of the projected flux
  std::array<std::size_t, 3> _shape_jGEam1;
  std::vector<T> _data_jumpG_Eam1;

  // Cell-wise solutions (explicit setp)
  std::vector<T> _c_ta_div, _cj_ta_ea, _cj_ta_ea_interm;

  // --- The equation system
  // Marker for addition mean-value constraint
  bool _meanvalue_condition_required;

  // Boundary markers
  std::vector<std::int8_t> _boundary_markers;

  // Equation system
  std::array<std::size_t, 2> _shape_Te, _shape_Be;
  std::vector<T> _data_Te, _data_Be, _data_Ce, _data_Le;

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> _A, _A_rec, _B, _Ainv_t_B,
      _C;
  Eigen::Matrix<T, Eigen::Dynamic, 1> _L, _Ainv_t_fu, _u_sigma, _u_c;

  // Solver
  Eigen::LLT<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> _solver_A;
  Eigen::PartialPivLU<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
      _solver_C;
};
} // namespace dolfinx_eqlb