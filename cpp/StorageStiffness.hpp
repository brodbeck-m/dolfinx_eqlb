#pragma once

#include <algorithm>
#include <iostream>
#include <memory>
#include <span>
#include <vector>

template <typename T>
class StorageStiffness
{
public:
  /// Initialization
  ///
  /// Storage of element constributions to stiffness and
  /// penalisation.
  ///
  /// @param ncells      Numbe of cells on current processor
  /// @param ndofs_elmt  Number of DOFs on element
  /// @param ndofs_pen   Number of DOFs of constraining field
  StorageStiffness(int ncells, int ndofs_elmt, int ndofs_cons)
      : _data_stf(ncells * ndofs_elmt * ndofs_elmt, 0),
        _data_pen(ncells * ndofs_cons, 0), _offset_stf(ncells + 1, 0),
        _offset_pen(ncells + 1, 0), _is_evaluated(ncells, 0)
  {
    // Set offset stiffness
    const int dim = ndofs_elmt * ndofs_elmt;

    std::generate(_offset_stf.begin(), _offset_stf.end(),
                  [n = 0, dim]() mutable { return dim * (n++); });

    // Set offset penalisation
    std::generate(_offset_pen.begin(), _offset_pen.end(),
                  [n = 0, ndofs_cons]() mutable { return ndofs_cons * (n++); });
  }

  /// Copy constructor
  StorageStiffness(const StorageStiffness& list) = default;

  /// Move constructor
  StorageStiffness(StorageStiffness&& list) = default;

  /// Destructor
  ~StorageStiffness() = default;

  /* Setter functions */
  /// Mark cell as evaluated
  /// @param cell_i Patch-local cell-id
  void mark_cell_evaluated(int cell_i) { _is_evaluated[cell_i] = 1; }

  /* Getter functions */
  /// Check if cell is evaluated
  /// @param cell_i Patch-local cell-id
  /// @return Evaluation stataus (0->False, 1->Tue)
  std::int8_t evaluation_status(int cell_i) { return _is_evaluated[cell_i]; }

  /// Extract element-stiffness of cell
  /// @param cell_i Patch-local cell-id
  /// @return Stiffness of cell_i
  std::span<T> stiffness_elmt(int cell_i)
  {
    return std::span<T>(_data_stf.data() + _offset_stf[cell_i],
                        _offset_stf[cell_i + 1] - _offset_stf[cell_i]);
  }

  /// Extract element-stiffness of cell (constant version)
  /// @param cell_i Patch-local cell-id
  /// @return Stiffness of cell_i
  std::span<const T> stiffness_elmt(int cell_i) const
  {
    return std::span<const T>(_data_stf.data() + _offset_stf[cell_i],
                              _offset_stf[cell_i + 1] - _offset_stf[cell_i]);
  }

  /// Extract penalty terms of cell
  /// @param cell_i Patch-local cell-id
  /// @return Penalty of cell_i
  std::span<T> penalty_elmt(int cell_i)
  {
    return std::span<T>(_data_pen.data() + _offset_pen[cell_i],
                        _offset_pen[cell_i + 1] - _offset_pen[cell_i]);
  }

  /// Extract penalty terms of cell (constant version)
  /// @param cell_i Patch-local cell-id
  /// @return Penalty of cell_i
  std::span<const T> penalty_elmt(int cell_i) const
  {
    return std::span<const T>(_data_pen.data() + _offset_pen[cell_i],
                              _offset_pen[cell_i + 1] - _offset_pen[cell_i]);
  }

private:
  std::vector<T> _data_stf, _data_pen;
  std::vector<std::int32_t> _offset_stf, _offset_pen;
  std::vector<std::int8_t> _is_evaluated;
};