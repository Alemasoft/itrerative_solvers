import 'dart:math';

import 'package:equatable/equatable.dart';

import 'cholesky.dart';
import 'dense_matrix.dart';
import 'lu_decomposition.dart';
import 'matrix.dart';
import 'vector.dart';

class SparseEntry<T extends num> extends Equatable {
  final int row;
  final int col;
  final T value;

  SparseEntry(this.row, this.col, this.value);

  @override
  List<Object?> get props => [row, col, value];
}

/// A sparse matrix.
///
/// A sparse matrix is a matrix that is stored in memory as a 2D array.
class SparseMatrix<T extends num> implements Matrix<T> {
  /// The cholseky decomposition of this matrix.
  final Cholesky<T> _cholesky = Cholesky<T>();

  /// The LU decomposition of this matrix.
  final LUDecomposition<T> _luDecomposition = LUDecomposition<T>();

  /// The inverse of this matrix.
  SparseMatrix<T>? _inverse;

  /// The transposed of this matrix.
  SparseMatrix<T>? _transposed;

  /// The determinant of this matrix.
  T? _determinant;

  /// The rank of this matrix.
  int? _rank;

  int _rows;

  int _cols;

  /// The data of the matrix.
  ///
  /// The data is stored as a list of rows.
  Set<SparseEntry<T>> _data;

  /// Whether the matrix is diagonal.
  bool? _isDiagonal;

  /// Whether the matrix is lower triangular.
  bool? _isLowerTriangular;

  /// Whether the matrix is upper triangular.
  bool? _isUpperTriangular;

  /// Whether the matrix is singular.
  bool? _isSingular;

  /// Whether the matrix is square.
  bool? _isSquare;

  /// Whether the matrix is symmetric.
  bool? _isSymmetric;

  /// Whether the matrix is positive definite.
  bool? _isPositiveDefinite;

  /// Constructs a new diagonal matrix.
  SparseMatrix.diagonal(Vector<T> diagonal)
      : _data = {},
        _rows = diagonal.length,
        _cols = diagonal.length {
    if (diagonal.isEmpty) {
      return;
    }
    _data = List<SparseEntry<T>>.generate(
        diagonal.length, (i) => SparseEntry<T>(i, i, diagonal[i])).toSet();
  }

  /// Constructs an empty sparse matrix.
  ///
  /// Should only be used if the matrix is going to be filled with values.
  /// Consider using [SparseMatrix.zero] instead.
  SparseMatrix.empty()
      : _isDiagonal = false,
        _isLowerTriangular = false,
        _isUpperTriangular = false,
        _isSingular = false,
        _isSquare = false,
        _isSymmetric = false,
        _isPositiveDefinite = false,
        _data = {},
        _rows = 0,
        _cols = 0;

  /// Constructs a sparse matrix from a list of columns.
  SparseMatrix.fromCols(List<Vector<T>> cols)
      : _data = {},
        _rows = cols.isEmpty ? 0 : cols[0].length,
        _cols = cols.length {
    if (cols.isEmpty) {
      return;
    }
    for (int i = 0; i < cols.length; i++) {
      if (cols[i].length != cols[0].length) {
        throw ArgumentError.value(
            cols[i].length,
            "cols",
            "All columns must have length ${cols[0].length}. "
                "Column $i has length ${cols[i].length}.");
      }
    }
    for (int i = 0; i < cols.length; i++) {
      for (int j = 0; j < cols[i].length; j++) {
        if (cols[i][j] != 0) {
          _data.add(SparseEntry<T>(j, i, cols[i][j]));
        }
      }
    }
  }

  /// Constructs a sparse matrix from a list of columns represented as lists.
  ///
  /// This is a convenience constructor that calls
  /// [SparseMatrix.fromCols] with a list of [Vector]s.
  SparseMatrix.fromColsList(List<List<T>> cols)
      : this.fromCols(cols.map((e) => Vector.fromList(e)).toList());

  SparseMatrix.fromMatrix(Matrix<T> matrix)
      : _data = {},
        _rows = matrix.rows,
        _cols = matrix.cols {
    if (matrix is SparseMatrix<T>) {
      _data = matrix._data;
      return;
    }
    for (int i = 0; i < matrix.rows; i++) {
      for (int j = 0; j < matrix.cols; j++) {
        if (matrix[i][j] != 0) {
          _data.add(SparseEntry<T>(i, j, matrix[i][j]));
        }
      }
    }
  }

  /// Constructs a sparse matrix from a list of rows.
  SparseMatrix.fromRows(List<Vector<T>> rows)
      : _data = {},
        _rows = rows.length,
        _cols = rows.isEmpty ? 0 : rows[0].length {
    if (rows.isEmpty) {
      return;
    }
    for (int i = 0; i < rows.length; i++) {
      if (rows[i].length != rows[0].length) {
        throw ArgumentError.value(
            rows,
            "rows",
            "All rows must have length ${rows[0].length}. "
                "Row $i has length ${rows[i].length}.");
      }
    }
    for (int i = 0; i < rows.length; i++) {
      for (int j = 0; j < rows[i].length; j++) {
        if (rows[i][j] != 0) {
          _data.add(SparseEntry<T>(i, j, rows[i][j]));
        }
      }
    }
  }

/*
  void _updateStats() {
    _isSquare = true;
    _isSymmetric = true;
    _isDiagonal = true;
    _isLowerTriangular = true;
    _isUpperTriangular = true;
    if (rows == cols) _isSquare = true;
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < i; j++) {
        if (_data[i][j] != _data[j][i]) _isSymmetric = false;
        if (i != j && _data[i][j] != 0 as T) _isDiagonal = false;
        if (_data[i][j] != 0 as T) _isUpperTriangular = false;
      }
      for (int z = i + 1; z < cols; z++) {
        if (_data[i][z] != 0 as T) _isLowerTriangular = false;
      }
    }
    _isPositiveDefinite = _checkPositiveDefinite();
    _isSingular = _checkSingular();
  }
*/

  /// Constructs a sparse matrix from a list of rows represented as lists.
  ///
  /// This is a convenience constructor that calls
  /// [SparseMatrix.fromRows] with a list of [Vector]s.
  SparseMatrix.fromRowsList(List<List<T>> rows)
      : this.fromRows(rows.map((e) => Vector.fromList(e)).toList());

  /// Constructs a new identity matrix.
  ///
  /// [size] must be greater or equal to 0.
  /// If [size] is 0, an empty matrix is returned.
  SparseMatrix.identity(int size)
      : _data = {},
        _rows = size,
        _cols = size {
    if (size < 0) {
      throw ArgumentError.value(size, "size", "Must be greater than 0.");
    }
    if (size == 0) {
      return;
    }
    _data =
        List<SparseEntry<T>>.generate(size, (i) => SparseEntry<T>(i, i, 1 as T))
            .toSet();
  }

  /// Constructs a new matrix with random values.
  ///
  /// [rows] and [cols] must be greater or equal to 0,
  /// otherwise an [ArgumentError] is thrown.
  ///
  /// If [rows] or [cols] is 0, both must be 0,
  /// otherwise an [ArgumentError] is thrown.
  ///
  /// If [rows] and [cols] is 0, an empty matrix is returned.
  /// [generator] is a function that takes an index and returns a random value.
  SparseMatrix.random(int rows, int cols, T Function(int index) generator)
      : _data = {},
        _rows = rows,
        _cols = cols {
    if (rows < 0) {
      throw ArgumentError.value(rows, "rows", "Must be greater or equal to 0.");
    }
    if (cols < 0) {
      throw ArgumentError.value(cols, "cols", "Must be greater or equal to 0.");
    }
    if ((rows == 0 || cols == 0) && rows != cols) {
      throw ArgumentError(
          "If rows or cols is 0, both must be 0. Got rows = $rows, cols = $cols.");
    }
    if (rows == 0 && cols == 0) {
      return;
    }
    _data = List<SparseEntry<T>>.generate(rows * cols,
        (i) => SparseEntry<T>(i ~/ cols, i % cols, generator(i))).toSet();
  }

  /// Constructs a new symmetric matrix with random values.
  ///
  /// [size] must be greater or equal to 0,
  /// otherwise an [ArgumentError] is thrown.
  SparseMatrix.randomSymmetric(int size, T Function(int index) generator)
      : _data = {},
        _rows = size,
        _cols = size {
    if (size < 0) {
      throw ArgumentError.value(size, "size", "Must be greater than 0.");
    }
    if (size == 0) {
      return;
    }
    _data = List<SparseEntry<T>>.generate(size * size,
        (i) => SparseEntry<T>(i ~/ size, i % size, generator(i))).toSet();
  }

  /// Constructs a new sparse matrix with only the secondary diagonal set.
  ///
  /// [diagonal] must be a vector with length greater or equal to 0,
  /// otherwise an [ArgumentError] is thrown.
  SparseMatrix.secondDiagonal(Vector<T> diagonal)
      : _data = {},
        _rows = diagonal.length,
        _cols = diagonal.length {
    if (diagonal.length < 0) {
      throw ArgumentError.value(
          diagonal.length, "diagonal", "Must be greater than 0.");
    }
    if (diagonal.length == 0) {
      return;
    }
    _data = List<SparseEntry<T>>.generate(diagonal.length,
        (i) => SparseEntry<T>(i, diagonal.length - i - 1, diagonal[i])).toSet();
  }

  /// Constructs a new sparse matrix filled with zeros.
  ///
  /// [rows] must be greater or equal to 0,
  /// otherwise an [ArgumentError] is thrown.
  ///
  /// If [cols] is null, it is set to [rows].
  /// If [cols] is greater than 0, [rows] must be greater than 0,
  /// otherwise an [ArgumentError] is thrown.
  ///
  /// If [rows] and [cols] is 0 (or null), an empty matrix is returned.
  SparseMatrix.zero(int rows, [int? cols])
      : _data = {},
        _rows = rows,
        _cols = cols ?? rows {
    if (cols != null && cols < 1 && rows > 0) {
      throw ArgumentError.value(cols, "cols", "Must be greater than 0.");
    }
    if (rows < 0) {
      throw ArgumentError.value(rows, "rows", "Must be greater or equal to 0.");
    }
    return;
  }

  /// The number of columns in the matrix.
  @override
  int get cols => _cols;

  List<SparseEntry<T>> get data => List.unmodifiable(_data);

  /// Returns the determinant of the matrix.
  ///
  /// If the matrix is not square, an [Exception] is thrown.
  /// If the matrix is singular, 0 is returned.
  /// If the matrix is diagonal, upper triangular or lower triangular,
  /// the determinant is calculated using the diagonal elements.
  /// Otherwise, the LU decomposition is used to calculate the determinant.
  /// The result is cached for performance.
  @override
  T get determinant {
    if (_determinant != null) return _determinant!;
    if (!isSquare) throw Exception('Matrix is not square');
    if (isSingular) return 0 as T;
    if (isDiagonal || isUpperTriangular || isLowerTriangular) {
      T det = 1 as T;
      for (int i = 0; i < rows; i++) {
        det = det * this[i][i] as T;
      }
      return det;
    }
    _determinant = _luDecomposition.determinant;
    return _determinant!;
  }

  /// Returns the inverse of the matrix.
  ///
  /// If the matrix is not square, an [Exception] is thrown.
  /// If the matrix is singular, an [Exception] is thrown.
  /// If the matrix is diagonal, upper triangular or lower triangular,
  /// the inverse is calculated using the diagonal elements.
  /// Otherwise, the inverse is calculated using LU decomposition.
  /// The result is cached for performance.
  @override
  Matrix<T> get inverse {
    if (_inverse != null) return _inverse!;
    if (!isSquare) throw Exception('Matrix is not square');
    if (isSingular) throw Exception('Matrix is singular');
    if (isDiagonal || isUpperTriangular || isLowerTriangular) {
      Matrix<T> inv = SparseMatrix.zero(rows, cols);
      for (int i = 0; i < rows; i++) {
        if (this[i][i] == 0 as T) throw Exception('Matrix is singular');
        inv[i][i] = 1 / this[i][i] as T;
      }
      return inv;
    }
    return _luDecomposition.inverse;
  }

  /// Returns whether the matrix is diagonal or not.
  ///
  /// A matrix is diagonal if all the elements outside the main diagonal
  /// are zero.
  /// The result is cached for performance.
  @override
  bool get isDiagonal {
    if (_isDiagonal == null) {
      _isDiagonal = true;
      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
          if (i != j && this[i][j] != 0 as T) _isDiagonal = false;
        }
      }
    }
    return _isDiagonal!;
  }

  /// Returns whether the matrix is lower triangular or not.
  ///
  /// A matrix is lower triangular if all the elements above the main diagonal
  /// are zero.
  /// The result is cached for performance.
  @override
  bool get isLowerTriangular {
    if (_isLowerTriangular == null) {
      _isLowerTriangular = true;
      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
          if (i < j && this[i][j] != 0 as T) _isLowerTriangular = false;
        }
      }
    }
    return _isLowerTriangular!;
  }

  /// Returns whether the matrix is positive definite or not.
  ///
  /// A matrix is positive definite if it is symmetric and all its eigenvalues
  /// are greater than zero.
  /// The result is cached for performance.
  @override
  bool get isPositiveDefinite {
    _isPositiveDefinite ??= _checkPositiveDefinite();
    return _isPositiveDefinite!;
  }

  /// Returns whether the matrix is singular or not.
  ///
  /// A matrix is singular if its determinant is zero.
  /// The result is cached for performance.
  @override
  bool get isSingular {
    _isSingular ??= _checkSingular();
    return _isSingular!;
  }

  /// Returns whether the matrix is square or not.
  ///
  /// A matrix is square if the number of rows is equal
  /// to the number of columns.
  ///
  /// The result is cached for performance.
  @override
  bool get isSquare {
    _isSquare ??= rows == cols;
    return _isSquare!;
  }

  /// Returns whether the matrix is symmetric or not.
  ///
  /// A matrix is symmetric if it is equal to its transpose.
  /// The result is cached for performance.
  @override
  bool get isSymmetric {
    if (_isSymmetric == null) {
      _isSymmetric = true;
      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
          if (this[i][j] != this[j][i]) _isSymmetric = false;
        }
      }
    }
    return _isSymmetric!;
  }

  /// Returns whether the matrix is upper triangular or not.
  ///
  /// A matrix is upper triangular if all the elements below the main diagonal
  /// are zero.
  /// The result is cached for performance.
  @override
  bool get isUpperTriangular {
    if (_isUpperTriangular == null) {
      _isUpperTriangular = true;
      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
          if (i > j && this[i][j] != 0 as T) _isUpperTriangular = false;
        }
      }
    }
    return _isUpperTriangular!;
  }

  /// The number of rows in the matrix.
  @override
  int get rows => _rows;

  /// The size of the matrix calculated as rows * cols.
  @override
  int get size => rows * cols;

  /// Returns the transpose of the matrix.
  ///
  /// The transpose of a matrix is an operator which flips a matrix over its diagonal.
  @override
  Matrix<T> get transpose {
    if (_transposed != null) return _transposed!;
    SparseMatrix<T> transposed = copy();
    for (int i = 0; i < rows; i++) {
      for (int j = i + 1; j < cols; j++) {
        T temp = transposed[i][j];
        transposed[i][j] = transposed[j][i];
        transposed[j][i] = temp;
      }
    }
    _transposed = transposed;
    return _transposed!;
  }

  /// Returns the results of the moltiplication of this matrix with [other].
  ///
  /// The number of columns in this matrix must be equal to the number
  /// of rows in [other], otherwise an [ArgumentError] is thrown.
  @override
  Matrix<T> operator *(Matrix<T> other) {
    if (cols != other.rows) {
      throw ArgumentError.value(
          other.rows,
          "other",
          "Must have the same number of rows as this matrix has columns. "
              "Got ${other.rows} rows, expected $cols rows.");
    }
    if (size == 0) return copy();
    SparseMatrix<T> result = SparseMatrix<T>.zero(rows, other.cols);
    for (int i = 0; i < result.rows; i++) {
      for (int j = 0; j < result.cols; j++) {
        for (int k = 0; k < cols; k++) {
          result[i][j] = result[i][j] + ([i][k] * other[k][j]) as T;
        }
      }
    }
    return result;
  }

  /// Returns the results of the addition of this matrix with [other].
  ///
  /// The two matrices must have the same size,
  /// otherwise an [ArgumentError] is thrown.
  @override
  Matrix<T> operator +(Matrix<T> other) {
    if (other.size != size) {
      throw ArgumentError.value(
          other, "other", "Must be the same size as this matrix.");
    }
    if (size == 0) return copy();
    SparseMatrix<T> result = other.copy().toSparse();
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        result[i][j] = result[i][j] + other[i][j] as T;
      }
    }
    return result;
  }

  /// Returns the results of the subtraction of this matrix with [other].
  ///
  /// The two matrices must have the same size,
  /// otherwise an [ArgumentError] is thrown.
  @override
  Matrix<T> operator -(Matrix<T> other) {
    if (other.size != size) {
      throw ArgumentError.value(other.size, "other",
          "Must be the same size as this matrix $size. Got ${other.size}.");
    }
    SparseMatrix<T> result = other.copy().toSparse();
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        result[i][j] = result[i][j] - other[i][j] as T;
      }
    }
    return result;
  }

  /// Returns the vector at the given [row].
  ///
  /// If [row] is greater than the number of rows in the matrix,
  /// an [ArgumentError] is thrown.
  @override
  Vector<T> operator [](int row) {
    if (row >= rows) {
      throw ArgumentError.value(row, "row", "Must be less than $rows");
    }
    return Vector<T>.fromIterable(
        _data.where((element) => element.row == row).map((e) => e.value));
  }

  /// Sets the vector at the given [row] to [value].
  ///
  /// If [row] is greater than the number of rows in the matrix,
  /// an [ArgumentError] is thrown.
  @override
  void operator []=(int row, Vector<T> value) {
    if (row >= rows) {
      throw ArgumentError.value(row, "row", "Must be less than $rows");
    }
    for (int i = 0; i < value.length; i++) {
      if (value[i] != 0) {
        if (!_data.add(SparseEntry<T>(row, i, value[i]))) {
          _data
              .removeWhere((element) => element.row == row && element.col == i);
          _data.add(SparseEntry<T>(row, i, value[i]));
        }
      }
    }
  }

  /// Appends the given [col] to the matrix.
  ///
  /// The length of the [col] must be equal to the number of rows in the matrix,
  /// otherwise an [Exception] is thrown.
  @override
  void addCol(Vector<T> col) {
    if (col.length != rows) {
      throw Exception('Column length must be equal to rows');
    }
    for (int i = 0; i < rows; i++) {
      if (col[i] != 0) _data.add(SparseEntry<T>(i, cols, col[i]));
    }
    _resetStats();
  }

  /// Appends the given [row] to the matrix.
  ///
  /// The length of the [row] must be equal to the number
  /// of columns in the matrix, otherwise an [Exception] is thrown.
  @override
  void addRow(Vector<T> row) {
    if (row.length != cols) {
      throw Exception('Row length must be equal to cols');
    }
    for (int i = 0; i < cols; i++) {
      if (row[i] != 0) _data.add(SparseEntry<T>(rows, i, row[i]));
    }
    _resetStats();
  }

  /// The condition number of the matrix.
  ///
  /// Uses the 2-norm.
  @override
  T cond() {
    if (!isSquare) throw Exception('Matrix is not square');
    return norm2() * inverse.norm2() as T;
  }

  /// Returns a copy of this matrix.
  ///
  /// The copy is a deep copy, meaning that the data is copied.
  @override
  SparseMatrix<T> copy() {
    SparseMatrix<T> m = SparseMatrix<T>.fromRows(List.from(_data));
    m._isDiagonal = _isDiagonal;
    m._isLowerTriangular = _isLowerTriangular;
    m._isUpperTriangular = _isUpperTriangular;
    m._isSquare = _isSquare;
    m._isSymmetric = _isSymmetric;
    return m;
  }

  /// Returns the element at the given [row] and [col].
  ///
  /// If [row] or [col] is greater than the number of rows or columns in the
  /// matrix, an [ArgumentError] is thrown.
  @override
  T get(int row, int col) {
    if (row >= rows) {
      throw ArgumentError.value(row, "Row", 'Index out of bounds');
    }
    if (col >= cols) {
      throw ArgumentError.value(col, "Col", 'Index out of bounds');
    }
    return this[row][col];
  }

  /// Returns the 1-norm of the matrix.
  ///
  /// If the matrix is not square, an [Exception] is thrown.
  @override
  T norm1() {
    if (!isSquare) throw Exception('Matrix is not square');
    T result = 0 as T;
    for (int i = 0; i < rows; i++) {
      T sum = 0 as T;
      for (int j = 0; j < cols; j++) {
        sum = sum + this[i][j].abs() as T;
      }
      if (sum > result) result = sum;
    }
    return result;
  }

  /// Returns the 2-norm of the matrix.
  ///
  /// If the matrix is not square, an [Exception] is thrown.
  @override
  T norm2() {
    if (!isSquare) throw Exception('Matrix is not square');
    T result = 0 as T;
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        result = result + this[i][j] * this[i][j] as T;
      }
    }
    return sqrt(result) as T;
  }

  /// Returns the infinity-norm of the matrix.
  ///
  /// If the matrix is not square, an [Exception] is thrown.
  @override
  T normInf() {
    if (!isSquare) throw Exception('Matrix is not square');
    T result = 0 as T;
    for (int i = 0; i < rows; i++) {
      T sum = 0 as T;
      for (int j = 0; j < cols; j++) {
        sum = sum + this[j][i].abs() as T;
      }
      if (sum > result) result = sum;
    }
    return result;
  }

  /// Returns the p-norm of the matrix.
  ///
  /// If the matrix is not square, an [Exception] is thrown.
  @override
  T normP(T p) {
    if (!isSquare) throw Exception('Matrix is not square');
    T result = 0 as T;
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        result = result + pow(this[i][j], p) as T;
      }
    }
    return pow(result, 1 / p) as T;
  }

  /// Returns the determinant of the matrix.
  ///
  /// The rank is cached for performance.
  @override
  int rank() {
    if (_rank != null) return _rank!;
    int rank = 0;
    List<bool> rowSelected = List.filled(rows, false);
    for (int i = 0; i < cols; ++i) {
      int j;
      for (j = 0; j < rows; ++j) {
        if (!rowSelected[j] && this[j][i].abs() > 0) break;
      }

      if (j != rows) {
        ++rank;
        rowSelected[j] = true;
        for (int p = i + 1; p < cols; ++p) {
          this[j][p] = this[j][p] / this[j][i] as T;
        }
        for (int k = 0; k < rows; ++k) {
          if (k != j && this[k][i].abs() > 0) {
            for (int p = i + 1; p < cols; ++p) {
              this[k][p] = this[k][p] - this[j][p] * this[k][i] as T;
            }
          }
        }
      }
    }
    return rank;
  }

  /// Removes a column from the matrix and returns it as a [Vector].
  ///
  /// If the column index is out of bounds, an [ArgumentError] is thrown.
  @override
  Vector<T> removeCol(int col) {
    if (col >= cols || col < 0) {
      throw ArgumentError.value(col, "Col", 'Index out of bounds');
    }
    Vector<T> result = Vector<T>.fromIterable(
        _data.where((element) => element.col == col).map((e) => e.value));
    _data.removeWhere((element) => element.col == col);
    _resetStats();
    return result;
  }

  /// Removes a row from the matrix and returns it as a [Vector].
  ///
  /// If the row index is out of bounds, an [ArgumentError] is thrown.
  @override
  Vector<T> removeRow(int row) {
    if (row >= rows || row < 0) {
      throw ArgumentError.value(row, "Row", 'Index out of bounds');
    }
    Vector<T> result = Vector<T>.fromIterable(
        _data.where((element) => element.row == row).map((e) => e.value));
    _data.removeWhere((element) => element.row == row);
    _resetStats();
    return result;
  }

  ///Returns the results of the division of the matrix by a scalar.
  ///
  /// If the scalar is zero, an [ArgumentError] is thrown.
  @override
  Matrix<T> scalarDiv(T scalar) {
    if (scalar == 0 as T) {
      throw ArgumentError.value(scalar, 'scalar', 'Cannot divide by zero');
    }
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        this[i][j] = this[i][j] / scalar as T;
      }
    }
    _resetStats();
    return this;
  }

  /// Returns the results of the multiplication of the matrix by a scalar.
  ///
  /// If the scalar is zero this equals to [Matrix.zero].
  @override
  Matrix<T> scalarProd(T scalar) {
    if (scalar == 0 as T) {
      _data = {};
      _resetStats();
      return this;
    }
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        this[i][j] = this[i][j] * scalar as T;
      }
    }
    _resetStats();
    return this;
  }

  /// Returns the results of the subtraction of a scalar to the matrix.
  ///
  /// If the scalar is zero nothing happens.
  @override
  Matrix<T> scalarSub(T scalar) {
    if (scalar == 0 as T) return this;
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        this[i][j] = this[i][j] - scalar as T;
      }
    }
    _resetStats();
    return this;
  }

  /// Returns the results of the addition of a scalar to the matrix.
  ///
  /// If the scalar is zero nothing happens.
  @override
  Matrix<T> scalarSum(T scalar) {
    if (scalar == 0 as T) return this;
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        this[i][j] = this[i][j] + scalar as T;
      }
    }
    _resetStats();
    return this;
  }

  /// Sets the value of the matrix at the given position and returns the old value.
  ///
  /// If the row or column index is out of bounds, an [ArgumentError] is thrown.
  /// If the value is the same as the current one, nothing happens.
  ///
  @override
  T set(int row, int col, T value) {
    if (row >= rows) {
      throw ArgumentError.value(row, "Row", 'Index out of bounds');
    }
    if (col >= cols) {
      throw ArgumentError.value(col, "Col", 'Index out of bounds');
    }
    T oldValue = this[row][col];
    if (oldValue == value) return oldValue;
    this[row][col] = value;
    _resetStats();
    return oldValue;
  }

  /// Returns a new [DenseMatrix] containing the same data as this matrix.
  ///
  /// Defaults to [DenseMatrix.fromMatrix].
  @override
  DenseMatrix<T> toDense() {
    return DenseMatrix.fromMatrix(this);
  }

  /// Returns this matrix as a [SparseMatrix].
  ///
  /// Defaults to [this].
  @override
  SparseMatrix<T> toSparse() {
    return this;
  }

  /// Returns the trace of the matrix.
  ///
  /// If the matrix is not square, an [Exception] is thrown.
  /// The trace of a square matrix is the sum of the elements on the main diagonal.
  @override
  T trace() {
    if (!isSquare) throw Exception('Matrix is not square');
    T result = 0 as T;
    for (int i = 0; i < rows; i++) {
      result = result + this[i][i] as T;
    }
    return result;
  }

  /// Checks if the matrix is positive definite
  /// by using the Cholesky decomposition.
  bool _checkPositiveDefinite() {
    try {
      _cholesky.decompose(this);
      return true;
    } catch (e) {
      print(e);
      return false;
    }
  }

  /// Checks if the matrix is singular by using the LU decomposition.
  bool _checkSingular() {
    if (determinant == 0) return true;
    try {
      _luDecomposition.decompose(this);
      return true;
    } catch (e) {
      print(e);
      return false;
    }
  }

  /// Resets the cached statistics of this matrix.
  ///
  /// This method should be called when the matrix is modified.
  void _resetStats() {
    _isDiagonal = null;
    _isLowerTriangular = null;
    _isUpperTriangular = null;
    _isSquare = cols == rows;
    _isSymmetric = null;
    _isPositiveDefinite = null;
    _inverse = null;
    _transposed = null;
    _isSingular = null;
  }
}
