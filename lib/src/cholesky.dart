import 'dart:math';

import 'dense_matrix.dart';
import 'matrix.dart';
import 'sparse_matrix.dart';

/// A Cholesky decomposition of a symmetric positive definite matrix.
///
/// The Cholesky decomposition of a symmetric positive definite matrix A is
/// a lower triangular matrix L such that A = L * L'.
class Cholesky<T extends num> {
  /// The lower triangular matrix of the Cholesky decomposition.
  Matrix<T> L;

  /// Creates a new Cholesky decomposition.
  Cholesky() : L = SparseMatrix.empty();

  /// Returns the Cholesky decomposition of a symmetric positive definite matrix.
  ///
  /// The matrix must be square and symmetric and positive definite.
  /// If the matrix is not positive definite (but is square and symmetric)
  /// an attempt is made to run the decomposition anyway, but an [ArgumentError]
  /// is thrown if the decomposition fails.
  Matrix<T> decompose(Matrix<T> A) {
    if (!A.isSquare) throw ArgumentError("The matrix must be square.");
    if (!A.isSymmetric) throw ArgumentError("The matrix must be symmetric.");
    if (A is DenseMatrix<T>) return _choleskyDense(A);
    if (A is SparseMatrix<T>) return _choleskySparse(A);
    throw ArgumentError("The matrix type is not supported.");
  }

  /// Returns a DenseMatrix representing the cholesky decomposition of A.
  ///
  /// The matrix must be square and symmetric and positive definite otherwise an
  /// [ArgumentError] is thrown.
  Matrix<T> _choleskyDense(Matrix<T> A) {
    L = DenseMatrix.zero(A.rows);
    for (int i = 0; i < A.rows; i++) {
      for (int j = 0; j < (i + 1); j++) {
        T sum = A[i][j];
        for (int k = 0; k < j; k++) {
          sum = sum - L[i][k] * L[j][k] as T;
        }
        if (sum.isNegative || sum.isNaN) {
          throw ArgumentError("The matrix must be positive definite.");
        }
        if (i == j) {
          L[i][j] = sqrt(sum) as T;
        } else {
          L[i][j] = sum / L[j][j] as T;
        }
      }
    }
    return L;
  }

  /// Returns a SparseMatrix representing the cholesky decomposition of A.
  ///
  /// The matrix must be square and symmetric and positive definite otherwise an
  /// [ArgumentError] is thrown.
  Matrix<T> _choleskySparse(Matrix<T> A) {
    L = SparseMatrix.zero(A.rows);
    for (int i = 0; i < A.rows; i++) {
      for (int j = 0; j < (i + 1); j++) {
        T sum = A[i][j];
        for (int k = 0; k < j; k++) {
          sum = sum - L[i][k] * L[j][k] as T;
        }
        if (sum.isNegative || sum.isNaN) {
          throw ArgumentError("The matrix must be positive definite.");
        }
        if (i == j) {
          L[i][j] = sqrt(sum) as T;
        } else {
          L[i][j] = sum / L[j][j] as T;
        }
      }
    }
    return L;
  }
}
