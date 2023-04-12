import 'dense_matrix.dart';
import 'matrix.dart';
import 'sparse_matrix.dart';

/// The LU decomposition of a square matrix A.
///
/// This is a decomposition of the form A = LU, where L is a lower triangular
/// matrix and U is an upper triangular matrix.
class LUDecomposition<T extends num> {
  /// The lower and upper triangular matrices.
  Matrix<T> L, U;

  /// Creates a new LU decomposition.
  LUDecomposition()
      : L = SparseMatrix.empty(),
        U = SparseMatrix.empty();

  T get determinant => L.determinant * U.determinant as T;

  Matrix<T> get inverse => L.inverse * U.inverse;

  /// Decomposes the given matrix A into L and U.
  ///
  /// The matrix A must be square.
  Matrix<T> decompose(Matrix<T> A) {
    if (!A.isSquare) throw ArgumentError("The matrix must be square.");
    if (A is DenseMatrix<T>) return _luDense(A);
    if (A is SparseMatrix<T>) return _luSparse(A);
    throw ArgumentError("The matrix type is not supported.");
  }

  /// Returns a DenseMatrix representing the LU decomposition of matrix A.
  Matrix<T> _luDense(DenseMatrix<T> A) {
    L = DenseMatrix.identity(A.rows);
    U = DenseMatrix.zero(A.rows);
    int n = A.rows;
    for (int i = 0; i < n; i++) {
      for (int k = i; k < n; k++) {
        T sum = A[i][k];
        for (int j = 0; j < i; j++) {
          sum = sum - L[i][j] * U[j][k] as T;
        }
        U[i][k] = sum;
      }
      for (int k = i; k < n; k++) {
        if (i == k) {
          L[i][i] = 1 as T;
        } else {
          T sum = A[k][i];
          for (int j = 0; j < i; j++) {
            sum = sum - L[k][j] * U[j][i] as T;
          }
          L[k][i] = sum / U[i][i] as T;
        }
      }
    }
    return L * U;
  }

  /// Returns a SparseMatrix representing the LU decomposition of matrix A.
  Matrix<T> _luSparse(SparseMatrix<T> A) {
    L = SparseMatrix.identity(A.rows);
    U = SparseMatrix.zero(A.rows);
    int n = A.rows;
    for (int i = 0; i < n; i++) {
      for (int k = i; k < n; k++) {
        T sum = A[i][k];
        for (int j = 0; j < i; j++) {
          sum = sum - L[i][j] * U[j][k] as T;
        }
        U[i][k] = sum;
      }
      for (int k = i; k < n; k++) {
        if (i == k) {
          L[i][i] = 1 as T;
        } else {
          T sum = A[k][i];
          for (int j = 0; j < i; j++) {
            sum = sum - L[k][j] * U[j][i] as T;
          }
          L[k][i] = sum / U[i][i] as T;
        }
      }
    }
    return L * U;
  }
}
