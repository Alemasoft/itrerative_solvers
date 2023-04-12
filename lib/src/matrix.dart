import 'package:iterative_solvers/iterative_solvers.dart';

abstract class Matrix<T extends num> {
  Matrix.diagonal(Vector<T> diagonal);
  Matrix.empty();

  Matrix.fromCols(List<Vector<T>> cols);
  Matrix.fromColsList(List<List<T>> cols);

  Matrix.fromMatrix(Matrix<T> other);
  Matrix.fromRows(List<Vector<T>> rows);

  Matrix.fromRowsList(List<List<T>> rows);

  Matrix.identity(int size);

  Matrix.random(int rows, int cols);
  Matrix.randomSymmetric(int size);
  Matrix.secondDiagonal(Vector<T> diagonal);
  Matrix.zero(int rows, int cols);

  int get cols;

  T get determinant;
  Matrix<T> get inverse;

  bool get isDiagonal;
  bool get isLowerTriangular;
  bool get isPositiveDefinite;

  bool get isSingular;

  bool get isSquare;
  bool get isSymmetric;

  bool get isUpperTriangular;
  int get rows;
  int get size;

  Matrix<T> get transpose;
  Matrix<T> operator *(covariant other);

  Matrix<T> operator +(covariant other);
  Matrix<T> operator -(covariant other);
  Vector<T> operator [](int row);

  void operator []=(int row, Vector<T> value);
  void addCol(Vector<T> col);
  void addRow(Vector<T> row);
  T cond();

  Matrix<T> copy();
  T get(int row, int col);
  T norm1();

  T norm2();
  T normInf();
  T normP(T p);
  int rank();
  Vector<T> removeCol(int col);
  Vector<T> removeRow(int row);
  Matrix<T> scalarDiv(T scalar);

  Matrix<T> scalarProd(T scalar);
  Matrix<T> scalarSub(T scalar);
  Matrix<T> scalarSum(T scalar);
  T set(int row, int col, T value);
  DenseMatrix<T> toDense();
  SparseMatrix<T> toSparse();
  T trace();
}
