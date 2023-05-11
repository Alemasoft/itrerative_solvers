import 'package:iterative_solvers/src/exception.dart';
import 'package:ml_linalg/linalg.dart';

extension MatrixHelper on Matrix {
  List<double> get diagonal {
    if (!isSquare) throw IterativeSolverException("Matrix is not square");
    List<double> d = [];
    for (int i = 0; i < length; i++) {
      for (int j = 0; j < length; j++) {
        if (i == j) {
          d.add(this[i][j]);
        }
      }
    }
    return d;
  }
}
