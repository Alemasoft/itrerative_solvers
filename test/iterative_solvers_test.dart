import 'package:iterative_solvers/src/jacobi.dart';
import 'package:iterative_solvers/src/solver_config.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:test/test.dart';

void main() {
  group('JacobiSolver', () {
    JacobiSolver jacobiSolver = JacobiSolver(SolverConfig(verbose: true));
    test("Jacobi simple", () {
      Matrix A = Matrix.fromList([
        [4, -1, -1],
        [-2, 6, 1],
        [-1, 1, 7],
      ]);
      Vector b = Vector.fromList([3, 9, -6]);
      Vector x = Vector.fromList([1, 2, -1]);
      for (double tolerace in tolerances) {
        jacobiSolver.config.tolerance = tolerace;
        jacobiSolver.solve(a: A, b: b, x: x);
      }
    });
  });
}
