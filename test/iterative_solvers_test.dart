import 'package:iterative_solvers/iterative_solvers.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:test/test.dart';

void main() {
  group('JacobiSolver', () {
    JacobiSolver jacobiSolver = JacobiSolver(SolverConfig(verbose: true));
    setUp(() {
      print("\n\nNew test\n\n");
    });
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
    test("3k X 3k matrix tolerance $tolerance4", () {
      DateTime dt = DateTime.now();
      Matrix A = MtxMatrix.fromFile("dati/spa2.mtx", dType: DType.float64);
      DateTime dt2 = DateTime.now();
      print(dt2.difference(dt).toString());
      Vector b = Vector.filled(3000, 1.0, dtype: DType.float64);
      Vector f = (A * b).toVector();
      print(DateTime.now().difference(dt2).toString());
      jacobiSolver.config.tolerance = tolerance4;
      Vector x = jacobiSolver.solve(a: A, b: f);
      assert(x.isNotEmpty);
    });
    test("3k X 3k matrix tolerance $tolerance6", () {
      DateTime dt = DateTime.now();
      Matrix A = MtxMatrix.fromFile("dati/spa2.mtx", dType: DType.float64);
      DateTime dt2 = DateTime.now();
      print(dt2.difference(dt).toString());
      Vector b = Vector.filled(3000, 1.0, dtype: DType.float64);
      Vector f = (A * b).toVector();
      print(DateTime.now().difference(dt2).toString());
      jacobiSolver.config.tolerance = tolerance6;
      Vector x = jacobiSolver.solve(a: A, b: f);
      assert(x.isNotEmpty);
    });
    test("3k X 3k matrix tolerance $tolerance8", () {
      DateTime dt = DateTime.now();
      Matrix A = MtxMatrix.fromFile("dati/spa2.mtx", dType: DType.float64);
      DateTime dt2 = DateTime.now();
      print(dt2.difference(dt).toString());
      Vector b = Vector.filled(3000, 1.0, dtype: DType.float64);
      Vector f = (A * b).toVector();
      print(DateTime.now().difference(dt2).toString());
      jacobiSolver.config.tolerance = tolerance8;
      Vector x = jacobiSolver.solve(a: A, b: f);
      assert(x.isNotEmpty);
    });
    test("3k X 3k matrix tolerance $tolerance10", () {
      DateTime dt = DateTime.now();
      Matrix A = MtxMatrix.fromFile("dati/spa2.mtx", dType: DType.float64);
      DateTime dt2 = DateTime.now();
      print(dt2.difference(dt).toString());
      Vector b = Vector.filled(3000, 1.0, dtype: DType.float64);
      Vector f = (A * b).toVector();
      print(DateTime.now().difference(dt2).toString());
      jacobiSolver.config.tolerance = tolerance10;
      Vector x = jacobiSolver.solve(a: A, b: f);
      assert(x.isNotEmpty);
    });
  });
}
