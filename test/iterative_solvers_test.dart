import 'package:iterative_solvers/iterative_solvers.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:test/test.dart';

void main() {
  const bool verb = false;

  group('JacobiSolver', () {
    JacobiSolver jacobiSolver = JacobiSolver(SolverConfig(verbose: verb));
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
      Matrix A = MtxMatrix.fromFile("dati/spa2.mtx", dType: DType.float64);
      Vector b = Vector.filled(3000, 1.0, dtype: DType.float64);
      Vector f = (A * b).toVector();
      jacobiSolver.config.tolerance = tolerance4;
      Vector x = jacobiSolver.solve(a: A, b: f, x: b);
      assert(x.isNotEmpty);
    });
    test("3k X 3k matrix tolerance $tolerance6", () {
      Matrix A = MtxMatrix.fromFile("dati/spa2.mtx", dType: DType.float64);

      Vector b = Vector.filled(3000, 1.0, dtype: DType.float64);
      Vector f = (A * b).toVector();

      jacobiSolver.config.tolerance = tolerance6;
      Vector x = jacobiSolver.solve(a: A, b: f, x: b);
      assert(x.isNotEmpty);
    });
    test("3k X 3k matrix tolerance $tolerance8", () {
      Matrix A = MtxMatrix.fromFile("dati/spa2.mtx", dType: DType.float64);

      Vector b = Vector.filled(3000, 1.0, dtype: DType.float64);
      Vector f = (A * b).toVector();

      jacobiSolver.config.tolerance = tolerance8;
      Vector x = jacobiSolver.solve(a: A, b: f, x: b);
      assert(x.isNotEmpty);
    });
    test("3k X 3k matrix tolerance $tolerance10", () {
      Matrix A = MtxMatrix.fromFile("dati/spa2.mtx", dType: DType.float64);

      Vector b = Vector.filled(3000, 1.0, dtype: DType.float64);
      Vector f = (A * b).toVector();

      jacobiSolver.config.tolerance = tolerance10;
      Vector x = jacobiSolver.solve(a: A, b: f, x: b);
      assert(x.isNotEmpty);
    });
  });
  group('GaussSeidelSolver', () {
    GaussSeidelSolver gaussSeidelSolver =
        GaussSeidelSolver(SolverConfig(verbose: verb));
    setUp(() {
      print("\n\nNew test\n\n");
    });
    test("Gauss simple", () {
      Matrix A = Matrix.fromList([
        [4, -1, -1],
        [-2, 6, 1],
        [-1, 1, 7],
      ]);
      Vector b = Vector.fromList([3, 9, -6]);
      Vector x = Vector.fromList([1, 2, -1]);
      for (double tolerace in tolerances) {
        gaussSeidelSolver.config.tolerance = tolerace;
        gaussSeidelSolver.solve(a: A, b: b, x: x);
      }
    });
    test("3k X 3k matrix tolerance $tolerance4", () {
      Matrix A = MtxMatrix.fromFile("dati/spa2.mtx", dType: DType.float64);

      Vector b = Vector.filled(3000, 1.0, dtype: DType.float64);
      Vector f = (A * b).toVector();

      gaussSeidelSolver.config.tolerance = tolerance4;
      Vector x = gaussSeidelSolver.solve(a: A, b: f, x: b);
      assert(x.isNotEmpty);
    });
    test("3k X 3k matrix tolerance $tolerance6", () {
      Matrix A = MtxMatrix.fromFile("dati/spa2.mtx", dType: DType.float64);

      Vector b = Vector.filled(3000, 1.0, dtype: DType.float64);
      Vector f = (A * b).toVector();

      gaussSeidelSolver.config.tolerance = tolerance6;
      Vector x = gaussSeidelSolver.solve(a: A, b: f, x: b);
      assert(x.isNotEmpty);
    });
    test("3k X 3k matrix tolerance $tolerance8", () {
      Matrix A = MtxMatrix.fromFile("dati/spa2.mtx", dType: DType.float64);

      Vector b = Vector.filled(3000, 1.0, dtype: DType.float64);
      Vector f = (A * b).toVector();

      gaussSeidelSolver.config.tolerance = tolerance8;
      Vector x = gaussSeidelSolver.solve(a: A, b: f, x: b);
      assert(x.isNotEmpty);
    });
    test("3k X 3k matrix tolerance $tolerance10", () {
      Matrix A = MtxMatrix.fromFile("dati/spa2.mtx", dType: DType.float64);

      Vector b = Vector.filled(3000, 1.0, dtype: DType.float64);
      Vector f = (A * b).toVector();

      gaussSeidelSolver.config.tolerance = tolerance10;
      Vector x = gaussSeidelSolver.solve(a: A, b: f, x: b);
      assert(x.isNotEmpty);
    });
  });

  group('Gradient', () {
    GradientSolver gradientSolver = GradientSolver(SolverConfig(verbose: verb));
    setUp(() {
      print("\n\nNew test\n\n");
    });
    test("Gradient simple", () {
      Matrix A = Matrix.fromList([
        [4, -1, -1],
        [-2, 6, 1],
        [-1, 1, 7],
      ]);
      Vector b = Vector.fromList([3, 9, -6]);
      Vector x = Vector.fromList([1, 2, -1]);
      for (double tolerace in tolerances) {
        gradientSolver.config.tolerance = tolerace;
        gradientSolver.solve(a: A, b: b, x: x);
      }
    });
    test("3k X 3k matrix tolerance $tolerance4", () {
      Matrix A = MtxMatrix.fromFile("dati/spa2.mtx", dType: DType.float64);

      Vector b = Vector.filled(3000, 1.0, dtype: DType.float64);
      Vector f = (A * b).toVector();

      gradientSolver.config.tolerance = tolerance4;
      Vector x = gradientSolver.solve(a: A, b: f, x: b);
      assert(x.isNotEmpty);
    });
    test("3k X 3k matrix tolerance $tolerance6", () {
      Matrix A = MtxMatrix.fromFile("dati/spa2.mtx", dType: DType.float64);

      Vector b = Vector.filled(3000, 1.0, dtype: DType.float64);
      Vector f = (A * b).toVector();

      gradientSolver.config.tolerance = tolerance6;
      Vector x = gradientSolver.solve(a: A, b: f, x: b);
      assert(x.isNotEmpty);
    });
    test("3k X 3k matrix tolerance $tolerance8", () {
      Matrix A = MtxMatrix.fromFile("dati/spa2.mtx", dType: DType.float64);

      Vector b = Vector.filled(3000, 1.0, dtype: DType.float64);
      Vector f = (A * b).toVector();

      gradientSolver.config.tolerance = tolerance8;
      Vector x = gradientSolver.solve(a: A, b: f, x: b);
      assert(x.isNotEmpty);
    });
    test("3k X 3k matrix tolerance $tolerance10", () {
      Matrix A = MtxMatrix.fromFile("dati/spa2.mtx", dType: DType.float64);

      Vector b = Vector.filled(3000, 1.0, dtype: DType.float64);
      Vector f = (A * b).toVector();

      gradientSolver.config.tolerance = tolerance10;
      Vector x = gradientSolver.solve(a: A, b: f, x: b);
      assert(x.isNotEmpty);
    });
  });

  group('ConjGradient', () {
    ConjGradientSolver conjGradientSolver =
        ConjGradientSolver(SolverConfig(verbose: verb));
    setUp(() {
      print("\n\nNew test\n\n");
    });
    test("ConjGradient simple", () {
      Matrix A = Matrix.fromList([
        [4, -1, -1],
        [-2, 6, 1],
        [-1, 1, 7],
      ]);
      Vector b = Vector.fromList([3, 9, -6]);
      Vector x = Vector.fromList([1, 2, -1]);
      for (double tolerace in tolerances) {
        conjGradientSolver.config.tolerance = tolerace;
        conjGradientSolver.solve(a: A, b: b, x: x);
      }
    });
    test("3k X 3k matrix tolerance $tolerance4", () {
      Matrix A = MtxMatrix.fromFile("dati/spa2.mtx", dType: DType.float64);

      Vector b = Vector.filled(3000, 1.0, dtype: DType.float64);
      Vector f = (A * b).toVector();

      conjGradientSolver.config.tolerance = tolerance4;
      Vector x = conjGradientSolver.solve(a: A, b: f, x: b);
      assert(x.isNotEmpty);
    });
    test("3k X 3k matrix tolerance $tolerance6", () {
      Matrix A = MtxMatrix.fromFile("dati/spa2.mtx", dType: DType.float64);

      Vector b = Vector.filled(3000, 1.0, dtype: DType.float64);
      Vector f = (A * b).toVector();

      conjGradientSolver.config.tolerance = tolerance6;
      Vector x = conjGradientSolver.solve(a: A, b: f, x: b);
      assert(x.isNotEmpty);
    });
    test("3k X 3k matrix tolerance $tolerance8", () {
      Matrix A = MtxMatrix.fromFile("dati/spa2.mtx", dType: DType.float64);

      Vector b = Vector.filled(3000, 1.0, dtype: DType.float64);
      Vector f = (A * b).toVector();

      conjGradientSolver.config.tolerance = tolerance8;
      Vector x = conjGradientSolver.solve(a: A, b: f, x: b);
      assert(x.isNotEmpty);
    });
    test("3k X 3k matrix tolerance $tolerance10", () {
      Matrix A = MtxMatrix.fromFile("dati/spa2.mtx", dType: DType.float64);

      Vector b = Vector.filled(3000, 1.0, dtype: DType.float64);
      Vector f = (A * b).toVector();

      conjGradientSolver.config.tolerance = tolerance10;
      Vector x = conjGradientSolver.solve(a: A, b: f, x: b);
      assert(x.isNotEmpty);
    });
  });
}
