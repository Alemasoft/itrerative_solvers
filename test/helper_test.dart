import 'package:iterative_solvers/src/common/helper.dart';
import 'package:iterative_solvers/src/solver_config.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:test/test.dart';

bool equalsWithAnyTolerance(double a, double b) {
  for (double tolerance in tolerances) {
    if (a == b || (a - b).abs() < tolerance) {
      return true;
    }
  }
  print("a: $a, b: $b, a-b: ${a - b}");
  return false;
}

void main() {
  group('Matrix Helper', () {
    test("Diagonal", () {
      Matrix A = Matrix.fromList([
        [4, -1, -1],
        [-2, 6, 1],
        [-1, 1, 7],
      ]);
      List<double> d = A.diagonal;
      expect(d, [4, 6, 7]);
    });
    test("From mtx", () {
      Matrix A = MatrixHelper.fromMtx("dati/spa1.mtx");
      expect(A.isSquare, true);
      expect(A.isNotEmpty, true);
      expect(A.columnCount, 1000);
      expect(A.rowCount, 1000);
      expect(equalsWithAnyTolerance(A[195][0], 0.0), true);
      expect(equalsWithAnyTolerance(A[378][0], 0.109), true);
      expect(equalsWithAnyTolerance(A[505][505], 982), true);
      expect(equalsWithAnyTolerance(A[999][999], 149), true);
    });
    test("From mtx", () {
      Matrix A = MatrixHelper.fromMtx("dati/spa2.mtx");
      expect(A.columnCount, 3000);
      expect(A.rowCount, 3000);
      expect(A.isSquare, true);
      expect(A.isNotEmpty, true);
      expect(equalsWithAnyTolerance(A[370][0], 0.125), true);
      expect(equalsWithAnyTolerance(A[1720][2291], 0.156), true);
    });
  });
}
