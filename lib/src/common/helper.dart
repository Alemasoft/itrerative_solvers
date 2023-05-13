import 'dart:io';
import 'dart:typed_data';

import 'package:ml_linalg/linalg.dart';

import 'exception.dart';

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

  static Matrix fromMtx(String path, {DType dType = DType.float32}) {
    File file = File(path);
    List<String> lines = file.readAsLinesSync();
    List<String> header = lines[0].split("  ");
    int rows = int.tryParse(header[0]) ?? 0;
    int cols = int.tryParse(header[1]) ?? 0;
    int entries = int.tryParse(header[2]) ?? 0;
    List<Float64List> matrix = List.generate(rows, (_) => Float64List(cols));
    if (entries == 0) return Matrix.fromList(matrix, dtype: dType);
    for (int i = 1; i <= entries; i++) {
      List<String> line = lines[i].split("  ");
      int row = int.parse(line[0]) - 1;
      int col = int.parse(line[1]) - 1;
      double value = double.parse(line[2]);
      matrix[row][col] = value;
    }
    return Matrix.fromList(matrix, dtype: dType);
  }
}
