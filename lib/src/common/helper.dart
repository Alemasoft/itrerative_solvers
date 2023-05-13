import 'dart:io';
import 'dart:typed_data';

import 'package:ml_linalg/linalg.dart';

import 'exception.dart';

extension MtxMatrix on Matrix {
  static Matrix fromFile(String path, {DType dType = DType.float64}) {
    File file = File(path);
    List<String> lines = file.readAsLinesSync();
    List<String> header = lines[0].split("  ");
    int? rows = int.tryParse(header[0]);
    int? cols = int.tryParse(header[1]);
    int? entries = int.tryParse(header[2]);
    if (rows == null || cols == null || entries == null) {
      throw IterativeSolverException("Invalid header in mtx file");
    }
    if (rows == 0 || cols == 0) {
      return Matrix.empty(dtype: dType);
    }
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

extension MatrixCasting on Matrix {
  Matrix to32() {
    if (dtype == DType.float32) return this;
    return Matrix.fromFlattenedList(asFlattenedList, rowCount, columnCount,
        dtype: DType.float32);
  }

  Matrix to64() {
    if (dtype == DType.float64) return this;
    return Matrix.fromFlattenedList(asFlattenedList, rowCount, columnCount,
        dtype: DType.float64);
  }
}
