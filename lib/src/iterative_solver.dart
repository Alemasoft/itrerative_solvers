import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

abstract class IterativeSolver {
  int get iterations;

  Vector get solution;

  double get relativeError;

  DateTime get startTime;

  DateTime get endTime;

  Duration get duration;

  String? get error;

  bool get hasError => error != null;

  Vector solve({
    required Matrix a,
    required Vector b,
    required Vector x,
  });
}
