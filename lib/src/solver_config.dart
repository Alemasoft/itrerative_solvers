const double _tolerance4 = 1e-4;
const double _tolerance6 = 1e-6;
const double _tolerance8 = 1e-8;
const double _tolerance10 = 1e-10;
const double _maxIterations = 20000;

const List<double> tolerances = [
  _tolerance4,
  _tolerance6,
  _tolerance8,
  _tolerance10
];

class SolverConfig {
  double tolerance;

  double maxIterations;

  bool verbose = false;

  SolverConfig({double? maxIterations, double? tolerance, bool? verbose})
      : maxIterations = maxIterations ?? _maxIterations,
        tolerance = tolerance ?? _tolerance4,
        verbose = verbose ?? false;

  SolverConfig.tolerance4({double? maxIterations, bool? verbose})
      : maxIterations = maxIterations ?? _maxIterations,
        tolerance = _tolerance4,
        verbose = verbose ?? false;

  SolverConfig.tolerance6({double? maxIterations, bool? verbose})
      : maxIterations = maxIterations ?? _maxIterations,
        tolerance = _tolerance6,
        verbose = verbose ?? false;

  SolverConfig.tolerance8({double? maxIterations, bool? verbose})
      : maxIterations = maxIterations ?? _maxIterations,
        tolerance = _tolerance8,
        verbose = verbose ?? false;

  SolverConfig.tolerance10({double? maxIterations, bool? verbose})
      : maxIterations = maxIterations ?? _maxIterations,
        tolerance = _tolerance10,
        verbose = verbose ?? false;

  @override
  String toString() {
    return 'SolverConfig{tolerance: $tolerance, maxIterations: $maxIterations, verbose: $verbose}';
  }
}
