const double tolerance4 = 1e-4;
const double tolerance6 = 1e-6;
const double tolerance8 = 1e-8;
const double tolerance10 = 1e-10;
const double _maxIterations = 20000;

const List<double> tolerances = [
  tolerance4,
  tolerance6,
  tolerance8,
  tolerance10
];

class SolverConfig {
  double tolerance;

  double maxIterations;

  bool verbose = false;

  SolverConfig({double? maxIterations, double? tolerance, bool? verbose})
      : maxIterations = maxIterations ?? _maxIterations,
        tolerance = tolerance ?? tolerance4,
        verbose = verbose ?? false;

  SolverConfig.tolerance4({double? maxIterations, bool? verbose})
      : maxIterations = maxIterations ?? _maxIterations,
        tolerance = tolerance4,
        verbose = verbose ?? false;

  SolverConfig.tolerance6({double? maxIterations, bool? verbose})
      : maxIterations = maxIterations ?? _maxIterations,
        tolerance = tolerance6,
        verbose = verbose ?? false;

  SolverConfig.tolerance8({double? maxIterations, bool? verbose})
      : maxIterations = maxIterations ?? _maxIterations,
        tolerance = tolerance8,
        verbose = verbose ?? false;

  SolverConfig.tolerance10({double? maxIterations, bool? verbose})
      : maxIterations = maxIterations ?? _maxIterations,
        tolerance = tolerance10,
        verbose = verbose ?? false;

  @override
  String toString() {
    return 'SolverConfig{tolerance: $tolerance, maxIterations: $maxIterations, verbose: $verbose}';
  }
}
