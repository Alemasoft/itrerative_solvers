//Some common tolerance values
const double tolerance10 = 1e-10;
const double tolerance12 = 1e-12;
const double tolerance14 = 1e-14;
const double tolerance16 = 1e-16;
const double tolerance18 = 1e-18;
const double tolerance6 = 1e-6;
const double tolerance8 = 1e-8;

/// Stores the default configuration for the iterative solvers.
///
/// The default configuration is:
/// ```dart
/// IterativeSolversConfig.defaultConfig = IterativeSolversConfig(
///  maxIterations: 100,
///  tolerance: 1e-6,
///  verbose: false,
///  );
///  ```
///  You can change the default configuration by changing the values of the
///  [IterativeSolversConfig.defaultConfig] object.
///  ```dart
///  IterativeSolversConfig.defaultConfig.maxIterations = 1000;
///  IterativeSolversConfig.defaultConfig.tolerance = 1e-8;
///  IterativeSolversConfig.defaultConfig.verbose = true;
///  ```
///  The [maxIterations] and [tolerance] parameters are used to determine when
///  the iterative solver should stop iterating. The [verbose] parameter is used
///  to print the iteration number and the residual norm at each iteration.
///
class IterativeSolversConfig {
  /// The default configuration for the iterative solvers.
  static IterativeSolversConfig _defaultConfig = IterativeSolversConfig(
    maxIterations: 100,
    tolerance: 1e-6,
    verbose: false,
  );

  static IterativeSolversConfig get defaultConfig => _defaultConfig;

  /// Sets the default configuration for the iterative solvers.
  ///
  /// The [config] must have [maxIterations] and [tolerance] greater than 0.
  static set defaultConfig(IterativeSolversConfig config) {
    if (config.maxIterations <= 0) {
      throw ArgumentError('maxIterations must be greater than 0');
    }
    if (config.tolerance <= 0) {
      throw ArgumentError('tolerance must be greater than 0');
    }
    _defaultConfig = config;
  }

  /// The maximum number of iterations.
  int maxIterations;

  /// The tolerance used to determine when the iterative solver should stop
  double tolerance;

  /// If true, the iteration number and the residual norm will be printed at
  /// each iteration.
  bool verbose;

  /// Creates a new [IterativeSolversConfig] object.
  IterativeSolversConfig({
    required this.maxIterations,
    required this.tolerance,
    required this.verbose,
  });

  /// Creates a new [IterativeSolversConfig] object from a [json] object.
  factory IterativeSolversConfig.fromJson(Map<String, dynamic> json) {
    return IterativeSolversConfig(
      maxIterations: json['maxIterations'],
      tolerance: json['tolerance'],
      verbose: json['verbose'],
    );
  }

  /// Returns a [Map] representation of this [IterativeSolversConfig] object.
  Map<String, dynamic> toJson() {
    return {
      'maxIterations': maxIterations,
      'tolerance': tolerance,
      'verbose': verbose,
    };
  }

  /// Returns a [String] representation of this [IterativeSolversConfig] object.
  ///
  /// The [String] representation is in the form:
  /// ```dart
  /// "IterativeSolversConfig{maxIterations: $maxIterations, tolerance: $tolerance, verbose: $verbose}"
  /// ```
  /// where $maxIterations, $tolerance and $verbose are the values of the
  /// [maxIterations], [tolerance] and [verbose] parameters, respectively.
  @override
  String toString() {
    return 'IterativeSolversConfig{maxIterations: $maxIterations, tolerance: $tolerance, verbose: $verbose}';
  }
}
