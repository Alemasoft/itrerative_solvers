import 'dart:async';

import 'package:logging/logging.dart';
import 'package:ml_linalg/linalg.dart';

import '../../iterative_solvers.dart';
import '../iterative_solver.dart';

class GaussSeidelSolver implements IterativeSolver {
  SolverConfig _config;

  int _iterations;

  Vector _solution;

  double _relativeError;

  DateTime _startTime;

  DateTime _endTime;

  Duration _duration;

  String? _error;

  final StreamController<Vector> _solutionStreamController =
      StreamController<Vector>.broadcast();

  Stream<Vector> get solutionStream => _solutionStreamController.stream;

  final StreamController<int> _iterationStreamController =
      StreamController<int>.broadcast();

  Stream<int> get iterationStream => _iterationStreamController.stream;

  final Logger _logger;

  GaussSeidelSolver([SolverConfig? solverConfig])
      : _config = SolverConfig(),
        _iterations = 0,
        _solution = Vector.empty(dtype: DType.float64),
        _relativeError = 0,
        _startTime = DateTime.now(),
        _endTime = DateTime.now(),
        _duration = Duration.zero,
        _error = "Not runned yet",
        _logger = Logger("GaussSeidelSolver") {
    config = solverConfig;
  }

  SolverConfig get config => _config;

  @override
  int get iterations => _iterations;

  @override
  Vector get solution => _solution;

  @override
  double get relativeError => _relativeError;

  @override
  DateTime get startTime => _startTime;

  @override
  DateTime get endTime => _endTime;

  @override
  Duration get duration => _duration;

  set config(SolverConfig? c) {
    if (c == null) return;
    _config = c;
    _logger.clearListeners();
    if (c.verbose) {
      hierarchicalLoggingEnabled = true;
      _logger.level = Level.ALL;
      _logger.onRecord.listen((record) {
        print(
            '${record.loggerName} | ${record.message}${record.error != null ? ": ${record.error}" : ""}');
      });
    }
  }

  void _clear() {
    _iterations = 0;
    _solution = Vector.empty(dtype: DType.float64);
    _relativeError = 0;
    _error = null;
    _logger.info("Solver cleared");
  }

  bool _checkConvergence(Vector phi, Vector previousPhi, int k) {
    if (k == config.maxIterations) {
      _error =
          "Convergence not reached in $k iterations out of ${config.maxIterations}";
      _logger.severe(_error);
      throw IterativeSolverException(_error!);
    }
    double res = (phi - previousPhi).norm();
    _logger.fine("Residual", res);
    return (res < config.tolerance);
  }

  @override
  Vector solve({required Matrix a, required Vector b, Vector? x}) {
    _logger.fine("New solve call");
    _logger.fine("config", _config.toString());
    _clear();
    int k = 0;
    _startTimer();
    int n = b.length;
    Vector phi = Vector.zero(n, dtype: DType.float64);
    bool convergenceReached = false;
    while (!convergenceReached) {
      _logger.fine("Iteration $k");
      Vector xk = Vector.fromList(phi.toList(), dtype: DType.float64);
      for (int i = 0; i < n; i++) {
        double sigma = 0;
        for (int j = 0; j < n; j++) {
          if (j != i) {
            sigma += a[i][j] * phi[j];
          }
        }
        phi = phi.set(i, (b[i] - sigma) / a[i][i]);
      }
      k++;
      _iterationStreamController.add(k);
      _solutionStreamController.add(phi);
      convergenceReached = _checkConvergence(phi, xk, k);
      _logger.fine("Convergence: $convergenceReached");
    }
    _endTimer();
    _iterations = k;
    _solution = phi;
    _logger.info("Solver solution", phi);
    if (x != null) {
      _relativeError = _calculateRelativeError(x, phi);
      _logger.info("Relative error", _relativeError);
    }
    return phi;
  }

  double _calculateRelativeError(
      Vector correctSolution, Vector approximatedSolution) {
    double error =
        (correctSolution - approximatedSolution).norm(Norm.euclidean, true) /
            correctSolution.norm(Norm.euclidean, true);
    return error;
  }

  @override
  String? get error => _error;

  @override
  bool get hasError => _error != null;

  void _startTimer() {
    _startTime = DateTime.now();
    _logger.info("Timer started", _startTime.toIso8601String());
  }

  void _endTimer() {
    _endTime = DateTime.now();
    _duration = _endTime.difference(_startTime);
    _logger.info("Timer ended", _endTime.toIso8601String());
    _logger.info("Solver ended with elapsed time", duration.toString());
  }
}
