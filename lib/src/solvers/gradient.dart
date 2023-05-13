import 'dart:async';

import 'package:logging/logging.dart';
import 'package:ml_linalg/linalg.dart';

import '../common/exception.dart';
import '../iterative_solver.dart';
import '../solver_config.dart';

class GradientSolver implements IterativeSolver {
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

  GradientSolver([SolverConfig? solverConfig])
      : _config = SolverConfig(),
        _iterations = 0,
        _solution = Vector.empty(dtype: DType.float64),
        _relativeError = 0,
        _startTime = DateTime.now(),
        _endTime = DateTime.now(),
        _duration = Duration.zero,
        _error = "Not runned yet",
        _logger = Logger("GradientSolver") {
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

  @override
  Vector solve({required Matrix a, required Vector b, Vector? x}) {
    _logger.fine("New solve call");
    _logger.fine("config", _config.toString());
    _clear();
    int n = b.length;
    Vector x0 = Vector.zero(n, dtype: DType.float64);
    Matrix r = Matrix.column(b.toList(), dtype: DType.float64) - (a * x0);
    int k = 0;
    bool convergenceReached = false;
    _startTimer();
    while (!convergenceReached) {
      _logger.fine("Iteration $k");
      Matrix gamma = r.transpose() * r / (r.transpose() * (a * r).toVector());
      x0 = x0 + gamma * r;
      convergenceReached = _checkConvergence(r.toVector(), k);
      _logger.fine("Convergence: $convergenceReached");
      if (convergenceReached) break;
      r = r - gamma * (a * r).toVector();
      k++;
      _iterationStreamController.add(k);
      _solutionStreamController.add(x0);
    }
    _endTimer();
    _iterations = k;
    _solution = x0;
    _logger.info("Solver solution", x0);
    if (x != null) {
      _relativeError = _calculateRelativeError(x, x0);
      _logger.info("Relative error", _relativeError);
    }
    return x0;
  }

  bool _checkConvergence(Vector residual, int k) {
    if (k == config.maxIterations) {
      _error =
          "Convergence not reached in $k iterations out of ${config.maxIterations}";
      _logger.severe(_error);
      throw IterativeSolverException(_error!);
    }
    double res = residual.norm(Norm.euclidean, true);
    _logger.fine("Residual", res);
    return res < config.tolerance;
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
