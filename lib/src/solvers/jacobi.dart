import 'dart:async';

import 'package:logging/logging.dart';
import 'package:ml_linalg/linalg.dart';

import '../common/exception.dart';
import '../iterative_solver.dart';
import '../solver_config.dart';

class JacobiSolver implements IterativeSolver {
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

  JacobiSolver([SolverConfig? solverConfig])
      : _config = SolverConfig(),
        _iterations = 0,
        _solution = Vector.empty(dtype: DType.float64),
        _relativeError = 0,
        _startTime = DateTime.now(),
        _endTime = DateTime.now(),
        _duration = Duration.zero,
        _error = "Not runned yet",
        _logger = Logger("JacobiSolver") {
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
    _startTimer();
    int k = 0;
    bool convergenceReached = false;
    Vector solution = Vector.zero(b.length, dtype: DType.float64);
    while (!convergenceReached) {
      _logger.fine("Iteration $k");
      Vector xk = Vector.fromList(solution.toList(), dtype: DType.float64);
      for (int i = 0; i < b.length; i++) {
        double sigma = 0;
        for (int j = 0; j < b.length; j++) {
          if (j != i) {
            sigma += a[i][j] * xk[j];
          }
        }
        solution = solution.set(i, (b[i] - sigma) / a[i][i]);
      }
      _logger.fine("Solution", solution.sum());
      k++;
      _iterationStreamController.add(k);
      _solutionStreamController.add(solution);
      convergenceReached = _checkConvergence(solution, a, b, k);
      _logger.fine("Convergence: $convergenceReached");
    }
    _endTimer();
    _iterations = k;
    _solution = solution;
    _logger.info("Solver solution", solution);
    if (x != null) {
      _relativeError = _calculateRelativeError(x, _solution);
      _logger.info("Relative error", _relativeError);
    }
    return solution;
  }

  bool _checkConvergence(Vector x, Matrix a, Vector b, int k) {
    if (x == b) return true;
    if (k == config.maxIterations) {
      _error =
          "Convergence not reached in $k iterations out of ${config.maxIterations}";
      _logger.severe(_error);
      throw IterativeSolverException(_error!);
    }
    double res = ((a * x).toVector() - b).norm(Norm.euclidean, true) /
        b.norm(Norm.euclidean, true);
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
