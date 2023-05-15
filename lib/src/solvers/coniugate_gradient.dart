import 'dart:async';
import 'dart:math';

import 'package:iterative_solvers/src/common/helper.dart';
import 'package:logging/logging.dart';
import 'package:ml_linalg/linalg.dart';

import '../common/exception.dart';
import '../iterative_solver.dart';
import '../solver_config.dart';

class ConjGradientSolver implements IterativeSolver {
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

  ConjGradientSolver([SolverConfig? solverConfig])
      : _config = SolverConfig(),
        _iterations = 0,
        _solution = Vector.empty(dtype: DType.float64),
        _relativeError = 0,
        _startTime = DateTime.now(),
        _endTime = DateTime.now(),
        _duration = Duration.zero,
        _error = "Not runned yet",
        _logger = Logger("ConjGradientSolver") {
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
    } else {
      hierarchicalLoggingEnabled = true;
      _logger.level = Level.INFO;
      _logger.clearListeners();
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
    _logger.fine("Solver cleared");
  }

  @override
  Vector solve({required Matrix a, required Vector b, Vector? x}) {
    _logger.info("Solver opened");
    _logger.info("config", _config.toString());
    _clear();
    final int n = b.length;
    Vector x0 = Vector.zero(n, dtype: DType.float64);
    _startTimer();
    Vector r = b - a * x0;
    Vector p = r;
    double rsold = r.dot(r);
    int k = 0;
    try {
      for (int i = 0; i < b.length; i++) {
        if (k == config.maxIterations) {
          _error =
              "Convergence not reached in $k iterations out of ${config.maxIterations}";
          _logger.severe(_error);
          throw IterativeSolverException(_error!);
        }
        k = i + 1;
        _iterationStreamController.add(k);
        _logger.fine("Iteration $k");
        Vector ap = (a * p).to64().toVector();
        double alpha = rsold / (p.dot(ap));
        x0 = x0 + p * alpha;
        r = r - ap * alpha;
        double rsnew = r.dot(r);
        _solutionStreamController.add(x0);
        _logger.fine("Residual", sqrt(rsnew));
        if (sqrt(rsnew) < config.tolerance) {
          _logger.fine("Convergence: true");
          break;
        }
        _logger.fine("Convergence: false");
        p = r + p * (rsnew / rsold);
        rsold = rsnew;
      }
    } catch (e) {
      _error = e.toString();
      _logger.severe(_error);
      throw IterativeSolverException(_error!);
    } finally {
      _endTimer();
      _iterations = k;
      _solution = x0;
      _logger.info("Solver solution", x0);
      _logger.info("Solver iterations", _iterations);
      if (x != null) {
        _relativeError = _calculateRelativeError(x, x0);
        _logger.info("Relative error", _relativeError);
      }
    }
    _logger.info("Solver closed\n\n");
    return x0;
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
