import 'package:iterative_solvers/src/helper.dart';
import 'package:iterative_solvers/src/iterative_solver.dart';
import 'package:iterative_solvers/src/solver_config.dart';
import 'package:logging/logging.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

import 'exception.dart';

class JacobiSolver implements IterativeSolver {
  SolverConfig _config;

  Vector _r;

  int _iterations;

  Vector _solution;

  double _relativeError;

  DateTime _startTime;

  DateTime _endTime;

  Duration _duration;

  String? _error;

  final Logger _logger;

  JacobiSolver([SolverConfig? solverConfig])
      : _config = SolverConfig(),
        _r = Vector.empty(),
        _iterations = 0,
        _solution = Vector.empty(),
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

  bool _canRun(Vector b) {
    if (iterations >= _config.maxIterations) {
      throw IterativeSolverException("Convergence not reached");
    }
    if (_r.isEmpty) return true;
    if (_r.norm() / b.norm() >= _config.tolerance) return false;
    return true;
  }

  void _clear() {
    _r = Vector.empty();
    _iterations = 0;
    _solution = Vector.empty();
    _relativeError = 0;
    _error = null;
    _logger.info("Solver cleared");
  }

  @override
  Vector solve({required Matrix a, required Vector b, Vector? x}) {
    try {
      _logger.info("\n\nNew run\n\n");
      _clear();
      _logger.info("Current config", config);
      _logger.info("Matrix A", a);
      _logger.info("Vector b", b);
      _solution = Vector.zero(b.length);
      Matrix P = Matrix.diagonal(a.diagonal);
      _logger.info("Matrix P", P);
      Matrix N = (a * -1) - P;
      _logger.info("Matrix N", N);
      _startTimer();
      while (_canRun(b)) {
        _r = (a * _solution).toVector() + b;
        _solution += P.inverse() * _r;
        _logger.fine("Iteration (${iterations + 1})", _solution);
        _iterations++;
      }
    } catch (e) {
      _logger.severe(e.toString());
      _error = e.toString();
    } finally {
      _endTimer();
    }
    if (!hasError) {
      _logger.info("Solver found this solution", solution);
      if (x != null) {
        _relativeError = (_solution - x).norm() / x.norm();
        _logger.info("Correct solution was", x);
        _logger.info("Solver ended with relative error", _relativeError);
      }
    }
    return solution;
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
