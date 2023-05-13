class IterativeSolverException implements Exception {
  IterativeSolverException(this.message);

  final String message;

  @override
  String toString() => message;
}
