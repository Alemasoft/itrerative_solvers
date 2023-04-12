import 'dart:math';

/// A generic vector class that can be used to represent a vector of any type.
///
/// [T] is the type of the vector. It can be any type that extends [num].
///
/// ```dart
/// Vector<int> v = Vector<int>.fromList([1, 2, 3]);
/// Vector<double> v = Vector<double>.fromList([1.0, 2.0, 3.0]);
/// Vector<num> v = Vector<num>.fromList([1, 2.0, 3]);
/// ```
class Vector<T extends num> {
  List<T> _data = [];

  Vector.empty() {
    _data = [];
  }

  Vector.filled(int length, T fillValue) {
    _data = List<T>.filled(length, fillValue);
  }

  factory Vector.fromIterable(Iterable<T> iterable) {
    return Vector<T>._(iterable.toList());
  }

  factory Vector.fromList(List<T> list) {
    return Vector<T>._(list);
  }

  Vector.random(int length, T Function(int index) generator) {
    _data = List<T>.generate(length, generator);
  }

  Vector.zero(int length) : this.filled(length, 0 as T);

  Vector._(List<T> list) {
    _data = list;
  }

  T get first {
    return _data.first;
  }

  set first(T value) {
    _data.first = value;
  }

  bool get isEmpty => _data.isEmpty;

  bool get isNotEmpty => _data.isNotEmpty;

  bool get isPositive => _data.every((element) => element > 0);

  Iterator<T> get iterator => _data.iterator;

  T get last {
    return _data.last;
  }

  set last(T value) {
    _data.last = value;
  }

  int get length => _data.length;

  set length(int value) {
    if (value < 0) {
      throw RangeError("Vector length cannot be negative");
    }
    _data.length = value;
  }

  Iterable<T> get reversed => _data.reversed;

  T get single => _data.single;

  Vector<T> operator *(Vector<T> other) {
    for (int i = 0; i < length; i++) {
      _data[i] = _data[i] * other._data[i] as T;
    }
    return Vector.fromList(_data);
  }

  Vector<T> operator +(Vector<T> other) {
    return Vector.fromList(_data + other._data);
  }

  T operator [](int index) {
    return _data[index];
  }

  void operator []=(int index, T value) {
    _data[index] = value;
  }

  void add(T value) {
    List<T> list = List<T>.filled(length, value);
    _data = _data + list;
  }

  void addAll(Vector<T> other) {
    for (T val in other._data) {
      add(val);
    }
  }

  bool any(bool Function(T element) test) {
    return _data.any(test);
  }

  void append(T value) {
    _data.add(value);
  }

  void appendAll(Iterable<T> iterable) {
    _data.addAll(iterable);
  }

  Map<int, T> asMap() {
    return _data.asMap();
  }

  List cast<R>() {
    return _data.cast<R>();
  }

  void clear() {
    return _data.clear();
  }

  bool contains(Object? element) {
    return _data.contains(element);
  }

  Vector<T> crossProduct(Vector<T> other) {
    if (length != 3 || other.length != 3) {
      throw Exception("Cross product is only defined for 3D vectors");
    }
    if (length != other.length) {
      throw Exception("Vectors must have the same length");
    }
    return Vector.fromList([
      _data[1] * other._data[2] - _data[2] * other._data[1] as T,
      _data[2] * other._data[0] - _data[0] * other._data[2] as T,
      _data[0] * other._data[1] - _data[1] * other._data[0] as T,
    ]);
  }

  T elementAt(int index) {
    return _data.elementAt(index);
  }

  bool every(bool Function(T element) test) {
    return every(test);
  }

  Iterable<T> expand(Iterable<T> Function(T element) toElements) {
    return _data.expand(toElements);
  }

  void fillRange(int start, int end, [T? fillValue]) {
    _data.fillRange(start, end, fillValue);
  }

  T firstWhere(bool Function(T element) test, {T Function()? orElse}) {
    return _data.firstWhere(test, orElse: orElse);
  }

  T fold(T initialValue, T Function(T previousValue, T element) combine) {
    return _data.fold(initialValue, combine);
  }

  Iterable<T> followedBy(Iterable<T> other) {
    return _data.followedBy(other);
  }

  void forEach(void Function(T element) action) {
    _data.forEach(action);
  }

  Iterable<T> getRange(int start, int end) {
    return _data.getRange(start, end);
  }

  int indexOf(T element, [int start = 0]) {
    return _data.indexOf(element, start);
  }

  int indexWhere(bool Function(T element) test, [int start = 0]) {
    return _data.indexWhere(test, start);
  }

  void insert(int index, T element) {
    return _data.insert(index, element);
  }

  void insertAll(int index, Iterable<T> iterable) {
    return _data.insertAll(index, iterable);
  }

  String join([String separator = ""]) {
    return _data.join(separator);
  }

  int lastIndexOf(T element, [int? start]) {
    return _data.lastIndexOf(element, start);
  }

  int lastIndexWhere(bool Function(T element) test, [int? start]) {
    return _data.lastIndexWhere(test, start);
  }

  T lastWhere(bool Function(T element) test, {T Function()? orElse}) {
    return _data.lastWhere(test, orElse: orElse);
  }

  Iterable<T> map(T Function(T e) toElement) {
    return _data.map(toElement);
  }

  T norm() {
    T sum = 0 as T;
    for (int i = 0; i < length; i++) {
      sum = sum + (_data[i] * _data[i]) as T;
    }
    return sqrt(sum) as T;
  }

  T norm1() {
    T sum = 0 as T;
    for (int i = 0; i < length; i++) {
      sum = sum + _data[i].abs() as T;
    }
    return sum;
  }

  Vector<T> normalize() {
    T norm = this.norm();
    for (int i = 0; i < length; i++) {
      _data[i] = _data[i] / norm as T;
    }
    return Vector.fromList(_data);
  }

  Vector<T> normalize1() {
    T norm = this.norm1();
    for (int i = 0; i < length; i++) {
      _data[i] = _data[i] / norm as T;
    }
    return Vector.fromList(_data);
  }

  Vector<T> normalizeInf() {
    T norm = this.normInf();
    for (int i = 0; i < length; i++) {
      _data[i] = _data[i] / norm as T;
    }
    return Vector.fromList(_data);
  }

  T normInf() {
    T max = 0 as T;
    for (int i = 0; i < length; i++) {
      if (_data[i].abs() > max) {
        max = _data[i].abs() as T;
      }
    }
    return max;
  }

  T reduce(T Function(T value, T element) combine) {
    return _data.reduce(combine);
  }

  bool remove(Object? value) {
    return _data.remove(value);
  }

  T removeAt(int index) {
    return _data.removeAt(index);
  }

  T removeLast() {
    return _data.removeLast();
  }

  void removeRange(int start, int end) {
    _data.removeRange(start, end);
  }

  void removeWhere(bool Function(T element) test) {
    return _data.removeWhere(test);
  }

  void replaceRange(int start, int end, Iterable<T> replacements) {
    return _data.replaceRange(start, end, replacements);
  }

  void retainWhere(bool Function(T element) test) {
    return _data.retainWhere(test);
  }

  T scalarProduct(Vector<T> other) {
    T sum = 0 as T;
    for (int i = 0; i < length; i++) {
      sum = sum + (_data[i] * other._data[i]) as T;
    }
    return sum;
  }

  void setAll(int index, Iterable<T> iterable) {
    _data.setAll(index, iterable);
  }

  void setRange(int start, int end, Iterable<T> iterable, [int skipCount = 0]) {
    _data.setRange(start, end, iterable, skipCount);
  }

  void shuffle([Random? random]) {
    _data.shuffle(random);
  }

  T singleWhere(bool Function(T element) test, {T Function()? orElse}) {
    return _data.singleWhere(test, orElse: orElse);
  }

  Iterable<T> skip(int count) {
    return _data.skip(count);
  }

  Iterable<T> skipWhile(bool Function(T value) test) {
    return _data.skipWhile(test);
  }

  void sort([int Function(T a, T b)? compare]) {
    _data.sort(compare);
  }

  List<T> sublist(int start, [int? end]) {
    return sublist(start, end);
  }

  Iterable<T> take(int count) {
    return _data.take(count);
  }

  Iterable<T> takeWhile(bool Function(T value) test) {
    return _data.takeWhile(test);
  }

  List<T> toList({bool growable = true}) {
    return _data.toList(growable: growable);
  }

  Set<T> toSet() {
    return _data.toSet();
  }

  Iterable<T> where(bool Function(T element) test) {
    return _data.where(test);
  }

  Iterable whereType<Y>() {
    return _data.whereType<Y>();
  }
}

extension VectorExtension on List<num> {
  Vector toVector() {
    return Vector.fromList(this);
  }
}
