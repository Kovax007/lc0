/*
  Fidelity comparison of two backends on the same network and positions.
  Added for the lc0ex backend lane: the correctness gate for chunked and
  quantised artifacts, compared position by position against a reference.

  The built-in "check" backend cannot serve this purpose: it is registered
  through the legacy NetworkFactory and cannot instantiate backends that
  register through BackendManager (lc0ex-cuda among them).
*/

#pragma once

namespace lczero {

class BackendCompare {
 public:
  void Run();
};

}  // namespace lczero
