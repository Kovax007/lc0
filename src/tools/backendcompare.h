/*
  Fidelity comparison of two backends on the same network and positions.
  Added for the lc0ex/FP8 backend lane (§4.6 of DESIGN_lc0ex_fp8_backend_0903).

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
