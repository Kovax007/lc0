/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#pragma once

#include <memory>

#include "neural/backend.h"
#include "utils/optionsdict.h"

namespace lczero {

// Creates a backend wrapper that merges the batches of concurrent callers into
// a single parent computation, the way the legacy `multiplexing` network does
// for old-API networks. Selfplay evaluates one small batch per game thread —
// mean NN batch 11 with 24 games — which leaves every kernel latency-bound and
// re-reads the whole weight set per evaluation. Merging trades a little
// per-game latency, which datagen does not care about, for a batch the GPU can
// actually fill.
//
// `max_batch` caps the merged batch. `wait_us` is how long a worker lingers for
// more work before running a batch it could already run; 0 runs immediately.
// `threads` is the number of merging workers.
std::unique_ptr<Backend> CreateBatchMergingBackend(
    std::unique_ptr<Backend> parent, int max_batch, int wait_us, int threads);

// Wraps `parent` according to the shared backend options, or returns it
// unchanged when batch merging is switched off (the default).
std::unique_ptr<Backend> MaybeWrapWithBatchMerging(
    std::unique_ptr<Backend> parent, const OptionsDict& opts);

}  // namespace lczero
