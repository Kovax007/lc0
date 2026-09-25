/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2021 The LCZero Authors

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

#include "neural/backend.h"
#include "search/classic/node.h"
#include "trainingdata/trainingdata_v6.h"
#include "trainingdata/childdata.h"
#include "trainingdata/trainingdata_v7.h"
#include "trainingdata/trainingdata_v8.h"
#include "trainingdata/writer.h"

namespace lczero {

class V6TrainingDataArray {
 public:
  // `emit_v7` writes 8396-byte V7 records instead of 8356-byte V6 ones.
  // `emit_v8` writes 11036-byte V8 records: the V7 record byte for byte plus
  // the 128-slot per-legal-move table (implies emit_v7).
  // `record_child_q` additionally packs the search's top-K per-move Q into
  // reserved[4..7] (requires emit_v7); it stays on under V8 so the two
  // independently written encoders cross-check each other on every record.
  // All default OFF, so a binary built from this tree reproduces cv2 byte for
  // byte unless asked otherwise.
  V6TrainingDataArray(FillEmptyHistory white_fill_empty_history,
                      FillEmptyHistory black_fill_empty_history,
                      pblczero::NetworkFormat::InputFormat input_format,
                      bool emit_v7 = false, bool record_child_q = false,
                      int child_q_min_visits = 1, bool emit_v8 = false,
                      int recipe_id = 0)
      : fill_empty_history_{white_fill_empty_history, black_fill_empty_history},
        input_format_(input_format),
        emit_v7_(emit_v7 || emit_v8),
        emit_v8_(emit_v8),
        record_child_q_(record_child_q && (emit_v7 || emit_v8)),
        child_q_min_visits_(child_q_min_visits < 1 ? 1 : child_q_min_visits),
        recipe_id_(recipe_id) {}

  // Add a chunk.
  // `root_data`, when non-null and emit_v8 is on, supplies the per-child
  // search data for the V8 block. It must be aligned with `legal_moves`.
  void Add(const classic::Node* node, const PositionHistory& history,
           classic::Eval best_eval, classic::Eval played_eval,
           bool best_is_proven, Move best_move, Move played_move,
           const std::vector<std::tuple<float, float>>& visits,
           std::span<Move> legal_moves,
           const std::optional<EvalResult>& nneval,
           const RootSearchData* root_data = nullptr);

  // Writes training data to a file.
  void Write(TrainingDataWriter* writer, GameResult result,
             bool adjudicated) const;

 private:
  // Fills the V8 block (header + 128-slot table) of an already-populated V7
  // prefix. Throws on a move-frame mismatch rather than writing a record
  // whose indices nobody can trust.
  void FillV8Block(V8TrainingData* out, const RootSearchData* root_data,
                   std::span<Move> legal_moves, int transform);

  // Always stored as V8; Write() emits the V7 or V6 prefix when the
  // corresponding flag is off, which is byte-identical to the older writers.
  // The unconditional 11 KB record costs ~30% more in-flight RAM than V7 even
  // when V8 is off; at 24 parallel games that is ~13 MB per selfplay process,
  // which is cheaper than carrying two record types through this class.
  std::vector<V8TrainingData> training_data_;
  FillEmptyHistory fill_empty_history_[2];
  pblczero::NetworkFormat::InputFormat input_format_;
  bool emit_v7_ = false;
  bool emit_v8_ = false;
  bool record_child_q_ = false;
  // Write-time visit floor: a child below it is never a candidate. Not
  // recoverable after the fact -- a V7 record stores no per-child visit
  // count, so this is the only confidence control the format allows.
  int child_q_min_visits_ = 1;
  // Stamped into every V8 record. The flags alone do not identify a recipe.
  int recipe_id_ = 0;
};

}  // namespace lczero
