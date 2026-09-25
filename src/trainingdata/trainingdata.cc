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

#include "trainingdata/trainingdata.h"

#include <algorithm>
#include <numeric>
#include <vector>

#include "trainingdata/childq.h"
#include "trainingdata/trainingdata_v8.h"
#include "utils/exception.h"

namespace lczero {

namespace {
std::tuple<float, float> DriftCorrect(float q, float d) {
  // Training data doesn't have a high number of nodes, so there shouldn't be
  // too much drift. Highest known value not caused by backend bug was 1.5e-7.
  const float allowed_eps = 0.000001f;
  if (q > 1.0f) {
    if (q > 1.0f + allowed_eps) {
      CERR << "Unexpectedly large drift in q " << q;
    }
    q = 1.0f;
  }
  if (q < -1.0f) {
    if (q < -1.0f - allowed_eps) {
      CERR << "Unexpectedly large drift in q " << q;
    }
    q = -1.0f;
  }
  if (d > 1.0f) {
    if (d > 1.0f + allowed_eps) {
      CERR << "Unexpectedly large drift in d " << d;
    }
    d = 1.0f;
  }
  if (d < 0.0f) {
    if (d < 0.0f - allowed_eps) {
      CERR << "Unexpectedly large drift in d " << d;
    }
    d = 0.0f;
  }
  float w = (1.0f - d + q) / 2.0f;
  float l = w - q;
  // Assume q drift is rarer than d drift and apply all correction to d.
  if (w < 0.0f || l < 0.0f) {
    float drift = 2.0f * std::min(w, l);
    if (drift < -allowed_eps) {
      CERR << "Unexpectedly large drift correction for d based on q. " << drift;
    }
    d += drift;
    // Since q is in range -1 to 1 - this correction should never push d outside
    // of range, but precision could be lost in calculations so just in case.
    if (d < 0.0f) {
      d = 0.0f;
    }
  }
  return {q, d};
}
}  // namespace

// Fills the V8 block: the 16-byte header plus the 128-slot per-legal-move
// table. Everything here is search-produced and recipe-independent, so a V8
// corpus can be re-targeted with any kernel this programme has built.
//
// SLOT ORDER is raw root visits descending, prior descending on ties. That is
// deliberately the SAME order the packed 6-child window uses, which is what
// makes the two-encoder cross-check (invariant v) exact and free. De-forced N
// is stored per slot, so nothing is lost by not sorting on it -- ordering
// carries no information the record does not already hold. The design doc
// specifies de-forced order and asserts the two orders coincide; they do not
// (the packer sorts on edge.GetN()), and preserving the free cross-check is
// worth more than the ordering choice.
void V6TrainingDataArray::FillV8Block(V8TrainingData* out,
                                      const RootSearchData* root_data,
                                      std::span<Move> legal_moves,
                                      int transform) {
  // The reply index is written in the CHILD's own frame with transform 0.
  // Under a canonical input format the child can carry a different board
  // transform from the root, which would silently mix frames. Refuse rather
  // than write a corpus nobody can decode.
  if (IsCanonicalFormat(input_format_)) {
    throw Exception(
        "V8 training data requires a non-canonical input format: the reply "
        "index would otherwise mix board transforms between root and child");
  }

  out->v8_layout_version = v8::kLayoutVersion;
  out->recipe_id = static_cast<uint16_t>(recipe_id_);
  out->provenance_flags = 0;  // rescorer-owned
  out->plies_until_progress = v8::kNotComputed;  // rescorer-owned

  // Every array starts fully sentinelled, so an early return still leaves a
  // well-formed "no data" block rather than a half-written one.
  for (int k = 0; k < v8::kSlots; ++k) {
    out->idx[k] = v8::kNoIdx;
    out->prior[k] = v8::kNoPrior;
    out->q[k] = v8::kNoQ;
    out->d[k] = v8::kNoD;
    out->m[k] = v8::kNoM;
    out->n_raw[k] = 0;
    out->n_deforced[k] = 0;
    out->reply_idx[k] = v8::kNoIdx;
    out->reply_q[k] = v8::kNoQ;
    out->reply_n[k] = 0;
  }

  size_t n_legal = 0;
  for (float p : out->probabilities) {
    if (p >= 0.0f) ++n_legal;
  }
  out->n_legal = static_cast<uint8_t>(std::min<size_t>(n_legal, 255));

  if (root_data == nullptr ||
      root_data->children.size() != legal_moves.size()) {
    // block_flags stays 0: "no V8 data", which §5.3 (vi) requires a reader to
    // accept. A V8 record with an empty block is valid; a V8 record with a
    // half-filled one is not.
    out->block_flags = 0;
    return;
  }

  struct Slot {
    uint16_t nn_idx;
    const RootChildData* c;
    // Index into legal_moves / root_data->children. Kept so the DAG
    // diagnostics can be permuted by the SAME sort as n_raw -- if
    // child_node_n[k] and n_raw[k] ever described different moves, the
    // undercount measurement would read as noise and nothing would flag it.
    size_t src;
  };
  std::vector<Slot> slots;
  slots.reserve(legal_moves.size());
  for (size_t j = 0; j < legal_moves.size(); ++j) {
    const uint16_t nn_idx = MoveToNNIndex(legal_moves[j], transform);
    // Same self-check the packed child-Q window does: an index the policy
    // target never filled means the move enumerations disagree, which is the
    // silent-permutation class. Refuse the record.
    if (nn_idx >= 1858 || out->probabilities[nn_idx] < 0.0f) {
      throw Exception(
          "V8 block: a legal move maps to a policy index absent from the "
          "target -- move frame or transform mismatch");
    }
    slots.push_back({nn_idx, &root_data->children[j], j});
  }

  std::stable_sort(slots.begin(), slots.end(),
                   [](const Slot& a, const Slot& b) {
                     if (a.c->n_raw != b.c->n_raw) return a.c->n_raw > b.c->n_raw;
                     return a.c->prior > b.c->prior;
                   });

  const bool overflow = slots.size() > static_cast<size_t>(v8::kSlots);
  const size_t n_stored = std::min<size_t>(slots.size(), v8::kSlots);
  out->n_stored = static_cast<uint8_t>(n_stored);

  bool any_reply = false;
  for (size_t k = 0; k < n_stored; ++k) {
    const RootChildData& c = *slots[k].c;
    out->idx[k] = slots[k].nn_idx;
    if (c.prior >= 0.0f) out->prior[k] = v8::EncodePrior(c.prior);
    out->n_raw[k] = v8::EncodeCount(c.n_raw);
    if (c.n_raw > 0) {
      out->q[k] = v8::EncodeSigned(c.q);
      out->d[k] = v8::EncodeSigned(c.d);
      if (root_data->m_present) out->m[k] = v8::EncodeM(c.m);
    }
    if (root_data->deforced_present) {
      out->n_deforced[k] = v8::EncodeCount(c.n_deforced);
    }
    if (c.reply_idx != 0xFFFF && c.reply_n > 0) {
      out->reply_idx[k] = c.reply_idx;
      out->reply_q[k] = v8::EncodeSigned(c.reply_q);
      out->reply_n[k] = v8::EncodeCount(c.reply_n);
      any_reply = true;
    }
  }

  uint16_t flags = v8::kBlockValid;
  if (root_data->prior_present) flags |= v8::kPriorPresent;
  if (root_data->deforced_present) flags |= v8::kDeforcedPresent;
  if (root_data->m_present) flags |= v8::kMPresent;
  if (root_data->fe_active) flags |= v8::kFeActive;
  if (overflow) flags |= v8::kOverflow;
  if (any_reply) flags |= v8::kReplyPresent;
  if (root_data->dag_searched) flags |= v8::kDagSearched;
  if (root_data->dag_target_corrected) flags |= v8::kDagTargetCorrected;
  if (root_data->dag.rule50_bucketed_key) flags |= v8::kDagRule50Keyed;
  out->block_flags = flags;

  // ---- DAG diagnostics into reserved_v8 (bit 9 raised by SetDagDiag) ------
  // Nothing here runs for a classic search: `present` is false, reserved_v8
  // stays zero and bit 9 stays clear, so a tree-era record is byte-identical
  // to what this writer produced before.
  if (root_data->dag.present) {
    const DagDiagnostics& dg = root_data->dag;
    V8DagDiag diag{};
    diag.dag_diag_version = 1;
    diag.key_flags =
        static_cast<uint8_t>((std::clamp(dg.cache_history_length, 0, 3) & 0x3) |
                             (dg.rule50_bucketed_key ? 0x4 : 0x0));
    diag.nn_evals = v8::EncodeCount(dg.nn_evals);
    diag.tt_hits = v8::EncodeCount(dg.tt_hits);
    diag.tt_hit_backups = v8::EncodeCount(dg.tt_hit_backups);
    diag.root_children_preexisting =
        v8::EncodeCount(dg.root_children_preexisting);

    // The six largest undercounts (n_node - n_edge), descending, reported as
    // SLOT indices so the edge visits, prior, Q and move of each entry are
    // read from the arrays the record already carries.
    //
    // Ranked by the undercount, NOT by visits: an under-counted move has few
    // edge visits by definition, so it sorts low in n_raw order and a
    // top-6-by-visits block would miss it exactly where it is worst.
    for (int k = 0; k < 6; ++k) {
      diag.undercount_slot[k] = 0xFF;
      diag.undercount_node_n[k] = 0;
    }
    if (dg.child_node_n.size() == root_data->children.size()) {
      // (deficit, slot) over the stored slots; n_node >= n_edge by definition,
      // so a negative deficit is a bug in the search, not a datum -- clamp at
      // zero and let the ranking drop it rather than storing a wrapped uint.
      std::vector<std::pair<uint32_t, size_t>> rank;
      rank.reserve(n_stored);
      for (size_t k = 0; k < n_stored; ++k) {
        const uint32_t n_node = dg.child_node_n[slots[k].src];
        const uint32_t n_edge = slots[k].c->n_raw;
        rank.emplace_back(n_node > n_edge ? n_node - n_edge : 0u, k);
      }
      std::stable_sort(rank.begin(), rank.end(),
                       [](const auto& a, const auto& b) {
                         return a.first > b.first;
                       });
      for (size_t k = 0; k < std::min<size_t>(rank.size(), 6); ++k) {
        if (rank[k].first == 0) break;  // nothing under-counted below here
        diag.undercount_slot[k] =
            static_cast<uint8_t>(std::min<size_t>(rank[k].second, 254));
        diag.undercount_node_n[k] =
            v8::EncodeCount(dg.child_node_n[slots[rank[k].second].src]);
      }
    }

    // argmax_index_by_node_n is an index into children; report it as a SLOT so
    // a consumer can compare it against slot 0 without re-deriving the sort.
    diag.argmax_slot_by_node_n = 0xFF;
    if (dg.argmax_index_by_node_n >= 0) {
      for (size_t k = 0; k < n_stored; ++k) {
        if (slots[k].src == static_cast<size_t>(dg.argmax_index_by_node_n)) {
          diag.argmax_slot_by_node_n = static_cast<uint8_t>(std::min<size_t>(k, 254));
          break;
        }
      }
    }

    diag.max_rule50_delta =
        static_cast<uint8_t>(std::min<uint32_t>(dg.max_rule50_delta, 255));
    diag.rule50_bucket_cross_hits =
        v8::EncodeCount(dg.rule50_bucket_cross_hits);
    diag.dag_nodes_at_start_div16 = v8::EncodeCount(dg.dag_nodes_at_start / 16);

    SetDagDiag(out, diag);
  }

  // Layout 2 recipe constants, in MEvaluator's constructor order after pst
  // (SPEC_v8_recipe_block_0902.md §3). Order is load-bearing -- see the header.
  out->recipe_pst = root_data->policy_softmax_temp;
  out->recipe_ml_slope = root_data->ml_slope;
  out->recipe_ml_max_effect = root_data->ml_max_effect;
  out->recipe_ml_constant = root_data->ml_constant_factor;
  out->recipe_ml_scaled = root_data->ml_scaled_factor;
  out->recipe_ml_quadratic = root_data->ml_quadratic_factor;
  out->recipe_ml_threshold = root_data->ml_threshold;

  out->fe_visits = v8::EncodeCount(root_data->fe_visits);
  const float cp = root_data->cpuct_at_root * 1000.0f;
  out->cpuct_x1000 =
      cp > 0.0f ? v8::EncodeCount(static_cast<uint64_t>(cp + 0.5f)) : 0;
}

void V6TrainingDataArray::Write(TrainingDataWriter* writer, GameResult result,
                                bool adjudicated) const {
  if (training_data_.empty()) return;
  // Base estimate off of best_m.  If needed external processing can use a
  // different approach.
  float m_estimate = training_data_.back().best_m + training_data_.size() - 1;
  for (auto chunk : training_data_) {
    bool black_to_move = chunk.side_to_move_or_enpassant;
    if (IsCanonicalFormat(static_cast<pblczero::NetworkFormat::InputFormat>(
            chunk.input_format))) {
      black_to_move = (chunk.invariance_info & (1u << 7)) != 0;
    }
    if (result == GameResult::WHITE_WON) {
      chunk.result_q = black_to_move ? -1 : 1;
      chunk.result_d = 0;
    } else if (result == GameResult::BLACK_WON) {
      chunk.result_q = black_to_move ? 1 : -1;
      chunk.result_d = 0;
    } else {
      chunk.result_q = 0;
      chunk.result_d = 1;
    }
    if (adjudicated) {
      chunk.invariance_info |= 1u << 5;  // Game adjudicated.
    }
    if (adjudicated && result == GameResult::UNDECIDED) {
      chunk.invariance_info |= 1u << 4;  // Max game length exceeded.
    }
    chunk.plies_left = m_estimate;
    m_estimate -= 1.0f;
    if (emit_v8_) {
      writer->WriteChunk(chunk);
    } else if (emit_v7_) {
      // Slice to the V7 prefix: the V8 block is appended AFTER reserved[8]
      // precisely so this slice is byte-identical to the cv3 writer.
      writer->WriteChunk(static_cast<const V7TrainingData&>(chunk));
    } else {
      // Slice to the V6 prefix. Byte-identical to the pre-cv3 writer, so a
      // binary from this tree with --training-data-v7=false reproduces cv2.
      writer->WriteChunk(static_cast<const V6TrainingData&>(chunk));
    }
  }
}

void V6TrainingDataArray::Add(
    const classic::Node* node, const PositionHistory& history, classic::Eval best_eval,
    classic::Eval played_eval, bool best_is_proven, Move best_move,
    Move played_move, const std::vector<std::tuple<float, float>>& visits,
    std::span<Move> legal_moves, const std::optional<EvalResult>& nneval,
    const RootSearchData* root_data) {
  V8TrainingData result{};  // value-init: the whole V7/V8 tail starts at zero
  const auto& position = history.Last();

  // Set version.
  result.version = emit_v8_ ? 8 : (emit_v7_ ? 7 : 6);
  result.input_format = input_format_;

  // Populate planes.
  int transform;
  InputPlanes planes = EncodePositionForNN(
      input_format_, history, 8, fill_empty_history_[position.IsBlackToMove()],
      &transform);
  int plane_idx = 0;
  for (auto& plane : result.planes) {
    plane = ReverseBitsInBytes(planes[plane_idx++].mask);
  }

  // Set illegal moves to have -1 probability.
  std::fill(std::begin(result.probabilities), std::end(result.probabilities),
            -1);
  // Set moves probabilities according to their relative amount of visits.
  // Compute Kullback-Leibler divergence in nats (between policy and visits).
  float kld_sum = 0;
  float total = 0.0;
  float total_n = std::accumulate(visits.begin(), visits.end(), 0.0f,
                                  [](float sum, const auto& child) {
                                    return sum + std::get<0>(child);
                                  });
  // Prevent garbage/invalid training data from being uploaded to server.
  // It's possible to have N=0 when there is only one legal move in position
  // (due to smart pruning).
  if (total_n == 0 && node->GetNumEdges() != 1) {
    throw Exception("Search generated invalid data!");
  }
  auto move_iter = legal_moves.begin();
  for (const auto& child : visits) {
    if (move_iter == legal_moves.end()) {
      throw Exception("More visited children than legal moves");
    }
    const Move move = *move_iter++;
    const float fracv = std::get<0>(child) / total_n;
    const float P = std::get<1>(child);
    if (fracv > 0 && P > 0) {
      kld_sum += fracv * std::log(fracv / P);
    }
    total += P;
    result.probabilities[MoveToNNIndex(move, transform)] = fracv;
  }
  if (total > 0) {
    // Add small epsilon for backward compatibility with earlier value of 0.
    auto epsilon = std::numeric_limits<float>::min();
    kld_sum = std::max(kld_sum + std::log(total), 0.0f) + epsilon;
  }
  result.policy_kld = kld_sum;

  const auto& castlings = position.GetBoard().castlings();
  // Populate castlings.
  // For non-frc trained nets, just send 1 like we used to.
  uint8_t our_queen_side = 1;
  uint8_t our_king_side = 1;
  uint8_t their_queen_side = 1;
  uint8_t their_king_side = 1;
  // If frc trained, send the bit mask representing rook position.
  if (Is960CastlingFormat(input_format_)) {
    our_queen_side <<= castlings.our_queenside_rook.idx;
    our_king_side <<= castlings.our_kingside_rook.idx;
    their_queen_side <<= castlings.their_queenside_rook.idx;
    their_king_side <<= castlings.their_kingside_rook.idx;
  }

  result.castling_us_ooo = castlings.we_can_000() ? our_queen_side : 0;
  result.castling_us_oo = castlings.we_can_00() ? our_king_side : 0;
  result.castling_them_ooo = castlings.they_can_000() ? their_queen_side : 0;
  result.castling_them_oo = castlings.they_can_00() ? their_king_side : 0;

  // Other params.
  if (IsCanonicalFormat(input_format_)) {
    result.side_to_move_or_enpassant =
        position.GetBoard().en_passant().as_int() >> 56;
    if ((transform & FlipTransform) != 0) {
      result.side_to_move_or_enpassant =
          ReverseBitsInBytes(result.side_to_move_or_enpassant);
    }
    // Send transform in deprecated move count so rescorer can reverse it to
    // calculate the actual move list from the input data.
    result.invariance_info =
        transform | (position.IsBlackToMove() ? (1u << 7) : 0u);
  } else {
    result.side_to_move_or_enpassant = position.IsBlackToMove() ? 1 : 0;
    result.invariance_info = 0;
  }
  if (best_is_proven) {
    result.invariance_info |= 1u << 3;  // Best node is proven best;
  }
  result.dummy = 0;
  result.rule50_count = position.GetRule50Ply();

  // Game result is undecided.
  result.result_q = 0;
  result.result_d = 1;

  classic::Eval orig_eval;
  if (nneval) {
    orig_eval.wl = nneval->q;
    orig_eval.d = nneval->d;
    orig_eval.ml = nneval->m;
  } else {
    orig_eval.wl = std::numeric_limits<float>::quiet_NaN();
    orig_eval.d = std::numeric_limits<float>::quiet_NaN();
    orig_eval.ml = std::numeric_limits<float>::quiet_NaN();
  }

  // Aggregate evaluation WL.
  result.root_q = -node->GetWL();
  result.best_q = best_eval.wl;
  result.played_q = played_eval.wl;
  result.orig_q = orig_eval.wl;

  // Draw probability of WDL head.
  result.root_d = node->GetD();
  result.best_d = best_eval.d;
  result.played_d = played_eval.d;
  result.orig_d = orig_eval.d;

  std::tie(result.best_q, result.best_d) =
      DriftCorrect(result.best_q, result.best_d);
  std::tie(result.root_q, result.root_d) =
      DriftCorrect(result.root_q, result.root_d);
  std::tie(result.played_q, result.played_d) =
      DriftCorrect(result.played_q, result.played_d);

  result.root_m = node->GetM();
  result.best_m = best_eval.ml;
  result.played_m = played_eval.ml;
  result.orig_m = orig_eval.ml;

  result.visits = node->GetN();
  if (position.IsBlackToMove()) {
    best_move.Flip();
    played_move.Flip();
  }
  result.best_idx = MoveToNNIndex(best_move, transform);
  result.played_idx = MoveToNNIndex(played_move, transform);
  result.q_st = 0.0f;

  // Per-move ("child") Q for the top-K root children, packed into the
  // reserved[4..7] window. This is the search's own per-edge evaluation --
  // the de-biased sibling of the visit-derived policy target, which the
  // record has never carried. See trainingdata/childq.h for the layout.
  //
  // Ordering is by visit count descending, prior descending as the
  // tie-break, matching Search::GetBestChildrenNoTemperature.
  //
  // Q is stored VERBATIM: edge Q is already in the root side-to-move frame,
  // the same frame as best_q/played_q, so it must NOT be negated.
  if (record_child_q_) {
    struct Cand {
      uint16_t nn_idx;
      uint32_t n;
      float p;
      float q;
    };
    std::vector<Cand> cands;
    cands.reserve(node->GetNumEdges());
    for (const auto& edge : node->Edges()) {
      const uint32_t n = edge.GetN();
      // Write-time visit floor. A child below it is not a candidate at all,
      // so the stored set is the top-K by visits AMONG children clearing the
      // floor. n == 0 is always excluded: no search evidence, and inventing a
      // value would be worse than storing nothing.
      if (n == 0 || n < static_cast<uint32_t>(child_q_min_visits_)) continue;
      const uint16_t idx = MoveToNNIndex(edge.GetMove(false), transform);
      // Self-check on the move frame. probabilities[] was filled from
      // `legal_moves` through the same MoveToNNIndex/transform above, and
      // illegal slots are left at -1. If an edge maps to a slot that was
      // never filled, the two enumerations disagree and the packed indices
      // would be silently wrong -- exactly the failure class that a castling
      // or transform mismatch produces. Drop the frame rather than write
      // corrupt per-move data.
      if (idx >= 1858 || result.probabilities[idx] < 0.0f) {
        throw Exception(
            "child-Q: root edge maps to a move index absent from the policy "
            "target -- move frame or transform mismatch");
      }
      cands.push_back({idx, n, edge.GetP(), edge.GetQ(0.0f, 0.0f)});
    }
    const size_t k = std::min<size_t>(childq::kMaxChildren, cands.size());
    if (k > 0) {
      std::partial_sort(cands.begin(), cands.begin() + k, cands.end(),
                        [](const Cand& a, const Cand& b) {
                          if (a.n != b.n) return a.n > b.n;
                          return a.p > b.p;
                        });
      childq::Child children[childq::kMaxChildren];
      for (size_t i = 0; i < k; ++i) {
        children[i].nn_idx = cands[i].nn_idx;
        children[i].q = cands[i].q;
      }
      childq::Encode(children, static_cast<int>(k),
                     childq::kSourceDatagenSearch, &result.reserved[4]);
    }
  }

  if (emit_v8_) FillV8Block(&result, root_data, legal_moves, transform);

  // Unknown here - will be filled in once the full data has been collected.
  result.plies_left = 0;
  training_data_.push_back(result);
}

}  // namespace lczero
