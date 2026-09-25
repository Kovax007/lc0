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

// V8 training record: the V7 record byte-for-byte, plus a fixed 128-slot
// per-legal-move table appended after reserved[8].
//
// WHY THE BLOCK IS APPENDED AND NOT INSERTED
// ------------------------------------------
// Inserting it after probabilities[] would shift every V7 offset (planes,
// the 15 floats, visits, q_st, reserved) and break every existing reader,
// the rescorer's struct inheritance and the op1 patcher. Field ORDER inside
// a packed struct carries no meaning to a consumer; INDEX SPACE does. The
// index space here is the 1858 nn-index of probabilities[], i.e. the output
// space of the attention policy map, so an expanded child_q[idx] lines up
// with policy logit idx with no conversion.
//
// WHY A SLOT TABLE AND NOT DENSE 1858-WIDE ARRAYS
// -----------------------------------------------
// Dense would be 30,772 B raw per record (3.7x V7). A 128-slot compact table
// is 2,640 B (1.2x V7) and compresses to the same size as dense, because a
// sparse dense array and a slot table carry the same information. The pool
// holds the record as-is; the loader expands to 1858-space only when a
// position is selected for a batch.
//
// STRUCT-OF-ARRAYS, not array-of-structs: same-typed values sit contiguous,
// which compresses better and expands as per-array gathers. Unused-slot
// sentinels become runs.
//
// WHY EVERY SLOT CARRIES ITS OWN INDEX
// ------------------------------------
// It would be 256 B cheaper to imply the slot order (movegen order, or the
// de-forced ordering recomputed in the loader). That is the silent-
// permutation failure class, and it removes the two cheapest checks we have:
// the writer's `probabilities[idx] >= 0` self-check, and the cross-check of
// slots 0..5 against the independently-written packed 6-child window.
//
// SIGN FRAME. q[], d[] and reply_q[] are all in the ROOT side-to-move frame,
// the same frame as best_q/played_q. reply_idx[] is the ONLY field in the
// other frame: it is indexed as the opponent would index it in the child
// position, exactly like opp_played_idx. See the frame note on reply_idx.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "trainingdata/trainingdata_v7.h"
#include "utils/cppattributes.h"

namespace lczero {

namespace v8 {

// Slot count. 108 is the no-promotion legal-move bound (Chinchalkar 1996);
// multi-queen positions can reach 218 in theory but real games above 100 are
// vanishingly rare. Overflow is flagged, not silent.
inline constexpr int kSlots = 128;

// 1 = the original block. 2 adds the recipe constants in reserved_v8, which
// are what a re-target needs and the per-move arrays cannot supply: the QM a
// kernel sees is Q + MEvaluator::GetMUtility(M, root_m, root_q, six
// moves-left constants), and the search prior is prior^(1/PST). Without them
// a corpus is interpretable only through an external registry that can drift.
inline constexpr int kLayoutVersion = 2;

// Sentinels. One per array, so an unused slot is unambiguous in every column.
inline constexpr uint16_t kNoIdx = 0xFFFF;
inline constexpr uint16_t kNoPrior = 0xFFFF;
inline constexpr int16_t kNoQ = INT16_MIN;
inline constexpr int16_t kNoD = INT16_MIN;
inline constexpr uint16_t kNoM = 0xFFFF;
inline constexpr uint16_t kNotComputed = 0xFFFF;  // plies_until_progress

// block_flags bits.
enum BlockFlag : uint16_t {
  kBlockValid = 1u << 0,
  kPriorPresent = 1u << 1,
  kDeforcedPresent = 1u << 2,
  kMPresent = 1u << 3,
  kFeActive = 1u << 4,
  kOverflow = 1u << 5,     // n_legal > kSlots, table truncated
  kReplyPresent = 1u << 6,
  kReplyExternal = 1u << 7,  // 0 = from the search tree, 1 = external probe
  // --- DAG datagen (2026-09-19). Bits 8..15 were free and zero in every
  // record written so far, so a reader that does not know them sees exactly
  // what it saw before.
  kDagSearched = 1u << 8,        // searched with dag_classic, not classic
  kDagDiagPresent = 1u << 9,     // reserved_v8[0..29] holds a V8DagDiag
  kDagTargetCorrected = 1u << 10,  // max(N_eq, N_edge) applied to transposed
                                   // root edges (the undercount fix)
  kDagRule50Keyed = 1u << 11,    // the TT key carries a rule50 bucket
};

// provenance_flags bits. Mirrors reserved[0] as bits; the rescorer writes both.
enum ProvenanceFlag : uint16_t {
  kProvTb = 1u << 0,
  kProvNoiseTrigger = 1u << 1,
  kProvCrossPly = 1u << 2,
  kProvOp1 = 1u << 3,
  kProvSfRescore = 1u << 4,
  kProvForcedPly = 1u << 5,
  kProvPostFork = 1u << 6,
  kProvLegacyClass4 = 1u << 7,
};

// Datagen target-recipe registry. The record says which recipe produced its
// policy target, so a mixed pool can never be trained on unknowingly.
enum RecipeId : uint16_t {
  kRecipeUnknown = 0,
  kRecipeCv = 1,
  kRecipeCv2 = 2,
  kRecipeCv3 = 3,
  kRecipeCv4A = 4,
  kRecipeCv4B = 5,
};

// ---- quantisation ---------------------------------------------------------
// Deliberately tiny and header-only so the writer, any C++ reader and the
// python decoders all describe the same arithmetic.

// prior: round(-log2(P) * 2048). P == 1 -> 0. Clamped to 65534 so that
// 0xFFFF stays an unambiguous "absent" (65534 is P = 2^-31.99, unreachable).
inline uint16_t EncodePrior(float p) {
  if (!(p > 0.0f)) return 65534;  // also catches NaN
  if (p >= 1.0f) return 0;
  const double v = -std::log2(static_cast<double>(p)) * 2048.0;
  if (!(v > 0.0)) return 0;
  if (v >= 65534.0) return 65534;
  return static_cast<uint16_t>(v + 0.5);
}
inline float DecodePrior(uint16_t v) {
  if (v == kNoPrior) return -1.0f;
  if (v == 0) return 1.0f;
  return static_cast<float>(std::exp2(-static_cast<double>(v) / 2048.0));
}

// q, d, reply_q: round(x * 32767). INT16_MIN is the sentinel, so the encoded
// range is [-32767, 32767] and every representable value round-trips.
inline int16_t EncodeSigned(float x) {
  if (!(x >= -1.0f)) x = -1.0f;  // also catches NaN
  if (x > 1.0f) x = 1.0f;
  const float scaled = x * 32767.0f;
  int v = static_cast<int>(scaled >= 0.0f ? scaled + 0.5f : scaled - 0.5f);
  if (v < -32767) v = -32767;
  if (v > 32767) v = 32767;
  return static_cast<int16_t>(v);
}
inline float DecodeSigned(int16_t v) {
  return v == kNoQ ? 0.0f : static_cast<float>(v) / 32767.0f;
}

// m: round(M_plies * 32), saturating. 0xFFFF is the sentinel, so the top
// representable value is 65534/32 = 2047.94 plies.
inline uint16_t EncodeM(float m) {
  if (!(m > 0.0f)) return 0;  // also catches NaN
  const float scaled = m * 32.0f;
  if (scaled >= 65534.0f) return 65534;
  return static_cast<uint16_t>(scaled + 0.5f);
}
inline float DecodeM(uint16_t v) {
  return v == kNoM ? -1.0f : static_cast<float>(v) / 32.0f;
}

// Counts saturate at 65534 rather than wrapping. At 800 visits this is
// unreachable; a future high-visit regime must not silently alias.
inline uint16_t EncodeCount(uint64_t n) {
  return n > 65534 ? static_cast<uint16_t>(65534) : static_cast<uint16_t>(n);
}

}  // namespace v8

// ---- DAG search diagnostics (2026-09-19) ---------------------------------
//
// WHY THIS IS NOT A LAYOUT BUMP.
//
// It occupies `reserved_v8[0..29]`, which every record written to date leaves
// zero, and it changes no existing field, no offset and not the 11,036-byte
// stride. Its presence is announced by `block_flags` bit 9, NOT by
// `v8_layout_version`.
//
// That distinction is load-bearing. The training loader's .so SILENTLY ZEROES
// a V8 block whose layout version it does not recognise, which is why every
// launch is gated on `v8_meta[:,0] == 2`. Bumping the version to 3 to announce
// a block that lives in previously-reserved bytes would therefore not "warn"
// anybody -- it would make the loader discard the entire per-move table for
// every DAG-era record, and the gate would report the zeroing as a data fault.
// A capability flag is additive; a version is a contract, and this breaks none.
//
// WHAT IT IS FOR. A DAG search shares nodes between parents, which makes three
// quantities un-reconstructable from the record as it stands: how much of the
// visit budget was spent vs inherited (a transposition hit costs no NN eval),
// how far a root edge's own visit count lags the shared node's, and whether
// the shared node was reached at a rule50 the stored value does not describe.
// The first is an effort bias, the second a policy-target bias, the third a
// value-label bias. None is visible in a tree-era record because none exists
// there.
//
// Deliberately NOT stored, because it is recomputable offline from fields the
// record already carries: N_eq per slot (from q[], prior[], cpuct_x1000,
// visits and the recipe constants -- the same PUCT inversion n_deforced[]
// uses), hence the corrected target and its KL against the shipped
// probabilities[]; resign/adjudication; game length and phase.
#pragma pack(push, 1)
struct V8DagDiag {
  uint8_t dag_diag_version;  // 1

  // bits 0-1: cache_history_length in effect (the NN eval is computed on one
  //           path's history and reused on others; the DAG also shares the
  //           BACKUPS, so the reuse is deeper than the NNCache's)
  // bit 2:    rule50 bucket included in the TT key
  // bits 3-7: reserved
  uint8_t key_flags;

  uint16_t nn_evals;        // NN evaluations this search actually made.
                            // `visits` is a constant 800 in this corpus, so
                            // this is the only field that shows the budget
                            // buying more effective depth in the endgame than
                            // in the opening.
  uint16_t tt_hits;         // transposition-table hits during this search
  uint16_t tt_hit_backups;  // root visits that ended at a TT hit (no new node)
  uint16_t root_children_preexisting;  // root children whose LowNode already
                                       // existed when the search started

  // THE UNDERCOUNT, RANKED BY THE UNDERCOUNT -- not by visit rank.
  //
  // The first version of this block stored the node visits of slots 0..5, i.e.
  // the six MOST-VISITED root edges. That indexes the wrong thing. An
  // under-counted move is by definition one with FEW edge visits behind many
  // node visits: its evidence arrived through a transposition, so PUCT has not
  // yet allocated it the edge visits its shared Q implies. Such a move sorts
  // LOW in n_raw order -- beyond slot 5 in any position with a few real
  // candidates -- so a top-6-by-visits block would systematically miss exactly
  // the cases it exists to measure, and would report the undercount as absent
  // precisely where it is worst.
  //
  // So: the six root children with the largest (n_node - n_edge), descending.
  // `undercount_slot[k]` is the SLOT index (into idx[]/n_raw[]/q[]...), so the
  // edge visits, prior, Q and move of each entry are already in the record and
  // are not duplicated here; `undercount_node_n[k]` is that child's shared-node
  // visit count. Entry 0 is therefore the maximum over ALL children, which is
  // why the old `max_node_minus_edge` field is gone rather than kept.
  uint8_t undercount_slot[6];      // 0xFF = no such entry
  uint16_t undercount_node_n[6];
  uint8_t argmax_slot_by_node_n;   // 0xFF if none; an argmax flip vs slot 0 is
                                   // the undercount changing the played move

  uint8_t max_rule50_delta;  // largest |rule50(path) - rule50(first touch)|
                             // over this search's TT hits, saturating at 255
  uint16_t rule50_bucket_cross_hits;  // hits where the path was at rule50 >= 80
                                      // and the stored node was not, or the
                                      // reverse -- the dangerous subset, and
                                      // the one that also produces the
                                      // terminal-vs-non-terminal disagreement
  uint16_t dag_nodes_at_start_div16;  // DAG size inherited from earlier
                                      // searches of this game, / 16
} PACKED_STRUCT;
#pragma pack(pop)

static_assert(sizeof(V8DagDiag) == 34, "V8DagDiag must be 34 bytes");
static_assert(offsetof(V8DagDiag, nn_evals) == 2, "");
static_assert(offsetof(V8DagDiag, tt_hits) == 4, "");
static_assert(offsetof(V8DagDiag, tt_hit_backups) == 6, "");
static_assert(offsetof(V8DagDiag, root_children_preexisting) == 8, "");
static_assert(offsetof(V8DagDiag, undercount_slot) == 10, "");
static_assert(offsetof(V8DagDiag, undercount_node_n) == 16, "");
static_assert(offsetof(V8DagDiag, argmax_slot_by_node_n) == 28, "");
static_assert(offsetof(V8DagDiag, max_rule50_delta) == 29, "");
static_assert(offsetof(V8DagDiag, rule50_bucket_cross_hits) == 30, "");
static_assert(offsetof(V8DagDiag, dag_nodes_at_start_div16) == 32, "");

#pragma pack(push, 1)

struct V8TrainingData : V7TrainingData {
  // ---- block header (16 bytes, offsets 8396..8411) ----
  uint16_t v8_layout_version;   // v8::kLayoutVersion. Bump on ANY block change.
  uint16_t recipe_id;           // v8::RecipeId
  uint16_t provenance_flags;    // v8::ProvenanceFlag bits (rescorer-owned)
  uint8_t n_legal;              // == count of probabilities[] >= 0
  uint8_t n_stored;             // slots in use; == n_legal unless kOverflow
  uint16_t plies_until_progress;  // rescorer-owned; ships at kNotComputed
  uint16_t block_flags;         // v8::BlockFlag bits
  uint16_t fe_visits;           // --forced-exploration-visits in effect, 0 = off
  uint16_t cpuct_x1000;         // root cpuct used by the de-force inversion

  // ---- the slot table (struct-of-arrays, offsets 8412..10971) ----
  // Slot k is the same move in every array. Slot order is raw root visits
  // descending, prior descending on ties -- identical to the ordering of the
  // packed 6-child window, which is what makes invariant (v) a free check.
  uint16_t idx[v8::kSlots];         // nn-index 0..1857 in the ROOT frame
  uint16_t prior[v8::kSlots];       // net prior, temperature undone
  int16_t q[v8::kSlots];            // root side-to-move frame
  int16_t d[v8::kSlots];
  uint16_t m[v8::kSlots];           // child moves-left, plies
  uint16_t n_raw[v8::kSlots];       // raw root visits (FE + noise included)
  uint16_t n_deforced[v8::kSlots];  // PUCT-inverted N'; 0 if prune-forced off

  // The second ply. reply_idx[k] is the most-visited grandchild below child k
  // -- the reply the search believes in. NOTE THE FRAME: it is indexed as the
  // OPPONENT would index it in the child position, the same frame as
  // opp_played_idx, because that is the frame the move is stored in. Convert
  // with a fixed mirror_idx[1858] bijection; no board or movegen needed.
  // reply_q is stored ALREADY converted to the root frame, so every Q column
  // in this record shares one sign convention.
  uint16_t reply_idx[v8::kSlots];
  int16_t reply_q[v8::kSlots];
  uint16_t reply_n[v8::kSlots];

  // ---- recipe constants (layout 2), offsets 10972..10999 ----
  // The seven scalars the SEARCH consumed that left no other trace in the
  // record. Without them the stored `prior` is uninterpretable: it is
  // renorm(p^PST), so recovering the search-consumed prior needs PST; and QM
  // = Q + MEvaluator::GetMUtility(...) needs all six moves-left constants.
  //
  // ⚠ FIELD ORDER IS MEvaluator's CONSTRUCTOR ORDER after pst (search.cc:90-101),
  // per SPEC_v8_recipe_block_0902.md §3 -- NOT params.h declaration order.
  // The two differ, and a transposition here is silent and catastrophic:
  // ml_slope is 0.007 while ml_max_effect is 0.3, a 43x error that nothing
  // downstream would flag.
  //
  // float32, NOT quantised: MovesLeftSlope is 0.0027 at default, which a
  // uint16 x1000 encoding would round to 3 -- an 11% error in a constant that
  // multiplies the entire M-utility. 28 bytes of a constant-per-corpus record
  // compress to nothing.
  //
  // SENTINEL: pst == 0.0f means the block is absent/unknown (PST 0 is outside
  // lc0's [0.1, 10] range, so it cannot be a real value). A layout-2 record
  // with pst == 0 is a writer fault and must fail the gate.
  float recipe_pst;           // policy-softmax-temp        [0.1, 10]
  float recipe_ml_slope;      // m_slope_      moves-left-slope             [0, 1]
  float recipe_ml_max_effect; // m_cap_        moves-left-max-effect        [0, 1]
  float recipe_ml_constant;   // a_constant_   moves-left-constant-factor  [-1, 1]
  float recipe_ml_scaled;     // a_linear_     moves-left-scaled-factor    [-2, 2]
  float recipe_ml_quadratic;  // a_square_     moves-left-quadratic-factor [-1, 1]
  float recipe_ml_threshold;  // q_threshold_  moves-left-threshold         [0, 1]

  uint8_t reserved_v8[36];  // all zero
} PACKED_STRUCT;

#pragma pack(pop)

static_assert(sizeof(V8TrainingData) == 11036, "Wrong struct size");

// Offsets are the on-disk contract. V8TrainingData inherits, so it is not a
// standard-layout type and offsetof on it is "conditionally supported" --
// which every compiler this project builds with supports and does correctly.
// The alternative (no offset checks at all) trades a compiler warning for a
// corpus nobody can decode.
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winvalid-offsetof"
#endif

// Offsets are the on-disk contract. A compiler that inserted padding, or a
// field reordered by a well-meaning edit, is caught here and not by a corpus.
static_assert(offsetof(V8TrainingData, v8_layout_version) == 8396, "");
static_assert(offsetof(V8TrainingData, provenance_flags) == 8400, "");
static_assert(offsetof(V8TrainingData, n_legal) == 8402, "");
static_assert(offsetof(V8TrainingData, n_stored) == 8403, "");
static_assert(offsetof(V8TrainingData, plies_until_progress) == 8404, "");
static_assert(offsetof(V8TrainingData, block_flags) == 8406, "");
static_assert(offsetof(V8TrainingData, fe_visits) == 8408, "");
static_assert(offsetof(V8TrainingData, cpuct_x1000) == 8410, "");
static_assert(offsetof(V8TrainingData, idx) == 8412, "");
static_assert(offsetof(V8TrainingData, prior) == 8668, "");
static_assert(offsetof(V8TrainingData, q) == 8924, "");
static_assert(offsetof(V8TrainingData, d) == 9180, "");
static_assert(offsetof(V8TrainingData, m) == 9436, "");
static_assert(offsetof(V8TrainingData, n_raw) == 9692, "");
static_assert(offsetof(V8TrainingData, n_deforced) == 9948, "");
static_assert(offsetof(V8TrainingData, reply_idx) == 10204, "");
static_assert(offsetof(V8TrainingData, reply_q) == 10460, "");
static_assert(offsetof(V8TrainingData, reply_n) == 10716, "");
static_assert(offsetof(V8TrainingData, recipe_pst) == 10972, "");
static_assert(offsetof(V8TrainingData, recipe_ml_slope) == 10976, "");
static_assert(offsetof(V8TrainingData, recipe_ml_max_effect) == 10980, "");
static_assert(offsetof(V8TrainingData, recipe_ml_constant) == 10984, "");
static_assert(offsetof(V8TrainingData, recipe_ml_scaled) == 10988, "");
static_assert(offsetof(V8TrainingData, recipe_ml_quadratic) == 10992, "");
static_assert(offsetof(V8TrainingData, recipe_ml_threshold) == 10996, "");
static_assert(offsetof(V8TrainingData, reserved_v8) == 11000, "");

// The diagnostic block is written into reserved_v8 and read back by offset by
// the python decoders, so pin the ABSOLUTE offsets here too. If either struct
// moves, this is what fails -- not a corpus.
static_assert(offsetof(V8TrainingData, reserved_v8) + offsetof(V8DagDiag, nn_evals) == 11002, "");
static_assert(offsetof(V8TrainingData, reserved_v8) + offsetof(V8DagDiag, undercount_slot) == 11010, "");
static_assert(offsetof(V8TrainingData, reserved_v8) + offsetof(V8DagDiag, undercount_node_n) == 11016, "");
static_assert(offsetof(V8TrainingData, reserved_v8) + offsetof(V8DagDiag, dag_nodes_at_start_div16) == 11032, "");
static_assert(offsetof(V8TrainingData, reserved_v8) + sizeof(V8DagDiag) == 11034, "");
static_assert(sizeof(V8TrainingData::reserved_v8) - sizeof(V8DagDiag) == 2,
              "2 bytes of reserved_v8 must remain free after the diag block");

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

// Copies the diagnostic block into reserved_v8 and raises bit 9. memcpy, not a
// reinterpret_cast onto the array: reserved_v8 is a uint8_t[36] and writing a
// V8DagDiag through it would be an aliasing violation that -O2 is entitled to
// reorder. The 6 trailing bytes are left zero.
inline void SetDagDiag(V8TrainingData* out, const V8DagDiag& diag) {
  std::memcpy(out->reserved_v8, &diag, sizeof(diag));
  out->block_flags |= v8::kDagDiagPresent;
}

}  // namespace lczero
