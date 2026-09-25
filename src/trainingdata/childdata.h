/*
  Per-root-child search data for the V8 training record.

  WHY THIS IS A SEPARATE PASS AND NOT A RETURN VALUE OF GetVisitDistribution
  -------------------------------------------------------------------------
  GetVisitDistribution has five exit paths (grill9, k2/hybrid, pp, raw visits,
  and the un-pruned fallback) and it is the function that decides the RECORDED
  TARGET. Everything in this struct is independent of which kernel ran -- it is
  what the search saw, not what the recipe made of it. Threading an out-param
  through all five paths would put the target computation at risk for data
  that does not depend on it, and gate 6 (off-flag byte identity against the
  cv3 binary) would then have to re-prove the whole target.

  So Search::GetRootChildData() walks the root edges once more, under its own
  lock, and recomputes the de-forced counts from the same inputs. That costs
  one extra pass per recorded position (once per played move, not per playout)
  and it is deterministic: same tree, same params, same numbers.

  ORDER. The vector is aligned with the `legal_moves` span the caller passed,
  exactly like GetVisitDistribution's return value. The V8 writer does the
  sort into slot order; the search does not, because the caller's alignment is
  what lets the writer cross-check every index against probabilities[].
*/

#pragma once

#include <cstdint>
#include <vector>

namespace lczero {

struct RootChildData {
  uint16_t nn_idx = 0xFFFF;  // ROOT frame, index space of probabilities[]
  // The net's own policy with the softmax temperature UNDONE (the backend
  // applies it; `nneval->p` is already sharpened), renormalised over legal
  // moves. Temperature-free, so it compares across recipes.
  float prior = -1.0f;
  // Valid iff n_raw > 0. The child's own evaluation, in the ROOT side-to-move
  // frame -- the same frame as best_q/played_q. Never negated.
  float q = 0.0f;
  float d = 0.0f;
  float m = 0.0f;  // plies
  uint32_t n_raw = 0;
  uint32_t n_deforced = 0;

  // The second ply: the most-visited grandchild below this child.
  // ⚠ FRAME: reply_idx is in the CHILD's side-to-move frame (the frame the
  // move is stored in, and the frame opp_played_idx uses), while reply_q has
  // already been converted to the ROOT frame. Do not mix them up.
  uint16_t reply_idx = 0xFFFF;
  float reply_q = 0.0f;
  uint32_t reply_n = 0;
};

// Host-side DAG search diagnostics, in natural widths. The writer saturates
// them into V8DagDiag; the search fills them without thinking about the wire
// format, exactly as it does for the per-child arrays.
//
// Every counter here is meaningless for a tree search and is only filled by
// dag_classic. `present` stays false for a classic search, so a tree-era
// record keeps reserved_v8 all-zero and block_flags bit 9 clear.
struct DagDiagnostics {
  bool present = false;

  int cache_history_length = 0;   // 0..3, into key_flags bits 0-1
  bool rule50_bucketed_key = false;

  uint64_t nn_evals = 0;
  uint64_t tt_hits = 0;
  uint64_t tt_hit_backups = 0;
  uint32_t root_children_preexisting = 0;

  // NODE visits of the root children, in the SAME order as
  // RootSearchData::children (the caller's legal_moves order). The search
  // reports every child and the WRITER decides what fits in 34 bytes -- it
  // ranks by (n_node - n_edge), because ranking by visits would miss the
  // under-counted moves by construction (see V8DagDiag). Keeping the raw
  // vector here also means the quarantine run's sidecar can dump all 128
  // without a second collection path.
  std::vector<uint32_t> child_node_n;

  // No max_node_minus_edge: entry 0 of the writer's undercount ranking IS the
  // maximum, and a second copy of a derived quantity is a second thing that
  // can disagree with the first.
  int argmax_index_by_node_n = -1;  // index into children, -1 if none

  uint32_t max_rule50_delta = 0;
  uint64_t rule50_bucket_cross_hits = 0;
  uint64_t dag_nodes_at_start = 0;
};

struct RootSearchData {
  std::vector<RootChildData> children;  // aligned with the caller's legal_moves
  bool prior_present = false;
  bool deforced_present = false;
  bool m_present = false;
  bool fe_active = false;
  uint32_t fe_visits = 0;
  float cpuct_at_root = 0.0f;

  // Recipe constants the per-move arrays cannot supply. QM = Q + MUtility(...)
  // needs all six moves-left values; recovering the search prior from the
  // stored (temperature-undone) one needs the PST. Stored so a corpus is
  // self-describing instead of registry-dependent.
  float policy_softmax_temp = 0.0f;
  float ml_max_effect = 0.0f;
  float ml_threshold = 0.0f;
  float ml_slope = 0.0f;
  float ml_constant_factor = 0.0f;
  float ml_scaled_factor = 0.0f;
  float ml_quadratic_factor = 0.0f;

  // Filled only by dag_classic. See DagDiagnostics.
  DagDiagnostics dag;
  bool dag_searched = false;        // block_flags bit 8
  bool dag_target_corrected = false;  // block_flags bit 10
};

}  // namespace lczero
