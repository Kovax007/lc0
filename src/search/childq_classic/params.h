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

// The childq_classic compartment = classic's search, verbatim, plus the
// child-Q consumers. Every classic option is inherited unchanged (same names,
// same defaults), the child-Q options are added, and with them at their
// defaults the search is bit-identical to classic (gated on 1,000 positions).
// classic itself is never edited: its stoppers and time management are used
// as they are, exactly as dag_classic does.

#pragma once

#include "search/childq_prior.h"
#include "search/classic/params.h"
#include "search/classic/stoppers/factory.h"
#include "search/classic/stoppers/stoppers.h"
#include "search/classic/stoppers/timemgr.h"

namespace lczero {
namespace childq_classic {

using ContemptMode = classic::ContemptMode;
using classic::IterationStats;
using classic::MakeTimeManager;
using classic::MemoryWatchingStopper;
using classic::PopulateTimeManagementOptions;
using classic::RunType;
using classic::SearchStopper;
using classic::StoppersHints;
using classic::TimeManager;

class SearchParams : public classic::SearchParams {
 public:
  SearchParams(const OptionsDict& options);
  SearchParams(const SearchParams&) = delete;

  // classic's options, then the child-Q ones.
  static void Populate(OptionsParser* options);

  const ChildQParams& GetChildQ() const { return kChildQ; }
  bool GetChildQVerboseStats() const { return kChildQVerboseStats; }

  // The forced-exploration arms (RULING 09-28 section 3 + child-Q addendum 2):
  // E2 = a larger forced budget in balanced-but-sharp roots, read once at a
  // checkpoint of the root search; E3 = part of the budget re-aimed by the
  // child-Q head's sigma. All off by default (the search is then classic's).
  struct ForcedArm {
    int sharp_checkpoint = 0;   // unforced root visits at the read; 0 = off
    float sharp_max_q = 0.5f;   // |Q(best child)| below this = balanced
    float sharp_min_gap = 0.1f; // Q(best) - Q(second) at least this = sharp
    // Gate v2: what "balanced" means at the checkpoint. kQ is the test above
    // (recipes 6-9). kMuS and kUnc read the ROOT's accumulated W / D / L: in
    // the WDL model (the result is logistic with mean mu and spread s, a draw
    // is a result inside [-1, 1]) or as 1 - max(W, D, L). A balanced endgame
    // is mostly a dead draw; these tests leave it out.
    enum class SharpMode { kQ, kMuS, kUnc };
    SharpMode sharp_mode = SharpMode::kQ;
    float sharp_max_mu = 1.5f;    // kMuS: |mu| at most this
    float sharp_min_s = 0.5f;     // kMuS: s at least this
    float sharp_min_unc = 0.30f;  // kUnc: 1 - max(W, D, L) at least this
    int sharp_visits = 0;       // forced budget in sharp roots; 0 = read only
    float childq_share = 0.0f;  // E3: share of the budget the head allocates
    float childq_temp = 0.1f;   // E3: temperature on q_hat - max q_hat
    // The recorded target's de-forcing with U from the de-forced total (one
    // fixed-point step) instead of the raw one: invariant to the forced budget.
    bool deforce_self_consistent = false;
    // The stopper patch (both off by default).
    // The forced-visit estimate that the visit stopper subtracts counts an
    // edge BELOW its PUCT-natural count as negative; with the clamp such an
    // edge counts as 0, so a search never ends below its visit limit.
    bool estimate_clamp = false;
    // The stop decision is taken only by the search worker at the end of an
    // iteration, never by the watchdog thread in the middle of one: it is then
    // a function of the tree alone. For visit-limited selfplay.
    bool stop_at_iteration_boundary = false;
  };
  const ForcedArm& GetForcedArm() const { return kForcedArm; }

  static const OptionId kChildQModeId;
  static const OptionId kChildQFpuWeightId;
  static const OptionId kChildQPriorScaleId;
  static const OptionId kChildQPriorBiasSlopeId;
  static const OptionId kChildQVisitValueVarId;
  static const OptionId kChildQPriorMinId;
  static const OptionId kChildQPriorMaxId;
  static const OptionId kChildQSigmaFloorId;
  static const OptionId kChildQOptimismId;
  static const OptionId kChildQFloorId;
  static const OptionId kChildQAtRootId;
  static const OptionId kChildQBackupId;
  static const OptionId kChildQBackupWeightId;
  static const OptionId kChildQBackupTempId;
  static const OptionId kChildQPriorMixId;
  static const OptionId kChildQPriorTempId;
  static const OptionId kChildQPriorMixAtRootId;
  static const OptionId kChildQPriorBlendId;
  static const OptionId kChildQVerboseStatsId;
  static const OptionId kFeSharpCheckpointId;
  static const OptionId kFeSharpMaxQId;
  static const OptionId kFeSharpMinGapId;
  static const OptionId kFeSharpModeId;
  static const OptionId kFeSharpMaxMuId;
  static const OptionId kFeSharpMinSId;
  static const OptionId kFeSharpMinUncId;
  static const OptionId kFeSharpVisitsId;
  static const OptionId kFeChildQShareId;
  static const OptionId kFeChildQTempId;
  static const OptionId kDeforceSelfConsistentId;
  static const OptionId kFeEstimateClampId;
  static const OptionId kStopAtIterationBoundaryId;

 private:
  const ChildQParams kChildQ;
  const bool kChildQVerboseStats;
  const ForcedArm kForcedArm;
};

}  // namespace childq_classic
}  // namespace lczero
