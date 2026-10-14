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

#include "search/childq_classic/params.h"

#include <algorithm>
#include <string>
#include <vector>

namespace lczero {
namespace childq_classic {

const OptionId SearchParams::kChildQModeId{
    "childq-mode", "ChildQMode",
    "Use the network's child-Q head (a value estimate for every move, made at "
    "the parent) in PUCT selection. \"off\" leaves the search identical to "
    "classic. \"floor\": an unvisited child starts at max(child-Q, FPU) -- the "
    "head can bring a first visit forward, never delay it. \"fpu\": an "
    "unvisited child starts at --childq-fpu-weight times its child-Q plus the "
    "rest times the FPU value. \"prior\": every child's Q is shrunk towards "
    "its child-Q, which counts as n0 = prior-scale / sigma^2 visits. Needs a "
    "backend that serves the head."};
const OptionId SearchParams::kChildQFpuWeightId{
    "childq-fpu-weight", "ChildQFpuWeight",
    "With --childq-mode=fpu, the weight of the child-Q in the value of an "
    "unvisited child; the rest is the FPU value."};
const OptionId SearchParams::kChildQPriorScaleId{
    "childq-prior-scale", "ChildQPriorScale",
    "With --childq-mode=prior, the child-Q counts as this value / sigma^2 "
    "visits, clipped to [--childq-prior-min, --childq-prior-max]. Replaced "
    "by the SHUSS scale when --childq-prior-bias-slope and "
    "--childq-visit-value-var are both set."};
const OptionId SearchParams::kChildQPriorBiasSlopeId{
    "childq-prior-bias-slope", "ChildQPriorBiasSlope",
    "SHUSS pseudo-count (Fabiano & Cazenave 2021, Lemma 1): how much the head "
    "compresses true gaps between moves (0.5 = half). With "
    "--childq-visit-value-var, prior-scale = slope * variance. 0 = unused."};
const OptionId SearchParams::kChildQVisitValueVarId{
    "childq-visit-value-var", "ChildQVisitValueVar",
    "SHUSS pseudo-count: the per-visit leaf-value variance of the served net. "
    "0 = unused."};
const OptionId SearchParams::kChildQPriorMinId{
    "childq-prior-min", "ChildQPriorMin",
    "Lower bound of the child-Q pseudo-count with --childq-mode=prior."};
const OptionId SearchParams::kChildQPriorMaxId{
    "childq-prior-max", "ChildQPriorMax",
    "Upper bound of the child-Q pseudo-count with --childq-mode=prior."};
const OptionId SearchParams::kChildQSigmaFloorId{
    "childq-sigma-floor", "ChildQSigmaFloor",
    "The child-Q standard deviation is used as at least this value."};
const OptionId SearchParams::kChildQOptimismId{
    "childq-optimism", "ChildQOptimism",
    "With --childq-mode other than off, adds this value * sigma / sqrt(1 + N) "
    "to the selection value of a child with N visits; negative = pessimism "
    "(an uncertain head estimate waits for evidence)."};
const OptionId SearchParams::kChildQFloorId{
    "childq-floor", "ChildQFloor",
    "With --childq-mode=fpu or prior, an unvisited child never starts below "
    "the FPU value. Always on with --childq-mode=floor."};
const OptionId SearchParams::kChildQAtRootId{
    "childq-at-root", "ChildQAtRoot",
    "With --childq-mode other than off: false = the root's children are "
    "selected as without a head (the root keeps its own FPU and forced "
    "exploration); the head acts below the root only."};
const OptionId SearchParams::kChildQBackupId{
    "childq-backup", "ChildQBackup",
    "1-ply backup: at expansion a node's value becomes (1 - w) * V + w * A, A "
    "the aggregate of its children's child-Q: \"best\" (max), \"best-sigma\" "
    "(max of child-Q - sigma), \"policy-mean\" (sum P * child-Q) or "
    "\"softmax\" "
    "(softmax(child-Q / --childq-backup-temp)-weighted mean). Independent of "
    "--childq-mode."};
const OptionId SearchParams::kChildQBackupWeightId{
    "childq-backup-weight", "ChildQBackupWeight",
    "The weight w of --childq-backup."};
const OptionId SearchParams::kChildQBackupTempId{
    "childq-backup-temp", "ChildQBackupTemp",
    "Temperature of --childq-backup=softmax."};
const OptionId SearchParams::kChildQPriorMixId{
    "childq-prior-mix", "ChildQPriorMix",
    "Blends the policy with the child-Q over legal moves, P' ~ P^(1 - mix) * "
    "exp(mix * child-Q / --childq-prior-temp), before Dirichlet noise. 0 is "
    "off. Independent of --childq-mode."};
const OptionId SearchParams::kChildQPriorTempId{
    "childq-prior-temp", "ChildQPriorTemp",
    "Temperature of the child-Q term of --childq-prior-mix."};
const OptionId SearchParams::kChildQPriorMixAtRootId{
    "childq-prior-mix-at-root", "ChildQPriorMixAtRoot",
    "--childq-prior-mix at the root; negative = the same as below it."};
const OptionId SearchParams::kChildQPriorBlendId{
    "childq-prior-blend", "ChildQPriorBlend",
    "Form of --childq-prior-mix: \"geometric\" (P^(1 - mix) * exp(mix * "
    "child-Q / temp)) or \"linear\" ((1 - mix) * P + mix * softmax(child-Q / "
    "temp), which never takes a move below (1 - mix) of its prior)."};
const OptionId SearchParams::kChildQVerboseStatsId{
    "childq-verbose-stats", "ChildQVerboseStats",
    "Adds every move's child-Q and sigma (CQ, CS) to the verbose move stats, "
    "and its selection value (SQ) when --childq-mode is not off."};
const OptionId SearchParams::kFeSharpCheckpointId{
    "forced-exploration-sharp-checkpoint", "ForcedExplorationSharpCheckpoint",
    "E2 / stratum read: once the root has this many visits beyond its forced "
    "minimums, read Q of the most-visited child and its gap to the "
    "second-most-visited one. 0 = no read."};
const OptionId SearchParams::kFeSharpMaxQId{
    "forced-exploration-sharp-max-q", "ForcedExplorationSharpMaxQ",
    "A root is balanced when |Q(most-visited child)| is below this."};
const OptionId SearchParams::kFeSharpMinGapId{
    "forced-exploration-sharp-min-gap", "ForcedExplorationSharpMinGap",
    "A root is sharp when Q(most-visited) - Q(second-most-visited) is at "
    "least this."};
const OptionId SearchParams::kFeSharpModeId{
    "forced-exploration-sharp-mode", "ForcedExplorationSharpMode",
    "What makes a root balanced at the sharp checkpoint. q = |Q(most-visited "
    "child)| below --forced-exploration-sharp-max-q. mu_s = the root's W / D / "
    "L in the WDL model: |mu| at most --forced-exploration-sharp-max-mu and "
    "the spread s at least --forced-exploration-sharp-min-s. unc = 1 - max(W, "
    "D, L) at least --forced-exploration-sharp-min-unc. The gap test "
    "(--forced-exploration-sharp-min-gap) applies in every mode."};
const OptionId SearchParams::kFeSharpMaxMuId{
    "forced-exploration-sharp-max-mu", "ForcedExplorationSharpMaxMu",
    "Sharp mode mu_s: the largest |mu| (in draw-window units; 1 = a 50 % win "
    "chance) of a balanced root."};
const OptionId SearchParams::kFeSharpMinSId{
    "forced-exploration-sharp-min-s", "ForcedExplorationSharpMinS",
    "Sharp mode mu_s: the smallest outcome spread s of a contested root (at "
    "mu = 0, s 0.3 is a 93 % draw and s 1.0 a 46 % one)."};
const OptionId SearchParams::kFeSharpMinUncId{
    "forced-exploration-sharp-min-unc", "ForcedExplorationSharpMinUnc",
    "Sharp mode unc: the smallest 1 - max(W, D, L) of a contested root."};
const OptionId SearchParams::kFeSharpVisitsId{
    "forced-exploration-sharp-visits", "ForcedExplorationSharpVisits",
    "E2: the forced-exploration budget of a balanced-but-sharp root (every "
    "child's forced minimum is scaled by this / --forced-exploration-visits). "
    "0 = read only."};
const OptionId SearchParams::kFeChildQShareId{
    "forced-exploration-childq-share", "ForcedExplorationChildQShare",
    "E3: this share of the forced budget is allocated by the child-Q head, in "
    "proportion to sigma * exp((q_hat - max q_hat) / "
    "--forced-exploration-childq-temp); the rest keeps the sqrt(P) split. "
    "Needs a backend that serves the head; 0 = off."};
const OptionId SearchParams::kFeEstimateClampId{
    "forced-exploration-estimate-clamp", "ForcedExplorationEstimateClamp",
    "The visit limit counts playouts minus an estimate of the forced visits. "
    "That estimate adds, per root move, its visits minus its PUCT-natural "
    "count, which is negative for a move below its natural count, so a search "
    "can end below the limit. With this on such a move counts as 0 and the "
    "visit limit is a floor on unforced visits."};
const OptionId SearchParams::kStopAtIterationBoundaryId{
    "stop-at-iteration-boundary", "StopAtIterationBoundary",
    "The stop decision (visit, playout and time limits) is taken only by the "
    "search worker at the end of an iteration, never by the watchdog thread in "
    "the middle of one, so it depends on the tree alone. Meant for "
    "visit-limited selfplay with one search thread."};
const OptionId SearchParams::kDeforceSelfConsistentId{
    "policy-target-deforce-self-consistent",
    "PolicyTargetDeforceSelfConsistent",
    "De-force the recorded policy target with U = cpuct * sqrt(the de-forced "
    "total) (one fixed-point step) instead of the raw root visits, which "
    "include the forced ones; the target then no longer grows flatter with "
    "the forced budget."};
const OptionId SearchParams::kFeChildQTempId{
    "forced-exploration-childq-temp", "ForcedExplorationChildQTemp",
    "E3: temperature of the head's allocation."};

void SearchParams::Populate(OptionsParser* options) {
  classic::SearchParams::Populate(options);
  std::vector<std::string> childq_mode = {"off", "floor", "fpu", "prior"};
  options->Add<ChoiceOption>(kChildQModeId, childq_mode) = "off";
  options->Add<FloatOption>(kChildQFpuWeightId, 0.0f, 1.0f) = 1.0f;
  options->Add<FloatOption>(kChildQPriorScaleId, 0.001f, 100.0f) = 0.01f;
  options->Add<FloatOption>(kChildQPriorBiasSlopeId, 0.0f, 10.0f) = 0.0f;
  options->Add<FloatOption>(kChildQVisitValueVarId, 0.0f, 4.0f) = 0.0f;
  options->Add<FloatOption>(kChildQPriorMinId, 0.0f, 1000.0f) = 0.25f;
  options->Add<FloatOption>(kChildQPriorMaxId, 0.0f, 1000.0f) = 8.0f;
  options->Add<FloatOption>(kChildQSigmaFloorId, 0.0f, 1.0f) = 0.1f;
  options->Add<FloatOption>(kChildQOptimismId, -5.0f, 5.0f) = 0.0f;
  options->Add<BoolOption>(kChildQFloorId) = false;
  options->Add<BoolOption>(kChildQAtRootId) = true;
  // Not "max": a UCI option-syntax keyword, which cutechess drops from a
  // combo's var list (the value is then rejected and the option left at off).
  std::vector<std::string> childq_backup = {"off", "best", "best-sigma",
                                            "policy-mean", "softmax"};
  options->Add<ChoiceOption>(kChildQBackupId, childq_backup) = "off";
  options->Add<FloatOption>(kChildQBackupWeightId, 0.0f, 1.0f) = 0.5f;
  options->Add<FloatOption>(kChildQBackupTempId, 0.001f, 10.0f) = 0.05f;
  options->Add<FloatOption>(kChildQPriorMixId, 0.0f, 1.0f) = 0.0f;
  options->Add<FloatOption>(kChildQPriorTempId, 0.001f, 10.0f) = 0.1f;
  options->Add<FloatOption>(kChildQPriorMixAtRootId, -1.0f, 1.0f) = -1.0f;
  std::vector<std::string> childq_blend = {"geometric", "linear"};
  options->Add<ChoiceOption>(kChildQPriorBlendId, childq_blend) = "geometric";
  options->Add<BoolOption>(kChildQVerboseStatsId) = false;
  options->Add<IntOption>(kFeSharpCheckpointId, 0, 100000) = 0;
  options->Add<FloatOption>(kFeSharpMaxQId, 0.0f, 1.0f) = 0.5f;
  options->Add<FloatOption>(kFeSharpMinGapId, 0.0f, 2.0f) = 0.1f;
  options->Add<ChoiceOption>(
      kFeSharpModeId, std::vector<std::string>{"q", "mu_s", "unc"}) = "q";
  options->Add<FloatOption>(kFeSharpMaxMuId, 0.0f, 100.0f) = 1.5f;
  options->Add<FloatOption>(kFeSharpMinSId, 0.0f, 10.0f) = 0.5f;
  options->Add<FloatOption>(kFeSharpMinUncId, 0.0f, 1.0f) = 0.30f;
  options->Add<IntOption>(kFeSharpVisitsId, 0, 100000) = 0;
  options->Add<FloatOption>(kFeChildQShareId, 0.0f, 1.0f) = 0.0f;
  options->Add<FloatOption>(kFeChildQTempId, 0.001f, 10.0f) = 0.1f;
  options->Add<BoolOption>(kDeforceSelfConsistentId) = false;
  options->Add<BoolOption>(kFeEstimateClampId) = false;
  options->Add<BoolOption>(kStopAtIterationBoundaryId) = false;
}

namespace {
ChildQParams ChildQParamsFromOptions(const OptionsDict& options) {
  ChildQParams params;
  const std::string mode = options.Get<std::string>(SearchParams::kChildQModeId);
  params.mode = mode == "fpu" || mode == "floor" ? ChildQMode::kFpu
                : mode == "prior"                ? ChildQMode::kPrior
                                                 : ChildQMode::kOff;
  params.fpu_weight = options.Get<float>(SearchParams::kChildQFpuWeightId);
  params.floor = options.Get<bool>(SearchParams::kChildQFloorId);
  params.at_root = options.Get<bool>(SearchParams::kChildQAtRootId);
  const std::string backup =
      options.Get<std::string>(SearchParams::kChildQBackupId);
  params.backup = backup == "best"          ? ChildQBackup::kMax
                  : backup == "best-sigma"  ? ChildQBackup::kMaxSigma
                  : backup == "policy-mean" ? ChildQBackup::kPolicyMean
                  : backup == "softmax"     ? ChildQBackup::kSoftmax
                                            : ChildQBackup::kOff;
  params.backup_weight =
      options.Get<float>(SearchParams::kChildQBackupWeightId);
  params.backup_temp = options.Get<float>(SearchParams::kChildQBackupTempId);
  if (mode == "floor") {
    // max(q_hat, FPU): the head's full estimate, never below FPU.
    params.fpu_weight = 1.0f;
    params.floor = true;
  }
  params.prior_scale = options.Get<float>(SearchParams::kChildQPriorScaleId);
  params.prior_bias_slope =
      options.Get<float>(SearchParams::kChildQPriorBiasSlopeId);
  params.visit_value_var =
      options.Get<float>(SearchParams::kChildQVisitValueVarId);
  params.prior_min = options.Get<float>(SearchParams::kChildQPriorMinId);
  params.prior_max = std::max(
      params.prior_min, options.Get<float>(SearchParams::kChildQPriorMaxId));
  params.sigma_floor = options.Get<float>(SearchParams::kChildQSigmaFloorId);
  params.optimism = options.Get<float>(SearchParams::kChildQOptimismId);
  params.prior_mix = options.Get<float>(SearchParams::kChildQPriorMixId);
  params.prior_temp = options.Get<float>(SearchParams::kChildQPriorTempId);
  params.prior_mix_root =
      options.Get<float>(SearchParams::kChildQPriorMixAtRootId);
  params.prior_linear =
      options.Get<std::string>(SearchParams::kChildQPriorBlendId) == "linear";
  return params;
}

SearchParams::ForcedArm ForcedArmFromOptions(const OptionsDict& options) {
  SearchParams::ForcedArm arm;
  arm.sharp_checkpoint = options.Get<int>(SearchParams::kFeSharpCheckpointId);
  arm.sharp_max_q = options.Get<float>(SearchParams::kFeSharpMaxQId);
  arm.sharp_min_gap = options.Get<float>(SearchParams::kFeSharpMinGapId);
  const auto mode = options.Get<std::string>(SearchParams::kFeSharpModeId);
  arm.sharp_mode = mode == "mu_s"  ? SearchParams::ForcedArm::SharpMode::kMuS
                   : mode == "unc" ? SearchParams::ForcedArm::SharpMode::kUnc
                                   : SearchParams::ForcedArm::SharpMode::kQ;
  arm.sharp_max_mu = options.Get<float>(SearchParams::kFeSharpMaxMuId);
  arm.sharp_min_s = options.Get<float>(SearchParams::kFeSharpMinSId);
  arm.sharp_min_unc = options.Get<float>(SearchParams::kFeSharpMinUncId);
  arm.sharp_visits = options.Get<int>(SearchParams::kFeSharpVisitsId);
  arm.childq_share = options.Get<float>(SearchParams::kFeChildQShareId);
  arm.childq_temp = options.Get<float>(SearchParams::kFeChildQTempId);
  arm.deforce_self_consistent =
      options.Get<bool>(SearchParams::kDeforceSelfConsistentId);
  arm.estimate_clamp = options.Get<bool>(SearchParams::kFeEstimateClampId);
  arm.stop_at_iteration_boundary =
      options.Get<bool>(SearchParams::kStopAtIterationBoundaryId);
  return arm;
}
}  // namespace

SearchParams::SearchParams(const OptionsDict& options)
    : classic::SearchParams(options),
      kChildQ(ChildQParamsFromOptions(options)),
      kChildQVerboseStats(options.Get<bool>(kChildQVerboseStatsId)),
      kForcedArm(ForcedArmFromOptions(options)) {}

}  // namespace childq_classic
}  // namespace lczero
