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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>

namespace lczero {

// Search-side consumers of the network's child-Q head, written ONCE against
// per-edge inputs so that every search compartment (childq_classic now,
// childq_dag next) instantiates the same arithmetic.
//
// For every legal move the head gives q_hat, the expected value of the child
// position, and sigma, its standard deviation, both in the frame of the side
// to move in the position the move is played from -- the frame in which a
// parent reads the Q of its children. q_hat is EDGE-owned: it is evaluated at
// the parent. A consumer reads it for a child with no evidence of its own and
// never writes it into the child node (the DAG composition rule: backups stay
// node-owned; a pooled transposition Q always beats the head's guess).

enum class ChildQMode {
  kOff,    // Selection is unchanged.
  kFpu,    // An unvisited child starts at w * q_hat + (1 - w) * FPU.
  kPrior,  // Every child's Q is shrunk towards q_hat with a pseudo-count n0.
};

// The 1-ply backup (Kovax 09-27: "child-Q visits every child and lets update
// the tree"): at expansion a node's value becomes (1 - w) * V + w * A, A an
// aggregate of its children's q_hat, so the head's look at every child is
// backed up through the tree like a visit. Selection is not touched.
enum class ChildQBackup {
  kOff,
  // UCI values "best" / "best-sigma": "max" is a UCI option-syntax keyword
  // (spin min/max) and cutechess drops it from a combo's var list.
  kMax,         // max q_hat: 1-ply minimax (biased up by the head's noise).
  kMaxSigma,    // max (q_hat - sigma): minimax on pessimistic estimates.
  kPolicyMean,  // sum P q_hat: the value the policy's own choice would get.
  kSoftmax,     // softmax(q_hat / backup_temp)-weighted mean.
};

struct ChildQParams {
  ChildQMode mode = ChildQMode::kOff;
  float fpu_weight = 1.0f;
  // Pseudo-count n0 = clamp(prior_scale / sigma^2, prior_min, prior_max).
  //
  // SHUSS (Fabiano & Cazenave, ACG 2021, Lemma 1): the optimal pseudo-count
  // of an exterior score is t' = (bias slope) * (per-visit value variance) /
  // (score error variance). With sigma^2 the head's own error variance,
  // prior_scale = bias_slope * visit_value_var. When both are given (> 0)
  // they REPLACE prior_scale; the slope is how much the head compresses true
  // gaps (≈ 0.5 measured 09-11: median top-2 gap 0.27 vs the 1-ply value's
  // 0.57), the variance is a per-visit leaf-value statistic of the served net.
  float prior_scale = 0.01f;
  float prior_bias_slope = 0.0f;
  float visit_value_var = 0.0f;
  float prior_min = 0.25f;
  float prior_max = 8.0f;
  // Sigma is used as max(sigma, sigma_floor).
  float sigma_floor = 0.1f;
  // Bonus optimism * sigma / sqrt(1 + n), with kFpu and kPrior; negative =
  // pessimism (q_hat - kappa * sigma: an uncertain estimate waits for evidence).
  float optimism = 0.0f;
  // An unvisited child never starts below FPU: the head can bring a move's
  // first visit forward, never delay it. `floor` mode = kFpu, w 1, floor on,
  // i.e. max(q_hat, FPU) -- the 09-12 ruling's always-on form.
  bool floor = false;
  // false: the root's children are selected exactly as without a head (the
  // root keeps its own FPU -- in selfplay absolute 1.0, every root move
  // visited -- and forced exploration); the head acts below the root only.
  // Datagen's root exploration removes the net's biases from the targets.
  bool at_root = true;
  ChildQBackup backup = ChildQBackup::kOff;
  float backup_weight = 0.5f;
  float backup_temp = 0.05f;
  // Policy blend; 0 is off. Independent of the mode. Geometric:
  // P' ~ P^(1 - prior_mix) * exp(prior_mix * q_hat / prior_temp) -- a sharp
  // head can squeeze a move the policy likes to ~0. Linear:
  // P' = (1 - prior_mix) * P + prior_mix * softmax(q_hat / prior_temp) --
  // every move keeps at least (1 - prior_mix) of its prior.
  float prior_mix = 0.0f;
  // The mix at the root; negative = prior_mix. A root carried over from the
  // previous move keeps the priors it was expanded with.
  float prior_mix_root = -1.0f;
  float prior_temp = 0.1f;
  bool prior_linear = false;

  float PriorMix(bool root) const {
    return root && prior_mix_root >= 0.0f ? prior_mix_root : prior_mix;
  }

  bool AffectsSelection() const { return mode != ChildQMode::kOff; }
  bool AffectsBackup() const {
    return backup != ChildQBackup::kOff && backup_weight > 0.0f;
  }
  bool AffectsPolicy() const {
    return prior_mix > 0.0f || prior_mix_root > 0.0f;
  }
  bool Enabled() const {
    return AffectsSelection() || AffectsPolicy() || AffectsBackup();
  }
  float EffectivePriorScale() const {
    return (prior_bias_slope > 0.0f && visit_value_var > 0.0f)
               ? prior_bias_slope * visit_value_var
               : prior_scale;
  }
};

// Edge storage. The mean is kept on a linear 16-bit grid over [-1, 1], sigma on
// an 8-bit logarithmic grid over [0, 1] where code 0 means "no child-Q".
inline uint16_t EncodeChildQ(float q) {
  if (!std::isfinite(q)) q = 0.0f;
  return static_cast<uint16_t>(
      std::lround((std::clamp(q, -1.0f, 1.0f) + 1.0f) * 32767.5f));
}

inline float DecodeChildQ(uint16_t code) { return code / 32767.5f - 1.0f; }

inline uint8_t EncodeChildSigma(float sigma) {
  if (!(sigma >= 0.0f)) return 0;  // Negative or NaN: absent.
  const float x =
      std::log1p(std::min(sigma, 1.0f) * 100.0f) / std::log1p(100.0f);
  return static_cast<uint8_t>(1 + std::lround(254.0f * x));
}

inline float DecodeChildSigma(uint8_t code) {
  if (code == 0) return -1.0f;
  return std::expm1((code - 1) / 254.0f * std::log1p(100.0f)) / 100.0f;
}

// The value PUCT selection uses for a child that has a child-Q, with kFpu or
// kPrior. `n` is the CHILD NODE's completed visits (in a tree the edge's; in a
// DAG the shared node's N over all parents -- rule 2: the head only fills a
// child with no evidence); `q`, `wl` and `d` are its search values when
// n > 0, `fpu` is the value the search gives an unvisited child. Terminal
// children keep their exact value.
inline float ChildQSelectionQ(const ChildQParams& params, float child_q,
                              float child_sigma, uint32_t n, bool terminal,
                              float q, float wl, float d, float draw_score,
                              float fpu) {
  if (terminal) return q;
  const float sigma = std::max(child_sigma, params.sigma_floor);
  float value;
  if (params.mode == ChildQMode::kFpu) {
    value = n > 0 ? q
                  : params.fpu_weight * child_q +
                        (1.0f - params.fpu_weight) * fpu;
  } else {
    const float n0 =
        std::clamp(params.EffectivePriorScale() / std::max(sigma * sigma, 1e-6f),
                   params.prior_min, params.prior_max);
    if (n > 0) {
      value = (n * wl + n0 * child_q) / (n + n0) + draw_score * d;
    } else {
      value = n0 > 0.0f ? child_q : fpu;
    }
  }
  if (n == 0 && params.floor) value = std::max(value, fpu);
  return value + params.optimism * sigma / std::sqrt(1.0f + n);
}

// The aggregate A of the 1-ply backup over a node's legal moves: `policy` the
// node's priors, `child_q` / `child_sigma` the head, all in the node's
// side-to-move frame (the frame of its own value).
inline float ChildQBackupAggregate(const ChildQParams& params,
                                   std::span<const float> policy,
                                   std::span<const float> child_q,
                                   std::span<const float> child_sigma) {
  const size_t count = std::min(policy.size(), child_q.size());
  double best = std::numeric_limits<double>::lowest();
  for (size_t i = 0; i < count; ++i) {
    double v = child_q[i];
    if (params.backup == ChildQBackup::kMaxSigma && i < child_sigma.size()) {
      v -= std::max(child_sigma[i], 0.0f);
    }
    best = std::max(best, v);
  }
  if (params.backup == ChildQBackup::kMax ||
      params.backup == ChildQBackup::kMaxSigma) {
    return static_cast<float>(best);
  }
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < count; ++i) {
    const double w = params.backup == ChildQBackup::kPolicyMean
                         ? static_cast<double>(policy[i])
                         : std::exp((child_q[i] - best) / params.backup_temp);
    num += w * child_q[i];
    den += w;
  }
  return den > 0.0 ? static_cast<float>(num / den) : static_cast<float>(best);
}

// Linear blend, in place: P'_a = (1 - mix) * P_a + mix * softmax(child_q /
// temp)_a over the moves given (the legal ones; P sums to 1 over them).
inline void ApplyChildQPriorMixLinear(std::span<float> policy,
                                      std::span<const float> child_q,
                                      float mix, float temp) {
  constexpr size_t kMaxMoves = 256;
  const size_t count = std::min({policy.size(), child_q.size(), kMaxMoves});
  if (count == 0) return;
  std::array<double, kMaxMoves> weight;
  double max_logit = std::numeric_limits<double>::lowest();
  for (size_t i = 0; i < count; ++i) {
    weight[i] = static_cast<double>(child_q[i]) / temp;
    max_logit = std::max(max_logit, weight[i]);
  }
  double total = 0.0;
  for (size_t i = 0; i < count; ++i) {
    total += weight[i] = std::exp(weight[i] - max_logit);
  }
  for (size_t i = 0; i < count; ++i) {
    policy[i] = static_cast<float>(std::min(
        (1.0 - mix) * static_cast<double>(policy[i]) + mix * weight[i] / total,
        1.0));
  }
}

// Blends a node's policy with the child-Q of the same moves, in place:
// P'_a ~ P_a^(1 - mix) * exp(mix * child_q_a / temp), normalised over the moves
// given (the legal ones); with `linear`, ApplyChildQPriorMixLinear.
inline void ApplyChildQPriorMix(std::span<float> policy,
                                std::span<const float> child_q, float mix,
                                float temp, bool linear = false) {
  if (linear) return ApplyChildQPriorMixLinear(policy, child_q, mix, temp);
  constexpr size_t kMaxMoves = 256;
  const size_t count = std::min({policy.size(), child_q.size(), kMaxMoves});
  std::array<double, kMaxMoves> weight;
  std::array<bool, kMaxMoves> excluded;
  double max_logit = std::numeric_limits<double>::lowest();
  for (size_t i = 0; i < count; ++i) {
    // P^(1 - mix) of a zero-policy move is zero unless mix is 1.
    excluded[i] = mix < 1.0f && !(policy[i] > 0.0f);
    if (excluded[i]) continue;
    weight[i] = mix * static_cast<double>(child_q[i]) / temp;
    if (mix < 1.0f) {
      weight[i] += (1.0 - mix) * std::log(static_cast<double>(policy[i]));
    }
    max_logit = std::max(max_logit, weight[i]);
  }
  if (max_logit == std::numeric_limits<double>::lowest()) return;
  double total = 0.0;
  for (size_t i = 0; i < count; ++i) {
    if (!excluded[i]) total += weight[i] = std::exp(weight[i] - max_logit);
  }
  for (size_t i = 0; i < count; ++i) {
    policy[i] = excluded[i]
                    ? 0.0f
                    : static_cast<float>(std::min(weight[i] / total, 1.0));
  }
}

}  // namespace lczero
