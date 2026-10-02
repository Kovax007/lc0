/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess. If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include "network_fingerprint.h"

#include <cstddef>

#include "utils/exception.h"

namespace lczero {
namespace lc0ex {
namespace {

using NetworkFormat = pblczero::NetworkFormat;

NetworkFormat::ActivationFunction ResolveDefaultActivation(
    NetworkFormat::DefaultActivation activation) {
  switch (activation) {
    case NetworkFormat::DEFAULT_ACTIVATION_RELU:
      return NetworkFormat::ACTIVATION_RELU;
    case NetworkFormat::DEFAULT_ACTIVATION_MISH:
      return NetworkFormat::ACTIVATION_MISH;
  }
  throw Exception("Unsupported default activation in network format.");
}

NetworkFormat::ActivationFunction ResolveActivation(
    NetworkFormat::ActivationFunction activation,
    NetworkFormat::ActivationFunction default_activation) {
  return activation == NetworkFormat::ACTIVATION_DEFAULT ? default_activation
                                                         : activation;
}

void MarkEmbedding(pblczero::Weights* weights) {
  weights->mutable_ip_emb_preproc_w();
  weights->mutable_ip_emb_preproc_b();
  weights->mutable_ip_emb_w();
  weights->mutable_ip_emb_b();
  weights->mutable_ip_emb_ln_gammas();
  weights->mutable_ip_emb_ln_betas();
  weights->mutable_ip_mult_gate();
  weights->mutable_ip_add_gate();

  auto* ffn = weights->mutable_ip_emb_ffn();
  ffn->mutable_dense1_w();
  ffn->mutable_dense1_b();
  ffn->mutable_dense2_w();
  ffn->mutable_dense2_b();
  weights->mutable_ip_emb_ffn_ln_gammas();
  weights->mutable_ip_emb_ffn_ln_betas();
}

void MarkEncoder(pblczero::Weights::EncoderLayer* encoder) {
  auto* mha = encoder->mutable_mha();
  auto* smolgen = mha->mutable_smolgen();
  smolgen->mutable_compress();
  smolgen->mutable_dense1_w();
  smolgen->mutable_dense1_b();
  smolgen->mutable_ln1_gammas();
  smolgen->mutable_ln1_betas();
  smolgen->mutable_dense2_w();
  smolgen->mutable_dense2_b();
  smolgen->mutable_ln2_gammas();
  smolgen->mutable_ln2_betas();

  mha->mutable_q_w();
  mha->mutable_q_b();
  mha->mutable_k_w();
  mha->mutable_k_b();
  mha->mutable_v_w();
  mha->mutable_v_b();
  mha->mutable_dense_w();
  mha->mutable_dense_b();

  encoder->mutable_ln1_gammas();
  encoder->mutable_ln1_betas();
  auto* ffn = encoder->mutable_ffn();
  ffn->mutable_dense1_w();
  ffn->mutable_dense1_b();
  ffn->mutable_dense2_w();
  ffn->mutable_dense2_b();
  encoder->mutable_ln2_gammas();
  encoder->mutable_ln2_betas();
}

void MarkPolicyHead(pblczero::Weights* weights) {
  auto* policy = weights->mutable_policy_heads()->mutable_vanilla();
  policy->mutable_ip_pol_w();
  policy->mutable_ip_pol_b();
  policy->mutable_ip2_pol_w();
  policy->mutable_ip2_pol_b();
  policy->mutable_ip3_pol_w();
  policy->mutable_ip3_pol_b();
  policy->mutable_ip4_pol_w();
}

void MarkValueHead(pblczero::Weights* weights) {
  auto* value = weights->mutable_value_heads()->mutable_winner();
  value->mutable_ip_val_w();
  value->mutable_ip_val_b();
  value->mutable_ip1_val_w();
  value->mutable_ip1_val_b();
  value->mutable_ip2_val_w();
  value->mutable_ip2_val_b();
}

void MarkMovesLeftHead(pblczero::Weights* weights) {
  weights->mutable_ip_mov_w();
  weights->mutable_ip_mov_b();
  weights->mutable_ip1_mov_w();
  weights->mutable_ip1_mov_b();
  weights->mutable_ip2_mov_w();
  weights->mutable_ip2_mov_b();
}

}  // namespace

pblczero::Net BuildNetworkFingerprint(const pblczero::Net& network) {
  pblczero::Net fingerprint;
  const auto& source_format = network.format().network_format();
  auto* target_format = fingerprint.mutable_format()->mutable_network_format();
  target_format->set_input(source_format.input());
  target_format->set_output(source_format.output());
  target_format->set_network(source_format.network());
  target_format->set_policy(source_format.policy());
  target_format->set_value(source_format.value());
  target_format->set_moves_left(source_format.moves_left());
  target_format->set_input_embedding(source_format.input_embedding());
  target_format->set_default_activation(source_format.default_activation());

  const auto default_activation =
      ResolveDefaultActivation(source_format.default_activation());
  target_format->set_ffn_activation(
      ResolveActivation(source_format.ffn_activation(), default_activation));
  target_format->set_smolgen_activation(ResolveActivation(
      source_format.smolgen_activation(), default_activation));

  const auto& source_weights = network.weights();
  auto* target_weights = fingerprint.mutable_weights();
  target_weights->set_headcount(source_weights.headcount());

  MarkEmbedding(target_weights);
  target_weights->mutable_smolgen_w();
  MarkMovesLeftHead(target_weights);
  MarkPolicyHead(target_weights);
  MarkValueHead(target_weights);

  for (std::size_t i = 0; i < source_weights.encoder().size(); ++i) {
    MarkEncoder(target_weights->add_encoder());
  }

  return fingerprint;
}

}  // namespace lc0ex
}  // namespace lczero
