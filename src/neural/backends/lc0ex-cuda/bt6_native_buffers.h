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

// The named buffers of an lc0ex program, built from a native BT6 net file
// (NetworkFormat.network = NETWORK_ATTENTIONBODY_BT6, FLOAT32 layers).
//
// An lc0ex program reads its weights as named buffers. For a net exported as
// an ONNX carrier those buffers arrive as ONNX initializers; a native BT6 file
// carries its weights as proto layers instead, so this builds the same buffers
// from them: the layers themselves, the folds the program serves (gated FFN
// pairs, the edge-attention temperature folds, the policy blend), and, for an
// int8 program, the integer weights from Weights.int8_calibration. The result
// is byte-identical to the buffers the lab carrier tools write for the same net.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "proto/net.pb.h"

namespace lczero {
namespace lc0ex {

// ONNX TensorProto data type numbers, as lc0ex buffers declare them.
inline constexpr int kBt6Float = 1;
inline constexpr int kBt6Uint8 = 2;
inline constexpr int kBt6Int32 = 6;
inline constexpr int kBt6Float16 = 10;

struct Bt6HostBuffer {
  int data_type = 0;
  std::vector<std::uint64_t> shape;
  std::vector<std::byte> bytes;
};

struct Bt6ServeOptions {
  // The served value head: winner, q, st, or a value_head_map key.
  std::string value_head = "winner";
  // The value head whose error output fills /value_error/ (programs with an
  // error output only).
  std::string error_head = "st";
  // The served policy: a geometric blend of named heads (name, weight), folded
  // into one head. A single head is {name, 1}.
  std::vector<std::pair<std::string, double>> policy = {{"vanilla", 1.0}};
  // The program's policy width D (the /policy/Q/w buffer's columns); 0 = the
  // sum of the blended heads' widths. Narrower blends are zero-padded.
  std::uint64_t policy_width = 0;
  bool int8 = false;   // the program serves int8 GEMM sites
  bool error = false;  // the program has an error output
};

// True for a native BT6 net file.
bool IsNativeBt6(const pblczero::Net& net);

// Every buffer the options select, by lc0ex buffer name.
std::unordered_map<std::string, Bt6HostBuffer> BuildBt6Buffers(
    const pblczero::Net& net, const Bt6ServeOptions& options);

// The shortest decimal that round-trips `value`, read as a double: a blend
// weight typed as 0.4315 folds as the double 0.4315, not as 0.4315f.
double DecimalOfFloat(float value);

// The names of the policy heads of a native BT6 file that lc0ex serves (the
// attention heads on the shared embedding), sorted (the order a blend of
// several heads is folded in).
std::vector<std::string> Bt6PolicyHeadNames(const pblczero::Net& net);

}  // namespace lc0ex
}  // namespace lczero
