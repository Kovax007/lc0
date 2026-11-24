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

// Every number here must come out bit-identical to the lab carrier tools, so
// the arithmetic follows them exactly: float64 folds rounded once to the
// buffer's type, float32 products where the composer multiplies in float32,
// FP16 rounded through float32 (round half to even), Python's exactly rounded
// fsum for the int8 activation step. Build without -ffast-math and without FMA
// contraction (the default x86-64 target has no FMA).

#include "bt6_native_buffers.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <functional>
#include <iterator>
#include <numeric>
#include <string_view>

#include "neural/tables/attention_policy_map.h"
#include "utils/exception.h"
#include "utils/fp16_utils.h"

namespace lczero {
namespace lc0ex {
namespace {

using Layer = pblczero::Weights::Layer;
using Int8Site = pblczero::Weights::Int8Site;

// Every BT6 head is an entry of PolicyHeads / ValueHeads; its BT6 fields say
// its kind (net.proto). The policy fold serves the attention heads that read
// the shared embedding (policy_heads.ip_pol_w): not a head with its own
// embedding (ip_pol_w), a child-Q head (cq_*) or a depth-attending head
// (depth_*). The value builder serves the plain value heads: not a
// depth-attending one (depth_*).
bool IsServedPolicyHead(const pblczero::Weights::PolicyHead& head) {
  return !head.has_ip_pol_w() && !head.has_cq_w_bins() &&
         !head.has_depth_square_attention() && head.depth_probe_size() == 0;
}

bool IsServedValueHead(const pblczero::Weights::ValueHead& head) {
  return !head.has_depth_square_attention() &&
         !head.has_depth_global_attention() && head.depth_probe_w_size() == 0;
}

// A layer decoded to the training tensor, row-major with shape `dims`.
struct Arr {
  std::vector<std::size_t> dims;
  std::vector<float> v;
  std::size_t size() const { return v.size(); }
};

std::uint64_t U(std::size_t x) { return static_cast<std::uint64_t>(x); }

// A FLOAT32 or FLOAT16 layer as the tensor it was written from. The BT6 writer
// stores the tensor's axes reversed (column-major order of `dims`; a kernel
// [in, out] is lc0's [out][in]), so this reverses them back. A FLOAT16 layer
// (the release file with FP16 weights) is widened exactly; its values are the
// FLOAT32 file's rounded to nearest even, so every FP16 buffer built from them
// is bit-identical to one built from the FLOAT32 file.
Arr Dec(const Layer& layer, const std::string& what) {
  const bool f16 = layer.encoding() == Layer::FLOAT16;
  if (layer.encoding() != Layer::FLOAT32 && !f16) {
    throw Exception("BT6 layer " + what +
                    ": lc0ex serves FLOAT32 and FLOAT16 BT6 layers only.");
  }
  const std::size_t width = f16 ? 2 : 4;
  const auto params = layer.params();
  if (params.empty() || params.size() % width != 0) {
    throw Exception("BT6 layer " + what + " is missing or truncated.");
  }
  const std::size_t n = params.size() / width;
  std::vector<std::size_t> dims(layer.dims().begin(), layer.dims().end());
  if (dims.empty()) dims = {n};
  if (std::accumulate(dims.begin(), dims.end(), std::size_t{1},
                      std::multiplies<std::size_t>()) != n) {
    throw Exception("BT6 layer " + what + ": dims do not match its size.");
  }
  std::vector<float> raw(n);
  if (f16) {
    for (std::size_t i = 0; i < n; ++i) {
      std::uint16_t h;
      std::memcpy(&h, params.data() + 2 * i, 2);
      raw[i] = FP16toFP32(h);
    }
  } else {
    std::memcpy(raw.data(), params.data(), params.size());
  }
  Arr out{dims, std::vector<float>(n)};
  if (dims.size() == 1) {
    out.v = std::move(raw);
  } else if (dims.size() == 2) {
    const std::size_t r = dims[0], c = dims[1];
    for (std::size_t i = 0; i < r; ++i) {
      for (std::size_t j = 0; j < c; ++j) out.v[i * c + j] = raw[j * r + i];
    }
  } else {
    std::vector<std::size_t> index(dims.size(), 0);
    for (std::size_t flat = 0; flat < n; ++flat) {
      std::size_t offset = 0, stride = 1;
      for (std::size_t d = 0; d < dims.size(); ++d) {
        offset += index[d] * stride;
        stride *= dims[d];
      }
      out.v[flat] = raw[offset];
      for (std::size_t d = dims.size(); d-- > 0;) {
        if (++index[d] < dims[d]) break;
        index[d] = 0;
      }
    }
  }
  return out;
}

// A dense layer's kernel shape [in, out], out = its bias's length.
std::vector<std::uint64_t> KernelShape(const Arr& w, const Arr& b,
                                       const std::string& what) {
  if (b.size() == 0 || w.size() % b.size() != 0) {
    throw Exception("BT6 layer " + what + ": kernel and bias do not match.");
  }
  return {U(w.size() / b.size()), U(b.size())};
}

// float32 -> float16 bits, round half to even, subnormals kept (numpy's
// npy_floatbits_to_halfbits). 0xffff (a NaN no finite input produces) marks
// overflow, infinity or NaN.
std::uint16_t HalfBits(float value) {
  std::uint32_t f;
  std::memcpy(&f, &value, 4);
  const std::uint32_t sign = (f & 0x80000000u) >> 16;
  std::uint32_t exp = f & 0x7f800000u;
  if (exp >= 0x47800000u) return 0xffff;
  if (exp <= 0x38000000u) {
    if (exp < 0x33000000u) return static_cast<std::uint16_t>(sign);
    exp >>= 23;
    std::uint32_t sig = 0x00800000u + (f & 0x007fffffu);
    sig >>= (113 - exp);
    if (((sig & 0x00003fffu) != 0x00001000u) || (f & 0x000007ffu)) {
      sig += 0x00001000u;
    }
    return static_cast<std::uint16_t>(sign + (sig >> 13));
  }
  const std::uint32_t hexp = (exp - 0x38000000u) >> 13;
  std::uint32_t sig = f & 0x007fffffu;
  if ((sig & 0x00003fffu) != 0x00001000u) sig += 0x00001000u;
  const std::uint32_t bits = hexp + (sig >> 13);
  if (bits >= 0x7c00u) return 0xffff;
  return static_cast<std::uint16_t>(sign + bits);
}

// Python's math.fsum: the correctly rounded sum (Shewchuk's partials, CPython's
// half-even fix-up across partials).
double FSum(const std::vector<double>& values) {
  std::vector<double> p;
  for (double x : values) {
    std::size_t i = 0;
    for (double y : p) {
      if (std::fabs(x) < std::fabs(y)) std::swap(x, y);
      const double hi = x + y;
      const double lo = y - (hi - x);
      if (lo != 0.0) p[i++] = lo;
      x = hi;
    }
    p.resize(i);
    if (x != 0.0) {
      if (!std::isfinite(x)) throw Exception("BT6 int8: a non-finite sum.");
      p.push_back(x);
    }
  }
  double hi = 0.0, lo = 0.0;
  std::size_t n = p.size();
  if (n > 0) {
    hi = p[--n];
    while (n > 0) {
      const double x = hi;
      const double y = p[--n];
      hi = x + y;
      lo = y - (hi - x);
      if (lo != 0.0) break;
    }
    if (n > 0 && ((lo < 0.0 && p[n - 1] < 0.0) || (lo > 0.0 && p[n - 1] > 0.0))) {
      const double y = lo * 2.0;
      const double x = hi + y;
      if (y == x - hi) hi = x;
    }
  }
  return hi;
}

std::vector<float> ConcatColumns(const std::vector<const Arr*>& parts) {
  const std::size_t rows = parts.at(0)->dims.at(0);
  std::size_t cols = 0;
  for (const auto* p : parts) {
    if (p->dims.size() != 2 || p->dims[0] != rows) {
      throw Exception("BT6: mismatched kernels in a concatenation.");
    }
    cols += p->dims[1];
  }
  std::vector<float> out(rows * cols);
  for (std::size_t r = 0; r < rows; ++r) {
    std::size_t o = 0;
    for (const auto* p : parts) {
      const std::size_t c = p->dims[1];
      std::copy_n(p->v.begin() + r * c, c, out.begin() + r * cols + o);
      o += c;
    }
  }
  return out;
}

std::vector<float> Concat(const std::vector<const Arr*>& parts) {
  std::vector<float> out;
  for (const auto* p : parts) out.insert(out.end(), p->v.begin(), p->v.end());
  return out;
}

// A gated FFN's bias as the program reads it: [0 | up].
std::vector<float> GateBias(const Arr& up_bias) {
  std::vector<float> out(2 * up_bias.size(), 0.0f);
  std::copy(up_bias.v.begin(), up_bias.v.end(), out.begin() + up_bias.size());
  return out;
}

std::vector<double> ToDouble(const std::vector<float>& v) {
  return std::vector<double>(v.begin(), v.end());
}

class Builder {
 public:
  Builder(const pblczero::Net& net, const Bt6ServeOptions& options)
      : w_(net.weights()),
        fmt_(net.format().network_format().bt6()),
        options_(options) {}

  std::unordered_map<std::string, Bt6HostBuffer> Build();

 private:
  // FP16 values are rounded through float32 first.
  template <typename T>
  void Put(const std::string& name, std::vector<std::uint64_t> shape,
           const std::vector<T>& values, int data_type = kBt6Float16);
  void PutBytes(const std::string& name, std::vector<std::uint64_t> shape,
                int data_type, std::vector<std::byte> bytes);
  std::vector<double> Offsets(const Layer& t_off, const std::string& what) const;
  const pblczero::Weights::PolicyHead& PolicyHead(const std::string& name) const;
  const pblczero::Weights::ValueHead& ValueHead(const std::string& name) const;
  void Dense(const std::string& stem, const Layer& w, const Layer& b,
             const std::string& what);

  void Embedding();
  void Prologue();
  void Encoders();
  void EdgeSites();
  void Heads();
  void Policy();
  void Int8();

  const pblczero::Weights& w_;
  const pblczero::Bt6Format& fmt_;
  const Bt6ServeOptions& options_;
  std::unordered_map<std::string, Bt6HostBuffer> out_;
  std::uint64_t d_ = 0;
  std::size_t heads_ = 0, head_dim_ = 0;
  double alpha_ = 1.0;
  std::vector<int> relidx_;
  Arr pair_in_;                   // pair_stream.p_in [C, pair_d_e]
  std::vector<double> pair_off_;  // [pair_d_e, 64, 64]
};

template <typename T>
void Builder::Put(const std::string& name, std::vector<std::uint64_t> shape,
                  const std::vector<T>& values, int data_type) {
  std::uint64_t n = 1;
  for (auto d : shape) n *= d;
  if (values.size() != n) {
    throw Exception("BT6 buffer " + name + ": " + std::to_string(values.size()) +
                    " values for its shape.");
  }
  std::vector<std::byte> bytes;
  if (data_type == kBt6Float16) {
    bytes.resize(2 * n);
    for (std::size_t i = 0; i < n; ++i) {
      const std::uint16_t h = HalfBits(static_cast<float>(values[i]));
      if (h == 0xffff) throw Exception("BT6 buffer " + name + " overflows FP16.");
      std::memcpy(bytes.data() + 2 * i, &h, 2);
    }
  } else if (data_type == kBt6Float) {
    bytes.resize(4 * n);
    for (std::size_t i = 0; i < n; ++i) {
      const float f = static_cast<float>(values[i]);
      std::memcpy(bytes.data() + 4 * i, &f, 4);
    }
  } else {
    throw Exception("BT6 buffer " + name + ": not a float type.");
  }
  PutBytes(name, std::move(shape), data_type, std::move(bytes));
}

void Builder::PutBytes(const std::string& name, std::vector<std::uint64_t> shape,
                       int data_type, std::vector<std::byte> bytes) {
  if (!out_.emplace(name, Bt6HostBuffer{data_type, std::move(shape),
                                        std::move(bytes)})
           .second) {
    throw Exception("BT6 buffer " + name + " built twice.");
  }
}

// The runtime gather t_off.T[:, relidx]: [channels, 64, 64], float64.
std::vector<double> Builder::Offsets(const Layer& t_off,
                                     const std::string& what) const {
  const Arr t = Dec(t_off, what);  // [225, channels]
  if (t.dims.size() != 2 || t.dims[0] != 225) {
    throw Exception("BT6 layer " + what + " is not [225, channels].");
  }
  const std::size_t channels = t.dims[1];
  std::vector<double> out(channels * 4096);
  for (std::size_t c = 0; c < channels; ++c) {
    for (std::size_t s = 0; s < 4096; ++s) {
      out[c * 4096 + s] = t.v[relidx_[s] * channels + c];
    }
  }
  return out;
}

const pblczero::Weights::PolicyHead& Builder::PolicyHead(
    const std::string& name) const {
  const auto& ph = w_.policy_heads();
  if (name == "vanilla") return ph.vanilla();
  if (name == "optimistic_st") return ph.optimistic_st();
  if (name == "soft") return ph.soft();
  if (name == "opponent") return ph.opponent();
  for (const auto& entry : ph.policy_head_map()) {
    if (entry.key() != name) continue;
    if (!IsServedPolicyHead(entry.value())) {
      throw Exception("BT6 policy head '" + name +
                      "' is not an attention head on the shared embedding "
                      "(own embedding, child-Q or depth-attending); lc0ex "
                      "does not serve it as the policy.");
    }
    return entry.value();
  }
  throw Exception("The BT6 file has no policy head '" + name + "'.");
}

const pblczero::Weights::ValueHead& Builder::ValueHead(
    const std::string& name) const {
  const auto& vh = w_.value_heads();
  if (name == "winner") return vh.winner();
  if (name == "q") return vh.q();
  if (name == "st") return vh.st();
  for (const auto& entry : vh.value_head_map()) {
    if (entry.key() != name) continue;
    if (!IsServedValueHead(entry.value())) {
      throw Exception("BT6 value head '" + name +
                      "' is depth-attending; lc0ex does not serve it.");
    }
    return entry.value();
  }
  throw Exception("The BT6 file has no value head '" + name + "'.");
}

void Builder::Dense(const std::string& stem, const Layer& w, const Layer& b,
                    const std::string& what) {
  const Arr kernel = Dec(w, what + "_w");
  const Arr bias = Dec(b, what + "_b");
  Put(stem + "/w", KernelShape(kernel, bias, what), kernel.v);
  Put(stem + "/b", {U(bias.size())}, bias.v);
}

std::unordered_map<std::string, Bt6HostBuffer> Builder::Build() {
  d_ = fmt_.d_model();
  heads_ = fmt_.heads();
  if (d_ == 0 || heads_ == 0 || d_ % heads_ != 0 ||
      w_.encoder_size() != fmt_.num_blocks()) {
    throw Exception("The BT6 file's format block does not match its weights.");
  }
  head_dim_ = d_ / heads_;
  alpha_ = fmt_.has_residual_alpha() ? fmt_.residual_alpha() : 1.0;
  relidx_.resize(4096);
  for (int q = 0; q < 64; ++q) {
    for (int k = 0; k < 64; ++k) {
      relidx_[q * 64 + k] = (q / 8 - k / 8 + 7) * 15 + (q % 8 - k % 8 + 7);
    }
  }
  pair_in_ = Dec(w_.pair_stream().p_in(), "pair_stream.p_in");
  if (pair_in_.dims.size() != 2) {
    throw Exception("BT6: pair_stream.p_in is not [C, pair_d_e].");
  }
  pair_off_ = Offsets(w_.pair_stream().t_off(), "pair_stream.t_off");
  Embedding();
  Prologue();
  Encoders();
  EdgeSites();
  Heads();
  Policy();
  if (options_.int8) Int8();
  return std::move(out_);
}

void Builder::Embedding() {
  const auto& ffn = w_.ip_emb_ffn();
  Dense("/attn_body/preproc", w_.ip_emb_preproc_w(), w_.ip_emb_preproc_b(),
        "ip_emb_preproc");
  Dense("/attn_body/matmul", w_.ip_emb_w(), w_.ip_emb_b(), "ip_emb");
  Put("/attn_body/ln0/scale", {d_}, Dec(w_.ip_emb_ln_gammas(), "ip_emb_ln_gammas").v);
  Put("/attn_body/ln0/bias", {d_}, Dec(w_.ip_emb_ln_betas(), "ip_emb_ln_betas").v);
  Arr mult = Dec(w_.ip_mult_gate(), "ip_mult_gate");
  for (auto& x : mult.v) x = std::max(x, 0.0f);
  Put("/attn_body/mult_gate/w", {64, d_}, mult.v);
  Put("/attn_body/position/w", {1, 64, d_}, Dec(w_.ip_add_gate(), "ip_add_gate").v);
  const Arr gate = Dec(ffn.dense_gate_w(), "ip_emb_ffn.dense_gate_w");
  const Arr up = Dec(ffn.dense1_w(), "ip_emb_ffn.dense1_w");
  const std::uint64_t dff = gate.dims.at(1);
  Put("/attn_body/ffn/dense1/w", {d_, 2 * dff}, ConcatColumns({&gate, &up}));
  Put("/attn_body/ffn/dense1/b", {2 * dff},
      GateBias(Dec(ffn.dense1_b(), "ip_emb_ffn.dense1_b")));
  Put("/attn_body/ffn/dense2/w", {dff, d_}, Dec(ffn.dense2_w(), "ip_emb_ffn.dense2_w").v);
  Put("/attn_body/ffn/dense2/b", {d_}, Dec(ffn.dense2_b(), "ip_emb_ffn.dense2_b").v);
  if (!fmt_.has_embedding_ffn_alpha()) {
    throw Exception("The BT6 file's format has no embedding_ffn_alpha.");
  }
  Put("/attn_body/ffn/alpha/w", {1}, std::vector<float>{fmt_.embedding_ffn_alpha()});
  Put("/attn_body/ln1/scale", {d_}, Dec(w_.ip_emb_ffn_ln_gammas(), "ip_emb_ffn_ln_gammas").v);
  Put("/attn_body/ln1/bias", {d_}, Dec(w_.ip_emb_ffn_ln_betas(), "ip_emb_ffn_ln_betas").v);
}

void Builder::Prologue() {
  const auto& es = w_.edge_stream();
  const Arr& pin = pair_in_;
  const Arr ein = Dec(es.p_in(), "edge_stream.p_in");
  const std::size_t channels = pin.dims[0];
  if (channels != 34 || ein.dims.size() != 2 || ein.dims[0] != channels) {
    throw Exception("BT6: lc0ex serves 34 attack-graph channels.");
  }
  Put("/prologue/channel_mix", {U(channels), U(pin.dims[1])}, pin.v, kBt6Float);
  Put("/prologue/offsets", {U(pin.dims[1]), 64, 64}, pair_off_, kBt6Float);
  Put("/prologue/edge_mix", {U(channels), U(ein.dims[1])}, ein.v, kBt6Float);
  Put("/prologue/edge_offsets", {U(ein.dims[1]), 64, 64},
      Offsets(es.t_off(), "edge_stream.t_off"), kBt6Float);
  // Each code group's table row = the sum of [pair p_in | edge p_in]'s rows
  // over the code's set bits.
  const std::size_t width = pin.dims[1] + ein.dims[1];
  std::vector<double> mix(channels * width);
  for (std::size_t c = 0; c < channels; ++c) {
    for (std::size_t j = 0; j < pin.dims[1]; ++j) {
      mix[c * width + j] = pin.v[c * pin.dims[1] + j];
    }
    for (std::size_t j = 0; j < ein.dims[1]; ++j) {
      mix[c * width + pin.dims[1] + j] = ein.v[c * ein.dims[1] + j];
    }
  }
  constexpr int kGroups[4][2] = {{0, 9}, {9, 8}, {17, 9}, {26, 8}};
  std::vector<double> codes;
  std::size_t rows = 0;
  for (const auto& group : kGroups) {
    const int first = group[0], bits = group[1];
    for (int code = 0; code < (1 << bits); ++code, ++rows) {
      for (std::size_t j = 0; j < width; ++j) {
        double sum = 0.0;
        for (int b = 0; b < bits; ++b) {
          if ((code >> b) & 1) sum += mix[(first + b) * width + j];
        }
        codes.push_back(sum);
      }
    }
  }
  Put("/prologue/mix_codes", {U(rows), U(width)}, codes, kBt6Float);
  Put("/encoder/qk_scale", {1},
      std::vector<double>{1.0 / std::sqrt(static_cast<double>(head_dim_))});
}

void Builder::Encoders() {
  const std::uint64_t h = heads_;
  const std::size_t edges = pair_in_.dims[0], pd = pair_in_.dims[1];
  const double qk = 1.0 / std::sqrt(static_cast<double>(head_dim_));
  for (std::size_t i = 0; i < w_.encoder_size(); ++i) {
    const auto& enc = w_.encoder(i);
    const auto& m = enc.mha();
    const auto& f = enc.ffn();
    const std::string p = "/encoder" + std::to_string(i);
    const std::string e = "encoder[" + std::to_string(i) + "].";
    std::vector<Arr> w = {Dec(m.q_w(), e + "q_w"), Dec(m.k_w(), e + "k_w"),
                          Dec(m.v_w(), e + "v_w")};
    std::vector<Arr> b = {Dec(m.q_b(), e + "q_b"), Dec(m.k_b(), e + "k_b"),
                          Dec(m.v_b(), e + "v_b")};
    if (m.has_output_gate_w()) w.push_back(Dec(m.output_gate_w(), e + "output_gate_w"));
    if (m.has_output_gate_b()) b.push_back(Dec(m.output_gate_b(), e + "output_gate_b"));
    std::vector<const Arr*> wp, bp;
    for (const auto& a : w) wp.push_back(&a);
    for (const auto& a : b) bp.push_back(&a);
    Put(p + "/mha/qkvg/w", {d_, U(w.size()) * d_}, ConcatColumns(wp));
    Put(p + "/mha/qkvg/b", {U(b.size()) * d_}, Concat(bp));
    Put(p + "/mha/out/w", {d_, d_}, Dec(m.dense_w(), e + "dense_w").v);
    Put(p + "/mha/out/b", {d_}, Dec(m.dense_b(), e + "dense_b").v);
    Put(p + "/mha/alpha/w", {1}, std::vector<float>{static_cast<float>(alpha_)});
    Put(p + "/ln1/scale", {d_}, Dec(enc.ln1_gammas(), e + "ln1_gammas").v);
    Put(p + "/ln1/bias", {d_}, Dec(enc.ln1_betas(), e + "ln1_betas").v);
    const Arr gate = Dec(f.dense_gate_w(), e + "dense_gate_w");
    const Arr up = Dec(f.dense1_w(), e + "dense1_w");
    const std::uint64_t dff = gate.dims.at(1);
    Put(p + "/ffn/dense1/w", {d_, 2 * dff}, ConcatColumns({&gate, &up}));
    Put(p + "/ffn/dense1/b", {2 * dff}, GateBias(Dec(f.dense1_b(), e + "dense1_b")));
    Put(p + "/ffn/dense2/w", {dff, d_}, Dec(f.dense2_w(), e + "dense2_w").v);
    Put(p + "/ffn/dense2/b", {d_}, Dec(f.dense2_b(), e + "dense2_b").v);
    Put(p + "/ffn/alpha/w", {1}, std::vector<float>{static_cast<float>(alpha_)});
    Put(p + "/ln2/scale", {d_}, Dec(enc.ln2_gammas(), e + "ln2_gammas").v);
    Put(p + "/ln2/bias", {d_}, Dec(enc.ln2_betas(), e + "ln2_betas").v);

    // Edge attention, with the per-head temperatures folded in (t_node into
    // the node terms, t_edge into the edge read); float64, rounded once.
    const Arr t_node = Dec(m.edge_t_node(), e + "edge_t_node");
    const Arr t_edge = Dec(m.edge_t_edge(), e + "edge_t_edge");
    if (t_node.size() != h || t_edge.size() != h) {
      throw Exception("BT6 " + e + "edge temperatures are not one per head.");
    }
    auto tempered = [&](const Arr& t, const Arr& table) {
      const std::size_t per = table.size() / h;
      std::vector<double> out(table.size());
      for (std::size_t a = 0; a < h; ++a) {
        for (std::size_t c = 0; c < per; ++c) {
          out[a * per + c] = static_cast<double>(t.v[a]) *
                             static_cast<double>(table.v[a * per + c]);
        }
      }
      return out;
    };
    std::vector<double> qks(h);
    for (std::size_t a = 0; a < h; ++a) qks[a] = static_cast<double>(t_node.v[a]) * qk;
    Put(p + "/mha/egt/qk_scale", {h}, qks, kBt6Float);
    Put(p + "/mha/egt/attack", {h, U(edges)},
        tempered(t_node, Dec(m.attack_w(), e + "attack_w")), kBt6Float);
    Put(p + "/mha/egt/key", {h, U(edges), U(head_dim_)},
        tempered(t_node, Dec(m.attack_gate_k(), e + "attack_gate_k")), kBt6Float);
    Put(p + "/mha/egt/query", {h, U(edges), U(head_dim_)},
        tempered(t_node, Dec(m.attack_gate_q(), e + "attack_gate_q")), kBt6Float);
    const Arr hm = Dec(m.pair_att_w(), e + "pair_att_w");  // [heads, pair_d_e]
    if (hm.size() != h * pd) throw Exception("BT6 " + e + "pair_att_w is not [heads, pair_d_e].");
    std::vector<double> sc(h * edges), cb(h * 4096);
    for (std::size_t a = 0; a < h; ++a) {
      const double t = t_node.v[a];
      for (std::size_t c = 0; c < edges; ++c) {
        double sum = 0.0;
        for (std::size_t k = 0; k < pd; ++k) {
          sum += static_cast<double>(hm.v[a * pd + k]) *
                 static_cast<double>(pair_in_.v[c * pd + k]);
        }
        sc[a * edges + c] = t * sum;
      }
      for (std::size_t s = 0; s < 4096; ++s) {
        double sum = 0.0;
        for (std::size_t k = 0; k < pd; ++k) {
          sum += static_cast<double>(hm.v[a * pd + k]) * pair_off_[k * 4096 + s];
        }
        cb[a * 4096 + s] = t * sum;
      }
    }
    Put(p + "/mha/egt/scaled_coefficients", {h, U(edges)}, sc, kBt6Float);
    Put(p + "/mha/egt/constant_bias", {h, 64, 64}, cb, kBt6Float);
    const Arr ew = Dec(m.edge_e_w(), e + "edge_e_w");
    Put(p + "/mha/egt/edge_read/w", {h, U(ew.size() / h)}, tempered(t_edge, ew), kBt6Float);
    const Arr door = Dec(m.edge_m_w(), e + "edge_m_w");
    Put(p + "/mha/egt/door/w", {h, U(door.size() / h)}, door.v, kBt6Float);
    const Arr gw = Dec(m.edge_g_w(), e + "edge_g_w");
    Put(p + "/mha/egt/gate/w", {h, U(gw.size() / h)}, gw.v, kBt6Float);
    Put(p + "/mha/egt/gate/b", {h}, Dec(m.edge_g_b(), e + "edge_g_b").v, kBt6Float);
  }
  Put("/encoder/final_norm/scale", {d_},
      Dec(w_.encoder_final_norm().gammas(), "encoder_final_norm.gammas").v);
  Put("/encoder/final_norm/bias", {d_},
      Dec(w_.encoder_final_norm().betas(), "encoder_final_norm.betas").v);
}

// The edge-stream update sites: each map served [out, in], which is the
// layer's stored order.
void Builder::EdgeSites() {
  const auto& es = w_.edge_stream();
  if (fmt_.edge_update_after_size() != es.update_size()) {
    throw Exception("BT6: edge_update_after does not match the edge-stream updates.");
  }
  for (std::size_t n = 0; n < es.update_size(); ++n) {
    const auto& u = es.update(n);
    const std::string p = "/encoder" + std::to_string(fmt_.edge_update_after(n)) + "/edge_site";
    const std::string e = "edge_stream.update[" + std::to_string(n) + "].";
    auto served = [&](const std::string& name, const Layer& layer, const std::string& what) {
      const Arr a = Dec(layer, e + what);  // [in, out]
      if (a.dims.size() != 2) throw Exception("BT6 " + e + what + " is not a kernel.");
      const std::size_t r = a.dims[0], c = a.dims[1];
      std::vector<float> out(r * c);
      for (std::size_t i = 0; i < r; ++i) {
        for (std::size_t j = 0; j < c; ++j) out[j * r + i] = a.v[i * c + j];
      }
      Put(p + name, {U(c), U(r)}, out, kBt6Float);
    };
    served("/readback/w", u.o_e(), "o_e");
    served("/triplet/value/w", u.tri_v(), "tri_v");
    served("/triplet/gate/w", u.tri_eg(), "tri_eg");
    const Arr gb = Dec(u.tri_eg_b(), e + "tri_eg_b");
    Put(p + "/triplet/gate/b", {U(gb.size())}, gb.v, kBt6Float);
    served("/triplet/out/w", u.tri_o(), "tri_o");
    served("/ffn/dense1/w", u.ffn_in(), "ffn_in");
    const Arr fb = Dec(u.ffn_in_b(), e + "ffn_in_b");
    Put(p + "/ffn/dense1/b", {U(fb.size())}, fb.v, kBt6Float);
    served("/ffn/dense1_rev/w", u.ffn_rev(), "ffn_rev");
    served("/ffn/dense2/w", u.ffn_out(), "ffn_out");
  }
}

void Builder::Heads() {
  const auto& v = ValueHead(options_.value_head);
  const std::string vn = "value_heads." + options_.value_head + ".";
  Dense("/value/embedding", v.ip_val_w(), v.ip_val_b(), vn + "ip_val");
  Dense("/value/dense1", v.ip1_val_w(), v.ip1_val_b(), vn + "ip1_val");
  Dense("/value/dense2", v.ip2_val_w(), v.ip2_val_b(), vn + "ip2_val");
  if (options_.error) {
    const auto& eh = ValueHead(options_.error_head);
    const std::string en = "value_heads." + options_.error_head + ".";
    if (!eh.has_ip_val_err_w()) {
      throw Exception("Value head '" + options_.error_head + "' has no error output.");
    }
    Dense("/value_error/embedding", eh.ip_val_w(), eh.ip_val_b(), en + "ip_val");
    Dense("/value_error/dense1", eh.ip1_val_w(), eh.ip1_val_b(), en + "ip1_val");
    Dense("/value_error/dense2", eh.ip_val_err_w(), eh.ip_val_err_b(), en + "ip_val_err");
  }
  Dense("/moves/embedding", w_.ip_mov_w(), w_.ip_mov_b(), "ip_mov");
  Dense("/moves/dense1", w_.ip1_mov_w(), w_.ip1_mov_b(), "ip1_mov");
  Dense("/moves/dense2", w_.ip2_mov_w(), w_.ip2_mov_b(), "ip2_mov");
}

// The served policy: the blend sum_i w_i logits_i of the named heads, folded
// into one head of width D. Queries w_i sqrt(D / d_i) q_i (float32, as the
// composer multiplies), keys k_i, promotion w_i P_i, pair read
// sqrt(D) sum_i w_i u_i / sqrt(d_i) (float64); the shared embedding padded
// with zero outputs to D. A single head is the blend {name, 1}.
void Builder::Policy() {
  const auto& ph = w_.policy_heads();
  if (options_.policy.empty()) throw Exception("BT6: no policy head selected.");
  const Arr E = Dec(ph.ip_pol_w(), "policy_heads.ip_pol_w");  // [d_model, e]
  const Arr Eb = Dec(ph.ip_pol_b(), "policy_heads.ip_pol_b");
  if (E.dims.size() != 2) throw Exception("BT6: policy_heads.ip_pol_w is not a kernel.");
  const std::size_t in = E.dims[0], e = E.dims[1];
  std::vector<std::size_t> widths;
  for (const auto& [name, weight] : options_.policy) {
    widths.push_back(Dec(PolicyHead(name).ip2_pol_w(), name + ".ip2_pol_w").dims.at(1));
  }
  const std::size_t sum = std::accumulate(widths.begin(), widths.end(), std::size_t{0});
  const std::size_t D = options_.policy_width ? options_.policy_width : sum;
  if (sum > D || e > D) {
    throw Exception("The policy blend is " + std::to_string(sum) +
                    " wide; this lc0ex program serves " + std::to_string(D) + ".");
  }
  std::vector<float> E2(in * D, 0.0f), b2(D, 0.0f), Q(D * D, 0.0f), K(D * D, 0.0f);
  std::vector<float> Qb(D, 0.0f), Kb(D, 0.0f), P(D * 4, 0.0f);
  std::vector<double> u(16, 0.0);
  for (std::size_t r = 0; r < in; ++r) {
    std::copy_n(E.v.begin() + r * e, e, E2.begin() + r * D);
  }
  std::copy(Eb.v.begin(), Eb.v.end(), b2.begin());
  std::size_t o = 0;
  for (std::size_t n = 0; n < options_.policy.size(); ++n) {
    const auto& [name, weight] = options_.policy[n];
    const auto& head = PolicyHead(name);
    const std::size_t di = widths[n];
    const float s = static_cast<float>(
        weight * std::sqrt(static_cast<double>(D) / static_cast<double>(di)));
    const float wf = static_cast<float>(weight);
    const Arr q = Dec(head.ip2_pol_w(), name + ".ip2_pol_w");  // [e, di]
    const Arr qb = Dec(head.ip2_pol_b(), name + ".ip2_pol_b");
    const Arr k = Dec(head.ip3_pol_w(), name + ".ip3_pol_w");
    const Arr kb = Dec(head.ip3_pol_b(), name + ".ip3_pol_b");
    const Arr pr = Dec(head.ip4_pol_w(), name + ".ip4_pol_w");  // [di, 4]
    const Arr pu = Dec(head.pair_u(), name + ".pair_u");
    if (q.dims.at(0) != e || k.size() != e * di || qb.size() != di ||
        kb.size() != di || pr.size() != di * 4 || pu.size() != 16) {
      throw Exception("Policy head '" + name + "' does not fit the shared embedding.");
    }
    for (std::size_t r = 0; r < e; ++r) {
      for (std::size_t c = 0; c < di; ++c) {
        Q[r * D + o + c] = s * q.v[r * di + c];
        K[r * D + o + c] = k.v[r * di + c];
      }
    }
    for (std::size_t c = 0; c < di; ++c) {
      Qb[o + c] = s * qb.v[c];
      Kb[o + c] = kb.v[c];
      for (std::size_t j = 0; j < 4; ++j) P[(o + c) * 4 + j] = wf * pr.v[c * 4 + j];
    }
    for (std::size_t t = 0; t < 16; ++t) {
      u[t] += std::sqrt(static_cast<double>(D)) * weight *
              static_cast<double>(pu.v[t]) / std::sqrt(static_cast<double>(di));
    }
    o += di;
  }
  std::vector<float> u32(16);
  for (std::size_t t = 0; t < 16; ++t) u32[t] = static_cast<float>(u[t]);
  const std::uint64_t d = D;
  Put("/policy/embedding/w", {U(in), d}, E2);
  Put("/policy/embedding/b", {d}, b2);
  Put("/policy/Q/w", {d, d}, Q);
  Put("/policy/Q/b", {d}, Qb);
  Put("/policy/K/w", {d, d}, K);
  Put("/policy/K/b", {d}, Kb);
  Put("/policy/promotion/w", {d, 4}, P);
  Put("/policy/mix/coefficients", {16}, u32);
  const double root = std::sqrt(static_cast<double>(D));
  const std::size_t edges = pair_in_.dims[0], pd = pair_in_.dims[1];
  if (pd != 16) throw Exception("BT6: the policy pair read expects pair_d_e 16.");
  std::vector<double> sc(edges), cb(4096);
  for (std::size_t c = 0; c < edges; ++c) {
    double sum = 0.0;
    for (std::size_t t = 0; t < pd; ++t) {
      sum += static_cast<double>(u32[t]) * static_cast<double>(pair_in_.v[c * pd + t]);
    }
    sc[c] = sum / root;
  }
  for (std::size_t s = 0; s < 4096; ++s) {
    double sum = 0.0;
    for (std::size_t t = 0; t < pd; ++t) {
      sum += static_cast<double>(u32[t]) * pair_off_[t * 4096 + s];
    }
    cb[s] = sum / root;
  }
  Put("/policy/pair/scaled_coefficients", {1, U(edges)}, sc);
  Put("/policy/pair/constant_bias", {1, 64, 64}, cb);
  Put("/policy/qk_scale", {1}, std::vector<double>{1.0 / root});
  // The gather from the 64 x 64 + 8 x 24 attention logits to the 1858 moves:
  // the inverse of kAttnPolicyMap.
  std::vector<std::int32_t> gather(1858, -1);
  for (std::size_t i = 0; i < std::size(kAttnPolicyMap); ++i) {
    const int move = kAttnPolicyMap[i];
    if (move < 0) continue;
    if (move >= 1858 || gather[move] != -1) {
      throw Exception("BT6: kAttnPolicyMap is not a map onto the 1858 moves.");
    }
    gather[move] = static_cast<std::int32_t>(i);
  }
  if (std::count(gather.begin(), gather.end(), -1) != 0) {
    throw Exception("BT6: kAttnPolicyMap does not cover the 1858 moves.");
  }
  std::vector<std::byte> map(gather.size() * 4);
  std::memcpy(map.data(), gather.data(), map.size());
  PutBytes("/policy/map/w", {1858}, kBt6Int32, std::move(map));
}

// int8 (Q1) from Weights.int8_calibration (lczero-training
// convert/int8_calibration.py): for a site with weight W [in, out], prescale s
// and activation reciprocal r, W' = s_j W[j, n]; step_n = max_j |W'| / 127
// (1 if 0); codes = round half to even (W' / step_n) clamped to +-127, stored
// [out, in]; scale_n = step_n D alpha with D = in / fsum_j(r_j s_j).
void Builder::Int8() {
  const auto& calib = w_.int8_calibration();
  if (calib.site_size() == 0) {
    throw Exception("This lc0ex program serves int8 and the BT6 file has no int8 recipe.");
  }
  std::unordered_map<std::uint64_t, const Int8Site*> sites;
  for (const auto& site : calib.site()) {
    sites[(static_cast<std::uint64_t>(site.block()) << 8) |
          static_cast<std::uint64_t>(site.kind())] = &site;
  }
  auto site_of = [&](std::size_t block, Int8Site::Kind kind) -> const Int8Site& {
    const auto it = sites.find((static_cast<std::uint64_t>(block) << 8) |
                               static_cast<std::uint64_t>(kind));
    if (it == sites.end()) {
      throw Exception("The BT6 int8 recipe has no " +
                      pblczero::Weights_Int8Site_Kind_Name(kind) + " site for block " +
                      std::to_string(block) + ".");
    }
    return *it->second;
  };
  // Writes stem/w and stem/scale; returns the site's r.
  auto quantise = [&](const std::string& stem, const std::vector<double>& matrix,
                      std::size_t rows, std::size_t cols, const Int8Site& site,
                      double alpha, const std::string& what) {
    const Arr s = Dec(site.s(), what + ".s");
    Arr r = Dec(site.r(), what + ".r");
    if (s.size() != rows || r.size() != rows || matrix.size() != rows * cols) {
      throw Exception("BT6 int8 site " + what + ": its vectors do not match the weight.");
    }
    std::vector<double> products(rows);
    for (std::size_t j = 0; j < rows; ++j) {
      products[j] = static_cast<double>(r.v[j]) * static_cast<double>(s.v[j]);
    }
    const double D = static_cast<double>(rows) / FSum(products);
    std::vector<double> step(cols, 0.0);
    for (std::size_t j = 0; j < rows; ++j) {
      const double sj = s.v[j];
      for (std::size_t n = 0; n < cols; ++n) {
        step[n] = std::max(step[n], std::fabs(matrix[j * cols + n] * sj));
      }
    }
    for (auto& x : step) {
      x = x / 127.0;
      if (!(x > 0.0)) x = 1.0;
    }
    std::vector<std::byte> codes(rows * cols);
    for (std::size_t j = 0; j < rows; ++j) {
      const double sj = s.v[j];
      for (std::size_t n = 0; n < cols; ++n) {
        const double c = std::clamp(std::nearbyint((matrix[j * cols + n] * sj) / step[n]),
                                    -127.0, 127.0);
        codes[n * rows + j] = static_cast<std::byte>(static_cast<std::int8_t>(c));
      }
    }
    std::vector<double> scale(cols);
    for (std::size_t n = 0; n < cols; ++n) scale[n] = step[n] * D * alpha;
    PutBytes(stem + "/w", {U(cols), U(rows)}, kBt6Uint8, std::move(codes));
    Put(stem + "/scale", {U(cols)}, scale, kBt6Float);
    return r;
  };
  for (std::size_t i = 0; i < w_.encoder_size(); ++i) {
    const auto& m = w_.encoder(i).mha();
    const auto& f = w_.encoder(i).ffn();
    const std::string p = "/encoder" + std::to_string(i);
    const std::string e = "int8 encoder" + std::to_string(i);
    const auto& attn_in = site_of(i, Int8Site::ATTN_IN);
    const auto& ffn_in = site_of(i, Int8Site::FFN_IN);
    Put(p + "/ln1/quant/r", {d_}, Dec(attn_in.r(), e + " attn_in.r").v, kBt6Float);
    Put(p + "/ln1/quant/m", {d_}, std::vector<float>(d_, 0.0f), kBt6Float);
    Put(p + "/ln2/quant/r", {d_}, Dec(ffn_in.r(), e + " ffn_in.r").v, kBt6Float);
    Put(p + "/ln2/quant/m", {d_}, std::vector<float>(d_, 0.0f), kBt6Float);

    // attn_in: [q | k | v | output_gate] columns.
    std::vector<Arr> w = {Dec(m.q_w(), e + " q_w"), Dec(m.k_w(), e + " k_w"),
                          Dec(m.v_w(), e + " v_w")};
    std::vector<Arr> b = {Dec(m.q_b(), e + " q_b"), Dec(m.k_b(), e + " k_b"),
                          Dec(m.v_b(), e + " v_b")};
    if (m.has_output_gate_w()) w.push_back(Dec(m.output_gate_w(), e + " output_gate_w"));
    if (m.has_output_gate_b()) b.push_back(Dec(m.output_gate_b(), e + " output_gate_b"));
    std::vector<const Arr*> wp, bp;
    for (const auto& a : w) wp.push_back(&a);
    for (const auto& a : b) bp.push_back(&a);
    const std::vector<double> qkv = ToDouble(ConcatColumns(wp));
    quantise(p + "/mha/qkv/i8", qkv, d_, qkv.size() / d_, attn_in, 1.0, e + " attn_in");
    const std::vector<float> qkv_b = Concat(bp);
    Put(p + "/mha/qkv/i8/bias", {U(qkv_b.size())}, qkv_b, kBt6Float);

    // attn_out: dense_w, alpha folded into the scale and the bias.
    const Arr out = Dec(m.dense_w(), e + " dense_w");
    const Arr r_out = quantise(p + "/mha/out/i8", ToDouble(out.v), d_, out.size() / d_,
                               site_of(i, Int8Site::ATTN_OUT), alpha_, e + " attn_out");
    std::vector<double> out_b = ToDouble(Dec(m.dense_b(), e + " dense_b").v);
    for (auto& x : out_b) x *= alpha_;
    Put(p + "/mha/out/i8/bias", {U(out_b.size())}, out_b, kBt6Float);
    Put(p + "/mha/out/i8/r", {d_}, r_out.v, kBt6Float);

    // ffn_in: (gate, up) column pairs; the bias is (0, up) per pair.
    const Arr gate = Dec(f.dense_gate_w(), e + " dense_gate_w");
    const Arr up = Dec(f.dense1_w(), e + " dense1_w");
    const std::size_t hidden = gate.dims.at(1);
    if (gate.size() != d_ * hidden || up.size() != gate.size()) {
      throw Exception("BT6 " + e + ": the FFN kernels do not pair.");
    }
    std::vector<double> pairs(d_ * 2 * hidden);
    for (std::size_t j = 0; j < d_; ++j) {
      for (std::size_t n = 0; n < hidden; ++n) {
        pairs[j * 2 * hidden + 2 * n] = gate.v[j * hidden + n];
        pairs[j * 2 * hidden + 2 * n + 1] = up.v[j * hidden + n];
      }
    }
    quantise(p + "/ffn/dense1/i8", pairs, d_, 2 * hidden, ffn_in, 1.0, e + " ffn_in");
    const Arr up_b = Dec(f.dense1_b(), e + " dense1_b");
    std::vector<float> pair_b(2 * up_b.size(), 0.0f);
    for (std::size_t n = 0; n < up_b.size(); ++n) pair_b[2 * n + 1] = up_b.v[n];
    Put(p + "/ffn/dense1/i8/bias", {U(pair_b.size())}, pair_b, kBt6Float);

    // ffn_mid: dense2_w, alpha folded into the scale and the bias.
    const Arr down = Dec(f.dense2_w(), e + " dense2_w");
    const Arr r_mid = quantise(p + "/ffn/dense2/i8", ToDouble(down.v), hidden,
                               down.size() / hidden, site_of(i, Int8Site::FFN_MID),
                               alpha_, e + " ffn_mid");
    Put(p + "/ffn/mid/i8/r", {U(r_mid.size())}, r_mid.v, kBt6Float);
    std::vector<double> down_b = ToDouble(Dec(f.dense2_b(), e + " dense2_b").v);
    for (auto& x : down_b) x *= alpha_;
    Put(p + "/ffn/dense2/i8/bias", {U(down_b.size())}, down_b, kBt6Float);
  }

  // D1: the served value head's WDL layer refit to the int8 trunk.
  const pblczero::Weights::ValueHeadRefit* refit = nullptr;
  for (const auto& entry : calib.d1()) {
    if (entry.key() == options_.value_head) refit = &entry;
  }
  if (!refit) {
    throw Exception("The BT6 file's int8 recipe has no D1 refit for value head '" +
                    options_.value_head + "'; serve that head with an FP16 program.");
  }
  const Arr head = Dec(ValueHead(options_.value_head).ip2_val_w(), "ip2_val_w");
  if (refit->has_w_start()) {
    const Arr start = Dec(refit->w_start(), "d1.w_start");
    if (start.size() != head.size()) {
      throw Exception("BT6 D1: the refit did not start from this value head's layer.");
    }
    for (std::size_t j = 0; j < head.size(); ++j) {
      if (std::fabs(start.v[j] - head.v[j]) > 1e-5f) {
        throw Exception("BT6 D1: the refit did not start from this value head's layer.");
      }
    }
  }
  const Arr W1 = Dec(refit->w(), "d1.w");
  const Arr b1 = Dec(refit->b(), "d1.b");
  Put("/value/dense2/w_d1", KernelShape(W1, b1, "d1"), W1.v);
  Put("/value/dense2/b_d1", {U(b1.size())}, b1.v);
}

}  // namespace

bool IsNativeBt6(const pblczero::Net& net) {
  return net.format().network_format().network() ==
         pblczero::NetworkFormat::NETWORK_ATTENTIONBODY_BT6;
}

std::unordered_map<std::string, Bt6HostBuffer> BuildBt6Buffers(
    const pblczero::Net& net, const Bt6ServeOptions& options) {
  if (!IsNativeBt6(net)) throw Exception("Not a native BT6 net file.");
  return Builder(net, options).Build();
}

double DecimalOfFloat(float value) {
  char text[64];
  const auto written = std::to_chars(text, text + sizeof(text), value);
  double result = 0.0;
  std::from_chars(text, written.ptr, result);
  return result;
}

std::vector<std::string> Bt6PolicyHeadNames(const pblczero::Net& net) {
  const auto& ph = net.weights().policy_heads();
  std::vector<std::string> names;
  if (ph.has_vanilla()) names.push_back("vanilla");
  if (ph.has_optimistic_st()) names.push_back("optimistic_st");
  if (ph.has_soft()) names.push_back("soft");
  if (ph.has_opponent()) names.push_back("opponent");
  for (const auto& entry : ph.policy_head_map()) {
    if (IsServedPolicyHead(entry.value())) names.emplace_back(entry.key());
  }
  std::sort(names.begin(), names.end());
  return names;
}

}  // namespace lc0ex
}  // namespace lczero
