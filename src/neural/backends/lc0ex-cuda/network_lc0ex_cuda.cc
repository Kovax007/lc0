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

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <map>
#include <cstdio>
#include <memory>
#include <mutex>
#include <optional>
#include <semaphore>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "neural/backend.h"
#include "neural/encoder.h"
#include "neural/factory.h"
#include "neural/network.h"
#include "neural/loader.h"
#include "neural/onnx/converter.h"
#include "neural/register.h"
#include "neural/shared_params.h"
#include "bt6_native_buffers.h"
#include "network_fingerprint.h"
#include "proto/lc0ex_metadata.pb.h"
#include "runtime/lc0ex_cuda.h"
#include "utils/atomic_vector.h"
#include "utils/exception.h"
#include "utils/fastmath.h"
#include "utils/logging.h"
#include "utils/trace.h"

namespace lczero {
namespace {

constexpr std::string_view kBackendName = "lc0ex-cuda";
constexpr std::string_view kInputMasksName = "/input/plane_masks";
constexpr std::string_view kInputValuesName = "/input/plane_values";
constexpr std::string_view kOutputPolicyName = "/output/policy";
constexpr std::string_view kOutputWdlName = "/output/wdl";
constexpr std::string_view kOutputMlhName = "/output/mlh";
constexpr std::size_t kNumOutputPolicy = 1858;
constexpr std::size_t kNumWdlOutputs = 3;

FillEmptyHistory ParseHistoryFill(const std::string& value) {
  if (value == "fen_only") return FillEmptyHistory::FEN_ONLY;
  if (value == "always") return FillEmptyHistory::ALWAYS;
  if (value == "no") return FillEmptyHistory::NO;
  throw Exception("Unknown history fill mode '" + value + "'.");
}

std::uint64_t DataTypeSize(pblczero::Buffer::DataType data_type) {
  switch (data_type) {
    case pblczero::Buffer::DATA_TYPE_F32:
      return sizeof(float);
    case pblczero::Buffer::DATA_TYPE_U8:
      return sizeof(std::uint8_t);
    case pblczero::Buffer::DATA_TYPE_F16:
      return sizeof(std::uint16_t);
    case pblczero::Buffer::DATA_TYPE_U64:
      return sizeof(std::uint64_t);
    case pblczero::Buffer::DATA_TYPE_BF16:
      return sizeof(std::uint16_t);
    default:
      throw Exception("Unsupported or unknown lc0ex buffer data type.");
  }
}

const lc0ex::BufferInfo* RequireProgramBuffer(
    const lc0ex::Program& program, std::string_view name,
    pblczero::Buffer::DataType data_type,
    std::initializer_list<std::uint64_t> shape) {
  const auto* buffer = program.FindBuffer(name);
  if (!buffer) {
    throw Exception("The lc0ex program '" + program.GetInfo().name +
                    "' has no buffer '" + std::string(name) + "'.");
  }
  if (buffer->data_type != data_type) {
    throw Exception("Data type mismatch for lc0ex program buffer '" +
                    std::string(name) + "'.");
  }
  if (buffer->shape.size() != shape.size()) {
    throw Exception("Shape mismatch for lc0ex program buffer '" +
                    std::string(name) + "'.");
  }

  std::uint64_t expected_size = DataTypeSize(data_type);
  std::size_t dimension_index = 0;
  for (const std::uint64_t dimension : shape) {
    if (buffer->shape[dimension_index++] != dimension) {
      throw Exception("Shape mismatch for lc0ex program buffer '" +
                      std::string(name) + "'.");
    }
    if (dimension != 0 &&
        expected_size > std::numeric_limits<std::uint64_t>::max() /
                            dimension) {
      throw Exception("Size overflow for lc0ex program buffer '" +
                      std::string(name) + "'.");
    }
    expected_size *= dimension;
  }
  if (buffer->size_bytes != expected_size) {
    throw Exception("Size mismatch for lc0ex program buffer '" +
                    std::string(name) + "'.");
  }
  return buffer;
}

struct ProgramSpec {
  std::size_t batch_size;
  const lc0ex::Program* program;
  const lc0ex::BufferInfo* input_masks;
  const lc0ex::BufferInfo* input_values;
  const lc0ex::BufferInfo* output_policy;
  const lc0ex::BufferInfo* output_wdl;
  const lc0ex::BufferInfo* output_mlh;
};

pblczero::NeuralExecutable LoadExecutableFile(const std::string& path) {
  std::ifstream file(path, std::ios::in | std::ios::binary);
  if (!file) {
    throw Exception("Cannot read lc0ex executable from " + path + ".");
  }

  std::string serialized((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
  if (file.bad()) {
    throw Exception("Error while reading lc0ex executable from " + path +
                    ".");
  }

  pblczero::NeuralExecutable executable;
  executable.ParseFromString(serialized);
  return executable;
}

// "sm_80" -> 80, "sm_120a" -> 120; -1 when the name has no sm_ number.
int ArchitectureNumber(std::string_view architecture) {
  if (architecture.substr(0, 3) != "sm_") return -1;
  int number = 0;
  const auto* first = architecture.data() + 3;
  const auto* last = architecture.data() + architecture.size();
  const auto [end, error] = std::from_chars(first, last, number);
  if (error != std::errc() || end == first) return -1;
  return number;
}

std::string CudaVersionName(std::uint32_t version) {
  if (version == 0) return "CUDA ?";
  return "CUDA " + std::to_string(version / 1000) + "." +
         std::to_string(version % 1000 / 10);
}

std::string DescribeProgram(const pblczero::Lc0exProgram& program) {
  std::string description = std::string(program.architecture()) + " " +
                            std::string(program.precision()) + " " +
                            CudaVersionName(program.cuda_version());
  if (program.has_graph()) {
    description += ", graph " + std::string(program.graph());
  }
  return description;
}

// With no lc0ex= path, the program comes from the net file
// (Net.lc0ex_programs): the one built for this GPU's architecture, of the
// requested precision, for a CUDA major version the driver runs; among
// several, the newest CUDA. precision=auto takes int8 when the value head is
// winner (int8 programs serve only that head) and FP16 otherwise. The chosen
// entry's recommended batch size and graph mode (empty when unset) are
// returned through the last two arguments.
pblczero::NeuralExecutable SelectCarriedProgram(
    const WeightsFile& weights, int gpu, const std::string& precision,
    const std::string& value_head, int* recommended_batch_size,
    std::string* graph_mode) {
  if (precision != "auto" && precision != "int8" && precision != "fp16") {
    throw Exception("Unknown lc0ex precision '" + precision +
                    "'; expected auto, int8 or fp16.");
  }
  if (weights.lc0ex_programs_size() == 0) {
    throw Exception(
        "The lc0ex-cuda backend needs a program: pass lc0ex=<file> in "
        "--backend-opts, or use a net file that carries lc0ex programs.");
  }
  const std::string device = lc0ex::DeviceArchitecture(gpu);
  const int device_number = ArchitectureNumber(device);
  const int driver = lc0ex::DriverVersion();
  const std::string wanted =
      precision != "auto" ? precision
                          : (value_head == "winner" ? "int8" : "fp16");

  const pblczero::Lc0exProgram* best = nullptr;
  for (const auto& program : weights.lc0ex_programs()) {
    if (ArchitectureNumber(program.architecture()) != device_number) continue;
    if (program.precision() != wanted) continue;
    if (program.cuda_version() / 1000 > static_cast<std::uint32_t>(driver) / 1000) {
      continue;
    }
    if (!best || program.cuda_version() > best->cuda_version()) best = &program;
  }
  if (!best) {
    std::string carried;
    for (const auto& program : weights.lc0ex_programs()) {
      carried += (carried.empty() ? "" : ", ") + DescribeProgram(program);
    }
    throw Exception("The net file carries lc0ex programs for " + carried +
                    "; none is " + wanted + " for this GPU (" + device +
                    ", driver " + CudaVersionName(driver) +
                    "). Pass lc0ex=<file> with a program built for it.");
  }

  pblczero::NeuralExecutable executable;
  executable.ParseFromString(best->executable());
  if (ArchitectureNumber(executable.target().architecture()) != device_number) {
    throw Exception("The carried lc0ex program '" +
                    std::string(best->name()) + "' is labelled " +
                    std::string(best->architecture()) + " but targets " +
                    std::string(executable.target().architecture()) + ".");
  }
  *recommended_batch_size = static_cast<int>(best->recommended_batch_size());
  *graph_mode = best->has_graph() ? std::string(best->graph()) : "";
  CERR << "lc0ex: using the program carried in the net file: "
       << best->name() << " (" << DescribeProgram(*best) << "; GPU " << device
       << ", driver " << CudaVersionName(driver) << ").";
  return executable;
}

void CheckNetworkFingerprint(const WeightsFile& weights,
                             const pblczero::NeuralExecutable& executable) {
  pblczero::Net executable_fingerprint;
  executable_fingerprint.ParseFromString(executable.metadata());

  const auto expected = executable_fingerprint.OutputAsString();
  if (lc0ex::BuildNetworkFingerprint(weights).OutputAsString() == expected) {
    return;
  }
  // A native BT6 file also runs the programs built from its lab carrier.
  if (lc0ex::IsNativeBt6(weights) &&
      lc0ex::BuildBt6CarrierFingerprint(weights).OutputAsString() == expected) {
    return;
  }
  throw Exception(
      "The lc0ex executable was created for a different network architecture.");
}

bool HasMatchingShape(const lc0ex::BufferInfo& buffer,
                      const pblczero::TensorProto& initializer) {
  if (initializer.dims_size() != buffer.shape.size()) {
    return false;
  }
  for (std::size_t i = 0; i < initializer.dims_size(); ++i) {
    const auto dimension = initializer.dims(i);
    if (dimension < 0 ||
        static_cast<std::uint64_t>(dimension) != buffer.shape[i]) {
      return false;
    }
  }
  return true;
}

void ValidateInitializer(const lc0ex::BufferInfo& buffer,
                         const pblczero::TensorProto& initializer) {
  if (static_cast<int>(buffer.data_type) !=
      static_cast<int>(initializer.data_type())) {
    throw Exception("Data type mismatch for lc0ex buffer '" + buffer.name +
                    "'.");
  }
  if (!HasMatchingShape(buffer, initializer)) {
    throw Exception("Shape mismatch for lc0ex buffer '" + buffer.name +
                    "'.");
  }
  if (static_cast<std::uint64_t>(initializer.raw_data().size()) !=
      buffer.size_bytes) {
    throw Exception("Size mismatch for lc0ex buffer '" + buffer.name +
                    "'.");
  }
}

WeightsToOnnxConverterOptions MakeConverterOptions(
    const OptionsDict& backend_options) {
  WeightsToOnnxConverterOptions converter_options;
  converter_options.opset = backend_options.GetOrDefault<int>("opset", 17);
  converter_options.ir = backend_options.GetOrDefault<int>("ir", -1);
  converter_options.alt_mish =
      backend_options.GetOrDefault<bool>("alt_mish", false);
  converter_options.alt_layernorm =
      backend_options.GetOrDefault<bool>("alt_layernorm", false);
  converter_options.no_shape =
      backend_options.GetOrDefault<bool>("no_shape", false);
  converter_options.policy_head =
      backend_options.GetOrDefault<std::string>("policy_head", "vanilla");
  converter_options.value_head =
      backend_options.GetOrDefault<std::string>("value_head", "winner");
  converter_options.no_wdl_softmax = true;

  std::string datatype;
  if (backend_options.Exists<std::string>("datatype")) {
    datatype = backend_options.Get<std::string>("datatype");
  } else {
    const bool fp16 = backend_options.GetOrDefault<bool>("fp16", true);
    datatype = fp16 ? "f16" : "f32";
  }
  converter_options.data_type =
      WeightsToOnnxConverterOptions::StringToDataType(datatype);
  return converter_options;
}

void CopyStridedHostTensor(
    const std::vector<std::uint64_t>& shape,
    const std::vector<std::int64_t>& dst_strides,
    const std::vector<std::int64_t>& src_strides,
    std::size_t elem_size,
    const std::byte* src,
    std::byte* dst) {
  if (shape.empty()) {
    std::memcpy(dst, src, elem_size);
    return;
  }
  std::size_t contiguous_dim = shape.size();
  std::size_t contiguous_bytes = elem_size;
  while (contiguous_dim > 0) {
    std::size_t dim = contiguous_dim - 1;
    if (dim == shape.size() - 1) {
      if (dst_strides[dim] == 1 && src_strides[dim] == 1) {
        contiguous_bytes *= shape[dim];
        contiguous_dim = dim;
      } else {
        break;
      }
    } else {
      if (dst_strides[dim] ==
              dst_strides[dim + 1] *
                  static_cast<std::int64_t>(shape[dim + 1]) &&
          src_strides[dim] ==
              src_strides[dim + 1] *
                  static_cast<std::int64_t>(shape[dim + 1])) {
        contiguous_bytes *= shape[dim];
        contiguous_dim = dim;
      } else {
        break;
      }
    }
  }

  auto copy_dim = [&](auto& self, std::size_t dim, const std::byte* s,
                      std::byte* d) -> void {
    if (dim >= contiguous_dim) {
      std::memcpy(d, s, contiguous_bytes);
      return;
    }
    for (std::uint64_t i = 0; i < shape[dim]; ++i) {
      self(self, dim + 1, s + i * src_strides[dim] * elem_size,
           d + i * dst_strides[dim] * elem_size);
    }
  };
  copy_dim(copy_dim, 0, src, dst);
}

std::vector<std::int64_t> DefaultStrides(
    const std::vector<std::uint64_t>& shape) {
  std::vector<std::int64_t> strides(shape.size(), 1);
  if (shape.empty()) return strides;
  for (std::size_t i = shape.size() - 1; i > 0; --i) {
    strides[i - 1] = strides[i] * static_cast<std::int64_t>(shape[i]);
  }
  return strides;
}

void CopyTensorToHostStaging(const lc0ex::BufferInfo& buffer,
                             std::span<const std::byte> source,
                             std::span<std::byte> destination_staging) {
  const auto elem_size = DataTypeSize(buffer.data_type);
  const auto src_strides = DefaultStrides(buffer.shape);

  if (buffer.offset_bytes >= destination_staging.size()) {
    throw Exception("Buffer '" + buffer.name +
                    "' offset exceeds persistent allocation bounds.");
  }

  CopyStridedHostTensor(
      buffer.shape, buffer.strides, src_strides, elem_size,
      source.data(), destination_staging.data() + buffer.offset_bytes);
}

// A native BT6 file (NETWORK_ATTENTIONBODY_BT6) carries its weights as proto
// layers, not as the program's buffers: build them (bt6_native_buffers.h) for
// what the program serves and what the backend options select.
//   value_head=winner|q|st       the served value head (default winner)
//   error_head=st                the head behind /value_error/ (programs with
//                                an error output; default st)
//   policy_<head>=<weight>, ...  the served policy: the geometric blend of the
//                                named heads, e.g. policy_grill=0.4315,
//                                policy_optimistic_st=0.5685
//   policy_head=<head>           one head (default vanilla), if no
//                                policy_<head> weights are given
// The blend's width is the program's (/policy/Q/w); int8 is served when the
// program has int8 sites, from the file's int8 recipe.
lc0ex::Bt6ServeOptions MakeBt6ServeOptions(const WeightsFile& weights,
                                           const lc0ex::Executable& executable,
                                           const OptionsDict& backend_options) {
  lc0ex::Bt6ServeOptions serve;
  serve.value_head =
      backend_options.GetOrDefault<std::string>("value_head", "winner");
  // Read even when the program has no error output, so a config that names it
  // works with every program.
  serve.error_head =
      backend_options.GetOrDefault<std::string>("error_head", "st");
  serve.policy.clear();
  for (const auto& head : lc0ex::Bt6PolicyHeadNames(weights)) {
    const std::string key = "policy_" + head;
    double weight = 0.0;
    if (backend_options.Exists<float>(key)) {
      weight = lc0ex::DecimalOfFloat(backend_options.Get<float>(key));
    } else if (backend_options.Exists<int>(key)) {
      weight = backend_options.Get<int>(key);
    } else {
      continue;
    }
    if (weight != 0.0) serve.policy.emplace_back(head, weight);
  }
  if (backend_options.Exists<std::string>("policy_head")) {
    if (!serve.policy.empty()) {
      throw Exception(
          "Give either policy_head or policy_<head> weights, not both.");
    }
    serve.policy = {{backend_options.Get<std::string>("policy_head"), 1.0}};
  } else if (serve.policy.empty()) {
    serve.policy = {{"vanilla", 1.0}};
  }
  const auto* policy_q = executable.FindBuffer("/policy/Q/w");
  if (!policy_q || policy_q->shape.size() != 2) {
    throw Exception("The lc0ex program has no /policy/Q/w [D, D] buffer.");
  }
  serve.policy_width = policy_q->shape[1];
  serve.int8 = executable.FindBuffer("/encoder0/mha/qkv/i8/w") != nullptr;
  serve.error = executable.FindBuffer("/value_error/embedding/w") != nullptr;
  return serve;
}

void UploadNativeBt6Weights(const WeightsFile& weights,
                            lc0ex::Executable& executable,
                            const OptionsDict& backend_options) {
  const auto serve = MakeBt6ServeOptions(weights, executable, backend_options);
  std::string policy;
  for (const auto& [head, weight] : serve.policy) {
    // The weight as typed: the shortest decimal of its float.
    std::array<char, 32> text{};
    const auto end = std::to_chars(text.data(), text.data() + text.size(),
                                   static_cast<float>(weight))
                         .ptr;
    policy += (policy.empty() ? "" : " + ") + std::string(text.data(), end) +
              " " + head;
  }
  CERR << "Building the lc0ex buffers from the native BT6 weights: value head "
       << serve.value_head << ", policy " << policy << " (width "
       << serve.policy_width << "), " << (serve.int8 ? "int8" : "FP16")
       << (serve.error ? ", error head " + serve.error_head : "") << ".";
  auto buffers = lc0ex::BuildBt6Buffers(weights, serve);

  std::vector<std::byte> staging(executable.GetPersistentAllocationSize(),
                                 std::byte{0});
  for (const auto& buffer : executable.GetBuffers()) {
    const auto it = buffers.find(buffer.name);
    if (it == buffers.end()) {
      throw Exception("The lc0ex buffer '" + buffer.name +
                      "' has no counterpart in the native BT6 weights.");
    }
    const auto& built = it->second;
    if (static_cast<int>(buffer.data_type) != built.data_type) {
      throw Exception("Data type mismatch for lc0ex buffer '" + buffer.name +
                      "'.");
    }
    if (built.shape != buffer.shape) {
      throw Exception("Shape mismatch for lc0ex buffer '" + buffer.name +
                      "'.");
    }
    if (built.bytes.size() != buffer.size_bytes) {
      throw Exception("Size mismatch for lc0ex buffer '" + buffer.name +
                      "'.");
    }
    CopyTensorToHostStaging(buffer, built.bytes, staging);
    buffers.erase(it);
  }
  // The builder writes every form a BT6 program may read (FP16 and int8 GEMM
  // weights, raw and folded coefficients), so a program leaves some unread:
  // one line, not a warning per buffer.
  if (!buffers.empty()) {
    std::vector<std::string> unread;
    for (const auto& [name, built] : buffers) unread.push_back(name);
    std::sort(unread.begin(), unread.end());
    std::string sample;
    for (std::size_t i = 0; i < unread.size() && i < 3; ++i) {
      sample += (i ? ", " : "") + unread[i];
    }
    CERR << unread.size()
         << " native BT6 buffers are not read by this program (expected; "
         << sample << (unread.size() > 3 ? ", ..." : "") << ").";
  }
  executable.CopyPersistentFromHost(staging);
}

void UploadWeights(const WeightsFile& weights, lc0ex::Executable& executable,
                   const OptionsDict& backend_options) {
  if (lc0ex::IsNativeBt6(weights)) {
    UploadNativeBt6Weights(weights, executable, backend_options);
    return;
  }
  std::optional<WeightsFile> converted_weights;
  if (!weights.has_onnx_model()) {
    CERR << "Converting weights to ONNX first.";
    converted_weights =
        ConvertWeightsToOnnx(weights, MakeConverterOptions(backend_options));
  }

  const auto& onnx_weights = converted_weights ? *converted_weights : weights;

  pblczero::ModelProto onnx;
  onnx.ParseFromString(onnx_weights.onnx_model().model());

  std::unordered_map<std::string, const pblczero::TensorProto*>
      initializers_by_name;
  initializers_by_name.reserve(onnx.graph().initializer_size());
  std::string duplicate_initializer;
  const bool unique_initializers = absl::c_all_of(
      onnx.graph().initializer(), [&](const auto& initializer) {
        const auto name = std::string(initializer.name());
        if (!initializers_by_name.emplace(name, &initializer).second) {
          duplicate_initializer = name;
          return false;
        }
        return true;
      });
  if (!unique_initializers) {
    throw Exception("The ONNX model contains duplicate initializer '" +
                    duplicate_initializer + "'.");
  }

  std::string missing_buffer;
  if (absl::c_any_of(executable.GetBuffers(), [&](const auto& buffer) {
        if (initializers_by_name.find(buffer.name) ==
            initializers_by_name.end()) {
          missing_buffer = buffer.name;
          return true;
        }
        return false;
      })) {
    throw Exception("The lc0ex buffer '" + missing_buffer +
                    "' has no corresponding ONNX initializer.");
  }

  CERR << "Uploading ONNX initializers to the lc0ex runtime.";
  std::vector<std::byte> staging(executable.GetPersistentAllocationSize(),
                                 std::byte{0});

  for (const auto& initializer : onnx.graph().initializer()) {
    const auto* buffer = executable.FindBuffer(initializer.name());
    if (!buffer) {
      CERR << "WARNING: ONNX initializer '" << initializer.name()
           << "' has no corresponding lc0ex buffer.";
      continue;
    }

    ValidateInitializer(*buffer, initializer);
    const auto source = std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(initializer.raw_data().data()),
        initializer.raw_data().size());
    CopyTensorToHostStaging(*buffer, source, staging);
  }

  executable.CopyPersistentFromHost(staging);
}

BackendAttributes MakeBackendAttributes(const WeightsFile& weights) {
  const auto& format = weights.format().network_format();
  // BackendAttributes has constructors on this branch, so no designated
  // initializers; `{}` zeroes every field the list below does not name.
  BackendAttributes attributes{};
  attributes.has_mlh =
      format.moves_left() != pblczero::NetworkFormat::MOVES_LEFT_NONE;
  attributes.has_wdl = format.output() == pblczero::NetworkFormat::OUTPUT_WDL;
  attributes.runs_on_cpu = false;
  attributes.concurrent_add_input = false;
  attributes.suggested_num_search_threads = 2;
  attributes.recommended_batch_size = 0;
  attributes.maximum_batch_size = 0;
  attributes.preferred_batch_step = 1;  // Network::GetPreferredBatchStep()'s default.
  return attributes;
}

void DecodeWdl(std::span<const float> logits, EvalResultPtr result) {
  const float maximum = std::max({logits[0], logits[1], logits[2]});
  const float win = std::exp(logits[0] - maximum);
  const float draw = std::exp(logits[1] - maximum);
  const float loss = std::exp(logits[2] - maximum);
  const float scale = 1.0f / (win + draw + loss);

  if (result.q) *result.q = (win - loss) * scale;
  if (result.d) *result.d = draw * scale;
}

// The semaphore admits `concurrency` threads; this hands each of them a
// distinct index so they address disjoint cached executions.
struct ExecutionSlotPool {
  std::mutex mutex;
  std::vector<bool> in_use;
};

class ExecutionPermit {
 public:
  ExecutionPermit(std::counting_semaphore<>& semaphore,
                  ExecutionSlotPool* pool = nullptr)
      : semaphore_(semaphore), pool_(pool) {
    semaphore_.acquire();
    if (pool_) {
      std::lock_guard<std::mutex> lock(pool_->mutex);
      for (std::size_t i = 0; i < pool_->in_use.size(); ++i) {
        if (!pool_->in_use[i]) {
          pool_->in_use[i] = true;
          index_ = i;
          break;
        }
      }
    }
  }

  ~ExecutionPermit() {
    if (pool_) {
      std::lock_guard<std::mutex> lock(pool_->mutex);
      pool_->in_use[index_] = false;
    }
    semaphore_.release();
  }

  ExecutionPermit(const ExecutionPermit&) = delete;
  ExecutionPermit& operator=(const ExecutionPermit&) = delete;

  std::size_t index() const { return index_; }

 private:
  std::counting_semaphore<>& semaphore_;
  ExecutionSlotPool* pool_ = nullptr;
  std::size_t index_ = 0;
};

class Lc0exCudaBackend;

class Lc0exCudaBackendComputation final : public BackendComputation {
 public:
  explicit Lc0exCudaBackendComputation(Lc0exCudaBackend* backend);

  size_t UsedBatchSize() const override { return entries_.size(); }

  AddInputResult AddInput(const EvalPosition& pos,
                          EvalResultPtr result) override;

  void ComputeBlocking(ComputationCallback callback) override;

 private:
  struct Entry {
    std::array<std::uint64_t, kInputPlanes> masks;
    std::array<float, kInputPlanes> values;
    MoveList legal_moves;
    EvalResultPtr result;
    int transform;
  };

  void DecodePolicy(const Entry& entry, std::span<const float> logits) const;

  Lc0exCudaBackend* backend_;
  AtomicVector<Entry> entries_;
};

// `auto` is the DAG graph (R34). R22 keyed this on the configured concurrency,
// because the graph then measured +25 % with one execution slot in flight and
// -9 % with two. That penalty was the graph serialising against *synchronous*
// host-to-device copies, not against the second slot: once the copies moved onto
// the slot's non-blocking stream the sign flipped, and the graph is now +7.2 %
// at two threads and batch 16 and never negative in a search. `linear` and `off`
// remain for diagnosis -- `off` in particular, because a graph-launched backend
// needs `nsys --cuda-graph-trace=node` to show its kernels at all.
lc0ex::GraphMode ResolveGraphMode(const std::string& value) {
  if (value == "dag" || value == "on") return lc0ex::GraphMode::kDag;
  if (value == "linear") return lc0ex::GraphMode::kLinear;
  if (value == "off") return lc0ex::GraphMode::kOff;
  if (value != "auto") {
    throw Exception("Unknown lc0ex graph mode '" + value +
                    "'; expected auto, dag, linear or off.");
  }
  return lc0ex::GraphMode::kDag;
}

class Lc0exCudaBackend final : public Backend {
 public:
  Lc0exCudaBackend(const WeightsFile& weights, const OptionsDict& options,
                   const OptionsDict& backend_options)
      : attributes_(MakeBackendAttributes(weights)),
        execution_semaphore_(
            backend_options.GetOrDefault<int>("concurrency", 1)),
        backend_options_(
            options.Get<std::string>(SharedBackendParams::kBackendOptionsId)),
        weights_path_(options.Get<std::string>(SharedBackendParams::kWeightsId)),
        input_format_(weights.format().network_format().input()) {
    UpdateConfiguration(options);
    Setup(weights, backend_options);
  }

  // The old-interface path (lc0ex-cuda-net, for demux). No search options:
  // demux encodes the inputs and applies the policy softmax itself.
  struct NetworkPath {};
  Lc0exCudaBackend(const WeightsFile& weights,
                   const OptionsDict& backend_options, NetworkPath)
      : attributes_(MakeBackendAttributes(weights)),
        execution_semaphore_(
            backend_options.GetOrDefault<int>("concurrency", 1)),
        input_format_(weights.format().network_format().input()) {
    Setup(weights, backend_options);
  }

  void Setup(const WeightsFile& weights, const OptionsDict& backend_options) {
    const int concurrency =
        std::max(1, backend_options.GetOrDefault<int>("concurrency", 1));
    execution_slot_pool_.in_use.assign(concurrency, false);
    executions_.resize(concurrency);

    const int gpu = backend_options.GetOrDefault<int>("gpu", 0);
    const std::string lc0ex_path =
        backend_options.GetOrDefault<std::string>("lc0ex", "");
    const std::string precision =
        backend_options.GetOrDefault<std::string>("precision", "auto");
    int carried_recommended_batch_size = 0;
    std::string carried_graph_mode;
    const auto executable_proto =
        !lc0ex_path.empty()
            ? LoadExecutableFile(lc0ex_path)
            : SelectCarriedProgram(
                  weights, gpu, precision,
                  backend_options.GetOrDefault<std::string>("value_head",
                                                            "winner"),
                  &carried_recommended_batch_size, &carried_graph_mode);
    CheckNetworkFingerprint(weights, executable_proto);

    // Off by default: it is a strict win only above the ladder's dense band,
    // which a search at --minibatch-size equal to the top dense rung never
    // reaches. Worth turning on for a minibatch-128 configuration.
    split_programs_ = backend_options.GetOrDefault<bool>("split_programs", false);
    // A carried program may name the graph mode it must run in; an explicit
    // graph= backend option still wins.
    std::string graph_mode =
        backend_options.GetOrDefault<std::string>("graph", "auto");
    if (graph_mode == "auto" && !carried_graph_mode.empty()) {
      graph_mode = carried_graph_mode;
    }
    runtime_ = lc0ex::CreateLc0exCudaRuntime(gpu, ResolveGraphMode(graph_mode));
    executable_ = runtime_->Load(executable_proto);
    InitializePrograms();
    // A carried program may name the batch size a search should use when
    // --minibatch-size=0 (the default); never above the largest it serves.
    if (carried_recommended_batch_size > 0 &&
        carried_recommended_batch_size < attributes_.maximum_batch_size) {
      attributes_.recommended_batch_size = carried_recommended_batch_size;
    }
    UploadWeights(weights, *executable_, backend_options);
  }

  ~Lc0exCudaBackend() override {
    // Members are destroyed in reverse declaration order, which would free the
    // Executable before the Executions that point into it: ~Lc0exCudaExecution
    // reads executable_->context_retained_, synchronises the slot's stream and
    // frees pinned staging in that context. Clearing the cache here is what
    // orders it correctly. Without this every lc0ex process segfaults on exit,
    // after all of its output, which is why it went unnoticed for three rounds.
    executions_.clear();
  }

  BackendAttributes GetAttributes() const override { return attributes_; }

  std::size_t MaxBatch() const { return programs_.back().batch_size; }
  bool HasMlhOutput() const { return programs_.front().output_mlh != nullptr; }
  // 40baf4c5 has no error output (the error-head commit is not in this base; D4 = off).
  bool HasErrOutput() const { return false; }

  // One batch of encoded positions (n <= MaxBatch()), outputs raw: policy and
  // WDL logits, moves left, error. The middle of
  // Lc0exCudaBackendComputation::ComputeBlocking, for the Network path.
  void EvaluatePlanes(const std::uint64_t* masks_in, const float* values_in,
                      std::size_t n, float* policy, float* wdl, float* mlh,
                      float* err) {
    LCTRACE_FUNCTION_SCOPE;
    const auto split = FindSplit(n);
    const ProgramSpec& program = split ? *split->first : FindProgram(n);
    const std::size_t padded =
        split ? split->first_count + split->second->batch_size
              : program.batch_size;
    // Padding slots hold copies of the last position (see ComputeBlocking).
    std::vector<std::uint64_t> masks(padded * kInputPlanes);
    std::vector<float> values(padded * kInputPlanes);
    for (std::size_t sample = 0; sample < padded; ++sample) {
      const std::size_t from = std::min(sample, n - 1) * kInputPlanes;
      std::copy_n(masks_in + from, kInputPlanes,
                  masks.begin() + sample * kInputPlanes);
      std::copy_n(values_in + from, kInputPlanes,
                  values.begin() + sample * kInputPlanes);
    }
    const auto mask_bytes =
        std::as_bytes(std::span<const std::uint64_t>(masks));
    const auto value_bytes = std::as_bytes(std::span<const float>(values));

    ExecutionPermit permit(execution_semaphore_, &execution_slot_pool_);
    const auto part = [&](const ProgramSpec& spec, std::size_t offset,
                          std::size_t count) {
      lc0ex::Execution& execution = GetExecution(permit.index(), spec);
      execution.GetBuffer(*spec.input_masks)
          .CopyFromHostAsync(
              mask_bytes.subspan(offset * kInputPlanes * sizeof(std::uint64_t)),
              spec.batch_size * kInputPlanes * sizeof(std::uint64_t));
      execution.GetBuffer(*spec.input_values)
          .CopyFromHostAsync(
              value_bytes.subspan(offset * kInputPlanes * sizeof(float)),
              spec.batch_size * kInputPlanes * sizeof(float));
      execution.Run();
      execution.GetBuffer(*spec.output_policy)
          .CopyToHostAsync(std::as_writable_bytes(std::span<float>(
                               policy + offset * kNumOutputPolicy,
                               count * kNumOutputPolicy)),
                           count * kNumOutputPolicy * sizeof(float));
      execution.GetBuffer(*spec.output_wdl)
          .CopyToHostAsync(std::as_writable_bytes(std::span<float>(
                               wdl + offset * kNumWdlOutputs,
                               count * kNumWdlOutputs)),
                           count * kNumWdlOutputs * sizeof(float));
      if (spec.output_mlh && mlh) {
        execution.GetBuffer(*spec.output_mlh)
            .CopyToHostAsync(std::as_writable_bytes(
                                 std::span<float>(mlh + offset, count)),
                             count * sizeof(float));
      }
      execution.Synchronize();
    };
    if (split) {
      part(*split->first, 0, split->first_count);
      part(*split->second, split->first_count, n - split->first_count);
    } else {
      part(program, 0, n);
    }
  }

  std::unique_ptr<BackendComputation> CreateComputation() override {
    return std::make_unique<Lc0exCudaBackendComputation>(this);
  }

  UpdateConfigurationResult UpdateConfiguration(
      const OptionsDict& options) override {
    Backend::UpdateConfiguration(options);
    if (backend_options_ !=
        options.Get<std::string>(SharedBackendParams::kBackendOptionsId)) {
      return NEED_RESTART;
    }
    if (weights_path_ !=
        options.Get<std::string>(SharedBackendParams::kWeightsId)) {
      return NEED_RESTART;
    }

    inverse_policy_temperature_ =
        1.0f / options.Get<float>(SharedBackendParams::kPolicySoftmaxTemp);
    fill_empty_history_ = ParseHistoryFill(
        options.Get<std::string>(SharedBackendParams::kHistoryFill));
    return UPDATE_OK;
  }

  // LC0EX_BATCH_HIST=1 records the batch sizes the search actually asks for.
  // The ladder rounds each one UP to the next compiled program, so the shape of
  // this histogram -- not the rung rates -- decides how much of the machine the
  // deployed artifact really uses. Printed once at exit.
  static void RecordBatch(std::size_t batch_size, std::size_t program_size) {
    static const bool enabled = [] {
      const char* value = std::getenv("LC0EX_BATCH_HIST");
      return value != nullptr && value[0] == '1';
    }();
    if (!enabled) return;
    static std::mutex mutex;
    static std::map<std::size_t, std::size_t> requested;
    static std::size_t asked = 0;
    static std::size_t served = 0;
    static bool registered = false;
    const std::lock_guard<std::mutex> lock(mutex);
    requested[batch_size]++;
    asked += batch_size;
    served += program_size;
    if (!registered) {
      registered = true;
      std::atexit([] {
        std::fprintf(stderr, "\n### lc0ex batch histogram (requested -> count)\n");
        for (const auto& [size, count] : requested) {
          std::fprintf(stderr, "%zu %zu\n", size, count);
        }
        std::fprintf(stderr, "### positions asked %zu, positions computed %zu, "
                             "ladder efficiency %.4f\n",
                     asked, served,
                     served ? static_cast<double>(asked) / served : 0.0);
      });
    }
  }

  const ProgramSpec& FindProgram(std::size_t batch_size) const {
    const auto iter = std::lower_bound(
        programs_.begin(), programs_.end(), batch_size,
        [](const ProgramSpec& program, std::size_t size) {
          return program.batch_size < size;
        });
    if (iter == programs_.end()) {
      throw Exception("NN input exceeds maximum lc0ex batch size of " +
                      std::to_string(attributes_.maximum_batch_size) + ".");
    }
    RecordBatch(batch_size, iter->batch_size);
    return *iter;
  }

  // Two programs instead of one padded program. For a formed batch `b` with
  // rungs r1 <= b < round_up, running r1 on the first r1 positions and the
  // smallest rung >= (b - r1) on the rest computes r1 + r2 padded positions
  // instead of round_up. Taken only when that is a strict saving by at least
  // `kSplitMargin`, because the split costs a second launch sequence and a
  // second output gather.
  //
  // Returns nullptr when the batch is on a rung, when no split helps, or when
  // the option is off.
  struct SplitPlan {
    const ProgramSpec* first;
    const ProgramSpec* second;
    std::size_t first_count;
  };

  std::optional<SplitPlan> FindSplit(std::size_t batch_size) const {
    if (!split_programs_) return std::nullopt;
    const auto up = std::lower_bound(
        programs_.begin(), programs_.end(), batch_size,
        [](const ProgramSpec& program, std::size_t size) {
          return program.batch_size < size;
        });
    if (up == programs_.end()) return std::nullopt;
    if (up->batch_size == batch_size) return std::nullopt;  // already exact
    if (up == programs_.begin()) return std::nullopt;       // below every rung

    const ProgramSpec& first = *(up - 1);          // largest rung < batch_size
    const std::size_t rest = batch_size - first.batch_size;
    const auto second = std::lower_bound(
        programs_.begin(), programs_.end(), rest,
        [](const ProgramSpec& program, std::size_t size) {
          return program.batch_size < size;
        });
    if (second == programs_.end()) return std::nullopt;

    const std::size_t split_padded = first.batch_size + second->batch_size;
    if (split_padded + kSplitMargin > up->batch_size) return std::nullopt;
    return SplitPlan{&first, &(*second), first.batch_size};
  }

  // One Execution per (concurrency slot, program), created on first use and
  // kept for the backend's lifetime. Rebuilding it per inference cost a slot
  // acquisition plus two heap vectors for each of the ~232 nodes, and with a
  // graph mode it would also rebuild the graph every time.
  lc0ex::Execution& GetExecution(std::size_t slot, const ProgramSpec& program) {
    auto& by_program = executions_[slot];
    const auto* key = program.program;
    const auto iter = by_program.find(key);
    if (iter != by_program.end()) return *iter->second;
    // Every program on this concurrency slot shares one device execution
    // slot: only one of them is ever in flight here.
    lc0ex::Execution* sibling =
        by_program.empty() ? nullptr : by_program.begin()->second.get();
    auto execution = executable_->CreateExecution(*program.program, sibling);
    auto* result = execution.get();
    by_program.emplace(key, std::move(execution));
    return *result;
  }

 private:
  void InitializePrograms() {
    for (const auto& program_info : executable_->GetPrograms()) {
      const auto* program = executable_->FindProgram(program_info.name);

      pblczero::ProgramMetadata metadata;
      metadata.ParseFromString(program_info.metadata);
      const std::size_t batch_size = metadata.batch_size();
      if (batch_size >
          static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw Exception("The lc0ex program '" + program_info.name +
                        "' has an unsupported batch size.");
      }

      const auto* input_masks = RequireProgramBuffer(
          *program, kInputMasksName, pblczero::Buffer::DATA_TYPE_U64,
          {batch_size, kInputPlanes});
      const auto* input_values = RequireProgramBuffer(
          *program, kInputValuesName, pblczero::Buffer::DATA_TYPE_F32,
          {batch_size, kInputPlanes});
      const auto* output_policy = RequireProgramBuffer(
          *program, kOutputPolicyName, pblczero::Buffer::DATA_TYPE_F32,
          {batch_size, kNumOutputPolicy});
      const auto* output_wdl = RequireProgramBuffer(
          *program, kOutputWdlName, pblczero::Buffer::DATA_TYPE_F32,
          {batch_size, kNumWdlOutputs});

      const auto* output_mlh = program->FindBuffer(kOutputMlhName);
      if (attributes_.has_mlh && !output_mlh) {
        throw Exception("The lc0ex program '" + program_info.name +
                        "' has no moves-left output buffer.");
      }
      if (output_mlh) {
        output_mlh = RequireProgramBuffer(
            *program, kOutputMlhName, pblczero::Buffer::DATA_TYPE_F32,
            {batch_size, 1});
      }

      programs_.push_back({batch_size, program, input_masks, input_values,
                           output_policy, output_wdl, output_mlh});
    }

    std::sort(programs_.begin(), programs_.end(),
              [](const ProgramSpec& lhs, const ProgramSpec& rhs) {
                return lhs.batch_size < rhs.batch_size;
              });
    attributes_.recommended_batch_size =
        static_cast<int>(programs_.back().batch_size);
    attributes_.maximum_batch_size =
        static_cast<int>(programs_.back().batch_size);
  }

  static constexpr std::size_t kSplitMargin = 8;

  BackendAttributes attributes_;
  bool split_programs_ = false;
  std::counting_semaphore<> execution_semaphore_;
  ExecutionSlotPool execution_slot_pool_;
  std::vector<std::unordered_map<const lc0ex::Program*,
                                 std::unique_ptr<lc0ex::Execution>>>
      executions_;
  const std::string backend_options_;
  const std::string weights_path_;
  const pblczero::NetworkFormat::InputFormat input_format_;
  float inverse_policy_temperature_ = 1.0f;
  FillEmptyHistory fill_empty_history_ = FillEmptyHistory::NO;
  std::unique_ptr<lc0ex::Runtime> runtime_;
  std::unique_ptr<lc0ex::Executable> executable_;
  std::vector<ProgramSpec> programs_;

  friend class Lc0exCudaBackendComputation;
};

BackendComputation::AddInputResult Lc0exCudaBackendComputation::AddInput(
    const EvalPosition& pos, EvalResultPtr result) {
  int transform = 0;
  const InputPlanes input = EncodePositionForNN(
      backend_->input_format_, pos.pos, kMoveHistory,
      backend_->fill_empty_history_, &transform);

  Entry entry{
      .masks = {},
      .values = {},
      .legal_moves = MoveList(pos.legal_moves.begin(), pos.legal_moves.end()),
      .result = result,
      .transform = transform,
  };
  for (std::size_t i = 0; i < kInputPlanes; ++i) {
    entry.masks[i] = input[i].mask;
    entry.values[i] = input[i].value;
  }
  entries_.emplace_back(std::move(entry));
  return ENQUEUED_FOR_EVAL;
}

Lc0exCudaBackendComputation::Lc0exCudaBackendComputation(
    Lc0exCudaBackend* backend)
    : backend_(backend),
      entries_(backend_->GetAttributes().maximum_batch_size) {}

void Lc0exCudaBackendComputation::DecodePolicy(
    const Entry& entry, std::span<const float> logits) const {
  float maximum = -std::numeric_limits<float>::infinity();
  for (std::size_t i = 0; i < entry.legal_moves.size(); ++i) {
    const std::size_t policy_index =
        MoveToNNIndex(entry.legal_moves[i], entry.transform);
    entry.result.p[i] = logits[policy_index];
    maximum = std::max(maximum, entry.result.p[i]);
  }

  float total = 0.0f;
  for (float& value : entry.result.p) {
    value = FastExp(
        (value - maximum) * backend_->inverse_policy_temperature_);
    total += value;
  }
  const float scale = total > 0.0f ? 1.0f / total : 1.0f;
  for (float& value : entry.result.p) value *= scale;
}

void Lc0exCudaBackendComputation::ComputeBlocking(
    ComputationCallback callback) {
  const std::size_t actual_batch = entries_.size();
  // The search counts waiting workers down in the callback, so it fires exactly
  // once per call, empty batch included.
  if (actual_batch == 0) {
    callback(ComputationEvent::FIRST_BACKEND_IDLE);
    return;
  }

  const auto split = backend_->FindSplit(actual_batch);
  const ProgramSpec& program =
      split ? *split->first : backend_->FindProgram(actual_batch);

  // A batch between rungs runs the next rung's program. Its padding slots must
  // hold real positions: left unset they keep whatever the planner last put in
  // that memory, which decodes as boards full of pieces -- their edge lists
  // overflow and every block takes the exact dense fallback for them (b65 on a
  // 128 rung: 15.0 ms instead of 11.6). Copies of the last position cost what a
  // real position costs.
  const std::size_t padded_batch =
      split ? split->first_count + split->second->batch_size
            : program.batch_size;
  std::vector<std::uint64_t> masks(padded_batch * kInputPlanes);
  std::vector<float> values(padded_batch * kInputPlanes);
  for (std::size_t sample = 0; sample < padded_batch; ++sample) {
    const std::size_t offset = sample * kInputPlanes;
    const auto& entry = entries_[std::min(sample, actual_batch - 1)];
    std::copy(entry.masks.begin(), entry.masks.end(), masks.begin() + offset);
    std::copy(entry.values.begin(), entry.values.end(),
              values.begin() + offset);
  }

  const auto mask_bytes = std::as_bytes(std::span<const std::uint64_t>(masks));
  const auto value_bytes = std::as_bytes(std::span<const float>(values));

  std::vector<float> policy(actual_batch * kNumOutputPolicy);
  std::vector<float> wdl(actual_batch * kNumWdlOutputs);
  std::vector<float> mlh;
  if (program.output_mlh) mlh.resize(actual_batch);

  const auto policy_bytes = std::as_writable_bytes(std::span<float>(policy));
  const auto wdl_bytes = std::as_writable_bytes(std::span<float>(wdl));

  // Everything is issued on this slot's stream, so the copies overlap another
  // slot's kernels. All of it, including the reads, completes at Synchronize().
  //
  // `part` runs `count` positions starting at `offset` on one program. The two
  // halves of a split share one device slot, so the first is fully synchronised
  // -- outputs already copied out -- before the second touches the allocation.
  {
    ExecutionPermit permit(backend_->execution_semaphore_,
                           &backend_->execution_slot_pool_);
    const auto part = [&](const ProgramSpec& spec, std::size_t offset,
                          std::size_t count) {
      lc0ex::Execution& execution =
          backend_->GetExecution(permit.index(), spec);
      execution.GetBuffer(*spec.input_masks)
          .CopyFromHostAsync(
              mask_bytes.subspan(offset * kInputPlanes * sizeof(std::uint64_t)),
              spec.batch_size * kInputPlanes * sizeof(std::uint64_t));
      execution.GetBuffer(*spec.input_values)
          .CopyFromHostAsync(
              value_bytes.subspan(offset * kInputPlanes * sizeof(float)),
              spec.batch_size * kInputPlanes * sizeof(float));
      execution.Run();
      execution.GetBuffer(*spec.output_policy)
          .CopyToHostAsync(
              policy_bytes.subspan(offset * kNumOutputPolicy * sizeof(float)),
              count * kNumOutputPolicy * sizeof(float));
      execution.GetBuffer(*spec.output_wdl)
          .CopyToHostAsync(
              wdl_bytes.subspan(offset * kNumWdlOutputs * sizeof(float)),
              count * kNumWdlOutputs * sizeof(float));
      if (spec.output_mlh) {
        const auto mlh_bytes = std::as_writable_bytes(std::span<float>(mlh));
        execution.GetBuffer(*spec.output_mlh)
            .CopyToHostAsync(mlh_bytes.subspan(offset * sizeof(float)),
                             count * sizeof(float));
      }
      execution.Synchronize();
    };

    if (split) {
      part(*split->first, 0, split->first_count);
      part(*split->second, split->first_count,
           actual_batch - split->first_count);
    } else {
      part(program, 0, actual_batch);
    }
  }
  // The device is idle from here on; decoding is host work (as wrapper.cc).
  callback(ComputationEvent::FIRST_BACKEND_IDLE);

  for (std::size_t sample = 0; sample < actual_batch; ++sample) {
    const Entry& entry = entries_[sample];
    DecodeWdl(std::span<const float>(
                  wdl.data() + sample * kNumWdlOutputs, kNumWdlOutputs),
              entry.result);
    if (!entry.result.p.empty()) {
      DecodePolicy(
          entry,
          std::span<const float>(policy.data() + sample * kNumOutputPolicy,
                                 kNumOutputPolicy));
    }
    if (entry.result.m) {
      *entry.result.m = mlh.empty() ? 0.0f : mlh[sample];
    }
    // No error output in the artifact: report zero uncertainty, as every other
    // backend does for a net without an error head (GetEVal()'s default).
    // Unwritten, the memcache hands the search an uninitialised value.
    if (entry.result.e) *entry.result.e = 0.0f;
  }
}

class Lc0exCudaBackendFactory final : public BackendFactory {
 public:
  int GetPriority() const override { return 1; }
  std::string_view GetName() const override { return kBackendName; }

  std::unique_ptr<Backend> Create(const OptionsDict& options) override {
    OptionsDict backend_options;
    backend_options.AddSubdictFromString(
        options.Get<std::string>(SharedBackendParams::kBackendOptionsId));

    const std::string weights_path =
        options.Get<std::string>(SharedBackendParams::kWeightsId);
    const std::optional<WeightsFile> weights = LoadWeights(weights_path);
    if (!weights) {
      throw Exception("The lc0ex-cuda backend requires a network file.");
    }

    auto backend = std::make_unique<Lc0exCudaBackend>(*weights, options,
                                                       backend_options);
    backend_options.CheckAllOptionsRead(std::string(kBackendName));
    return backend;
  }
};

REGISTER_BACKEND(Lc0exCudaBackendFactory)

// lc0ex under the old Network interface, so Menkib's demux can carry it.
class Lc0exCudaNetwork;

class Lc0exCudaNetworkComputation final : public NetworkComputation {
 public:
  explicit Lc0exCudaNetworkComputation(Lc0exCudaBackend* backend)
      : backend_(backend) {}

  void AddInput(InputPlanes&& input) override {
    if (input.size() != kInputPlanes) {
      throw Exception("lc0ex-cuda-net: unexpected input plane count.");
    }
    for (const auto& plane : input) {
      masks_.push_back(plane.mask);
      values_.push_back(plane.value);
    }
    ++batch_;
  }

  void ComputeBlocking() override {
    LCTRACE_FUNCTION_SCOPE;
    if (batch_ == 0) return;
    policy_.resize(batch_ * kNumOutputPolicy);
    std::vector<float> wdl(batch_ * kNumWdlOutputs);
    if (backend_->HasMlhOutput()) mlh_.resize(batch_);
    if (backend_->HasErrOutput()) err_.resize(batch_);
    const std::size_t top = backend_->MaxBatch();
    for (std::size_t start = 0; start < batch_; start += top) {
      const std::size_t n = std::min(top, batch_ - start);
      backend_->EvaluatePlanes(
          masks_.data() + start * kInputPlanes,
          values_.data() + start * kInputPlanes, n,
          policy_.data() + start * kNumOutputPolicy,
          wdl.data() + start * kNumWdlOutputs,
          mlh_.empty() ? nullptr : mlh_.data() + start,
          err_.empty() ? nullptr : err_.data() + start);
    }
    q_.resize(batch_);
    d_.resize(batch_);
    for (std::size_t sample = 0; sample < batch_; ++sample) {
      const float* logits = wdl.data() + sample * kNumWdlOutputs;
      const float maximum = std::max({logits[0], logits[1], logits[2]});
      const float win = std::exp(logits[0] - maximum);
      const float draw = std::exp(logits[1] - maximum);
      const float loss = std::exp(logits[2] - maximum);
      const float scale = 1.0f / (win + draw + loss);
      q_[sample] = (win - loss) * scale;
      d_[sample] = draw * scale;
    }
  }

  int GetBatchSize() const override { return static_cast<int>(batch_); }
  float GetQVal(int sample) const override { return q_[sample]; }
  float GetDVal(int sample) const override { return d_[sample]; }
  float GetEVal(int sample) const override {
    return err_.empty() ? 0.0f : err_[sample];
  }
  float GetPVal(int sample, int move_id) const override {
    return policy_[sample * kNumOutputPolicy + move_id];
  }
  float GetMVal(int sample) const override {
    return mlh_.empty() ? 0.0f : mlh_[sample];
  }

 private:
  Lc0exCudaBackend* backend_;
  std::size_t batch_ = 0;
  std::vector<std::uint64_t> masks_;
  std::vector<float> values_;
  std::vector<float> policy_;
  std::vector<float> q_;
  std::vector<float> d_;
  std::vector<float> mlh_;
  std::vector<float> err_;
};

class Lc0exCudaNetwork final : public Network {
 public:
  Lc0exCudaNetwork(const WeightsFile& weights, const OptionsDict& options)
      : backend_(weights, options, Lc0exCudaBackend::NetworkPath{}),
        capabilities_{weights.format().network_format().input(),
                      weights.format().network_format().output(),
                      weights.format().network_format().moves_left()},
        threads_(std::max(1, options.GetOrDefault<int>("concurrency", 1))),
        batch_step_(std::max(1, options.GetOrDefault<int>("batch_step", 8))) {}

  const NetworkCapabilities& GetCapabilities() const override {
    return capabilities_;
  }
  std::unique_ptr<NetworkComputation> NewComputation() override {
    return std::make_unique<Lc0exCudaNetworkComputation>(&backend_);
  }
  // One per execution slot; demux adds one more worker for a GPU child.
  int GetThreads() const override { return threads_; }
  int GetMiniBatchSize() const override {
    return static_cast<int>(backend_.MaxBatch());
  }
  int GetPreferredBatchStep() const override { return batch_step_; }

 private:
  Lc0exCudaBackend backend_;
  NetworkCapabilities capabilities_;
  int threads_;
  int batch_step_;
};

std::unique_ptr<Network> MakeLc0exCudaNetwork(
    const std::optional<WeightsFile>& weights, const OptionsDict& options) {
  if (!weights) {
    throw Exception("The lc0ex-cuda-net backend requires a network file.");
  }
  return std::make_unique<Lc0exCudaNetwork>(*weights, options);
}

REGISTER_NETWORK("lc0ex-cuda-net", MakeLc0exCudaNetwork, -900)

}  // namespace
}  // namespace lczero
