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
#include "neural/loader.h"
#include "neural/onnx/converter.h"
#include "neural/register.h"
#include "neural/shared_params.h"
#include "network_fingerprint.h"
#include "proto/lc0ex_metadata.pb.h"
#include "runtime/lc0ex_cuda.h"
#include "utils/atomic_vector.h"
#include "utils/exception.h"
#include "utils/fastmath.h"
#include "utils/logging.h"

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

void CheckNetworkFingerprint(const WeightsFile& weights,
                             const pblczero::NeuralExecutable& executable) {
  pblczero::Net executable_fingerprint;
  executable_fingerprint.ParseFromString(executable.metadata());

  const auto network_fingerprint = lc0ex::BuildNetworkFingerprint(weights);
  if (network_fingerprint.OutputAsString() !=
      executable_fingerprint.OutputAsString()) {
    throw Exception(
        "The lc0ex executable was created for a different network architecture.");
  }
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

void UploadWeights(const WeightsFile& weights, lc0ex::Executable& executable,
                   const OptionsDict& backend_options) {
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
  return {
      .has_mlh =
          format.moves_left() != pblczero::NetworkFormat::MOVES_LEFT_NONE,
      .has_wdl = format.output() == pblczero::NetworkFormat::OUTPUT_WDL,
      .runs_on_cpu = false,
      .suggested_num_search_threads = 2,
      .recommended_batch_size = 0,
      .maximum_batch_size = 0,
  };
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

  void ComputeBlocking() override;

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

    const int concurrency =
        std::max(1, backend_options.GetOrDefault<int>("concurrency", 1));
    execution_slot_pool_.in_use.assign(concurrency, false);
    executions_.resize(concurrency);

    const std::string lc0ex_path = backend_options.Get<std::string>("lc0ex");
    if (lc0ex_path.empty()) {
      throw Exception("The lc0ex-cuda backend requires an lc0ex path.");
    }

    const auto executable_proto = LoadExecutableFile(lc0ex_path);
    CheckNetworkFingerprint(weights, executable_proto);

    // Off by default: it is a strict win only above the ladder's dense band,
    // which a search at --minibatch-size equal to the top dense rung never
    // reaches. Worth turning on for a minibatch-128 configuration.
    split_programs_ = backend_options.GetOrDefault<bool>("split_programs", false);
    const int gpu = backend_options.GetOrDefault<int>("gpu", 0);
    runtime_ = lc0ex::CreateLc0exCudaRuntime(
        gpu, ResolveGraphMode(
                 backend_options.GetOrDefault<std::string>("graph", "auto")));
    executable_ = runtime_->Load(executable_proto);
    InitializePrograms();
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

void Lc0exCudaBackendComputation::ComputeBlocking() {
  const std::size_t actual_batch = entries_.size();
  if (actual_batch == 0) return;

  const auto split = backend_->FindSplit(actual_batch);
  const ProgramSpec& program =
      split ? *split->first : backend_->FindProgram(actual_batch);

  std::vector<std::uint64_t> masks(actual_batch * kInputPlanes);
  std::vector<float> values(actual_batch * kInputPlanes);
  for (std::size_t sample = 0; sample < actual_batch; ++sample) {
    const std::size_t offset = sample * kInputPlanes;
    std::copy(entries_[sample].masks.begin(), entries_[sample].masks.end(),
              masks.begin() + offset);
    std::copy(entries_[sample].values.begin(), entries_[sample].values.end(),
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
      lc0ex::Execution& execution = backend_->GetExecution(permit.index(), spec);
      execution.GetBuffer(*spec.input_masks)
          .CopyFromHostAsync(mask_bytes.subspan(offset * kInputPlanes *
                                                sizeof(std::uint64_t)),
                             count * kInputPlanes * sizeof(std::uint64_t));
      execution.GetBuffer(*spec.input_values)
          .CopyFromHostAsync(
              value_bytes.subspan(offset * kInputPlanes * sizeof(float)),
              count * kInputPlanes * sizeof(float));
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

}  // namespace
}  // namespace lczero
