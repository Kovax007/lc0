/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2020 The LCZero Authors

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
#include <atomic>
#include <cstdlib>
#include <future>
#include <new>
#include <queue>
#include <thread>

#include "neural/encoder.h"
#include "neural/factory.h"
#include "neural/shared_params.h"
#include "utils/atomic.h"
#include "utils/atomic_vector.h"
#include "utils/mutex.h"
#include "utils/numa.h"
#include "utils/trace.h"

namespace lczero {
namespace {

using Clock = std::chrono::high_resolution_clock;
#if __cpp_lib_hardware_interference_size >= 201703L
constexpr size_t kCacheLineSize = std::hardware_destructive_interference_size;
#else
constexpr size_t kCacheLineSize = 64;
#endif

class DemuxingComputation;

struct DemuxingWork {
  struct PredictedTimes {
    uint32_t prediction = 0;
    uint32_t original : 31 = 0;
    bool queue_was_idle_ : 1 = false;
  };
  DemuxingComputation* source_ = nullptr;
  int start_ = 0;
  int end_ = 0;
  PredictedTimes predicted_times_;

  DemuxingWork(int sample) : end_(sample) {}
  DemuxingWork(DemuxingComputation* source, int start, int end)
      : source_(source), start_(start), end_(end) {}

  void ProcessResults(std::unique_ptr<NetworkComputation>& computation);

  auto operator<=>(const DemuxingWork& b) const { return end_ <=> b.end_; }
};

class DemuxingComputation;
class DemuxingChildBackend;

class DemuxingChildBackend {
 public:
  using AssingType = std::tuple<BackendAttributes, const NetworkCapabilities*>;
  using AssignFuture = std::future<AssingType>;
  using AssignPromise = std::promise<AssingType>;

  ~DemuxingChildBackend();

  AssignFuture Assign(const std::string& name,
                      const std::optional<WeightsFile>& weights,
                      const OptionsDict& opts, std::span<int> shared_threads,
                      int shared_index = 0) {
    AssignPromise promise;
    auto rv = promise.get_future();
    assert(shared_index == 0 || shared_threads.empty());
    threads_.emplace_back([this, name, &weights, opts,
                           promise = std::move(promise), shared_threads,
                           shared_index]() mutable {
      try {
        int numa_node = opts.GetOrDefault<int>("numa_node", -1);
        bool numa_cuda_gpu = opts.GetOrDefault<bool>("numa_cuda_gpu", false);
        if (numa_node >= 0) {
          Numa::BindThreadToNode(numa_node);
          update_thread_bind_work_.start_ = numa_node;
        }
        int gpu = opts.GetOrDefault<int>("gpu", 0);
        if (numa_cuda_gpu) {
          Numa::BindThreadToCudaDevice(gpu);
          update_thread_bind_work_.end_ = gpu;
        }
        AssingType result = shared_index == 0
                                ? ConstructBackend(name, weights, opts)
                                : UseSharedBackend(this[shared_index].network_);
        std::vector<AssignFuture> other_backends;
        if (shared_threads.size() > 1) {
          std::get<0>(result).recommended_batch_size *= shared_threads.size();
          other_backends.reserve(shared_threads.size() - 1);
          int my_index = shared_threads.front();
          for (int i = 1; i < (int)shared_threads.size(); i++) {
            int index = shared_threads[i];
            other_backends.emplace_back(this[index - my_index].Assign(
                name, weights, opts, std::span<int>(), my_index - index));
          }
        }
        int nn_threads = opts.GetOrDefault<int>("demux_threads", 0);
        if (nn_threads == 0) {
          nn_threads = network_->GetThreads() + !network_->IsCpu();
        }
        while (!thread_init_ordering_.load(std::memory_order_acquire)) {
          SpinloopPause();
        }
        for (int i = 1; i < nn_threads; i++) {
          threads_.emplace_back([this, numa_node, numa_cuda_gpu, gpu]() {
            if (numa_node >= 0) {
              Numa::BindThreadToNode(numa_node);
            }
            if (numa_cuda_gpu) {
              Numa::BindThreadToCudaDevice(gpu);
            }
            Worker();
          });
        }
        for (auto& f : other_backends) {
          // Wait for the other shared backends and check for exceptions.
          f.get();
        }
        promise.set_value(std::move(result));
      } catch (...) {
        promise.set_exception(std::current_exception());
        return;
      }
      Worker();
    });
    thread_init_ordering_.store(1, std::memory_order_release);
    return rv;
  }

  AssingType UseSharedBackend(std::shared_ptr<Network> shared) {
    network_ = shared;
    BackendAttributes attrs(*network_);
    batch_times_ns_ =
        std::make_unique<std::atomic<uint32_t>[]>(attrs.maximum_batch_size + 1);
    IdlePrediction prediction(Clock::now().time_since_epoch().count(), 0);
    idle_prediction_.store(prediction, std::memory_order_relaxed);
    return {attrs, &network_->GetCapabilities()};
  }

  AssingType ConstructBackend(const std::string& name,
                              const std::optional<WeightsFile>& weights,
                              const OptionsDict& opts) {
    network_ = NetworkFactory::Get()->Create(name, weights, opts);
    BackendAttributes attrs(*network_);
    batch_times_ns_ =
        std::make_unique<std::atomic<uint32_t>[]>(attrs.maximum_batch_size + 1);
    IdlePrediction prediction(Clock::now().time_since_epoch().count(), 0);
    idle_prediction_.store(prediction, std::memory_order_relaxed);
    return {attrs, &network_->GetCapabilities()};
  }

  std::tuple<uint32_t, uint32_t, uint32_t> GetIdlePrediction(
      Clock::time_point now, int size1, int size2) const {
    IdlePrediction prediction =
        idle_prediction_.load(std::memory_order_relaxed);
    uint32_t cost1 = std::get<0>(GetBatchTime(size1));
    uint32_t cost2 = 0;
    if (size2 > 0) {
      cost2 = std::get<0>(GetBatchTime(size2));
    }

    if (prediction.queued_work_ns_ == 0) {
      return {0, cost1, cost2};
    }
    int64_t idle_time_ns =
        prediction.last_batch_completed_ - now.time_since_epoch().count();
    int64_t queued_work_ns = prediction.queued_work_ns_;
    return {std::max<int64_t>(0, idle_time_ns + queued_work_ns), cost1, cost2};
  }

  std::tuple<uint32_t, uint32_t> GetBatchTime(int batch_size) const {
    uint32_t batch_time, original_batch_time;
    original_batch_time = batch_time =
        batch_times_ns_[batch_size].load(std::memory_order_relaxed);
    if (batch_time == 0) {
      batch_time =
          batch_times_ns_[0].load(std::memory_order_relaxed) * batch_size;
    }
    return {batch_time, original_batch_time};
  }

  DemuxingWork::PredictedTimes AddBackendWork(int batch_size) {
    auto [batch_time, original_batch_time] = GetBatchTime(batch_size);
    auto idle = idle_prediction_.load(std::memory_order_relaxed);
    IdlePrediction new_idle = idle;
    bool backend_idle = false;
    do {
      backend_idle = new_idle.queued_work_ns_ == 0;
      if (idle.queued_work_ns_ == 0) {
        // Idle backend requires setting completion time to be when GPU gets the
        // work. We use CPU time to approximate the timing.
        auto now = Clock::now();
        new_idle.last_batch_completed_ = now.time_since_epoch().count();
      }
      new_idle.queued_work_ns_ += batch_time;
    } while (!idle_prediction_.compare_exchange_weak(
        idle, new_idle, std::memory_order_relaxed));
    return {batch_time, original_batch_time, backend_idle};
  }

  void StartComputationWhenIdle() {
    auto idle = idle_prediction_.load(std::memory_order_relaxed);
    IdlePrediction new_idle = idle;
    auto now = Clock::now();
    do {
      // Idle backend requires setting completion time to be when GPU gets the
      // work. We use CPU time to approximate the timing.
      new_idle.last_batch_completed_ = now.time_since_epoch().count();
    } while (!idle_prediction_.compare_exchange_weak(
        idle, new_idle, std::memory_order_relaxed));
  }

  constexpr static int BatchTimeUpdateWeightMul = 3;
  constexpr static int BatchTimeUpdateWeightDiv = 4;

  void CompleteBackendWork(int batch_size,
                           DemuxingWork::PredictedTimes predicted_times) {
    auto predicted_batch_time = predicted_times.prediction,
         original_batch_time = predicted_times.original;
    auto now = Clock::now();
    auto idle = idle_prediction_.load(std::memory_order_relaxed);
    IdlePrediction new_idle = idle;
    do {
      new_idle.last_batch_completed_ = now.time_since_epoch().count();
      new_idle.queued_work_ns_ -= predicted_batch_time;
    } while (!idle_prediction_.compare_exchange_weak(
        idle, new_idle, std::memory_order_relaxed));

    int32_t batch_time =
        now.time_since_epoch().count() - idle.last_batch_completed_;

    int32_t average_per_position = batch_time / batch_size;
    if (batch_time < 0) {
      // Protect against out of order clocks.
      return;
    }

    bool updated = false;
    if (original_batch_time == 0) {
      if (batch_times_ns_[batch_size].compare_exchange_strong(
              original_batch_time, batch_time, std::memory_order_relaxed)) {
        // There was no average yet so we updated it to match the first batch
        // time.
        updated = true;
      }
      // Other thread updated the prediction. We update the average using the
      // new prediction.
    }
    int32_t batch_time_delta =
        static_cast<int64_t>(batch_time - original_batch_time) *
        BatchTimeUpdateWeightMul / BatchTimeUpdateWeightDiv;
    if (batch_time_delta == 0) {
      // No need to update atomic value if prediction was accurate.
      updated = true;
    }
    if (!updated) {
      batch_times_ns_[batch_size].fetch_add(batch_time_delta,
                                            std::memory_order_relaxed);
    }

    uint32_t average = batch_times_ns_[0].load(std::memory_order_relaxed);
    if (average == 0) {
      if (batch_times_ns_[0].compare_exchange_strong(
              average, average_per_position, std::memory_order_relaxed)) {
        // There was no average yet so we updated it to match the first batch
        // time.
        return;
      }
    }
    int32_t average_delta =
        static_cast<int64_t>(average_per_position - average) *
        BatchTimeUpdateWeightMul / BatchTimeUpdateWeightDiv;
    batch_times_ns_[0].fetch_add(average_delta, std::memory_order_relaxed);
  }

  void Enqueue(DemuxingWork* work) {
    bool signal_atomic = false;
    {
      Mutex::Lock lock(mutex_);
      work->predicted_times_ = AddBackendWork(work->end_ - work->start_);
      queue_.push(work);
      State s = worker_state_.load(std::memory_order_relaxed);
      if (s == SLEEP) {
        signal_atomic = true;
        worker_state_.store(DATAQUEUED, std::memory_order_relaxed);
      }
    }
    if (signal_atomic) {
      worker_state_.notify_one();
    }
  }

  void Abort() {
    Mutex::Lock lock(mutex_);
    worker_state_.store(ABORTED, std::memory_order_relaxed);
    worker_state_.notify_all();
  }

  void Worker();

  void UpdateThreadBind() {
    if (update_thread_bind_work_.start_ == -1 &&
        update_thread_bind_work_.end_ == -1) {
      return;
    }
    update_thread_bind_work_.predicted_times_.prediction = threads_.size();
    {
      Mutex::Lock lock(mutex_);
      for (size_t i = 0; i < threads_.size(); i++) {
        queue_.push(&update_thread_bind_work_);
      }
      State s = worker_state_.load(std::memory_order_relaxed);
      if (s == SLEEP) {
        worker_state_.store(DATAQUEUED, std::memory_order_relaxed);
      }
    }
    worker_state_.notify_all();
  };

 private:
  struct IdlePrediction {
    uint64_t last_batch_completed_;
    uint64_t queued_work_ns_;

    IdlePrediction() = default;
    IdlePrediction(uint64_t last_batch_completed, uint64_t queued_work_ns)
        : last_batch_completed_(last_batch_completed),
          queued_work_ns_(queued_work_ns) {}
  };

  // Runtime constant variables
  std::vector<std::thread> threads_;
  std::shared_ptr<Network> network_;
  std::atomic<int> thread_init_ordering_ = 0;
  std::unique_ptr<std::atomic<uint32_t>[]> batch_times_ns_;
  // Atomically updated variables
  alignas(kCacheLineSize) std::atomic<IdlePrediction> idle_prediction_;
  // Mutex protected queue
  Mutex mutex_;
  enum State { SLEEP, DATAQUEUED, ABORTED };
  WaitableAtomic<State> worker_state_{SLEEP};
  std::queue<DemuxingWork*> queue_ GUARDED_BY(mutex_);
  DemuxingWork update_thread_bind_work_{nullptr, -1, -1};
};

class DemuxingBackend final : public Backend {
 public:
  static size_t CalculateBackendCount(const OptionsDict& backend_options) {
    const auto parents = backend_options.ListSubdicts();
    if (parents.empty()) {
      return std::max(
          1, backend_options.GetOrDefault<int>("shared_backend_threads", 1));
    }
    size_t count = 0;
    for (const auto& name : parents) {
      const auto& opts = backend_options.GetSubdict(name);
      size_t shared = opts.GetOrDefault<int>("shared_backend_threads", 1);
      count += std::max<size_t>(1, shared);
    }
    return count;
  }
  DemuxingBackend(const std::optional<WeightsFile>& weights,
                  const OptionsDict& options,
                  const OptionsDict& backend_options)
      : backends_(CalculateBackendCount(backend_options)),
        backend_opts_(
            options.Get<std::string>(SharedBackendParams::kBackendOptionsId)),
        weights_path_(
            options.Get<std::string>(SharedBackendParams::kWeightsId)) {
    int shared_threads = std::max(
        1, backend_options.GetOrDefault<int>("shared_backend_threads", 1));
    minimum_batch_step_ =
        backend_options.GetOrDefault<int>("min_batch_step", 1);
    core_reservation_counter_ = Numa::GetCoreReservationCounter();
    UpdateConfiguration(options);
    const auto parents = backend_options.ListSubdicts();
    std::vector<DemuxingChildBackend::AssignFuture> capabilities;
    capabilities.reserve(std::max<size_t>(1, parents.size()));
    if (parents.empty()) {
      // If options are empty, or multiplexer configured in root object,
      // initialize on root object and default backend.
      auto backends = NetworkFactory::Get()->GetBackendsList();
      std::vector<int> indexes;
      indexes.resize(shared_threads);
      assert(indexes.size() == backends_.size());
      std::generate(indexes.begin(), indexes.end(),
                    [n = 0]() mutable { return n++; });
      capabilities.emplace_back(AddBackend(
          0, backends[0], weights, backend_options, std::span(indexes)));
    }

    std::vector<std::tuple<unsigned, unsigned, std::vector<int>>>
        shared_indexes;
    // Interleave indexes if using shared backends.
    if (backends_.size() != parents.size()) {
      shared_indexes.reserve(parents.size());
      unsigned i = 0;
      for (const auto& name : parents) {
        const auto& opts = backend_options.GetSubdict(name);
        unsigned shared_threads =
            std::max(1, opts.GetOrDefault<int>("shared_backend_threads", 1));
        shared_indexes.emplace_back(shared_threads, i++, std::vector<int>());
      }
      std::sort(shared_indexes.begin(), shared_indexes.end(),
                [](const auto& a, const auto& b) {
                  return std::get<0>(a) > std::get<0>(b);
                });
      auto iter = shared_indexes.begin();
      unsigned src = 0;
      for (i = 0; i < backends_.size(); i++, iter++) {
        if (iter == shared_indexes.end()) {
          iter = shared_indexes.begin();
          src++;
        }
        while (std::get<0>(*iter) <= src) {
          iter++;
          if (iter == shared_indexes.end()) {
            iter = shared_indexes.begin();
            src++;
          }
        }
        std::get<2>(*iter).push_back(i);
      }
    }
    std::sort(shared_indexes.begin(), shared_indexes.end(),
              [](const auto& a, const auto& b) {
                return std::get<1>(a) < std::get<1>(b);
              });

    int i = 0;
    for (const auto& name : parents) {
      int index = i;
      std::span<int> shared_threads;
      if (!shared_indexes.empty()) {
        index = std::get<2>(shared_indexes[i]).front();
        shared_threads = std::span(std::get<2>(shared_indexes[i]));
      }
      capabilities.emplace_back(AddBackend(index, name, weights,
                                           backend_options.GetSubdict(name),
                                           shared_threads));
      i++;
    }
    i = 0;
    for (auto& future : capabilities) {
      assert(future.valid());
      auto [attr, caps] = future.get();
      if (i == 0) {
        i = 1;
        attrs_ = attr;
        input_format_ = caps->input_format;
      } else {
        attrs_ += attr;
        if (input_format_ != caps->input_format) {
          throw Exception("Incompatible input formats, " +
                          std::to_string(input_format_) + " vs " +
                          std::to_string(caps->input_format));
        }
      }
      if (attr.runs_on_cpu) {
        uses_cpu_backend_ = true;
      }
    }
    attrs_.maximum_batch_size =
        std::max(attrs_.recommended_batch_size, attrs_.maximum_batch_size);
    attrs_.maximum_batch_size = backend_options.GetOrDefault<int>(
        "max_batch", attrs_.maximum_batch_size);
    attrs_.recommended_batch_size =
        std::min(attrs_.maximum_batch_size, attrs_.recommended_batch_size);

    minimum_batch_step_ =
        std::clamp(minimum_batch_step_, 1, attrs_.recommended_batch_size);

    const int default_min_batch_step =
        attrs_.preferred_batch_step * (backends_.size() + 1) / 2;
    lower_limit_batch_step_ = backend_options.GetOrDefault<int>(
        "lower_limit_min_batch_step", default_min_batch_step);

    CalibratePerformance();

    CERR << "Demuxing backend initialized with " << backends_.size()
         << " backends, maximum batch size: " << attrs_.maximum_batch_size
         << ", recommended batch size: " << attrs_.recommended_batch_size
         << ", preferred batch step: " << attrs_.preferred_batch_step
         << ", minimum batch step: " << minimum_batch_step_
         << ", lower limit: " << lower_limit_batch_step_;
  }

  void CalibratePerformance();

  DemuxingChildBackend::AssignFuture AddBackend(
      int index, const std::string& name,
      const std::optional<WeightsFile>& weights, const OptionsDict& opts,
      std::span<int> shared_threads) {
    const std::string backend = opts.GetOrDefault<std::string>("backend", name);

    return backends_[index].Assign(backend, weights, opts, shared_threads);
  }

  std::unique_ptr<BackendComputation> CreateComputation() override;

  BackendAttributes GetAttributes() const override { return attrs_; }

  ~DemuxingBackend() { Abort(); }

  void Abort() {
    for (auto& b : backends_) {
      b.Abort();
    }
  }

  UpdateConfigurationResult UpdateConfiguration(
      const OptionsDict& options) override {
    auto rv = Backend::UpdateConfiguration(options);
    if (rv != UPDATE_OK) return rv;
    if (backend_opts_ !=
        options.Get<std::string>(SharedBackendParams::kBackendOptionsId)) {
      return NEED_RESTART;
    }
    if (weights_path_ !=
        options.Get<std::string>(SharedBackendParams::kWeightsId)) {
      return NEED_RESTART;
    }
    if (core_reservation_counter_ != Numa::GetCoreReservationCounter()) {
      for (auto& b : backends_) {
        b.UpdateThreadBind();
      }
      core_reservation_counter_ = Numa::GetCoreReservationCounter();
    }
    softmax_policy_temperature_ =
        1.0f / options.Get<float>(SharedBackendParams::kPolicySoftmaxTemp);
    fill_empty_history_ = EncodeHistoryFill(
        options.Get<std::string>(SharedBackendParams::kHistoryFill));
    return UPDATE_OK;
  }

 private:
  std::vector<DemuxingChildBackend> backends_;
  BackendAttributes attrs_;
  pblczero::NetworkFormat::InputFormat input_format_;
  size_t core_reservation_counter_ = 0;
  float softmax_policy_temperature_;
  FillEmptyHistory fill_empty_history_;
  int minimum_batch_step_ = 1;
  unsigned lower_limit_batch_step_ = 1;
  bool uses_cpu_backend_ = false;

  alignas(kCacheLineSize) Mutex load_balancing_mutex_;

  // Cache cold variables
  const std::string backend_opts_;
  const std::string weights_path_;

  friend class DemuxingComputation;
};

class DemuxingComputation final : public BackendComputation {
 public:
  DemuxingComputation(DemuxingBackend* backend)
      : backend_(backend), entries_(backend_->attrs_.maximum_batch_size) {}
  ~DemuxingComputation() {}

  AddInputResult AddInput(const EvalPosition& pos,
                          EvalResultPtr result) override {
    int transform;
    const size_t idx = entries_.emplace_back(NetworkComputationRequest{
        .input = EncodePositionForNN(backend_->input_format_, pos.pos, 8,
                                     backend_->fill_empty_history_, &transform),
        .nn_indices = NetworkComputationRequest::ToNNIndices(pos.legal_moves, transform),
        .result = result,
        .transform = 0});
    entries_[idx].transform = transform;
    return ENQUEUED_FOR_EVAL;
  }

  void ComputeBlocking(ComputationCallback callback) override;

  size_t UsedBatchSize() const override { return entries_.size(); }

  void NotifyFirstDone() {
    callback_(ComputationEvent::FIRST_BACKEND_IDLE);
    first_done_.notify_one();
  }

  void NotifyComplete() {
    // speed lane 10-31: read what this needs from *this BEFORE the decrement. The decrement that reaches 0 releases the
    // waiter in ComputeBlocking, whose search worker may free this computation at once; reading backend_ after it was a
    // use-after-free (GPF on rig2 with the engine pinned, demux over random children).
    const bool uses_cpu_backend = backend_->uses_cpu_backend_;
    int outstanding = dataready_.fetch_sub(1, std::memory_order_acq_rel);
    if (uses_cpu_backend && outstanding == 1) {
      last_done_.store(1, std::memory_order_release);
      last_done_.notify_one();
    }
  }

  void ProcessResults(const DemuxingWork& work,
                      std::unique_ptr<NetworkComputation>& computation);

 private:
  DemuxingBackend* backend_;
  AtomicVector<NetworkComputationRequest> entries_;
  std::vector<DemuxingWork> children_;
  ComputationCallback callback_;

  alignas(kCacheLineSize) std::atomic<int> dataready_ = 0;
  alignas(kCacheLineSize) std::atomic<int> first_done_ = 0;
  alignas(kCacheLineSize) std::atomic<int> last_done_ = 0;

  friend class DemuxingChildBackend;
};

void DemuxingWork::ProcessResults(
    std::unique_ptr<NetworkComputation>& computation) {
  source_->ProcessResults(*this, computation);
}

void DemuxingComputation::ProcessResults(
    const DemuxingWork& work,
    std::unique_ptr<NetworkComputation>& computation) {
  size_t size = work.end_ - work.start_;
  for (size_t i = 0; i < size; ++i) {
    entries_[work.start_ + i].ProcessResult(
        *computation, i, backend_->softmax_policy_temperature_);
  }
}

std::unique_ptr<BackendComputation> DemuxingBackend::CreateComputation() {
  return std::make_unique<DemuxingComputation>(this);
}

void DemuxingBackend::CalibratePerformance() {
  Position board{ChessBoard::kStartposBoard, 0, 0};
  EvalPosition pos{.pos = std::span(&board, 1), .legal_moves = {}};
  DemuxingComputation computation(this);
  for (int i = 0; i < attrs_.recommended_batch_size; i++) {
    computation.AddInput(pos, {});
  }
  computation.ComputeBlocking([](ComputationEvent) {});
}

DemuxingChildBackend::~DemuxingChildBackend() {
  while (!threads_.empty()) {
    threads_.back().join();
    threads_.pop_back();
  }
  while (!queue_.empty()) {
    queue_.front()->source_->NotifyComplete();
    queue_.pop();
  }
}

void DemuxingChildBackend::Worker() {
  while (worker_state_.load(std::memory_order_relaxed) != ABORTED) {
    DemuxingWork* work = nullptr;
    {
      worker_state_.wait(SLEEP, std::memory_order_relaxed);
      Mutex::Lock lock(mutex_);
      State s = worker_state_.load(std::memory_order_relaxed);
      if (queue_.empty()) {
        if (s == DATAQUEUED) {
          worker_state_.store(SLEEP, std::memory_order_relaxed);
        }
        continue;
      }
      work = queue_.front();
      queue_.pop();
      if (queue_.empty()) {
        if (s == DATAQUEUED) {
          worker_state_.store(SLEEP, std::memory_order_relaxed);
        }
      } else {
        if (threads_.size() > 1) {
          worker_state_.notify_one();
        }
      }
    }
    assert(work);
    if (work == &update_thread_bind_work_) {
      if (update_thread_bind_work_.start_ >= 0) {
        Numa::BindThreadToNode(update_thread_bind_work_.start_);
      }
      if (update_thread_bind_work_.end_ >= 0) {
        Numa::BindThreadToCudaDevice(update_thread_bind_work_.end_);
      }
      std::atomic_ref<uint32_t> prediction(
          update_thread_bind_work_.predicted_times_.prediction);
      prediction.fetch_sub(1, std::memory_order_relaxed);
      MonitorHelper monitor(prediction, 20000);
      monitor([](uint32_t value) { return value != 0; });
      continue;
    }
    LCTRACE_FUNCTION_SCOPE;
    auto computation = network_->NewComputation();
    auto& entries = work->source_->entries_;
    for (int i = work->start_; i < work->end_; i++) {
      computation->AddInput(std::move(entries[i].input));
    }
    if (work->predicted_times_.queue_was_idle_) {
      StartComputationWhenIdle();
    }
    computation->ComputeBlocking();
    // TODO: This should read the time from the backend which could use more
    // accurate GPU timers. CPU time has potential for random extra delay
    // sometimes.
    CompleteBackendWork(work->end_ - work->start_, work->predicted_times_);
    int expected = 0;
    if (work->source_->first_done_.compare_exchange_strong(
            expected, 1, std::memory_order_relaxed)) {
      work->source_->NotifyFirstDone();
    }
    work->ProcessResults(computation);
    work->source_->NotifyComplete();
  }
}

void DemuxingComputation::ComputeBlocking(ComputationCallback callback) {
  LCTRACE_FUNCTION_SCOPE;
  assert(UsedBatchSize() != 0);
  size_t used_batch_size = UsedBatchSize();
  callback_ = callback;
  // Calculate batch_step_ size split count.
  int step = backend_->attrs_.preferred_batch_step;
  if (used_batch_size <
          backend_->minimum_batch_step_ * backend_->backends_.size() &&
      used_batch_size > backend_->lower_limit_batch_step_) {
    step = backend_->minimum_batch_step_;
  }
  int splits = 1 + (used_batch_size - 1) / step;
  int last_split_size = (used_batch_size - 1) % step + 1;
  bool are_all_splits_full = last_split_size == step;
  // Calculate the minimum number of splits per backend.
  int split_size_per_backend = splits / backend_->backends_.size();
  // Calculate how many backends get extra work.
  int extra_split_backends =
      splits - split_size_per_backend * backend_->backends_.size();

  struct BackendSortingOrder {
    uint32_t idx;
    uint32_t work_left;
    std::array<uint32_t, 2> work_cost;
    auto operator<=>(const BackendSortingOrder& b) const {
      return work_left <=> b.work_left;
    }

    BackendSortingOrder() = default;
    BackendSortingOrder(uint32_t idx, uint32_t work_left,
                        std::array<uint32_t, 2> work_cost)
        : idx(idx), work_left(work_left), work_cost(work_cost) {}
  };

  // Do basic load balancing based on predicted time until idle.
  std::vector<BackendSortingOrder> backend_order;
  backend_order.reserve(backend_->backends_.size());

  Mutex::Lock lock(backend_->load_balancing_mutex_);

  auto now = Clock::now();
  int batch_size1 = extra_split_backends == 0 ? split_size_per_backend * step
                    : extra_split_backends > 1 ||
                            (extra_split_backends == 1 && are_all_splits_full)
                        ? (split_size_per_backend + 1) * step
                        : last_split_size + split_size_per_backend * step;
  int batch_size2 =
      are_all_splits_full || (extra_split_backends == 1 && !are_all_splits_full)
          ? 0
      : extra_split_backends == 0
          ? last_split_size + (split_size_per_backend - 1) * step
          : last_split_size + split_size_per_backend * step;
  for (uint32_t idx = 0; idx < backend_->backends_.size(); idx++) {
    const auto& b = backend_->backends_[idx];
    auto [idle_time, cost1, cost2] =
        b.GetIdlePrediction(now, batch_size1, batch_size2);
    backend_order.emplace_back(idx, idle_time, std::array{cost1, cost2});
  }

  auto begin = backend_order.begin();
  auto last = backend_order.begin() +
              (extra_split_backends > 0
                   ? extra_split_backends - (are_all_splits_full ? 0 : 1)
                   : backend_order.size() - (are_all_splits_full ? 0 : 1));
  // Select backends based on predicted end time of the largest batch size.
  if (extra_split_backends > 0 || !are_all_splits_full) {
    std::nth_element(
        begin, last, backend_order.end(),
        [](const BackendSortingOrder& a, const BackendSortingOrder& b) {
          if (a.work_left == 0 && b.work_left == 0) {
            return a.work_cost[0] < b.work_cost[0];
          }
          if (a.work_left == 0 || b.work_left == 0) {
            return a.work_left < b.work_left;
          }
          return a.work_left + a.work_cost[0] < b.work_left + b.work_cost[0];
        });
  }
  begin = last;

  if (!are_all_splits_full && extra_split_backends > 0) {
    // Select the shortest predicted end time for the batch with a partial
    // split to be the middle point.
    std::nth_element(
        begin, begin + 1, backend_order.end(),
        [](const BackendSortingOrder& a, const BackendSortingOrder& b) {
          if (a.work_left == 0 && b.work_left == 0) {
            return a.work_cost[0] < b.work_cost[0];
          }
          if (a.work_left == 0 || b.work_left == 0) {
            return a.work_left < b.work_left;
          }
          return a.work_left + a.work_cost[1] < b.work_left + b.work_cost[1];
        });
    begin = begin + 1;
  }

  // Find the first backend which got less work from the previous batch.
  size_t start_index = 0;
  size_t end_index = extra_split_backends;
  size_t work_start = 0;
  int work_items = split_size_per_backend > 0 ? backend_->backends_.size()
                                              : extra_split_backends;
  // First store the work item count and reserve memory from them.
  dataready_.store(work_items, std::memory_order_relaxed);
  children_.reserve(work_items);
  size_t i = start_index;
  // First send work to backends which get extra work.
  int split_size = split_size_per_backend + 1;
  for (; i != end_index; i++) {
    assert(work_start != UsedBatchSize());
    size_t next_i = i + 1;
    size_t idx = backend_order[i].idx;
    size_t work_end =
        next_i == end_index && !are_all_splits_full
            ? work_start + last_split_size + (split_size - 1) * step
            : work_start + split_size * step;
    work_end = std::min(work_end, UsedBatchSize());
    children_.emplace_back(this, work_start, work_end);
    backend_->backends_[idx].Enqueue(&children_.back());
    work_start = work_end;
  }
  // Queue remaining work items which don't get extra work.
  split_size--;
  if (split_size > 0) {
    for (; i != backend_order.size(); i++) {
      assert(work_start != UsedBatchSize());
      size_t work_end = work_start + split_size * step;
      size_t idx = backend_order[i].idx;
      work_end = std::min(work_end, UsedBatchSize());
      children_.emplace_back(this, work_start, work_end);
      backend_->backends_[idx].Enqueue(&children_.back());
      work_start = work_end;
    }
  }
  lock.unlock();
  assert(work_start == UsedBatchSize());
  assert(work_items == (int)children_.size());
  // Wait until all backends complete their work.
  if (!backend_->uses_cpu_backend_) {
    first_done_.wait(false, std::memory_order_acquire);
    // Use spinloop to reduce wake-up latency.
    static constexpr size_t kExpectedWaitNs = 1'000'000;
    MonitorHelper monitor(dataready_, kExpectedWaitNs);
    while (dataready_.load(std::memory_order_acquire) != 0) {
      monitor([](int value) { return value != 0; });
    }
  } else {
    last_done_.wait(0, std::memory_order_acquire);
  }
}

class DemuxingBackendFactory : public BackendFactory {
  std::unique_ptr<Backend> Create(const OptionsDict& options) override {
    const std::string backend_options_string =
        options.Get<std::string>(SharedBackendParams::kBackendOptionsId);
    OptionsDict backend_options;
    backend_options.AddSubdictFromString(backend_options_string);

    std::string net_path =
        options.Get<std::string>(SharedBackendParams::kWeightsId);
    std::optional<WeightsFile> weights = LoadWeights(net_path);
    return std::make_unique<DemuxingBackend>(weights, options, backend_options);
  }

  std::string_view GetName() const override {
    using namespace std::string_view_literals;
    return "demux"sv;
  }

  int GetPriority() const override { return -1001; }
};

BackendManager::Register register_demux(
    std::make_unique<DemuxingBackendFactory>());

}  // namespace
}  // namespace lczero
