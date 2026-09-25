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

#include "neural/batchmerge.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <map>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "neural/shared_params.h"
#include "utils/atomic_vector.h"
#include "utils/logging.h"

namespace lczero {
namespace {

// Every in-tree backend encodes the last 8 plies (`EncodePositionForNN(...,
// 8, ...)`), so that is all a merged input has to carry forward.
constexpr std::size_t kEncodedHistoryPlies = 8;

class BatchMergingBackend;

class MergingComputation : public BackendComputation {
 public:
  // AddInput is called concurrently by the search's task workers, which is
  // why every in-tree computation holds its entries in an AtomicVector rather
  // than a std::vector. Sized like the others, from the parent's advertised
  // maximum batch.
  MergingComputation(BatchMergingBackend* backend, std::size_t capacity)
      : backend_(backend), inputs_(capacity) {}

  size_t UsedBatchSize() const override { return inputs_.size(); }

  AddInputResult AddInput(const EvalPosition& pos,
                          EvalResultPtr result) override {
    // The spans in EvalPosition borrow the caller's scratch history and a
    // MoveList local to its loop body; both are reused or destroyed the
    // moment this call returns. The device backends get away with borrowing
    // because they encode inside AddInput -- we evaluate on a worker thread
    // later, so the data has to be copied here. (The legacy multiplexing
    // network has no such problem: its API hands over encoded planes by
    // value.) Only the plies the encoders actually read are kept.
    const std::size_t keep = std::min(pos.pos.size(), kEncodedHistoryPlies);
    inputs_.emplace_back(Input{
        std::vector<Position>(pos.pos.end() - keep, pos.pos.end()),
        MoveList(pos.legal_moves.begin(), pos.legal_moves.end()), result});
    return ENQUEUED_FOR_EVAL;
  }

  // Note: inputs_ are deliberately NOT cleared by ComputeBlocking(). The
  // device backends keep their entries too, and the search reads
  // UsedBatchSize() after computing (search.cc:2499); a computation that
  // emptied itself would make the gathering loop bail out every iteration.
  void ComputeBlocking() override;

  // Called by the worker thread with no lock held. The spans in EvalPosition
  // and the pointers in EvalResultPtr belong to the caller, which is parked in
  // ComputeBlocking() until we notify it, so they stay alive across this.
  void PopulateInto(BackendComputation* parent) const {
    for (const auto& input : inputs_) {
      parent->AddInput(EvalPosition{input.history, input.legal_moves},
                       input.result);
    }
  }

  void NotifyReady(std::exception_ptr error) {
    std::lock_guard<std::mutex> lock(mutex_);
    error_ = std::move(error);
    ready_ = true;
    ready_cv_.notify_all();
  }

 private:
  struct Input {
    std::vector<Position> history;
    MoveList legal_moves;
    EvalResultPtr result;
  };

  BatchMergingBackend* backend_;
  AtomicVector<Input> inputs_;

  std::mutex mutex_;
  std::condition_variable ready_cv_;
  bool ready_ = false;
  std::exception_ptr error_;
};

class BatchMergingBackend : public Backend {
 public:
  BatchMergingBackend(std::unique_ptr<Backend> parent, int max_batch,
                      int wait_us, int threads)
      : parent_(std::move(parent)),
        // A merged batch is populated into ONE parent computation, whose
        // capacity is the parent's maximum batch; packing past it would throw
        // inside the parent. So the merge cap never exceeds that maximum
        // (the fleet runs 64 on a 64-rung artifact: unchanged).
        max_batch_(std::min(std::max(1, max_batch),
                            std::max(1, parent_->GetAttributes()
                                            .maximum_batch_size))),
        linger_(std::max(0, wait_us)) {
    if (max_batch_ < static_cast<std::size_t>(max_batch)) {
      CERR << "Batch merging: --batch-merge-max-batch=" << max_batch
           << " exceeds the backend's maximum batch; using " << max_batch_
           << ".";
    }
    for (int i = 0; i < std::max(1, threads); ++i) {
      workers_.emplace_back([this]() { Worker(); });
    }
  }

  ~BatchMergingBackend() override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      abort_ = true;
    }
    cv_.notify_all();
    for (auto& worker : workers_) worker.join();
    // Release anything that queued while we were shutting down, so that no
    // caller is left parked in ComputeBlocking() forever.
    while (!queue_.empty()) {
      queue_.front()->NotifyReady(nullptr);
      queue_.pop();
    }
    ReportHistogram();
  }

  BackendAttributes GetAttributes() const override {
    return parent_->GetAttributes();
  }

  std::unique_ptr<BackendComputation> CreateComputation() override {
    return std::make_unique<MergingComputation>(
        this, parent_->GetAttributes().maximum_batch_size);
  }

  std::optional<EvalResult> GetCachedEvaluation(
      const EvalPosition& pos) override {
    return parent_->GetCachedEvaluation(pos);
  }

  UpdateConfigurationResult UpdateConfiguration(
      const OptionsDict& opts) override {
    return parent_->UpdateConfiguration(opts);
  }

  bool IsSameConfiguration(const OptionsDict& opts) const override {
    return parent_->IsSameConfiguration(opts);
  }

  void Enqueue(MergingComputation* computation) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      pending_ += computation->UsedBatchSize();
      queue_.push(computation);
    }
    // One wakeup is enough: a woken worker drains the whole queue up to
    // max_batch, so waking all of them (24 of them, in the configuration this
    // is meant for) just has 23 find an empty queue. A worker sitting in the
    // linger can absorb this notification, but it drains when the linger
    // expires, so the delay is bounded by wait_us -- and is zero at wait_us=0,
    // which is the coalesce-on-dequeue setting.
    cv_.notify_one();
  }

 private:
  void Worker() {
    while (true) {
      std::vector<MergingComputation*> children;
      std::size_t packed = 0;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]() { return abort_ || !queue_.empty(); });
        if (abort_) return;
        // Linger for more work rather than run a batch of one game's nodes.
        if (linger_.count() > 0 && pending_ < max_batch_) {
          cv_.wait_for(lock, linger_, [this]() {
            return abort_ || pending_ >= max_batch_;
          });
          if (abort_) return;
        }
        while (!queue_.empty()) {
          const std::size_t size = queue_.front()->UsedBatchSize();
          // A single caller larger than the cap still has to go through.
          if (packed != 0 && packed + size > max_batch_) break;
          children.push_back(queue_.front());
          queue_.pop();
          packed += size;
          pending_ -= size;
        }
      }
      if (children.empty()) continue;

      std::exception_ptr error;
      try {
        auto parent = parent_->CreateComputation();
        for (const auto* child : children) child->PopulateInto(parent.get());
        parent->ComputeBlocking();
      } catch (...) {
        error = std::current_exception();
      }
      for (auto* child : children) child->NotifyReady(error);

      {
        std::lock_guard<std::mutex> lock(histogram_mutex_);
        ++histogram_[packed];
      }
      // Datagen arms are stopped with a signal, so the destructor's report
      // never runs; emit periodically so a killed run still leaves the
      // merged-batch distribution behind.
      MaybeReportHistogram();
    }
  }

  void MaybeReportHistogram() {
    const auto now = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lock(histogram_mutex_);
      if (now - last_report_ < std::chrono::seconds(60)) return;
      last_report_ = now;
    }
    ReportHistogram();
  }

  void ReportHistogram() const {
    std::size_t batches = 0;
    std::size_t entries = 0;
    for (const auto& [size, count] : histogram_) {
      batches += count;
      entries += size * count;
    }
    if (batches == 0) return;
    CERR << "Batch merging: " << batches << " merged batches, " << entries
         << " evaluations, mean merged batch "
         << static_cast<double>(entries) / batches << ".";
    std::string line;
    for (const auto& [size, count] : histogram_) {
      line += " " + std::to_string(size) + ":" + std::to_string(count);
    }
    CERR << "Batch merging histogram (size:count):" << line;
  }

  std::unique_ptr<Backend> parent_;
  const std::size_t max_batch_;
  const std::chrono::microseconds linger_;

  std::mutex mutex_;
  std::condition_variable cv_;
  std::queue<MergingComputation*> queue_;
  std::size_t pending_ = 0;
  bool abort_ = false;
  std::vector<std::thread> workers_;

  mutable std::mutex histogram_mutex_;
  std::map<std::size_t, std::size_t> histogram_;
  std::chrono::steady_clock::time_point last_report_ =
      std::chrono::steady_clock::now();
};

void MergingComputation::ComputeBlocking() {
  if (inputs_.size() == 0) return;
  backend_->Enqueue(this);
  std::exception_ptr error;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    ready_cv_.wait(lock, [this]() { return ready_; });
    ready_ = false;
    error = std::move(error_);
    error_ = nullptr;
  }
  if (error) std::rethrow_exception(error);
}

}  // namespace

std::unique_ptr<Backend> CreateBatchMergingBackend(
    std::unique_ptr<Backend> parent, int max_batch, int wait_us, int threads) {
  return std::make_unique<BatchMergingBackend>(std::move(parent), max_batch,
                                               wait_us, threads);
}

std::unique_ptr<Backend> MaybeWrapWithBatchMerging(
    std::unique_ptr<Backend> parent, const OptionsDict& opts) {
  const int max_batch = opts.Get<int>(SharedBackendParams::kBatchMergeMaxBatch);
  if (max_batch <= 0) return parent;
  return CreateBatchMergingBackend(
      std::move(parent), max_batch,
      opts.Get<int>(SharedBackendParams::kBatchMergeWaitUs),
      opts.Get<int>(SharedBackendParams::kBatchMergeThreads));
}

}  // namespace lczero
