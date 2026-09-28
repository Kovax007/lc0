/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2025 The LCZero Authors

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

#include "neural/memcache.h"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <vector>

#include "neural/shared_params.h"
#include "utils/atomic_vector.h"
#include "utils/cache.h"
#include "utils/logging.h"
#include "utils/smallarray.h"

namespace lczero {
namespace {

// TODO For now it uses the hash of the current position, ignoring repetitions
// and history. We'll likely need to have configurable hash function that we'll
// also reuse as a tree hash key.
uint64_t ComputeEvalPositionHash(const EvalPosition& pos) {
  return pos.pos.back().Hash();
}

struct CachedValue {
  float q;
  float d;
  float m;
  uint8_t num_moves;
  std::unique_ptr<float[]> p;
  // Child-Q means followed by sigmas (2 * num_moves), when the backend has them.
  std::unique_ptr<float[]> childq;
};

void CachedValueToEvalResult(const CachedValue& cv, const EvalResultPtr& ptr) {
  if (ptr.d) *ptr.d = cv.d;
  if (ptr.q) *ptr.q = cv.q;
  if (ptr.m) *ptr.m = cv.m;
  std::copy(cv.p.get(), cv.p.get() + ptr.p.size(), ptr.p.begin());
  if (!ptr.cq.empty()) {
    const float* childq = cv.childq.get();
    std::copy(childq, childq + ptr.cq.size(), ptr.cq.begin());
    std::copy(childq + cv.num_moves, childq + cv.num_moves + ptr.cs.size(),
              ptr.cs.begin());
  }
}

class MemCache : public CachingBackend {
 public:
  MemCache(std::unique_ptr<Backend> wrapped, const OptionsDict& options)
      : wrapped_backend_(std::move(wrapped)),
        cache_(options.Get<int>(SharedBackendParams::kNNCacheSizeId)),
        // At least 1: a zero capacity would flush forever.
        max_batch_size_(std::max(
            1, wrapped_backend_->GetAttributes().maximum_batch_size)),
        has_childq_(wrapped_backend_->GetAttributes().has_childq) {}

  BackendAttributes GetAttributes() const override {
    return wrapped_backend_->GetAttributes();
  }
  std::unique_ptr<BackendComputation> CreateComputation() override;
  std::optional<EvalResult> GetCachedEvaluation(const EvalPosition&) override;

  void ClearCache() override { cache_.Clear(); }

  UpdateConfigurationResult UpdateConfiguration(
      const OptionsDict& options) override {
    auto ret = wrapped_backend_->UpdateConfiguration(options);
    if (ret == Backend::UPDATE_OK) {
      // Check if we need to clear the cache.
      if (!wrapped_backend_->IsSameConfiguration(options)) {
        cache_.Clear();
      }
    }
    return ret;
  }

  bool IsSameConfiguration(const OptionsDict& options) const override {
    return wrapped_backend_->IsSameConfiguration(options);
  }

  void SetCacheSize(size_t size) override { cache_.SetCapacity(size); }

 private:
  std::unique_ptr<Backend> wrapped_backend_;
  HashKeyedCache<CachedValue> cache_;
  const size_t max_batch_size_;
  // Cached entries with legal moves keep the child-Q head whenever the backend
  // has one, also for requests that do not ask for it (prefetch), so that a
  // later request that does is still a cache hit.
  const bool has_childq_;
  friend class MemCacheComputation;
};

// Every backend below this cache holds a computation in a fixed-capacity vector
// sized at the backend's maximum batch (lc0ex: the artifact's top rung, 64 for
// the datagen artifact), and throws "AtomicVector overflow" past it -- which,
// on a search thread, kills the process. Nothing upstream of here promises to
// stay under that number (the fleet has lost a process to it about once per
// 50 GPU-hours since 2026-09-08, cause of the >64 never pinned down), so this
// computation guarantees it: when the wrapped computation is full, it is run
// early, and a fresh one takes the next position. The split is INVISIBLE to
// the caller: UsedBatchSize() keeps counting the whole batch, and the early
// part's values reach the caller and the cache only in ComputeBlocking(), in
// the original order -- so a search sees exactly the return codes, sizes and
// cache state of an unsplit batch and takes the same path (gated: a forced
// split every 8 positions gives byte-identical selfplay games). A backend's
// rows do not depend on their batch companions (bitwise-checked for lc0ex on
// 09-12), so the values are the same too.
class MemCacheComputation : public BackendComputation {
 public:
  MemCacheComputation(std::unique_ptr<BackendComputation> wrapped_computation,
                      MemCache* memcache)
      : wrapped_computation_(std::move(wrapped_computation)),
        memcache_(memcache),
        entries_(memcache->max_batch_size_) {}

 private:
  struct Entry {
    uint64_t key;
    std::unique_ptr<CachedValue> value;
    EvalResultPtr result_ptr;
  };

  size_t UsedBatchSize() const override {
    // Shared lock: a concurrent FlushFull() replaces wrapped_computation_.
    std::shared_lock<std::shared_mutex> lock(flush_mutex_);
    return flushed_count_.load(std::memory_order_relaxed) +
           wrapped_computation_->UsedBatchSize();
  }
  virtual AddInputResult AddInput(const EvalPosition& pos,
                                  EvalResultPtr result) override {
    assert(pos.legal_moves.size() == result.p.size() || result.p.empty());
    const uint64_t hash = ComputeEvalPositionHash(pos);
    {
      HashKeyedCacheLock<CachedValue> lock(&memcache_->cache_, hash);
      // Sometimes search queries NN without passing the legal moves. It is
      // still cached in this case, but in subsequent queries we only return it
      // if legal moves are not passed again. Otherwise check the size to guard
      // against hash collisions.
      if (lock.holds_value() &&
          (pos.legal_moves.empty() ||
           (lock->p && lock->num_moves == pos.legal_moves.size() &&
            (result.cq.empty() || lock->childq)))) {
        CachedValueToEvalResult(**lock, result);
        return AddInputResult::FETCHED_IMMEDIATELY;
      }
    }
    // Reserve a slot before touching either vector, so concurrent AddInput
    // calls (search task workers) can never overrun the capacity together.
    while (true) {
      std::shared_lock<std::shared_mutex> lock(flush_mutex_);
      if (reserved_.fetch_add(1, std::memory_order_relaxed) <
          entries_.capacity()) {
        return AddReserved(hash, pos, result);
      }
      reserved_.fetch_sub(1, std::memory_order_relaxed);
      lock.unlock();
      FlushFull();
    }
  }

  virtual void ComputeBlocking() override {
    if (wrapped_computation_->UsedBatchSize() > 0) {
      wrapped_computation_->ComputeBlocking();
    }
    // Early-run entries first: the order an unsplit batch would insert them.
    for (auto& entry : flushed_) Publish(entry);
    for (auto& entry : entries_) Publish(entry);
  }

  AddInputResult AddReserved(uint64_t hash, const EvalPosition& pos,
                             EvalResultPtr result) {
    size_t entry_idx = entries_.emplace_back(
        Entry{hash, std::make_unique<CachedValue>(), result});
    auto& value = entries_[entry_idx].value;
    value->p.reset(pos.legal_moves.empty() ? nullptr
                                           : new float[pos.legal_moves.size()]);
    value->num_moves = pos.legal_moves.size();
    EvalResultPtr value_ptr{&value->q, &value->d, &value->m,
                            value->p ? std::span<float>{value->p.get(),
                                                        pos.legal_moves.size()}
                                     : std::span<float>{}};
    if (value->p && memcache_->has_childq_) {
      const size_t num_moves = pos.legal_moves.size();
      value->childq.reset(new float[2 * num_moves]);
      value_ptr.cq = {value->childq.get(), num_moves};
      value_ptr.cs = {value->childq.get() + num_moves, num_moves};
    }
    return wrapped_computation_->AddInput(pos, value_ptr);
  }

  // Delivers an entry's value to its caller and into the cache.
  void Publish(Entry& entry) {
    CachedValueToEvalResult(*entry.value, entry.result_ptr);
    memcache_->cache_.Insert(entry.key, std::move(entry.value));
  }

  // Runs the full wrapped computation now and starts a fresh one. Exclusive:
  // waits for every in-flight AddReserved to finish first.
  void FlushFull() {
    std::unique_lock<std::shared_mutex> lock(flush_mutex_);
    // Another thread may have flushed while this one waited for the lock.
    if (reserved_.load(std::memory_order_relaxed) < entries_.capacity()) {
      return;
    }
    const size_t flushed = entries_.size();
    wrapped_computation_->ComputeBlocking();
    // Held back, not published: publishing now would make a later duplicate
    // of one of these positions a cache hit, which it is not in an unsplit
    // batch, and the search would diverge.
    for (auto& entry : entries_) flushed_.push_back(std::move(entry));
    flushed_count_.fetch_add(flushed, std::memory_order_relaxed);
    entries_.clear();
    wrapped_computation_ = memcache_->wrapped_backend_->CreateComputation();
    reserved_.store(0, std::memory_order_relaxed);
    // Rare in the fleet, so every occurrence up to 16 is logged and then
    // every power of two: enough to count them from a datagen log without
    // flooding a test that forces one per batch.
    static std::atomic<uint64_t> total_flushes{0};
    const uint64_t n = total_flushes.fetch_add(1) + 1;
    if (n <= 16 || (n & (n - 1)) == 0) {
      CERR << "MemCache: a computation reached the backend's maximum batch ("
           << entries_.capacity() << "); ran " << flushed
           << " positions early and continued in a new batch (occurrence "
           << n << ").";
    }
  }

  std::unique_ptr<BackendComputation> wrapped_computation_;
  MemCache* memcache_;
  AtomicVector<Entry> entries_;
  // Entries of batches already run by FlushFull(), awaiting ComputeBlocking().
  std::vector<Entry> flushed_;
  std::atomic<size_t> flushed_count_{0};
  std::atomic<size_t> reserved_{0};
  mutable std::shared_mutex flush_mutex_;
};

std::unique_ptr<BackendComputation> MemCache::CreateComputation() {
  return std::make_unique<MemCacheComputation>(
      wrapped_backend_->CreateComputation(), this);
}
std::optional<EvalResult> MemCache::GetCachedEvaluation(
    const EvalPosition& pos) {
  const uint64_t hash = ComputeEvalPositionHash(pos);
  HashKeyedCacheLock<CachedValue> lock(&cache_, hash);
  if (!lock.holds_value() ||
      (!pos.legal_moves.empty() &&
       !(lock->p && lock->num_moves == pos.legal_moves.size()))) {
    return std::nullopt;
  }
  EvalResult result;
  result.d = lock->d;
  result.q = lock->q;
  result.m = lock->m;
  if (lock->p) {
    result.p.reserve(pos.legal_moves.size());
    std::copy(lock->p.get(), lock->p.get() + pos.legal_moves.size(),
              std::back_inserter(result.p));
  }
  if (lock->childq && !pos.legal_moves.empty()) {
    const size_t num_moves = pos.legal_moves.size();
    result.cq.assign(lock->childq.get(), lock->childq.get() + num_moves);
    result.cs.assign(lock->childq.get() + num_moves,
                     lock->childq.get() + 2 * num_moves);
  }
  return result;
}

}  // namespace

std::unique_ptr<CachingBackend> CreateMemCache(std::unique_ptr<Backend> wrapped,
                                               const OptionsDict& options) {
  return std::make_unique<MemCache>(std::move(wrapped), options);
}

}  // namespace lczero
