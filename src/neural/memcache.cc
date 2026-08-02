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

#include "neural/shared_params.h"
#include "utils/atomic_vector.h"
#include "utils/cache.h"
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
  // Only allocated when a caller asked for the extra policy heads and the
  // backend produced them, so an entry cached by a plain query cannot serve a
  // blending query.
  std::unique_ptr<float[]> p_optimistic;
  std::unique_ptr<float[]> p_soft;
};

// Whether the cached entry can serve a request that wants the given extra
// heads.
bool CachedValueSatisfies(const CachedValue& cv, const EvalResultPtr& ptr) {
  if (!ptr.p_optimistic.empty() && !cv.p_optimistic) return false;
  if (!ptr.p_soft.empty() && !cv.p_soft) return false;
  return true;
}

void CachedValueToEvalResult(const CachedValue& cv, const EvalResultPtr& ptr) {
  if (ptr.d) *ptr.d = cv.d;
  if (ptr.q) *ptr.q = cv.q;
  if (ptr.m) *ptr.m = cv.m;
  std::copy(cv.p.get(), cv.p.get() + ptr.p.size(), ptr.p.begin());
  if (cv.p_optimistic && !ptr.p_optimistic.empty()) {
    std::copy(cv.p_optimistic.get(),
              cv.p_optimistic.get() + ptr.p_optimistic.size(),
              ptr.p_optimistic.begin());
  }
  if (cv.p_soft && !ptr.p_soft.empty()) {
    std::copy(cv.p_soft.get(), cv.p_soft.get() + ptr.p_soft.size(),
              ptr.p_soft.begin());
  }
}

class MemCache : public CachingBackend {
 public:
  MemCache(std::unique_ptr<Backend> wrapped, const OptionsDict& options)
      : wrapped_backend_(std::move(wrapped)),
        cache_(options.Get<int>(SharedBackendParams::kNNCacheSizeId)),
        max_batch_size_(wrapped_backend_->GetAttributes().maximum_batch_size) {}

  BackendAttributes GetAttributes() const override {
    return wrapped_backend_->GetAttributes();
  }
  std::unique_ptr<BackendComputation> CreateComputation() override;
  std::optional<EvalResult> GetCachedEvaluation(const EvalPosition&) override;

  ExtraPolicyHeads SupportedExtraPolicyHeads() const override {
    return wrapped_backend_->SupportedExtraPolicyHeads();
  }

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
  friend class MemCacheComputation;
};

class MemCacheComputation : public BackendComputation {
 public:
  MemCacheComputation(std::unique_ptr<BackendComputation> wrapped_computation,
                      MemCache* memcache)
      : wrapped_computation_(std::move(wrapped_computation)),
        memcache_(memcache),
        entries_(memcache->max_batch_size_) {}

 private:
  size_t UsedBatchSize() const override {
    return wrapped_computation_->UsedBatchSize();
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
           (lock->p && lock->num_moves == pos.legal_moves.size())) &&
          CachedValueSatisfies(**lock, result)) {
        CachedValueToEvalResult(**lock, result);
        return AddInputResult::FETCHED_IMMEDIATELY;
      }
    }
    size_t entry_idx = entries_.emplace_back(
        Entry{hash, std::make_unique<CachedValue>(), result});
    auto& value = entries_[entry_idx].value;
    const size_t num_moves = pos.legal_moves.size();
    value->p.reset(pos.legal_moves.empty() ? nullptr : new float[num_moves]);
    value->num_moves = num_moves;
    const bool want_optimistic = value->p && !result.p_optimistic.empty();
    const bool want_soft = value->p && !result.p_soft.empty();
    if (want_optimistic) value->p_optimistic.reset(new float[num_moves]);
    if (want_soft) value->p_soft.reset(new float[num_moves]);
    return wrapped_computation_->AddInput(
        pos,
        EvalResultPtr{
            .q = &value->q,
            .d = &value->d,
            .m = &value->m,
            .p = value->p ? std::span<float>{value->p.get(), num_moves}
                          : std::span<float>{},
            .p_optimistic =
                want_optimistic
                    ? std::span<float>{value->p_optimistic.get(), num_moves}
                    : std::span<float>{},
            .p_soft = want_soft
                          ? std::span<float>{value->p_soft.get(), num_moves}
                          : std::span<float>{}});
  }

  virtual void ComputeBlocking() override {
    if (wrapped_computation_->UsedBatchSize() == 0) return;
    wrapped_computation_->ComputeBlocking();
    for (auto& entry : entries_) {
      CachedValueToEvalResult(*entry.value, entry.result_ptr);
      memcache_->cache_.Insert(entry.key, std::move(entry.value));
    }
  }

  struct Entry {
    uint64_t key;
    std::unique_ptr<CachedValue> value;
    EvalResultPtr result_ptr;
  };

  std::unique_ptr<BackendComputation> wrapped_computation_;
  MemCache* memcache_;
  AtomicVector<Entry> entries_;
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
  return result;
}

}  // namespace

std::unique_ptr<CachingBackend> CreateMemCache(std::unique_ptr<Backend> wrapped,
                                               const OptionsDict& options) {
  return std::make_unique<MemCache>(std::move(wrapped), options);
}

}  // namespace lczero
