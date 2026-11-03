/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors

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

#include <absl/cleanup/cleanup.h>
#include <absl/numeric/int128.h>
#include <hwy/highway.h>

#include <array>
#include <bit>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <execution>
#include <memory>
#include <mutex>
#include <numeric>
#include <version>

#include "utils/bititer.h"
#include "utils/mutex.h"

namespace hn = hwy::HWY_NAMESPACE;

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <sys/mman.h>
#endif

namespace lczero {

namespace dag_classic {
template <typename T>
class IntrusiveSharedPtr;
}  // namespace dag_classic

template <typename T>
struct IsManagedPointerType {
  static constexpr bool value = false;
};

template <typename U>
struct IsManagedPointerType<dag_classic::IntrusiveSharedPtr<U>> {
  static constexpr bool value = true;
};
template <typename U>
struct IsManagedPointerType<std::unique_ptr<U>> {
  static constexpr bool value = true;
};

template <typename T, bool is_managed_pointer_type>
struct HashValueToPointer {
  using type = T*;
};

template <typename T>
struct HashValueToPointer<T, true> {
  using type = typename T::element_type*;
};

// A hash-keyed cache. Thread-safe. Takes ownership of all values, which are
// deleted upon eviction; Implements per bucket LRU eviction. The cache uses
// shards to reduce lock contention.
// Does not support delete.
// Does not support replace!
// LRU eviction but tracked per bucket only.
template <class V>
class HashKeyedCache {
 public:
  static constexpr bool kIsManagedPointerType = IsManagedPointerType<V>::value;
  using element_type = V;
  using pointer = typename HashValueToPointer<V, kIsManagedPointerType>::type;

 private:
  static constexpr size_t kElementsInBucket = 32;
  static constexpr size_t kCacheLineSize = 64;
  static constexpr size_t kShardsPerThread = 16;
  // We need at least enough capacity make it practically impossible to fill a
  // bucket in a single computation.
  static constexpr size_t kMinimumSafeCapacity = 10000;

  using Entry = std::pair<uint64_t, element_type>;

  struct Bucket {
    Bucket() {
      for (size_t i = 0; i < kElementsInBucket; ++i) {
        entries[i].first = i + kElementsInBucket;
        low_byte[i] = i;
        least_recent_access[i] = kElementsInBucket - 1 - i;
      }
    }
    static constexpr hn::CappedTag<uint8_t, kElementsInBucket> byte_vec_tag{};
    alignas(kCacheLineSize) std::array<uint8_t, kElementsInBucket> low_byte;
    std::array<uint8_t, kElementsInBucket> least_recent_access;
    std::array<Entry, kElementsInBucket> entries;
  };

  struct Shard {
    alignas(kCacheLineSize) mutable SpinMutex mutex;
  };

  template <typename T>
  struct AlignedDeleter {
    size_t capacity;
    void operator()(T* ptr) const {
      for (size_t i = 0; i < capacity; ++i) {
        std::destroy_at(ptr + i);
      }
#if defined(_MSC_VER)
      _aligned_free(ptr);
#else
      std::free(ptr);
#endif
    }
  };

  using HashType = std::unique_ptr<Bucket[], AlignedDeleter<Bucket>>;

  uint8_t GetKeyLowByte(uint64_t key) { return (key & 0xff); }

  uint64_t MulHigh(uint64_t x, size_t y) const {
#if defined(_MSC_VER)
    return __umulh(x, y);
#else
    return static_cast<uint64_t>((static_cast<__uint128_t>(x) * y) >> 64);
#endif
  }

  Bucket* GetBucketAligned() {
    return std::assume_aligned<kCacheLineSize>(hash_.get());
  }

  const Bucket* GetBucketAligned() const {
    return const_cast<HashKeyedCache*>(this)->GetBucketAligned();
  }

  Bucket& GetBucket(uint64_t key) {
    size_t idx = MulHigh(key, capacity_);
    return GetBucketAligned()[idx];
  }

  const Shard* GetShards() const {
    const Bucket* last = GetBucketAligned() + capacity_;
    const Shard* first = reinterpret_cast<const Shard*>(last);
    return std::assume_aligned<kCacheLineSize>(first);
  }

  SpinMutex& GetShardMutex(const Bucket& bucket) const {
    size_t bucket_index = &bucket - GetBucketAligned();
    size_t idx = MulHigh(bucket_index, shard_mul_);
    return GetShards()[idx].mutex;
  }

  void UpdateLastAccess(Bucket& bucket, uint8_t current) {
    std::transform(
        std::execution::unseq, bucket.least_recent_access.begin(),
        bucket.least_recent_access.end(), bucket.least_recent_access.begin(),
        [current](uint8_t access) -> uint8_t {
          return access < current ? access + 1 : access == current ? 0 : access;
        });
    assert(
        std::all_of(std::execution::unseq, bucket.least_recent_access.begin(),
                    bucket.least_recent_access.end(),
                    [](uint8_t access) { return access < kElementsInBucket; }));
  }

  HashType AllocateHash(size_t capacity, size_t shards) const {
    if (capacity == 0) [[unlikely]] return nullptr;
    assert(capacity >= kMinimumSafeCapacity);
    size_t bytes = sizeof(Bucket) * capacity + sizeof(Shard) * shards;
    assert(bytes % kCacheLineSize == 0);
#if defined(_MSC_VER)
    void* ptr = _aligned_malloc(bytes, kCacheLineSize);
#else
    // Transparent huge pages for the big caches (a random bucket is otherwise
    // a TLB miss on 4 KB pages). On by default; LC0_HASH_HUGEPAGES=0 turns it off.
    static const bool huge = [] {
      const char* value = std::getenv("LC0_HASH_HUGEPAGES");
      return value == nullptr || value[0] != '0';
    }();
    constexpr size_t kHugePage = size_t{2} << 20;
    const size_t huge_bytes = (bytes + kHugePage - 1) / kHugePage * kHugePage;
    void* ptr = huge && bytes >= kHugePage
                    ? std::aligned_alloc(kHugePage, huge_bytes)
                    : std::aligned_alloc(kCacheLineSize, bytes);
    if (huge && ptr && bytes >= kHugePage) {
      madvise(ptr, huge_bytes, MADV_HUGEPAGE);
    }
#endif
    bool clean_exit = false;
    absl::Cleanup cleanup([ptr, &clean_exit]() {
      if (clean_exit) return;
#if defined(_MSC_VER)
      _aligned_free(ptr);
#else
      std::free(ptr);
#endif
    });
    for (size_t i = 0; i < capacity; ++i) {
      std::construct_at(reinterpret_cast<Bucket*>(ptr) + i);
    }
    for (size_t i = 0; i < shards; ++i) {
      std::construct_at(
          reinterpret_cast<Shard*>(reinterpret_cast<Bucket*>(ptr) + capacity) +
          i);
    }
    clean_exit = true;
    return HashType(reinterpret_cast<Bucket*>(ptr),
                    AlignedDeleter<Bucket>{capacity});
  }

  Entry* FindLocked(Bucket& bucket, uint64_t key) {
    uint8_t key_low_byte = GetKeyLowByte(key);
    const size_t lanes = hn::Lanes(bucket.byte_vec_tag);
    uint32_t mask = 0;
    static_assert(std::numeric_limits<decltype(mask)>::digits >=
                  kElementsInBucket);
    for (size_t chunk = 0; chunk < kElementsInBucket; chunk += lanes) {
      auto low_byte_vec =
          hn::Load(bucket.byte_vec_tag, bucket.low_byte.data() + chunk);
      auto needle = hn::Set(bucket.byte_vec_tag, key_low_byte);
      auto cmp_result = low_byte_vec == needle;
      mask |= hn::BitsFromMask(bucket.byte_vec_tag, cmp_result) << chunk;
    }

    for (auto bit : IterateBits(mask)) {
      if (bucket.entries[bit].first == key) {
        UpdateLastAccess(bucket, bucket.least_recent_access[bit]);
        return &bucket.entries[bit];
      }
    }
    return nullptr;
  }

  size_t CalculateCapacity(size_t capacity) const {
    return capacity == 0 ? 0
                         : std::max(kMinimumSafeCapacity,
                                    (capacity - 1) / sizeof(Bucket) + 1);
  }

  size_t CalculateShards(size_t threads) const {
    return std::max<size_t>(1, threads * kShardsPerThread);
  }

  size_t CalculateShardMul(size_t shards, size_t capacity) const {
    if (shards == 1) return 0;
    if (capacity == std::numeric_limits<size_t>::max()) return shards;
    // Calculate shards << 64 / capacity. This is used to calculate the shard
    // index from the bucket index.
    if (sizeof(size_t) == 8) {
      absl::uint128 shards_128 = absl::MakeUint128(shards, 0);
      absl::uint128 result = shards_128 / (capacity + 1);
      return static_cast<size_t>(result);
    } else {
      return (static_cast<uint64_t>(shards) << 32) / (capacity + 1);
    }
  }

  pointer ValueToPointer(element_type& value) {
    if constexpr (kIsManagedPointerType) {
      return value.get();
    } else {
      return &value;
    }
  }

 public:
  HashKeyedCache(size_t capacity, size_t threads)
      : capacity_(CalculateCapacity(capacity)),
        shards_(CalculateShards(threads)),
        shard_mul_(CalculateShardMul(shards_, capacity_)),
        hash_(AllocateHash(capacity_, CalculateShards(threads))) {}

  ~HashKeyedCache() {}

  // Inserts the element under key @key with value @val. Unless the key is
  // already in the cache. If the key is already in the cache, the new value is
  // silently ignored and the old value is kept. If the key is not in the cache,
  // the new value is moved to cache.
  template <typename... Args>
  std::pair<pointer, bool> Emplace(uint64_t key, Args&&... args) {
    if (capacity_ == 0) [[unlikely]] return {nullptr, false};
    auto& bucket = GetBucket(key);
    SpinMutex::Lock lock(GetShardMutex(bucket));
    uint8_t key_low_byte = GetKeyLowByte(key);

    // Check if key already exists in the bucket.
    Entry* entry = FindLocked(bucket, key);
    if (entry) {
      // Already exists.
      return {ValueToPointer(entry->second), false};
    }

    // If bucket is full, evict the least recently used entry.
    // Then insert the new entry to the bucket.
    uint32_t mask = 0;
    static_assert(std::numeric_limits<decltype(mask)>::digits >=
                  kElementsInBucket);
    for (size_t chunk = 0; chunk < kElementsInBucket;
         chunk += hn::Lanes(bucket.byte_vec_tag)) {
      auto lru_vec = hn::Load(bucket.byte_vec_tag,
                              bucket.least_recent_access.data() + chunk);
      auto needle = hn::Set(bucket.byte_vec_tag, kElementsInBucket - 1);
      auto cmp_result = hn::Eq(lru_vec, needle);
      mask |= hn::BitsFromMask(bucket.byte_vec_tag, cmp_result) << chunk;
    }
    assert(std::popcount(mask) == 1);
    size_t idx = std::countr_zero(mask);
    // Evict the entry.
    auto evicted = std::move(bucket.entries[idx].second);
    bucket.entries[idx].first = key;
    std::construct_at(&bucket.entries[idx].second, std::forward<Args>(args)...);
    bucket.low_byte[idx] = key_low_byte;
    UpdateLastAccess(bucket, kElementsInBucket - 1);

    // Evicted node is freed outside the lock to avoid random delays from memory
    // management blocking other threads.
    lock.unlock();
    return {ValueToPointer(bucket.entries[idx].second), true};
  }

  // Checks whether a key exists. Doesn't pin. Of course the next moment the
  // key may be evicted.
  bool ContainsKey(uint64_t key) {
    if (capacity_ == 0) [[unlikely]] return false;

    Bucket& bucket = GetBucket(key);
    SpinMutex::Lock lock(GetShardMutex(bucket));
    Entry* entry = FindLocked(bucket, key);
    return entry != nullptr;
  }

  // Looks up and pins the element by key. Returns nullptr if not found.
  pointer LookupAndPin(uint64_t key) {
    if (capacity_ == 0) [[unlikely]] return nullptr;

    Bucket& bucket = GetBucket(key);
    SpinMutex::Lock lock(GetShardMutex(bucket));
    Entry* entry = FindLocked(bucket, key);
    if (entry) {
      return ValueToPointer(entry->second);
    }
    return nullptr;
  }

  // Sets the capacity of the cache. If new capacity is less than current size
  // of the cache, oldest entries are evicted. In any case the hashtable is
  // rehashed.
  // There must be no concurrent access to the cache.
  void SetCapacity(size_t capacity, size_t threads) {
    if (CalculateCapacity(capacity) == capacity_ &&
        CalculateShards(threads) <= shards_) {
      shards_ = CalculateShards(threads);
      shard_mul_ = CalculateShardMul(shards_, capacity_);
      return;
    }

    auto old_hash =
        AllocateHash(CalculateCapacity(capacity),
                     std::max<size_t>(1, threads * kShardsPerThread));
    size_t old_capacity = CalculateCapacity(capacity);
    std::swap(hash_, old_hash);
    std::swap(capacity_, old_capacity);
    shards_ = CalculateShards(threads);
    shard_mul_ = CalculateShardMul(shards_, capacity_);

    // Rehash everything starting from the oldest entry. This will evict the
    // oldest entries first if any bucket becomes full.
    for (size_t age = kElementsInBucket; age-- > 0;) {
      for (size_t i = 0; i < old_capacity; ++i) {
        Bucket& bucket = old_hash[i];
        uint32_t mask = 0;
        static_assert(std::numeric_limits<decltype(mask)>::digits >=
                      kElementsInBucket);
        const size_t lanes = hn::Lanes(bucket.byte_vec_tag);
        for (size_t chunk = 0; chunk < kElementsInBucket; chunk += lanes) {
          auto access_vec = hn::Load(bucket.byte_vec_tag,
                                     bucket.least_recent_access.data() + chunk);
          auto needle = hn::Set(bucket.byte_vec_tag, age);
          auto cmp_result = hn::Eq(access_vec, needle);
          mask |= hn::BitsFromMask(bucket.byte_vec_tag, cmp_result) << chunk;
        }
        assert(std::popcount(mask) == 1);
        size_t idx = std::countr_zero(mask);
        Entry& entry = bucket.entries[idx];
        if (!entry.second) continue;
        Emplace(entry.first, std::move(entry.second));
      }
    }
  }

  // Clears the cache.
  // There must be no concurrent access to the cache.
  void Clear() {
    for (size_t i = 0; i < capacity_; ++i) {
      Bucket& bucket = GetBucketAligned()[i];
      for (size_t j = 0; j < kElementsInBucket; ++j) {
        bucket.entries[j].second.reset();
        bucket.entries[j].first = j + kElementsInBucket;
        bucket.low_byte[j] = j;
        bucket.least_recent_access[j] = kElementsInBucket - 1 - j;
      }
    }
  }

  size_t GetCapacity() const { return capacity_; }
  static constexpr size_t GetItemStructSize() { return sizeof(Bucket); }
  static constexpr size_t GetItemsInStructure() { return kElementsInBucket; }

  float GetLoadFactor() const {
    const size_t limit = 1000;
    size_t count = 0;
    size_t total = 0;
    std::unique_lock<SpinMutex> lock;
    SpinMutex* current_mutex = nullptr;
    for (size_t i = 0; i < capacity_ && total < limit; ++i) {
      const Bucket& bucket = GetBucketAligned()[i];
      SpinMutex& mutex = GetShardMutex(bucket);
      if (current_mutex != &mutex) {
        current_mutex = &mutex;
        if (lock.owns_lock()) {
          lock.unlock();
        }
        lock = std::unique_lock<SpinMutex>(mutex);
      }

      for (auto& entry : bucket.entries) {
        if (entry.second) {
          ++count;
        }
        if (++total >= limit) {
          break;
        }
      }
    }
    return static_cast<float>(count) / total;
  }

 private:
  // These are static when concurrent access happens.
  size_t capacity_;
  size_t shards_;
  size_t shard_mul_;
  HashType hash_;
};

}  // namespace lczero
