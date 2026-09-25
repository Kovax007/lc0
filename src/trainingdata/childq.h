/*
  Child-Q packing for the V7 training record's reserved[4..7] window.

  WHY THIS EXISTS
  ---------------
  The training data records a policy target (visit-derived) and a single root
  value, but no per-move value. Measured on the datagen regime dumps, the
  visit order and the per-move Q order agree only at Kendall tau 0.589 -- they
  are different objects, and the per-move one is the de-biased one (visit mass
  is prior-contaminated). reserved[4..7] is the only window in the V7 record
  that no other stage writes, so per-move Q goes there.

  THE 16-BYTE BUDGET AND WHY THE LAYOUT LOOKS LIKE THIS
  ----------------------------------------------------
  reserved[4..7] is declared `float`, but we store a packed byte payload. A
  packed payload reinterpreted as float can be NaN or Inf, which would poison
  any tool that takes statistics over reserved[] -- a silent-corruption class
  this project has been bitten by before. An IEEE-754 float32 is NaN/Inf iff
  its exponent is all ones, which requires BOTH bit 7 of byte 2 set AND bits
  0..6 of byte 3 set. So it is sufficient to keep bit 7 clear in byte 2 of
  every float, i.e. in payload bytes 2, 6, 10 and 14.

  The bit writer below therefore gives those four bytes a capacity of 7 bits
  and the other twelve a capacity of 8, for 12*8 + 4*7 = 124 usable bits.
  Every value packed through it is NaN-safe and Inf-safe BY CONSTRUCTION, for
  any field layout -- there is no runtime check to forget.

  Budget: 5 header bits + K * (11 + kQBits) <= 124.
  Move indices are 0..1857, which fits 11 bits (2047), with 0x7FF as the
  "empty slot" sentinel -- so the child count is free and needs no field.

    K=6, kQBits=8 -> 119 bits.  <-- shipped: 6 children at full 8-bit Q
    K=7, kQBits=6 -> 124 bits.      alternative: 7 children, coarser Q

  8-bit Q over [-1,1] is a step of 0.0078. Measured against a 100k-node
  reference, a stored child Q carries an intrinsic error of ~0.03, so 8 bits
  is already finer than the signal; K=7/kQBits=6 (step 0.0317) trades a 5%
  noise increase for +6pp of per-position Q-range coverage. Flip kMaxChildren
  and kQBits together to switch.

  SIGN FRAME: child Q is stored VERBATIM in the root side-to-move frame, the
  same frame as best_q/played_q. Verified empirically on the regime dumps:
  the engine's own bestmove child has Q ~ +frame_best_q in 1192 of 1195
  positions. Do not negate.
*/

#pragma once

#include <cstdint>
#include <cstring>

namespace lczero {
namespace childq {

// ---- layout parameters (the only things to change to re-shape the record) --
inline constexpr int kMaxChildren = 6;
inline constexpr int kQBits = 8;
// ---------------------------------------------------------------------------

inline constexpr int kIdxBits = 11;
inline constexpr uint16_t kNoMove = 0x7FF;  // 11-bit "empty slot" sentinel
inline constexpr int kHeaderBits = 5;       // 3 bits source + 2 bits version
inline constexpr int kLayoutVersion = 1;
inline constexpr int kUsableBits = 12 * 8 + 4 * 7;  // 124
static_assert(kHeaderBits + kMaxChildren * (kIdxBits + kQBits) <= kUsableBits,
              "child-Q layout does not fit the NaN-safe bit budget");

// Who computed these values. Stored so a consumer never trains on a mixture
// it cannot tell apart.
enum Source : uint8_t {
  kSourceNone = 0,          // no data (count is 0)
  kSourceDatagenSearch = 1, // the selfplay search's own per-edge Q
  kSourceRescoreTB = 2,     // exact tablebase values filled at rescore time
  kSourceConverted = 3,     // carried over by a format converter
};

struct Child {
  uint16_t nn_idx;  // index into probabilities[1858], same space, same transform
  float q;          // root side-to-move frame, [-1, 1]
};

namespace detail {

// Capacity of payload byte i: 7 bits for the four bytes whose bit 7 must stay
// clear to make a NaN exponent impossible, 8 for the rest.
inline constexpr int ByteCapacity(int i) {
  return (i == 2 || i == 6 || i == 10 || i == 14) ? 7 : 8;
}

class BitWriter {
 public:
  explicit BitWriter(uint8_t* buf) : buf_(buf) { std::memset(buf_, 0, 16); }
  // Writes `nbits` low bits of `value`, least-significant first.
  void Write(uint32_t value, int nbits) {
    for (int b = 0; b < nbits; ++b) {
      if ((value >> b) & 1u) buf_[byte_] |= static_cast<uint8_t>(1u << bit_);
      Advance();
    }
  }

 private:
  void Advance() {
    if (++bit_ >= ByteCapacity(byte_)) { bit_ = 0; ++byte_; }
  }
  uint8_t* buf_;
  int byte_ = 0;
  int bit_ = 0;
};

class BitReader {
 public:
  explicit BitReader(const uint8_t* buf) : buf_(buf) {}
  uint32_t Read(int nbits) {
    uint32_t v = 0;
    for (int b = 0; b < nbits; ++b) {
      if (buf_[byte_] & (1u << bit_)) v |= (1u << b);
      Advance();
    }
    return v;
  }

 private:
  void Advance() {
    if (++bit_ >= ByteCapacity(byte_)) { bit_ = 0; ++byte_; }
  }
  const uint8_t* buf_;
  int byte_ = 0;
  int bit_ = 0;
};

inline constexpr uint32_t kQLevels = (1u << kQBits) - 1;

inline uint32_t QuantiseQ(float q) {
  if (!(q >= -1.0f)) q = -1.0f;   // also catches NaN
  if (q > 1.0f) q = 1.0f;
  const float scaled = (q + 1.0f) * 0.5f * static_cast<float>(kQLevels);
  int v = static_cast<int>(scaled + 0.5f);
  if (v < 0) v = 0;
  if (v > static_cast<int>(kQLevels)) v = static_cast<int>(kQLevels);
  return static_cast<uint32_t>(v);
}

inline float DequantiseQ(uint32_t v) {
  return static_cast<float>(v) * 2.0f / static_cast<float>(kQLevels) - 1.0f;
}

}  // namespace detail

// Packs up to kMaxChildren children (caller supplies them best-first) into the
// 16 bytes starting at `reserved4`, which must be &frame.reserved[4].
// Returns the number actually stored.
inline int Encode(const Child* children, int count, Source source,
                  float* reserved4) {
  uint8_t buf[16];
  detail::BitWriter w(buf);
  if (count > kMaxChildren) count = kMaxChildren;
  if (count <= 0) source = kSourceNone;
  w.Write(static_cast<uint32_t>(source), 3);
  w.Write(static_cast<uint32_t>(kLayoutVersion), 2);
  for (int i = 0; i < kMaxChildren; ++i) {
    if (i < count && children[i].nn_idx < kNoMove) {
      w.Write(children[i].nn_idx, kIdxBits);
      w.Write(detail::QuantiseQ(children[i].q), kQBits);
    } else {
      w.Write(kNoMove, kIdxBits);
      w.Write(0, kQBits);
    }
  }
  std::memcpy(reserved4, buf, 16);
  return count;
}

// Unpacks. `out` must have room for kMaxChildren. Returns the number found.
inline int Decode(const float* reserved4, Source* source, Child* out) {
  uint8_t buf[16];
  std::memcpy(buf, reserved4, 16);
  detail::BitReader r(buf);
  const Source src = static_cast<Source>(r.Read(3));
  const uint32_t version = r.Read(2);
  if (source) *source = src;
  if (src == kSourceNone || version != kLayoutVersion) return 0;
  int n = 0;
  for (int i = 0; i < kMaxChildren; ++i) {
    const uint32_t idx = r.Read(kIdxBits);
    const uint32_t q = r.Read(kQBits);
    if (idx == kNoMove) continue;
    out[n].nn_idx = static_cast<uint16_t>(idx);
    out[n].q = detail::DequantiseQ(q);
    ++n;
  }
  return n;
}

// True iff the 16 bytes are all zero, i.e. no writer has claimed the window.
inline bool IsEmpty(const float* reserved4) {
  uint8_t buf[16];
  std::memcpy(buf, reserved4, 16);
  for (int i = 0; i < 16; ++i) if (buf[i] != 0) return false;
  return true;
}

}  // namespace childq
}  // namespace lczero
