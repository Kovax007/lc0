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
*/

// The child-Q head through demux: demux over `random` children with the
// synthetic child-Q head must return exactly what one `random` backend returns,
// and a backend registered with REGISTER_BACKEND (the route lc0ex-cuda takes)
// must be creatable as a demux child.

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "chess/board.h"
#include "chess/position.h"
#include "gtest/gtest.h"
#include "neural/backend.h"
#include "neural/factory.h"
#include "neural/register.h"
#include "neural/shared_params.h"
#include "utils/optionsdict.h"

namespace lczero {
namespace {

// A position with its history and legal moves, kept alive while EvalPosition
// spans point into it.
struct Sample {
  PositionHistory history;
  MoveList legal_moves;
};

// Deterministic positions: four starting FENs, each followed by a pseudo-random
// walk of 0 to 22 plies, so that samples differ in history as well as board.
std::vector<Sample> MakeSamples(size_t count) {
  const char* kFens[] = {
      ChessBoard::kStartposFen,
      "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
      "8/5pk1/6p1/8/3Q4/6P1/5PKP/q7 b - - 3 40",
  };
  std::vector<Sample> samples(count);
  uint64_t rng = 0x9e3779b97f4a7c15ULL;
  for (size_t i = 0; i < count; ++i) {
    ChessBoard board;
    int rule50 = 0;
    int game_ply = 0;
    board.SetFromFen(kFens[i % 4], &rule50, &game_ply);
    PositionHistory& history = samples[i].history;
    history.Reset(board, rule50, game_ply);
    for (size_t ply = 0; ply < i / 4 % 23; ++ply) {
      const MoveList moves = history.Last().GetBoard().GenerateLegalMoves();
      if (moves.empty()) break;
      rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
      history.Append(moves[(rng >> 33) % moves.size()]);
    }
    samples[i].legal_moves = history.Last().GetBoard().GenerateLegalMoves();
    while (samples[i].legal_moves.empty()) {
      history.Pop();
      samples[i].legal_moves = history.Last().GetBoard().GenerateLegalMoves();
    }
  }
  return samples;
}

std::vector<EvalPosition> AsEvalPositions(const std::vector<Sample>& samples) {
  std::vector<EvalPosition> positions;
  positions.reserve(samples.size());
  for (const Sample& sample : samples) {
    positions.push_back({sample.history.GetPositions(), sample.legal_moves});
  }
  return positions;
}

std::unique_ptr<Backend> CreateBackend(const std::string& backend,
                                       const std::string& backend_opts,
                                       const std::string& weights = "") {
  OptionsDict options;
  options.Set<std::string>(SharedBackendParams::kBackendId, backend);
  options.Set<std::string>(SharedBackendParams::kBackendOptionsId,
                           backend_opts);
  options.Set<std::string>(SharedBackendParams::kWeightsId, weights);
  options.Set<float>(SharedBackendParams::kPolicySoftmaxTemp, 1.359f);
  options.Set<std::string>(SharedBackendParams::kHistoryFill, "fen_only");
  return BackendManager::Get()->CreateFromParams(options);
}

// Exact equality: demux must hand back the child's own numbers, not a copy
// that went through any arithmetic.
void ExpectSameResult(const EvalResult& expected, const EvalResult& actual) {
  EXPECT_EQ(expected.q, actual.q);
  EXPECT_EQ(expected.d, actual.d);
  EXPECT_EQ(expected.m, actual.m);
  EXPECT_EQ(expected.p, actual.p);
  EXPECT_EQ(expected.cq, actual.cq);
  EXPECT_EQ(expected.cs, actual.cs);
}

bool SameResult(const EvalResult& a, const EvalResult& b) {
  return a.q == b.q && a.d == b.d && a.m == b.m && a.p == b.p && a.cq == b.cq &&
         a.cs == b.cs;
}

std::string RandomChildren(int children, const std::string& child_opts = "") {
  std::string result;
  for (int i = 0; i < children; ++i) {
    result += std::string(",") + static_cast<char>('a' + i) +
              "(backend=random" + child_opts + ")";
  }
  return result;
}

constexpr size_t kBatchSizes[] = {1, 2, 7, 64, 257};
constexpr int kBatchSteps[] = {1, 3, 64};

TEST(DemuxChildQ, RandomChildrenMatchSingleBackend) {
  auto reference = CreateBackend("random", "childq=true");
  ASSERT_TRUE(reference->GetAttributes().has_childq);
  for (int children : {2, 3}) {
    for (int batch_step : kBatchSteps) {
      auto demux =
          CreateBackend("demux", "batch_step=" + std::to_string(batch_step) +
                                     ",childq=true" + RandomChildren(children));
      ASSERT_TRUE(demux->GetAttributes().has_childq);
      for (size_t batch_size : kBatchSizes) {
        SCOPED_TRACE("children=" + std::to_string(children) +
                     " batch_step=" + std::to_string(batch_step) +
                     " batch_size=" + std::to_string(batch_size));
        ASSERT_LE(batch_size, static_cast<size_t>(
                                  demux->GetAttributes().maximum_batch_size));
        const auto samples = MakeSamples(batch_size);
        const auto positions = AsEvalPositions(samples);
        const auto expected = reference->EvaluateBatch(positions);
        const auto actual = demux->EvaluateBatch(positions);
        ASSERT_EQ(actual.size(), batch_size);
        for (size_t i = 0; i < batch_size; ++i) {
          SCOPED_TRACE("sample=" + std::to_string(i));
          const size_t moves = samples[i].legal_moves.size();
          ASSERT_EQ(actual[i].cq.size(), moves);
          ASSERT_EQ(actual[i].cs.size(), moves);
          // The synthetic head, not the "no head" defaults (0 and -1).
          for (size_t k = 0; k < moves; ++k) {
            EXPECT_GE(actual[i].cs[k], 0.05f);
            EXPECT_LE(actual[i].cs[k], 0.30f);
          }
          ExpectSameResult(expected[i], actual[i]);
        }
      }
    }
  }
}

// Children with different seeds give different numbers, so every field of a
// sample, child-Q included, must come from one and the same child.
TEST(DemuxChildQ, EverySampleComesWhollyFromOneChild) {
  constexpr int kChildren = 3;
  std::vector<std::unique_ptr<Backend>> references;
  std::string children;
  for (int i = 0; i < kChildren; ++i) {
    const std::string seed = "seed=" + std::to_string(i + 1);
    references.push_back(CreateBackend("random", "childq=true," + seed));
    children += std::string(",") + static_cast<char>('a' + i) +
                "(backend=random," + seed + ")";
  }
  for (int batch_step : kBatchSteps) {
    auto demux =
        CreateBackend("demux", "batch_step=" + std::to_string(batch_step) +
                                   ",childq=true" + children);
    for (size_t batch_size : kBatchSizes) {
      SCOPED_TRACE("batch_step=" + std::to_string(batch_step) +
                   " batch_size=" + std::to_string(batch_size));
      const auto samples = MakeSamples(batch_size);
      const auto positions = AsEvalPositions(samples);
      std::vector<std::vector<EvalResult>> expected;
      for (auto& reference : references) {
        expected.push_back(reference->EvaluateBatch(positions));
      }
      const auto actual = demux->EvaluateBatch(positions);
      std::vector<size_t> per_child(kChildren);
      for (size_t i = 0; i < batch_size; ++i) {
        SCOPED_TRACE("sample=" + std::to_string(i));
        int matches = 0;
        for (int c = 0; c < kChildren; ++c) {
          if (SameResult(expected[c][i], actual[i])) {
            ++matches;
            ++per_child[c];
          }
        }
        EXPECT_EQ(matches, 1);
      }
      const size_t groups = (batch_size + batch_step - 1) / batch_step;
      const size_t used = std::count_if(per_child.begin(), per_child.end(),
                                        [](size_t n) { return n > 0; });
      EXPECT_EQ(used, std::min<size_t>(groups, kChildren));
    }
  }
}

// Mixed children: demux reports has_childq = false, every other output still
// matches, and child-Q spans a caller passes anyway are left untouched for
// every sample, whichever child evaluated it.
TEST(DemuxChildQ, MixedChildrenReportNoChildQ) {
  auto reference = CreateBackend("random", "childq=true");
  for (int batch_step : {1, 3}) {
    for (const std::string children :
         {"childq=true,a(backend=random),b(backend=random,childq=false)",
          "a(backend=random),b(backend=random,childq=true),"
          "c(backend=random,childq=true)"}) {
      SCOPED_TRACE(children + " batch_step=" + std::to_string(batch_step));
      auto demux = CreateBackend(
          "demux", "batch_step=" + std::to_string(batch_step) + "," + children);
      EXPECT_FALSE(demux->GetAttributes().has_childq);

      for (size_t batch_size : kBatchSizes) {
        SCOPED_TRACE("batch_size=" + std::to_string(batch_size));
        const auto samples = MakeSamples(batch_size);
        const auto positions = AsEvalPositions(samples);
        auto expected = reference->EvaluateBatch(positions);
        const auto actual = demux->EvaluateBatch(positions);
        for (size_t i = 0; i < batch_size; ++i) {
          EXPECT_TRUE(actual[i].cq.empty());
          EXPECT_TRUE(actual[i].cs.empty());
          expected[i].cq.clear();
          expected[i].cs.clear();
          ExpectSameResult(expected[i], actual[i]);
        }

        constexpr float kUntouched = 42.0f;
        std::vector<EvalResult> requested(batch_size);
        auto computation = demux->CreateComputation();
        for (size_t i = 0; i < batch_size; ++i) {
          const size_t moves = samples[i].legal_moves.size();
          requested[i].p.resize(moves);
          requested[i].cq.assign(moves, kUntouched);
          requested[i].cs.assign(moves, kUntouched);
          computation->AddInput(positions[i], requested[i].AsPtr());
        }
        computation->ComputeBlocking();
        for (size_t i = 0; i < batch_size; ++i) {
          EXPECT_EQ(requested[i].q, expected[i].q);
          EXPECT_EQ(requested[i].p, expected[i].p);
          for (float value : requested[i].cq) EXPECT_EQ(value, kUntouched);
          for (float value : requested[i].cs) EXPECT_EQ(value, kUntouched);
        }
      }
    }
  }
  auto none = CreateBackend("demux", "a(backend=random),b(backend=random)");
  EXPECT_FALSE(none->GetAttributes().has_childq);
}

// A stand-in for lc0ex-cuda: registered with REGISTER_BACKEND, as lc0ex-cuda
// is, so only the Backend API knows it; reads its options the way
// Lc0exCudaBackendFactory does (a required `lc0ex`, `gpu`, then
// CheckAllOptionsRead(), so a demux-only option leaking into a child throws);
// and has a child-Q head. q is the child's gpu and cs encodes the sample and
// the move, so a test can tell who filled what.
constexpr char kRegisteredName[] = "demux-childq-registered-test";

struct RegisteredChild {
  std::string lc0ex;
  int gpu;
  std::string weights;
};

std::vector<RegisteredChild>& RegisteredChildren() {
  static std::vector<RegisteredChild> children;
  return children;
}

class RegisteredComputation : public BackendComputation {
 public:
  explicit RegisteredComputation(int gpu) : gpu_(gpu) {}
  size_t UsedBatchSize() const override { return entries_.size(); }
  AddInputResult AddInput(const EvalPosition& pos,
                          EvalResultPtr result) override {
    entries_.push_back({result, pos.pos.back().GetGamePly()});
    return ENQUEUED_FOR_EVAL;
  }
  void ComputeBlocking() override {
    for (const Entry& entry : entries_) {
      const EvalResultPtr& result = entry.result;
      *result.q = gpu_;
      *result.d = entry.ply;
      *result.m = 0;
      std::fill(result.p.begin(), result.p.end(), 1.0f / result.p.size());
      for (size_t i = 0; i < result.cq.size(); ++i) {
        result.cq[i] = gpu_;
        result.cs[i] = static_cast<float>(entry.ply * 100 + i);
      }
    }
  }

 private:
  struct Entry {
    EvalResultPtr result;
    int ply;
  };
  const int gpu_;
  std::vector<Entry> entries_;
};

class RegisteredBackend : public Backend {
 public:
  explicit RegisteredBackend(int gpu) : gpu_(gpu) {}
  BackendAttributes GetAttributes() const override {
    return {.has_mlh = true,
            .has_wdl = true,
            .runs_on_cpu = true,
            .suggested_num_search_threads = 1,
            .recommended_batch_size = 64,
            .maximum_batch_size = 256,
            .has_childq = true};
  }
  std::unique_ptr<BackendComputation> CreateComputation() override {
    return std::make_unique<RegisteredComputation>(gpu_);
  }

 private:
  const int gpu_;
};

class RegisteredBackendFactory : public BackendFactory {
 public:
  int GetPriority() const override { return -100000; }
  std::string_view GetName() const override { return kRegisteredName; }
  std::unique_ptr<Backend> Create(const OptionsDict& options) override {
    OptionsDict backend_options;
    backend_options.AddSubdictFromString(
        options.Get<std::string>(SharedBackendParams::kBackendOptionsId));
    RegisteredChild child{
        .lc0ex = backend_options.Get<std::string>("lc0ex"),
        .gpu = backend_options.GetOrDefault<int>("gpu", 0),
        .weights = options.Get<std::string>(SharedBackendParams::kWeightsId)};
    backend_options.CheckAllOptionsRead(kRegisteredName);
    RegisteredChildren().push_back(child);
    return std::make_unique<RegisteredBackend>(child.gpu);
  }
};

REGISTER_BACKEND(RegisteredBackendFactory)

TEST(DemuxChildQ, CreatesBackendApiChildren) {
  // Only the Backend API knows this backend; the old-API demux looked its
  // children up in NetworkFactory and failed with "Unknown backend".
  const auto networks = NetworkFactory::Get()->GetBackendsList();
  EXPECT_EQ(std::find(networks.begin(), networks.end(), kRegisteredName),
            networks.end());
  const auto backends = BackendManager::Get()->GetBackendNames();
  EXPECT_NE(std::find(backends.begin(), backends.end(), kRegisteredName),
            backends.end());

  for (int batch_step : {1, 2}) {
    SCOPED_TRACE("batch_step=" + std::to_string(batch_step));
    RegisteredChildren().clear();
    // The shape a maintainer types for lc0ex-cuda: shared options at the top
    // level, one subdict per GPU.
    auto demux = CreateBackend(
        "demux",
        "batch_step=" + std::to_string(batch_step) +
            ",lc0ex=/models/net.lc0ex"
            ",g0(backend=demux-childq-registered-test,gpu=0)"
            ",g1(backend=demux-childq-registered-test,gpu=1,demux_threads=2)",
        "parent-weights.pb.gz");
    ASSERT_EQ(RegisteredChildren().size(), 2u);
    for (int gpu = 0; gpu < 2; ++gpu) {
      EXPECT_EQ(RegisteredChildren()[gpu].lc0ex, "/models/net.lc0ex");
      EXPECT_EQ(RegisteredChildren()[gpu].gpu, gpu);
      EXPECT_EQ(RegisteredChildren()[gpu].weights, "parent-weights.pb.gz");
    }
    EXPECT_TRUE(demux->GetAttributes().has_childq);

    constexpr int kBatch = 9;
    std::vector<std::vector<Position>> histories;
    for (int i = 0; i < kBatch; ++i) {
      histories.push_back({Position(ChessBoard{}, 0, i)});
    }
    const MoveList moves =
        ChessBoard(ChessBoard::kStartposFen).GenerateLegalMoves();
    std::vector<EvalPosition> positions;
    for (const auto& history : histories) {
      positions.push_back({history, moves});
    }
    const auto results = demux->EvaluateBatch(positions);
    for (int i = 0; i < kBatch; ++i) {
      SCOPED_TRACE("sample=" + std::to_string(i));
      EXPECT_EQ(results[i].q, (i / batch_step) % 2);
      EXPECT_EQ(results[i].d, i);
      ASSERT_EQ(results[i].cq.size(), moves.size());
      for (size_t k = 0; k < moves.size(); ++k) {
        EXPECT_EQ(results[i].cq[k], results[i].q);
        EXPECT_EQ(results[i].cs[k], static_cast<float>(i * 100 + k));
      }
    }
  }
}

}  // namespace
}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  lczero::InitializeMagicBitboards();
  return RUN_ALL_TESTS();
}
