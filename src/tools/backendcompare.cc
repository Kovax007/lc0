/*
  Fidelity comparison of two backends over a list of real positions.

  Reports DISTRIBUTIONS (mean / p50 / p90 / p99 / max), not just means, of:
    - policy KL divergence  KL(ref || test)  over legal moves
    - top-1 agreement, and whether the reference's best move is in the test's
      top 3
    - |q| and |d| absolute error (value head)
    - |m| absolute error (moves-left head)

  Usage:
    lc0 backendcompare --weights=NET \
        --backend=lc0ex-cuda --backend-opts='lc0ex=ART,gpu=0,concurrency=1' \
        --ref-backend=cuda-fp16 --ref-backend-opts='gpu=0' \
        --fens=positions.txt --max-positions=10000 --batch-size=32
*/

#include "tools/backendcompare.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "chess/board.h"
#include "chess/position.h"
#include "neural/backend.h"
#include "neural/register.h"
#include "neural/shared_params.h"
#include "utils/optionsparser.h"

namespace lczero {
namespace {

const OptionId kRefBackendId{"ref-backend", "",
                             "Reference backend to compare against."};
const OptionId kRefBackendOptsId{"ref-backend-opts", "",
                                 "Reference backend options."};
const OptionId kRefWeightsId{
    "ref-weights", "",
    "Weights file for the reference backend. Defaults to --weights. Use it "
    "when the two sides are the same network in two representations -- an "
    "ONNX-wrapped net against a .pb.gz, or a Q/DQ export against its "
    "Q/DQ-stripped twin -- which no single -w can express."};
const OptionId kFensId{"fens", "",
                       "File with one FEN per line. Lines starting with '#' "
                       "and blank lines are skipped."};
const OptionId kMaxPositionsId{"max-positions", "",
                               "Maximum number of positions to compare."};
const OptionId kBatchSizeId{"batch-size", "",
                            "Positions per backend call."};
const OptionId kVerboseId{"verbose", "",
                          "Print the worst positions by policy KL."};

struct Stats {
  std::vector<double> v;
  void Add(double x) { v.push_back(x); }
  double Quantile(double q) {
    if (v.empty()) return 0.0;
    std::vector<double> s = v;
    std::sort(s.begin(), s.end());
    const size_t i = static_cast<size_t>(q * (s.size() - 1) + 0.5);
    return s[i];
  }
  double Mean() const {
    if (v.empty()) return 0.0;
    double t = 0.0;
    for (double x : v) t += x;
    return t / v.size();
  }
  double Max() const {
    if (v.empty()) return 0.0;
    return *std::max_element(v.begin(), v.end());
  }
};

void PrintRow(const std::string& name, Stats& s) {
  std::cout << std::left << std::setw(26) << name << std::right << std::fixed
            << std::setprecision(6) << std::setw(12) << s.Mean()
            << std::setw(12) << s.Quantile(0.50) << std::setw(12)
            << s.Quantile(0.90) << std::setw(12) << s.Quantile(0.99)
            << std::setw(12) << s.Max() << std::endl;
}

// KL(ref || test), guarded against zeros.
double PolicyKL(const std::vector<float>& ref, const std::vector<float>& test) {
  const double kEps = 1e-9;
  double kl = 0.0;
  for (size_t i = 0; i < ref.size() && i < test.size(); ++i) {
    const double p = ref[i];
    if (p <= kEps) continue;
    const double q = std::max(static_cast<double>(test[i]), kEps);
    kl += p * std::log(p / q);
  }
  return kl;
}

size_t ArgMax(const std::vector<float>& p) {
  return static_cast<size_t>(std::max_element(p.begin(), p.end()) - p.begin());
}

// Rank of index `idx` in `p` (0 = best).
size_t RankOf(const std::vector<float>& p, size_t idx) {
  size_t rank = 0;
  for (size_t i = 0; i < p.size(); ++i) {
    if (p[i] > p[idx]) ++rank;
  }
  return rank;
}

}  // namespace

void BackendCompare::Run() {
  OptionsParser options;
  SharedBackendParams::Populate(&options);
  options.Add<StringOption>(kRefBackendId) = "cuda-fp16";
  options.Add<StringOption>(kRefBackendOptsId) = "";
  options.Add<StringOption>(kRefWeightsId) = "";
  options.Add<StringOption>(kFensId) = "";
  options.Add<IntOption>(kMaxPositionsId, 1, 100000000) = 10000;
  options.Add<IntOption>(kBatchSizeId, 1, 1024) = 32;
  options.Add<BoolOption>(kVerboseId) = false;

  if (!options.ProcessAllFlags()) return;

  try {
    const OptionsDict& base = options.GetOptionsDict();

    const std::string fens_path = base.Get<std::string>(kFensId);
    if (fens_path.empty()) {
      std::cerr << "--fens=<file> is required." << std::endl;
      return;
    }
    std::ifstream in(fens_path);
    if (!in) {
      std::cerr << "Cannot open " << fens_path << std::endl;
      return;
    }

    const size_t max_positions =
        static_cast<size_t>(base.Get<int>(kMaxPositionsId));
    std::vector<std::string> fens;
    std::string line;
    while (fens.size() < max_positions && std::getline(in, line)) {
      while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
        line.pop_back();
      }
      if (line.empty() || line[0] == '#') continue;
      fens.push_back(line);
    }
    if (fens.empty()) {
      std::cerr << "No FENs read from " << fens_path << std::endl;
      return;
    }

    // The reference backend gets its own options dict: same weights and shared
    // parameters, different backend name and backend-opts.
    // NOTE: OptionsDict must NOT be copied here. Its `aliases_` member is
    // initialised to `{this}`, so a copy keeps a pointer to the ORIGINAL dict
    // and Get() resolves through it -- the copy would silently return the test
    // backend's options. A child dict with `base` as parent gives the intended
    // override-with-fallback semantics.
    OptionsDict ref_dict(&base);
    ref_dict.Set<std::string>(SharedBackendParams::kBackendOptionsId,
                              base.Get<std::string>(kRefBackendOptsId));
    const std::string ref_weights = base.Get<std::string>(kRefWeightsId);
    if (!ref_weights.empty()) {
      ref_dict.Set<std::string>(SharedBackendParams::kWeightsId, ref_weights);
    }

    const std::string test_name =
        base.Get<std::string>(SharedBackendParams::kBackendId);
    const std::string ref_name = base.Get<std::string>(kRefBackendId);

    std::cout << "# test backend      : " << test_name << "  opts='"
              << base.Get<std::string>(SharedBackendParams::kBackendOptionsId)
              << "'" << std::endl;
    std::cout << "# reference backend : " << ref_name << "  opts='"
              << base.Get<std::string>(kRefBackendOptsId) << "'" << std::endl;
    if (!ref_weights.empty()) {
      // Loud on purpose: every number below is then a comparison of two
      // different files, and a reader must not mistake it for one network.
      std::cout << "# reference weights : " << ref_weights
                << "   (DIFFERENT NET FROM --weights)" << std::endl;
    }

    std::unique_ptr<Backend> test_backend =
        BackendManager::Get()->CreateFromName(test_name, base);
    std::unique_ptr<Backend> ref_backend =
        BackendManager::Get()->CreateFromName(ref_name, ref_dict);

    const size_t batch_size =
        static_cast<size_t>(base.Get<int>(kBatchSizeId));
    const bool verbose = base.Get<bool>(kVerboseId);

    Stats kl, q_err, d_err, m_err, p_l1;
    size_t top1_match = 0, top3_match = 0, compared = 0, skipped = 0;
    std::vector<std::pair<double, std::string>> worst;

    for (size_t start = 0; start < fens.size(); start += batch_size) {
      const size_t end = std::min(start + batch_size, fens.size());

      // Histories must stay alive while the spans inside EvalPosition are used.
      std::vector<PositionHistory> histories(end - start);
      std::vector<MoveList> legal(end - start);
      std::vector<EvalPosition> batch;
      std::vector<std::string> batch_fens;
      batch.reserve(end - start);

      for (size_t i = start; i < end; ++i) {
        ChessBoard board;
        int rule50 = 0, gameply = 0;
        try {
          board.SetFromFen(fens[i], &rule50, &gameply);
        } catch (Exception&) {
          ++skipped;
          continue;
        }
        const size_t slot = batch.size();
        histories[slot].Reset(board, rule50, gameply);
        legal[slot] = histories[slot].Last().GetBoard().GenerateLegalMoves();
        if (legal[slot].empty()) {
          ++skipped;
          continue;
        }
        batch.push_back(
            EvalPosition{histories[slot].GetPositions(), legal[slot]});
        batch_fens.push_back(fens[i]);
      }
      if (batch.empty()) continue;

      std::vector<EvalResult> r_test = test_backend->EvaluateBatch(batch);
      std::vector<EvalResult> r_ref = ref_backend->EvaluateBatch(batch);

      for (size_t j = 0; j < batch.size(); ++j) {
        const std::vector<float>& pt = r_test[j].p;
        const std::vector<float>& pr = r_ref[j].p;
        if (pt.size() != pr.size() || pt.empty()) {
          ++skipped;
          continue;
        }
        const double this_kl = PolicyKL(pr, pt);
        kl.Add(this_kl);

        double l1 = 0.0;
        for (size_t k = 0; k < pt.size(); ++k) l1 += std::fabs(pt[k] - pr[k]);
        p_l1.Add(l1);

        const size_t ref_best = ArgMax(pr);
        const size_t test_best = ArgMax(pt);
        if (ref_best == test_best) ++top1_match;
        if (RankOf(pt, ref_best) < 3) ++top3_match;

        q_err.Add(std::fabs(r_test[j].q - r_ref[j].q));
        d_err.Add(std::fabs(r_test[j].d - r_ref[j].d));
        m_err.Add(std::fabs(r_test[j].m - r_ref[j].m));
        ++compared;

        if (verbose) worst.emplace_back(this_kl, batch_fens[j]);
      }
    }

    if (compared == 0) {
      std::cerr << "Nothing compared." << std::endl;
      return;
    }

    std::cout << "# positions compared: " << compared << "  (skipped "
              << skipped << ")" << std::endl
              << std::endl;
    std::cout << std::left << std::setw(26) << "metric" << std::right
              << std::setw(12) << "mean" << std::setw(12) << "p50"
              << std::setw(12) << "p90" << std::setw(12) << "p99"
              << std::setw(12) << "max" << std::endl;
    std::cout << std::string(86, '-') << std::endl;
    PrintRow("policy KL(ref||test)", kl);
    PrintRow("policy L1", p_l1);
    PrintRow("|q_test - q_ref|", q_err);
    PrintRow("|d_test - d_ref|", d_err);
    PrintRow("|m_test - m_ref|", m_err);
    std::cout << std::endl;
    std::cout << std::fixed << std::setprecision(4)
              << "top-1 agreement          : "
              << 100.0 * top1_match / compared << " %" << std::endl;
    std::cout << "ref best move in test top-3 : "
              << 100.0 * top3_match / compared << " %" << std::endl;

    if (verbose && !worst.empty()) {
      std::sort(worst.begin(), worst.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });
      std::cout << std::endl << "# worst 10 positions by policy KL" << std::endl;
      for (size_t i = 0; i < std::min<size_t>(10, worst.size()); ++i) {
        std::cout << std::setprecision(6) << worst[i].first << "  "
                  << worst[i].second << std::endl;
      }
    }
  } catch (Exception& ex) {
    std::cerr << ex.what() << std::endl;
  }
}

}  // namespace lczero
