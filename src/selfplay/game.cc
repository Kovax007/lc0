/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2021 The LCZero Authors

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

#include "selfplay/game.h"

#include <algorithm>
#include <cmath>

#include "search/classic/stoppers/common.h"
#include "search/classic/stoppers/factory.h"
#include "utils/random.h"

namespace lczero {

namespace {
const OptionId kReuseTreeId{"reuse-tree", "ReuseTree",
                            "Reuse the search tree between moves."};
const OptionId kResignPercentageId{
    "resign-percentage", "ResignPercentage",
    "Resign when win percentage drops below specified value."};
const OptionId kResignWDLStyleId{
    "resign-wdlstyle", "ResignWDLStyle",
    "If set, resign percentage applies to any output state being above "
    "100% minus the percentage instead of winrate being below."};
const OptionId kResignEarliestMoveId{"resign-earliest-move",
                                     "ResignEarliestMove",
                                     "Earliest move that resign is allowed."};
const OptionId kMinimumAllowedVistsId{
    "minimum-allowed-visits", "MinimumAllowedVisits",
    "Unless the selected move is the best move, temperature based selection "
    "will be retried until visits of selected move is greater than or equal to "
    "this threshold."};
const OptionId kUciChess960{
    "chess960", "UCI_Chess960",
    "Castling moves are encoded as \"king takes rook\"."};
const OptionId kSyzygyTablebaseId{
    "syzygy-paths", "SyzygyPath",
    "List of Syzygy tablebase directories, list entries separated by system "
    "separator (\";\" for Windows, \":\" for Linux).",
    's'};
const OptionId kOpeningStopProbId{
    "opening-stop-prob", "OpeningStopProb",
    "From each opening move, start a self-play game with probability max(p, "
    "1/n), where p is the value given and n the opening moves remaining."};
const OptionId kSelfplaySearchId{
    "selfplay-search", "SelfplaySearch",
    "The search every self-play move runs. classic = the fleet's search. "
    "childq-classic = classic's search verbatim plus the child-Q consumers "
    "(the --childq-* options); with those at their defaults it plays the same "
    "games as classic. Both players must use the same one."};

bool UsesChildQSearch(const PlayerOptions& white, const PlayerOptions& black) {
  const auto w = white.uci_options->Get<std::string>(kSelfplaySearchId);
  const auto b = black.uci_options->Get<std::string>(kSelfplaySearchId);
  if (w != b) {
    throw Exception("--selfplay-search must be the same for both players");
  }
  return w == "childq-classic";
}
}  // namespace

void SelfPlayGame::PopulateUciParams(OptionsParser* options) {
  options->Add<BoolOption>(kReuseTreeId) = false;
  options->Add<BoolOption>(kResignWDLStyleId) = false;
  options->Add<FloatOption>(kResignPercentageId, 0.0f, 100.0f) = 0.0f;
  options->Add<IntOption>(kResignEarliestMoveId, 0, 1000) = 0;
  options->Add<IntOption>(kMinimumAllowedVistsId, 0, 1000000) = 0;
  options->Add<BoolOption>(kUciChess960) = false;
  PopulateTimeManagementOptions(classic::RunType::kSelfplay, options);
  options->Add<StringOption>(kSyzygyTablebaseId);
  options->Add<FloatOption>(kOpeningStopProbId, 0.0f, 1.0f) = 0.0f;
#ifdef LC0_CHILDQ_CLASSIC
  options->Add<ChoiceOption>(
      kSelfplaySearchId,
      std::vector<std::string>{"classic", "childq-classic"}) = "classic";
#else
  options->Add<ChoiceOption>(kSelfplaySearchId,
                             std::vector<std::string>{"classic"}) = "classic";
#endif
}

SelfPlayGame::SelfPlayGame(PlayerOptions white, PlayerOptions black,
                           bool shared_tree, const Opening& opening)
    : options_{white, black},
      chess960_{white.uci_options->Get<bool>(kUciChess960) ||
                black.uci_options->Get<bool>(kUciChess960)},
      childq_search_{UsesChildQSearch(white, black)},
      training_data_(
          classic::SearchParams(*white.uci_options).GetHistoryFill(),
          classic::SearchParams(*black.uci_options).GetHistoryFill(),
          pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE,
          classic::SearchParams(*white.uci_options).GetTrainingDataV7(),
          classic::SearchParams(*white.uci_options).GetTrainingDataChildQ(),
          classic::SearchParams(*white.uci_options)
              .GetTrainingDataChildQMinVisits(),
          classic::SearchParams(*white.uci_options).GetTrainingDataV8(),
          classic::SearchParams(*white.uci_options).GetTrainingDataRecipeId()) {
  orig_fen_ = opening.start_fen;
  for (const auto& player : {white, black}) {
    if (player.search_limits.record_visits > player.search_limits.visits) {
      if (!childq_search_) {
        throw Exception(
            "--record-visits needs --selfplay-search=childq-classic");
      }
      if (player.uci_options->Get<int>(kMinimumAllowedVistsId) > 0) {
        throw Exception(
            "--record-visits cannot be combined with --minimum-allowed-visits: "
            "a re-picked move would come from the deeper tree.");
      }
    }
  }
  auto white_prob = white.uci_options->Get<float>(kOpeningStopProbId);
  auto black_prob = black.uci_options->Get<float>(kOpeningStopProbId);
  if (white_prob != black_prob && white_prob != 0 && black_prob != 0) {
    throw Exception("Stop probabilities must be both equal or zero!");
  }
#ifdef LC0_CHILDQ_CLASSIC
  if (childq_search_) {
    start_ply_ =
        InitTrees(cq_tree_, shared_tree, opening, white_prob, black_prob);
    return;
  }
#endif
  start_ply_ = InitTrees(tree_, shared_tree, opening, white_prob, black_prob);
}

template <typename TreeT>
int SelfPlayGame::InitTrees(std::shared_ptr<TreeT> (&tree)[2],
                            bool shared_tree, const Opening& opening,
                            float white_prob, float black_prob) {
  tree[0] = std::make_shared<TreeT>();
  tree[0]->ResetToPosition(orig_fen_, {});

  if (shared_tree) {
    tree[1] = tree[0];
  } else {
    tree[1] = std::make_shared<TreeT>();
    tree[1]->ResetToPosition(orig_fen_, {});
  }
  int ply = 0;
  for (Move m : opening.moves) {
    // For early exit from the opening, we support two cases: a) where both
    // sides have the same exit probability and b) where one side's exit
    // probability is zero. In the following formula, `positions` is the number
    // of possible exit points remaining, used for adjusting the exit
    // probability (to avoid favoring the last position).
    auto exit_prob_now = tree[0]->IsBlackToMove() ? black_prob : white_prob;
    auto exit_prob_next = tree[0]->IsBlackToMove() ? white_prob : black_prob;
    int positions = opening.moves.size() - ply + 1;
    if (exit_prob_now > 0.0f &&
        Random::Get().GetFloat(1.0f) <
            std::max(exit_prob_now,
                     exit_prob_now / (exit_prob_now * ((positions + 1) / 2) +
                                      exit_prob_next * (positions / 2)))) {
      break;
    }
    if (tree[0]->IsBlackToMove()) m.Flip();
    tree[0]->MakeMove(m);
    if (tree[0] != tree[1]) tree[1]->MakeMove(m);
    ply++;
  }
  return ply;
}

void SelfPlayGame::SetReplay(const Opening& line, float move_floor) {
  replaying_ = true;
  replay_moves_ = line.moves;
  replay_result_ = line.result;
  replay_move_floor_ = move_floor;
}

void SelfPlayGame::Play(int white_threads, int black_threads, bool training,
                        SyzygyTablebase* syzygy_tb, bool enable_resign) {
  // A replayed line is played to its end: at a proven root the best W is 1,
  // which would adjudicate the game after its first record.
  if (replaying_) enable_resign = false;
#ifdef LC0_CHILDQ_CLASSIC
  if (childq_search_) {
    PlayImpl(cq_tree_, cq_search_, white_threads, black_threads, training,
             syzygy_tb, enable_resign);
    return;
  }
#endif
  PlayImpl(tree_, search_, white_threads, black_threads, training, syzygy_tb,
           enable_resign);
}

template <typename TreeT, typename SearchT>
void SelfPlayGame::PlayImpl(std::shared_ptr<TreeT> (&tree)[2],
                            std::unique_ptr<SearchT>& search,
                            int white_threads, int black_threads,
                            bool training, SyzygyTablebase* syzygy_tb,
                            bool enable_resign) {
  bool blacks_move = tree[0]->IsBlackToMove();

  // The tournament-level tablebase (the syzygy_tb parameter, used by every
  // Search created below) is the only instance that is ever probed. The
  // per-game SyzygyTablebase re-init that used to live here was dead weight —
  // its object was never probed — and re-initializing a tablebase (145 mmaps
  // + entry allocations) concurrently from every starting game at high
  // parallelism corrupted the heap (reproducible free(): invalid next size
  // within minutes at --parallelism=24 with --syzygy-paths set).
  // Do moves while not end of the game. (And while not abort_)
  while (!abort_) {
    game_result_ = tree[0]->GetPositionHistory().ComputeGameResult();

    // If endgame, stop.
    if (game_result_ != GameResult::UNDECIDED) {
      replay_finished_ = replaying_;
      break;
    }
    if (!replaying_ &&
        tree[0]->GetPositionHistory().Last().GetGamePly() >= 450) {
      adjudicated_ = true;
      break;
    }
    // Replay: the ply of the line this position is at.
    const size_t replay_ply = tree[0]->GetPositionHistory().GetLength() - 1;
    if (replaying_ && replay_ply >= replay_moves_.size()) {
      // The line ended without a result on the board: it carries its own.
      game_result_ = replay_result_;
      if (game_result_ != GameResult::UNDECIDED) adjudicated_ = true;
      replay_finished_ = true;
      break;
    }
    // Initialize search.
    const int idx = blacks_move ? 1 : 0;
    // Record deeper than you play: the search runs past the decision budget.
    const bool record_deeper = options_[idx].search_limits.record_visits >
                               options_[idx].search_limits.visits;
    if (!options_[idx].uci_options->Get<bool>(kReuseTreeId)) {
      tree[idx]->TrimTreeAtHead();
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (abort_) break;
      auto stoppers = options_[idx].search_limits.MakeSearchStopper();
      classic::PopulateIntrinsicStoppers(stoppers.get(),
                                         *options_[idx].uci_options);

      std::unique_ptr<UciResponder> responder =
          std::make_unique<CallbackUciResponder>(
              options_[idx].best_move_callback, options_[idx].info_callback);

      // Disable the syzygy ROOT-move filter: when the position is within
      // tablebase range, hand the search all legal moves as `searchmoves`.
      // MakeRootMoveFilter treats a non-empty searchmoves list as an override
      // (search.cc "Search moves overrides tablebase"), so the root filter that
      // would otherwise delete TB-losing moves from the search — leaking the raw
      // prior into the policy target — is bypassed. Interior WDL adjudication is
      // unaffected (root_is_in_dtz_ stays false). The only thing given up is DTZ
      // root forcing on deep wins; the <=7-man value rescore restores those.
      MoveList root_filter;
      {
        const auto& board =
            tree[idx]->GetPositionHistory().Last().GetBoard();
        if (syzygy_tb &&
            static_cast<int>((board.ours() | board.theirs()).count()) <=
                syzygy_tb->max_cardinality()) {
          root_filter = board.GenerateLegalMoves();
        }
      }
      search = std::make_unique<SearchT>(
          *tree[idx], options_[idx].backend, std::move(responder),
          /* searchmoves */ root_filter,
          std::chrono::steady_clock::now(), std::move(stoppers),
          /* infinite */ false, /* ponder */ false,
          *options_[idx].uci_options, syzygy_tb);
      if (record_deeper) {
        search->SetDecisionVisits(options_[idx].search_limits.visits);
      }
      if (replaying_ && replay_move_floor_ > 0.0f &&
          options_[idx].search_limits.visits > 0) {
        search->SetReplayFloor(
            replay_moves_[replay_ply],
            static_cast<uint32_t>(std::lround(
                replay_move_floor_ * options_[idx].search_limits.visits)));
      }
    }

    // Do search.
    search->RunBlocking(blacks_move ? black_threads : white_threads);
    move_count_++;
    nodes_total_ += search->GetTotalPlayouts();
    if (abort_) break;

    Move best_move;
    bool best_is_terminal;
    const auto best_eval = search->GetBestEval(&best_move, &best_is_terminal);
    // The game's own reads (resign, adjudication, the eval statistics) are
    // those of the decision budget; best_eval, from the whole search, goes
    // into the record.
    const auto decision_eval =
        record_deeper ? search->GetDecisionEval() : best_eval;
    float eval = decision_eval.wl;
    eval = (eval + 1) / 2;
    if (eval < min_eval_[idx]) min_eval_[idx] = eval;
    const int move_number = tree[0]->GetPositionHistory().GetLength() / 2 + 1;
    auto best_w = (decision_eval.wl + 1.0f - decision_eval.d) / 2.0f;
    auto best_d = decision_eval.d;
    auto best_l = best_w - decision_eval.wl;
    max_eval_[0] = std::max(max_eval_[0], blacks_move ? best_l : best_w);
    max_eval_[1] = std::max(max_eval_[1], best_d);
    max_eval_[2] = std::max(max_eval_[2], blacks_move ? best_w : best_l);
    auto node = tree[idx]->GetCurrentHead();
    auto played_eval = best_eval;
    Move move;
    while (true) {
      move = replaying_ ? replay_moves_[replay_ply]
                        : search->GetBestMove().first;
      uint32_t max_n = 0;
      uint32_t cur_n = 0;

      for (auto& edge : node->Edges()) {
        if (edge.GetN() > max_n) {
          max_n = edge.GetN();
        }
        if (edge.GetMove(tree[idx]->IsBlackToMove()) == move) {
          cur_n = edge.GetN();
          played_eval.wl = edge.GetWL(-node->GetWL());
          played_eval.d = edge.GetD(node->GetD());
          played_eval.ml = edge.GetM(node->GetM() - 1) + 1;
        }
      }
      // If 'best move' is less than allowed visits and not max visits,
      // discard it and try again. (A replayed move is never discarded.)
      if (replaying_ || cur_n == max_n ||
          static_cast<int>(cur_n) >=
              options_[idx].uci_options->Get<int>(kMinimumAllowedVistsId)) {
        break;
      }
      PositionHistory history_copy = tree[idx]->GetPositionHistory();
      Move move_for_history = move;
      if (tree[idx]->IsBlackToMove()) move_for_history.Flip();
      history_copy.Append(move_for_history);
      // Ensure not to discard games that are already decided.
      if (history_copy.ComputeGameResult() == GameResult::UNDECIDED) {
        auto move_list_to_discard = GetMoves();
        move_list_to_discard.push_back(move);
        options_[idx].discarded_callback({orig_fen_, move_list_to_discard});
      }
      search->ResetBestMove();
    }

    if (training) {
      bool best_is_proof = best_is_terminal;  // But check for better moves.
      if (best_is_proof && best_eval.wl < 1) {
        auto best =
            (best_eval.wl == 0) ? GameResult::DRAW : GameResult::BLACK_WON;
        auto upper = best;
        for (const auto& edge : node->Edges()) {
          upper = std::max(edge.GetBounds().second, upper);
        }
        if (best < upper) {
          best_is_proof = false;
        }
      }
      // Append training data. The GameResult is later overwritten.
      std::vector<Move> legal_moves = tree[idx]
                                          ->GetPositionHistory()
                                          .Last()
                                          .GetBoard()
                                          .GenerateLegalMoves();
      auto root_visits = search->GetVisitDistribution(legal_moves);
      std::optional<EvalResult> nneval =
          options_[idx].backend->GetCachedEvaluation(EvalPosition{
              tree[idx]->GetPositionHistory().GetPositions(), legal_moves});
      // The per-child search data for the V8 block. Collected only when V8 is
      // on: it walks the root edges (and one ply below them) a second time,
      // which is free per played move but not free per playout.
      std::optional<RootSearchData> root_data;
      if (classic::SearchParams(*options_[idx].uci_options)
              .GetTrainingDataV8()) {
        root_data = search->GetRootChildData(legal_moves);
      }
      training_data_.Add(tree[idx]->GetCurrentHead(),
                         tree[idx]->GetPositionHistory(), best_eval,
                         played_eval, best_is_proof, best_move, move,
                         root_visits, legal_moves, nneval,
                         root_data ? &*root_data : nullptr);
      // The move of this ply was not the search's choice.
      if (replaying_) training_data_.MarkLastForcedPly();
    }

    // Adjudication happens after the training frame is appended so the
    // position whose search holds the proven evaluation (e.g. a found mate)
    // is included in the game record (upstream PR #2429).
    if (enable_resign && move_number >= options_[idx].uci_options->Get<int>(
                                            kResignEarliestMoveId)) {
      const float resignpct =
          options_[idx].uci_options->Get<float>(kResignPercentageId) / 100;
      if (options_[idx].uci_options->Get<bool>(kResignWDLStyleId)) {
        auto threshold = 1.0f - resignpct;
        if (best_w > threshold) {
          game_result_ =
              blacks_move ? GameResult::BLACK_WON : GameResult::WHITE_WON;
          adjudicated_ = true;
          break;
        }
        if (best_l > threshold) {
          game_result_ =
              blacks_move ? GameResult::WHITE_WON : GameResult::BLACK_WON;
          adjudicated_ = true;
          break;
        }
        if (best_d > threshold) {
          game_result_ = GameResult::DRAW;
          adjudicated_ = true;
          break;
        }
      } else {
        if (eval < resignpct) {  // always false when resignpct == 0
          game_result_ =
              blacks_move ? GameResult::WHITE_WON : GameResult::BLACK_WON;
          adjudicated_ = true;
          break;
        }
      }
    }
    // Must reset the search before mutating the tree.
    search.reset();

    // Add best move to the tree.
    if (tree[0]->IsBlackToMove()) move.Flip();
    tree[0]->MakeMove(move);
    if (tree[0] != tree[1]) tree[1]->MakeMove(move);
    blacks_move = !blacks_move;
  }
}

std::vector<Move> SelfPlayGame::GetMoves() const {
#ifdef LC0_CHILDQ_CLASSIC
  if (childq_search_) return MovesOf(*cq_tree_[0]);
#endif
  return MovesOf(*tree_[0]);
}

template <typename TreeT>
std::vector<Move> SelfPlayGame::MovesOf(const TreeT& tree) {
  std::vector<Move> moves;
  for (auto* node = tree.GetCurrentHead(); node != tree.GetGameBeginNode();
       node = node->GetParent()) {
    moves.push_back(node->GetParent()->GetEdgeToNode(node)->GetMove());
  }
  std::vector<Move> result;
  Position pos = tree.GetPositionHistory().Starting();
  while (!moves.empty()) {
    Move move = moves.back();
    moves.pop_back();
    pos = Position(pos, move);
    // Position already flipped, therefore flip the move if white to move.
    if (!pos.IsBlackToMove()) move.Flip();
    result.push_back(move);
  }
  return result;
}

float SelfPlayGame::GetWorstEvalForWinnerOrDraw() const {
  // TODO: This assumes both players have the same resign style.
  // Supporting otherwise involves mixing the meaning of worst.
  if (options_[0].uci_options->Get<bool>(kResignWDLStyleId)) {
    if (game_result_ == GameResult::WHITE_WON) {
      return std::max(max_eval_[1], max_eval_[2]);
    } else if (game_result_ == GameResult::BLACK_WON) {
      return std::max(max_eval_[1], max_eval_[0]);
    } else {
      return std::max(max_eval_[2], max_eval_[0]);
    }
  }
  if (game_result_ == GameResult::WHITE_WON) return min_eval_[0];
  if (game_result_ == GameResult::BLACK_WON) return min_eval_[1];
  return std::min(min_eval_[0], min_eval_[1]);
}

void SelfPlayGame::Abort() {
  std::lock_guard<std::mutex> lock(mutex_);
  abort_ = true;
  if (search_) search_->Abort();
#ifdef LC0_CHILDQ_CLASSIC
  if (cq_search_) cq_search_->Abort();
#endif
}

void SelfPlayGame::WriteTrainingData(TrainingDataWriter* writer) const {
  training_data_.Write(writer, game_result_, adjudicated_,
                       /* final_eval_as_result= */ replaying_);
}

std::unique_ptr<classic::ChainedSearchStopper>
SelfPlayLimits::MakeSearchStopper() const {
  auto result = std::make_unique<classic::ChainedSearchStopper>();

  // always set VisitsStopper to avoid exceeding the limit 4000000000, the
  // default value when visits = 0
  result->AddStopper(std::make_unique<classic::VisitsStopper>(
      std::max(visits, record_visits), false));
  if (playouts >= 0) {
    result->AddStopper(
        std::make_unique<classic::PlayoutsStopper>(playouts, false));
  }
  if (movetime >= 0) {
    result->AddStopper(std::make_unique<classic::TimeLimitStopper>(movetime));
  }
  return result;
}

}  // namespace lczero
