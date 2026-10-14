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

#pragma once

#include "chess/pgn.h"
#include "chess/position.h"
#include "chess/uciloop.h"
#include "neural/backend.h"
#include "search/classic/search.h"
#ifdef LC0_CHILDQ_CLASSIC
#include "search/childq_classic/search.h"
#endif
#include "search/classic/stoppers/stoppers.h"
#include "trainingdata/trainingdata.h"
#include "utils/optionsparser.h"

namespace lczero {

struct SelfPlayLimits {
  std::int64_t visits = -1;
  // Record deeper than you play: when above `visits`, the search runs to this
  // many visits and the record is written from that state, while the move and
  // the resign reads are those at `visits`. -1 = off.
  std::int64_t record_visits = -1;
  std::int64_t playouts = -1;
  std::int64_t movetime = -1;

  std::unique_ptr<classic::ChainedSearchStopper> MakeSearchStopper() const;
};

struct PlayerOptions {
  using OpeningCallback = std::function<void(const Opening&)>;
  // Backend to use by the player.
  Backend* backend;
  // Callback when player moves.
  CallbackUciResponder::BestMoveCallback best_move_callback;
  // Callback when player outputs info.
  CallbackUciResponder::ThinkingCallback info_callback;
  // Callback when player discards a selected move due to low visits.
  OpeningCallback discarded_callback;
  // User options dictionary.
  const OptionsDict* uci_options;
  // Limits to use for every move.
  SelfPlayLimits search_limits;
};

// Plays a single game vs itself.
class SelfPlayGame {
 public:
  // Player options may point to the same network/cache/etc.
  // If shared_tree is true, search tree is reused between players.
  // (useful for training games). Otherwise the tree is separate for black
  // and white (useful i.e. when they use different networks).
  SelfPlayGame(PlayerOptions white, PlayerOptions black, bool shared_tree,
               const Opening& opening);

  // Populate command line options that it uses.
  static void PopulateUciParams(OptionsParser* options);

  // Action replay (--replay-pgn; lc0 PR 1682): the search runs and is recorded
  // at every position of `line` as in any game, but the move played is the
  // line's. The game ends when the line does, with the line's result (if that
  // is UNDECIDED the last searched position's value becomes the result of
  // every record). Resign, the minimum-visits re-pick and the 450-ply cap are
  // off. `move_floor` > 0 gives the line's move a forced minimum of that share
  // of the visit budget at every root (--replay-move-floor). Call before
  // Play(); the game must have been built from the line's start position,
  // with the line's lead-in moves (if any) as its opening.
  void SetReplay(const Opening& line, float move_floor);
  // True when a replayed line was played to its end (not aborted).
  bool ReplayFinished() const { return replay_finished_; }

  // Starts the game and blocks until the game is finished.
  void Play(int white_threads, int black_threads, bool training,
            SyzygyTablebase* syzygy_tb, bool enable_resign = true);
  // Aborts the game currently played, doesn't matter if it's synchronous or
  // not.
  void Abort();

  // Number of ply used from the given opening.
  int GetStartPly() const { return start_ply_; }

  // Writes training data to a file.
  void WriteTrainingData(TrainingDataWriter* writer) const;
  // Number of training frames the game holds.
  size_t GetTrainingFrameCount() const { return training_data_.GetFrameCount(); }

  GameResult GetGameResult() const { return game_result_; }
  std::vector<Move> GetMoves() const;
  // Gets the eval which required the biggest swing up to get the final outcome.
  // Eval is the expected outcome in the range 0<->1.
  float GetWorstEvalForWinnerOrDraw() const;
  int move_count_ = 0;
  uint64_t nodes_total_ = 0;

 private:
  // The game loop and its helpers, written once for both searches
  // (--selfplay-search): classic's tree/search, or childq-classic's, which is
  // classic's verbatim plus the child-Q consumers.
  template <typename TreeT>
  int InitTrees(std::shared_ptr<TreeT> (&tree)[2], bool shared_tree,
                const Opening& opening, float white_prob, float black_prob);
  template <typename TreeT, typename SearchT>
  void PlayImpl(std::shared_ptr<TreeT> (&tree)[2],
                std::unique_ptr<SearchT>& search, int white_threads,
                int black_threads, bool training, SyzygyTablebase* syzygy_tb,
                bool enable_resign);
  template <typename TreeT>
  static std::vector<Move> MovesOf(const TreeT& tree);

  // options_[0] is for white player, [1] for black.
  PlayerOptions options_[2];
  // Node tree for player1 and player2. If the tree is shared between players,
  // tree_[0] == tree_[1].
  std::shared_ptr<classic::NodeTree> tree_[2];
  std::string orig_fen_;
  int start_ply_;

  // Search that is currently in progress. Stored in members so that Abort()
  // can stop it.
  std::unique_ptr<classic::Search> search_;
  bool abort_ = false;
  GameResult game_result_ = GameResult::UNDECIDED;
  bool adjudicated_ = false;
  // Action replay (SetReplay). The moves are in the frame GetBestMove()
  // returns, as the pgn readers give them.
  bool replaying_ = false;
  MoveList replay_moves_;
  GameResult replay_result_ = GameResult::UNDECIDED;
  float replay_move_floor_ = 0.0f;
  bool replay_finished_ = false;
  // Track minimum eval for each player so that GetWorstEvalForWinnerOrDraw()
  // can be calculated after end of game.
  float min_eval_[2] = {1.0f, 1.0f};
  // Track the maximum eval for white win, draw, black win for comparison to
  // actual outcome.
  float max_eval_[3] = {0.0f, 0.0f, 0.0f};
  const bool chess960_;
  // --selfplay-search=childq-classic: the game runs on cq_tree_ / cq_search_
  // and tree_ / search_ stay empty (and vice versa).
  const bool childq_search_;
#ifdef LC0_CHILDQ_CLASSIC
  std::shared_ptr<childq_classic::NodeTree> cq_tree_[2];
  std::unique_ptr<childq_classic::Search> cq_search_;
#endif
  std::mutex mutex_;

  // Training data to send.
  V6TrainingDataArray training_data_;

  std::unique_ptr<SyzygyTablebase> syzygy_tb_;
};

}  // namespace lczero
