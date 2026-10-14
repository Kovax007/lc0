/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2020 The LCZero Authors

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

#include <zlib.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <fstream>
#include <optional>

#include "chess/bitboard.h"
#include "chess/board.h"
#include "chess/position.h"
#include "utils/exception.h"
#include "utils/logging.h"

namespace lczero {

struct Opening {
  std::string start_fen = ChessBoard::kStartposFen;
  MoveList moves;
  // Only the replay reader fills these; the opening-book reader leaves them
  // at their defaults.
  GameResult result = GameResult::UNDECIDED;
  // Replay: the first `lead_in` moves are played without a search and without
  // a record, so that the first searched position has a real history.
  int lead_in = 0;
};

inline bool GzGetLine(gzFile file, std::string& line) {
  bool flag = false;
  char s[2000];
  line.clear();
  while (gzgets(file, s, sizeof(s))) {
    flag = true;
    line += s;
    auto r = line.find_last_of('\n');
    if (r != std::string::npos) {
      line.erase(r);
      break;
    }
  }
  return flag;
}

class PgnReader {
 public:
  void AddPgnFile(const std::string& filepath) {
    const gzFile file = gzopen(filepath.c_str(), "r");
    if (!file) {
      throw Exception(errno == ENOENT ? "Opening book file not found."
                                      : "Error opening opening book file.");
    }

    std::string line;
    bool in_comment = false;
    bool started = false;
    while (GzGetLine(file, line)) {
      // Check if we have a UTF-8 BOM. If so, just ignore it.
      // Only supposed to exist in the first line, but should not matter.
      if (line.substr(0, 3) == "\xEF\xBB\xBF") line = line.substr(3);
      if (!line.empty() && line.back() == '\r') line.pop_back();
      // TODO: support line breaks in tags to ensure they are properly ignored.
      if (line.empty() || line[0] == '[') {
        if (started) {
          Flush();
          started = false;
        }
        auto uc_line = line;
        std::transform(
            uc_line.begin(), uc_line.end(), uc_line.begin(),
            [](unsigned char c) { return std::toupper(c); }  // correct
        );
        if (uc_line.find("[FEN \"", 0) == 0) {
          auto start_trimmed = line.substr(6);
          cur_startpos_ = start_trimmed.substr(0, start_trimmed.find('"'));
          cur_board_.SetFromFen(cur_startpos_);
        }
        continue;
      }
      // Must have at least one non-tag non-empty line in order to be considered
      // a game.
      started = true;
      // Handle braced comments.
      int cur_offset = 0;
      while ((in_comment && line.find('}', cur_offset) != std::string::npos) ||
             (!in_comment && line.find('{', cur_offset) != std::string::npos)) {
        if (in_comment && line.find('}', cur_offset) != std::string::npos) {
          line = line.substr(0, cur_offset) +
                 line.substr(line.find('}', cur_offset) + 1);
          in_comment = false;
        } else {
          cur_offset = line.find('{', cur_offset);
          in_comment = true;
        }
      }
      if (in_comment) {
        line = line.substr(0, cur_offset);
      }
      // Trim trailing comment.
      if (line.find(';') != std::string::npos) {
        line = line.substr(0, line.find(';'));
      }
      if (line.empty()) continue;
      std::istringstream iss(line);
      std::string word;
      while (!iss.eof()) {
        word.clear();
        iss >> word;
        if (word.size() < 2) continue;
        // Trim move numbers from front.
        const auto idx = word.find('.');
        if (idx != std::string::npos) {
          bool all_nums = true;
          for (size_t i = 0; i < idx; i++) {
            if (word[i] < '0' || word[i] > '9') {
              all_nums = false;
              break;
            }
          }
          if (all_nums) {
            word = word.substr(idx + 1);
            // "12..." (the number of a Black move): drop the other dots too.
            const auto rest = word.find_first_not_of('.');
            word = rest == std::string::npos ? "" : word.substr(rest);
          }
        }
        // Pure move numbers can be skipped.
        if (word.size() < 2) continue;
        // Ignore score line.
        if (word == "1/2-1/2" || word == "1-0" || word == "0-1" || word == "*")
          continue;
        cur_game_.push_back(SanToMove(word, cur_board_));
        cur_board_.ApplyMove(cur_game_.back());
        // Board ApplyMove wants mirrored for black, but outside code wants
        // normal, so mirror it back again. The board is still the mover's
        // here: flipped means Black made the move (the opening may start with
        // Black to move, so the parity of the move count does not tell).
        if (cur_board_.flipped()) cur_game_.back().Flip();
        cur_board_.Mirror();
      }
    }
    if (started) {
      Flush();
    }
    gzclose(file);
  }
  std::vector<Opening> GetGames() const { return games_; }
  std::vector<Opening>&& ReleaseGames() { return std::move(games_); }

 protected:
  void Flush() {
    games_.push_back({cur_startpos_, cur_game_});
    cur_game_.clear();
    cur_board_.SetFromFen(ChessBoard::kStartposFen);
    cur_startpos_ = ChessBoard::kStartposFen;
  }

  static std::optional<PieceType> PieceToPieceType(int p) {
    switch (p) {
      case -1:
        return std::nullopt;
      case 2:
        return kQueen;
      case 3:
        return kBishop;
      case 4:
        return kKnight;
      case 5:
        return kRook;
      default:
        // 0 and 1 are pawn and king, which are not legal promotions, other
        // numbers don't correspond to a known piece type.
        CERR << "Unexpected promotion!!";
        throw Exception("Trying to create a move with illegal promotion.");
    }
  }

  Move SanToMove(const std::string& san, const ChessBoard& board) {
    int p = 0;
    size_t idx = 0;
    if (san[0] == 'K') {
      p = 1;
    } else if (san[0] == 'Q') {
      p = 2;
    } else if (san[0] == 'B') {
      p = 3;
    } else if (san[0] == 'N') {
      p = 4;
    } else if (san[0] == 'R') {
      p = 5;
    } else if (san.substr(0, 3) == "O-O") {
      Move m;
      auto king_board = board.kings() & board.ours();
      Square king_sq(File::FromIdx(GetLowestBit(king_board.as_int())), kRank1);
      m = Move::WhiteCastling(king_sq.file(),
                              san.substr(3, 2) == "-O"
                                  ? board.castlings().our_queenside_rook
                                  : board.castlings().our_kingside_rook);
      return m;
    }
    if (p != 0) idx++;
    // Formats e4 1e5 de5 d1e5 - with optional x's - followed by =Q for
    // promotions, and even more characters after that also optional.
    int r1 = -1;
    int c1 = -1;
    int r2 = -1;
    int c2 = -1;
    int p2 = -1;
    bool pPending = false;
    for (; idx < san.size(); idx++) {
      if (san[idx] == 'x') continue;
      if (san[idx] == '=') {
        pPending = true;
        continue;
      }
      if (san[idx] >= '1' && san[idx] <= '8') {
        r1 = r2;
        r2 = san[idx] - '1';
        continue;
      }
      if (san[idx] >= 'a' && san[idx] <= 'h') {
        c1 = c2;
        c2 = san[idx] - 'a';
        continue;
      }
      if (pPending) {
        if (san[idx] == 'Q') {
          p2 = 2;
        } else if (san[idx] == 'B') {
          p2 = 3;
        } else if (san[idx] == 'N') {
          p2 = 4;
        } else if (san[idx] == 'R') {
          p2 = 5;
        }
        pPending = false;
        break;
      }
      break;
    }
    if (r1 == -1 || c1 == -1) {
      // Need to find the from cell based on piece.
      int sr1 = r1;
      int sr2 = r2;
      if (board.flipped()) {
        if (sr1 != -1) sr1 = 7 - sr1;
        sr2 = 7 - sr2;
      }
      BitBoard searchBits;
      if (p == 0) {
        searchBits = (board.pawns() & board.ours());
      } else if (p == 1) {
        searchBits = (board.kings() & board.ours());
      } else if (p == 2) {
        searchBits = (board.queens() & board.ours());
      } else if (p == 3) {
        searchBits = (board.bishops() & board.ours());
      } else if (p == 4) {
        searchBits = (board.knights() & board.ours());
      } else if (p == 5) {
        searchBits = (board.rooks() & board.ours());
      }
      auto plm = board.GenerateLegalMoves();
      int pr1 = -1;
      int pc1 = -1;
      for (Square sq : searchBits) {
        if (sr1 != -1 && sq.rank().idx != sr1) continue;
        if (c1 != -1 && sq.file().idx != c1) continue;
        std::optional<PieceType> promotion = PieceToPieceType(p2);
        std::optional<Square> enpassant = std::nullopt;
        if (!board.en_passant().empty()) {
          auto sq = *board.en_passant().begin();
          enpassant = Square(sq.file(), kRank6);
        }
        Square to(File::FromIdx(c2), Rank::FromIdx(sr2));
        // Only a pawn takes en passant; a piece may move onto that square.
        Move move_to_find = promotion ? Move::WhitePromotion(sq, to, *promotion)
                            : p == 0 && enpassant && *enpassant == to
                                ? Move::WhiteEnPassant(sq, to)
                                : Move::White(sq, to);
        if (std::find(plm.begin(), plm.end(), move_to_find) == plm.end()) {
          continue;
        }
        if (pc1 != -1) {
          CERR << "Ambiguous!!";
          throw Exception("Opening book move seems ambiguous.");
        }
        pr1 = sq.rank().idx;
        pc1 = sq.file().idx;
      }
      if (pc1 == -1) {
        CERR << "No Match!!";
        throw Exception("Opening book move seems illegal.");
      }
      r1 = pr1;
      c1 = pc1;
      if (board.flipped()) {
        r1 = 7 - r1;
      }
    }
    std::optional<PieceType> promotion = PieceToPieceType(p2);

    std::optional<Square> enpassant = std::nullopt;
    if (!board.en_passant().empty()) {
      auto sq = *board.en_passant().begin();
      enpassant = Square(sq.file(), board.flipped() ? kRank3 : kRank6);
    }
    Square from(File::FromIdx(c1), Rank::FromIdx(r1));
    Square to(File::FromIdx(c2), Rank::FromIdx(r2));
    Move m = promotion ? Move::WhitePromotion(from, to, *promotion)
             : p == 0 && enpassant && *enpassant == to
                 ? Move::WhiteEnPassant(from, to)
                 : Move::White(from, to);
    if (board.flipped()) m.Flip();
    return m;
  }

  ChessBoard cur_board_{ChessBoard::kStartposFen};
  MoveList cur_game_;
  std::string cur_startpos_ = ChessBoard::kStartposFen;
  std::vector<Opening> games_;
};

// The reader behind selfplay --replay-pgn (lc0 PR 1682 "action replay" and
// PR 1786 "improvements for the pgn file reader", borg323). Against the
// opening-book reader above: the first move may be Black's, the game result is
// kept, a game ends at its result token, annotations and NAGs are skipped,
// "0-0" castles, variations are skipped (variation_ply < 0) or read as lines of
// their own, and errors name the line. The tag [LeadInPlies "k"] marks the
// first k plies of a game as its lead-in (Opening::lead_in). It is a separate
// class so that the book reader, which every opening book goes through, parses
// exactly as before.
class ReplayPgnReader : public PgnReader {
 public:
  // variation_ply >= 0: every variation becomes a game of its own that starts
  // that many plies before the branch point.
  void AddPgnFile(const std::string& filepath, int variation_ply = -1) {
    const gzFile file = gzopen(filepath.c_str(), "r");
    if (!file) {
      throw Exception(errno == ENOENT ? "Replay pgn file not found."
                                      : "Error opening replay pgn file.");
    }

    std::string line;
    bool in_comment = false;
    bool started = false;
    while (GzGetLine(file, line)) {
      line_no_++;
      if (line.substr(0, 3) == "\xEF\xBB\xBF") line = line.substr(3);
      if (!line.empty() && line.back() == '\r') line.pop_back();
      if (!line.empty() && line[0] == '[' && !in_comment) {
        if (started) {
          FlushGame("*");
          started = false;
        }
        auto uc_line = line;
        std::transform(uc_line.begin(), uc_line.end(), uc_line.begin(),
                       [](unsigned char c) { return std::toupper(c); });
        if (uc_line.find("[FEN \"", 0) == 0) {
          auto start_trimmed = line.substr(6);
          cur_startpos_ = start_trimmed.substr(0, start_trimmed.find('"'));
          cur_board_.SetFromFen(cur_startpos_);
        } else if (uc_line.find("[LEADINPLIES \"", 0) == 0) {
          const auto value = line.substr(14, line.find('"', 14) - 14);
          if (value.empty() ||
              value.find_first_not_of("0123456789") != std::string::npos ||
              value.size() > 4) {
            throw Exception("Bad LeadInPlies tag on line " +
                            std::to_string(line_no_) + " of the replay pgn.");
          }
          cur_lead_in_ = std::stoi(value);
        }
        continue;
      }
      // Handle braced comments.
      int cur_offset = 0;
      while ((in_comment && line.find('}', cur_offset) != std::string::npos) ||
             (!in_comment && line.find('{', cur_offset) != std::string::npos)) {
        if (in_comment && line.find('}', cur_offset) != std::string::npos) {
          line = line.substr(0, cur_offset) +
                 line.substr(line.find('}', cur_offset) + 1);
          in_comment = false;
        } else {
          cur_offset = line.find('{', cur_offset);
          in_comment = true;
        }
      }
      if (in_comment) {
        line = line.substr(0, cur_offset);
      }
      // Trim trailing comment.
      if (line.find(';') != std::string::npos) {
        line = line.substr(0, line.find(';'));
      }
      if (line.empty()) continue;
      std::istringstream iss(line);
      std::string word;
      std::string tmp_word;
      while (!iss.eof() || !tmp_word.empty()) {
        word.clear();
        if (tmp_word.empty()) {
          iss >> word;
        } else {
          word.swap(tmp_word);
        }
        if (word.empty()) continue;
        if (word[0] == '(') {
          if (cur_game_.empty()) {
            throw Exception("Variation with no move on line " +
                            std::to_string(line_no_) + " of the replay pgn.");
          }
          variation_stack_.push_back(cur_game_);
          cur_game_.pop_back();
          ReplayCurGame();
          tmp_word = word.substr(1);
          continue;
        } else if (word[0] == ')') {
          if (variation_stack_.empty()) {
            throw Exception("Too many closing parentheses on line " +
                            std::to_string(line_no_) + " of the replay pgn.");
          }
          if (variation_ply >= 0) {
            const int len = static_cast<int>(variation_stack_.back().size()) -
                            1 - variation_ply;
            if (len > 0) {
              int rule50;
              int moves;
              ChessBoard board;
              board.SetFromFen(cur_startpos_, &rule50, &moves);
              const int game_ply = 2 * moves - (board.flipped() ? 1 : 2);
              Position pos(board, rule50, game_ply);
              for (int i = 0; i < len; i++) {
                Move move = cur_game_[i];
                if (pos.IsBlackToMove()) move.Flip();
                pos = Position(pos, move);
              }
              games_.push_back(
                  {PositionToFen(pos),
                   MoveList(cur_game_.begin() + len, cur_game_.end()),
                   GameResult::UNDECIDED});
            } else {
              games_.push_back(
                  {cur_startpos_, cur_game_, GameResult::UNDECIDED});
            }
          }
          cur_game_ = variation_stack_.back();
          variation_stack_.pop_back();
          ReplayCurGame();
          tmp_word = word.substr(1);
          continue;
        }
        auto idx = word.find_first_of("()");
        if (idx != std::string::npos) {
          tmp_word = word.substr(idx);
          word.resize(idx);
        }
        // Trim move numbers from front ("12." and "12...").
        idx = word.find('.');
        if (idx != std::string::npos) {
          bool all_nums = true;
          for (size_t i = 0; i < idx; i++) {
            if (word[i] < '0' || word[i] > '9') {
              all_nums = false;
              break;
            }
          }
          if (all_nums) {
            word = word.substr(idx + 1);
            idx = word.find_first_not_of(".");
            if (idx == std::string::npos) continue;
            word = word.substr(idx);
          }
        }
        // Pure move numbers can be skipped (also empty).
        if (word.find_first_not_of("0123456789") == std::string::npos) continue;
        // Also skip spurious annotations.
        if (word.find_first_not_of("!?") == std::string::npos) continue;
        // Ignore "Numeric Annotation Glyph".
        if (word[0] == '$') continue;
        // Inside a variation that is not read.
        if (variation_ply < 0 && !variation_stack_.empty()) continue;
        // The result ends the game.
        if (word == "1/2-1/2" || word == "1-0" || word == "0-1" ||
            word == "*") {
          FlushGame(word);
          started = false;
          continue;
        }
        // "0-0" and "0-0-0" castle too.
        if (word.compare(0, 3, "0-0") == 0) {
          for (size_t i = 0; i < word.size() && i < 5; i++) {
            if (word[i] == '0') word[i] = 'O';
          }
        }
        Move move;
        try {
          move = SanToMove(word, cur_board_);
        } catch (const Exception& e) {
          throw Exception(std::string(e.what()) + " (move '" + word +
                          "', line " + std::to_string(line_no_) +
                          " of the replay pgn)");
        }
        const auto legal = cur_board_.GenerateLegalMoves();
        if (std::find(legal.begin(), legal.end(), move) == legal.end()) {
          throw Exception("Illegal move '" + word + "' on line " +
                          std::to_string(line_no_) + " of the replay pgn.");
        }
        cur_game_.push_back(move);
        cur_board_.ApplyMove(cur_game_.back());
        // The board wants the move in the mover's frame, the game list wants
        // it in the real one: flip it back when Black made it. (The book
        // reader assumes White moves first.)
        if (cur_board_.flipped()) cur_game_.back().Flip();
        cur_board_.Mirror();
        started = true;
      }
    }
    if (in_comment) {
      throw Exception("Comment reached the end of the replay pgn.");
    }
    if (started) {
      FlushGame("*");
    }
    gzclose(file);
  }

 private:
  void FlushGame(const std::string& termination) {
    if (!variation_stack_.empty()) {
      throw Exception("Variation not terminated on line " +
                      std::to_string(line_no_) + " of the replay pgn.");
    }
    auto result = GameResult::UNDECIDED;
    if (termination == "1/2-1/2") {
      result = GameResult::DRAW;
    } else if (termination == "1-0") {
      result = GameResult::WHITE_WON;
    } else if (termination == "0-1") {
      result = GameResult::BLACK_WON;
    }
    if (cur_lead_in_ > 0 &&
        cur_lead_in_ >= static_cast<int>(cur_game_.size())) {
      throw Exception("LeadInPlies leaves no move to replay in the game that "
                      "ends on line " +
                      std::to_string(line_no_) + " of the replay pgn.");
    }
    games_.push_back({cur_startpos_, cur_game_, result, cur_lead_in_});
    cur_game_.clear();
    cur_board_.SetFromFen(ChessBoard::kStartposFen);
    cur_startpos_ = ChessBoard::kStartposFen;
    cur_lead_in_ = 0;
  }

  // Sets the board to the position after cur_game_.
  void ReplayCurGame() {
    cur_board_.SetFromFen(cur_startpos_);
    for (auto move : cur_game_) {
      if (cur_board_.flipped()) move.Flip();
      cur_board_.ApplyMove(move);
      cur_board_.Mirror();
    }
  }

  int line_no_ = 0;
  int cur_lead_in_ = 0;
  std::vector<MoveList> variation_stack_;
};

}  // namespace lczero
