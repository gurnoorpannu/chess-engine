#pragma once
#include "board.h"
#include <vector>

// Generate all pseudo-legal moves (may leave own king in check)
std::vector<Move> generate_pseudo_legal(const Board& b);

// Generate all legal moves (filters out moves that leave own king in check)
std::vector<Move> generate_legal_moves(Board& b);

// Format a move as UCI string, e.g. "e2e4", "e7e8q"
std::string move_to_str(const Move& m);
