#pragma once
#include "board.h"

// Static evaluation from white's perspective (centipawns).
// Positive = good for white, negative = good for black.
int evaluate(const Board& b);

// Piece material values
constexpr int MAT[7] = {0, 100, 320, 330, 500, 900, 20000};

// Piece-square tables (A1=0 ... H8=63, white's perspective).
// For black pieces, mirror vertically: sq ^ 56
extern const int PST[7][64];
