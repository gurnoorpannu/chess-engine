#pragma once
#include "board.h"
#include "movegen.h"
#include <string>

constexpr int SCORE_INF  = 1'000'000;
constexpr int SCORE_MATE = 900'000;   // checkmate score; subtract ply for faster mates

struct SearchResult {
    Move      best{Move::null()};
    int       score{0};
    long long nodes{0};
};

// Serial negamax + alpha-beta pruning
SearchResult search_serial(Board& board, int depth);

// OpenMP root-parallel search (requires -DUSE_OMP and OpenMP)
#ifdef USE_OMP
SearchResult search_omp(Board& board, int depth);
#endif

// CUDA batch-evaluation search (requires -DUSE_CUDA and nvcc)
#ifdef USE_CUDA
SearchResult search_cuda(Board& board, int depth);
#endif
