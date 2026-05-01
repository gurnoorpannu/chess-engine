#ifdef USE_OMP
#include "search.h"
#include "eval.h"
#ifdef __APPLE__
#include "/opt/homebrew/opt/libomp/include/omp.h"
#else
#include <omp.h>
#endif
#include <algorithm>
#include <atomic>
#include <vector>

// Shared declaration from search_serial.cpp
int negamax(Board& b, int depth, int ply, int alpha, int beta, long long& nodes);

static int move_score(const Move& m, const Board& b) {
    if (m.flags & Move::CAPTURE) {
        int victim   = MAT[piece_type(b.sq[m.to])];
        int attacker = MAT[piece_type(b.sq[m.from])];
        return 10000 + victim - attacker / 10;
    }
    return 0;
}

static void order_moves(std::vector<Move>& moves, const Board& b) {
    std::stable_sort(moves.begin(), moves.end(), [&](const Move& a, const Move& bm) {
        return move_score(a, b) > move_score(bm, b);
    });
}

// ── OpenMP root-parallel search ───────────────────────────────────────────
//
// Strategy:
//   1. Evaluate the first (best-ordered) root move serially to get a tight
//      alpha lower bound before spawning threads. This dramatically improves
//      beta-pruning efficiency in the parallel phase.
//   2. Remaining moves are split across threads. Each thread reads the
//      shared alpha atomically; completed threads publish their score so
//      subsequent threads start with a tighter window.
SearchResult search_omp(Board& board, int depth) {
    auto moves = generate_legal_moves(board);
    if (moves.empty()) return {};

    order_moves(moves, board);

    const int n = (int)moves.size();
    SearchResult res;
    res.best = moves[0];

    // Phase 1: search best move serially to establish alpha
    {
        auto u    = board.make_move(moves[0]);
        int score = -negamax(board, depth - 1, 1, -SCORE_INF, SCORE_INF, res.nodes);
        board.undo_move(moves[0], u);
        res.score = score;
    }

    if (n == 1) return res;

    std::atomic<int>       global_alpha{res.score};
    int                    best_idx = 0;
    std::atomic<long long> total_nodes{res.nodes};

    // Phase 2: parallelize remaining moves with alpha = best score so far
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 1; i < n; i++) {
        Board     local = board;
        long long nodes = 0;

        int  alpha = global_alpha.load(std::memory_order_relaxed);
        auto u     = local.make_move(moves[i]);
        int  score = -negamax(local, depth - 1, 1, -SCORE_INF, -alpha, nodes);
        local.undo_move(moves[i], u);

        total_nodes.fetch_add(nodes, std::memory_order_relaxed);

        #pragma omp critical
        {
            if (score > global_alpha.load()) {
                global_alpha.store(score);
                best_idx = i;
            }
        }
    }

    int final_alpha = global_alpha.load();
    if (final_alpha > res.score) {
        res.score = final_alpha;
        res.best  = moves[best_idx];
    }
    res.nodes = total_nodes.load();
    return res;
}

#endif // USE_OMP
