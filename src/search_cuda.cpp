#ifdef USE_CUDA
// search_cuda.cpp — Hybrid CPU/GPU search
//
// Strategy:
//   Alpha-beta runs on the CPU down to depth N-1. At depth 1 (one ply before
//   leaves), instead of calling negamax recursively, we generate ALL child
//   positions, batch-evaluate them on the GPU in one kernel launch, and feed
//   the GPU scores back into the minimax. This gives the GPU thousands of
//   positions to evaluate in parallel.

#include "search.h"
#include "eval.h"
#include "../cuda/eval_cuda.cuh"
#include <algorithm>
#include <vector>

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

// Shared from search_serial.cpp
int negamax(Board& b, int depth, int ply, int alpha, int beta, long long& nodes);

// At depth 1: collect all child positions, batch-evaluate on GPU,
// return the best score from the current side's perspective.
static int gpu_leaf(Board& b, int ply, int alpha, int beta, long long& nodes) {
    auto moves = generate_legal_moves(b);
    if (moves.empty())
        return b.in_check() ? -(SCORE_MATE - ply) : 0;

    order_moves(moves, b);

    // Build batch of child positions
    std::vector<BoardData> batch;
    batch.reserve(moves.size());
    std::vector<bool> legal_flags(moves.size(), false);

    for (int i = 0; i < (int)moves.size(); i++) {
        auto u = b.make_move(moves[i]);
        // Position is always legal here (generate_legal_moves already filtered)
        BoardData bd;
        std::copy(b.sq, b.sq + 64, bd.sq);
        bd.wtm = b.wtm ? 1 : -1;
        batch.push_back(bd);
        legal_flags[i] = true;
        b.undo_move(moves[i], u);
    }

    nodes += (long long)batch.size();

    // GPU evaluates all child positions in parallel
    auto gpu_scores = gpu_evaluate_batch(batch);

    // Minimax over GPU scores
    int best = -SCORE_INF;
    for (int i = 0; i < (int)moves.size(); i++) {
        if (!legal_flags[i]) continue;
        // gpu_scores[i] is from the child's side perspective; negate for parent
        int score = -gpu_scores[i];
        if (score >= beta) return beta;
        if (score > best) best = score;
        if (score > alpha) alpha = score;
    }
    return best;
}

// Hybrid negamax: CPU alpha-beta at higher depths, GPU batch at depth == 1
static int negamax_cuda(Board& b, int depth, int ply, int alpha, int beta, long long& nodes) {
    nodes++;

    if (depth == 1)
        return gpu_leaf(b, ply, alpha, beta, nodes);

    if (depth == 0) {
        // Quiescence not wrapped in GPU here; fall back to CPU evaluate
        int score = evaluate(b);
        return b.wtm ? score : -score;
    }

    auto moves = generate_legal_moves(b);
    if (moves.empty())
        return b.in_check() ? -(SCORE_MATE - ply) : 0;

    order_moves(moves, b);

    for (const auto& m : moves) {
        auto u = b.make_move(m);
        int score = -negamax_cuda(b, depth - 1, ply + 1, -beta, -alpha, nodes);
        b.undo_move(m, u);

        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }
    return alpha;
}

// ── Root search ───────────────────────────────────────────────────────────
SearchResult search_cuda(Board& board, int depth) {
    cuda_init();

    auto moves = generate_legal_moves(board);
    if (moves.empty()) return {};

    order_moves(moves, board);

    SearchResult res;
    res.best  = moves[0];
    int alpha = -SCORE_INF;

    for (const auto& m : moves) {
        auto u = board.make_move(m);
        int score = -negamax_cuda(board, depth - 1, 1, -SCORE_INF, -alpha, res.nodes);
        board.undo_move(m, u);

        if (score > alpha) {
            alpha    = score;
            res.best = m;
        }
    }
    res.score = alpha;
    return res;
}

#endif // USE_CUDA
