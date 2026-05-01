#include "search.h"
#include "eval.h"
#include <algorithm>
#include <vector>

// ── Move ordering ─────────────────────────────────────────────────────────
// MVV-LVA: most-valuable-victim / least-valuable-attacker heuristic.
// Captures are sorted before quiet moves; among captures, prefer taking
// high-value pieces with low-value attackers.
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

// ── Quiescence search ─────────────────────────────────────────────────────
// Extend search until the position is "quiet" (no captures left), preventing
// the horizon effect where evaluation cuts off mid-exchange.
static int quiesce(Board& b, int alpha, int beta, long long& nodes) {
    nodes++;

    // Static evaluation from current side's perspective
    int stand_pat = evaluate(b);
    if (!b.wtm) stand_pat = -stand_pat;

    if (stand_pat >= beta) return beta;
    if (stand_pat > alpha) alpha = stand_pat;

    auto moves = generate_pseudo_legal(b);
    order_moves(moves, b);

    for (const auto& m : moves) {
        if (!(m.flags & Move::CAPTURE)) continue;

        auto u = b.make_move(m);
        bool legal = !b.square_attacked(b.king_sq(!b.wtm), b.wtm);
        int score = 0;
        if (legal) score = -quiesce(b, -beta, -alpha, nodes);
        b.undo_move(m, u);

        if (!legal) continue;
        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }
    return alpha;
}

// ── Negamax with alpha-beta pruning ───────────────────────────────────────
// Returns a score from the current side's perspective.
// depth = remaining plies; ply = distance from root (for mate scoring).
int negamax(Board& b, int depth, int ply, int alpha, int beta, long long& nodes) {
    nodes++;

    if (depth == 0)
        return quiesce(b, alpha, beta, nodes);

    auto moves = generate_legal_moves(b);

    if (moves.empty())
        return b.in_check() ? -(SCORE_MATE - ply) : 0; // checkmate or stalemate

    order_moves(moves, b);

    for (const auto& m : moves) {
        auto u = b.make_move(m);
        int score = -negamax(b, depth - 1, ply + 1, -beta, -alpha, nodes);
        b.undo_move(m, u);

        if (score >= beta) return beta; // beta cutoff (fail-high)
        if (score > alpha) alpha = score;
    }
    return alpha;
}

// ── Root search ───────────────────────────────────────────────────────────
SearchResult search_serial(Board& board, int depth) {
    auto moves = generate_legal_moves(board);
    if (moves.empty()) return {};

    order_moves(moves, board);

    SearchResult res;
    res.best  = moves[0];
    int alpha = -SCORE_INF;

    for (const auto& m : moves) {
        auto u = board.make_move(m);
        int score = -negamax(board, depth - 1, 1, -SCORE_INF, -alpha, res.nodes);
        board.undo_move(m, u);

        if (score > alpha) {
            alpha    = score;
            res.best = m;
        }
    }
    res.score = alpha;
    return res;
}
