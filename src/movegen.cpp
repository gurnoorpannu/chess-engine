#include "movegen.h"
#include <vector>

using ML = std::vector<Move>;

// ── Pawns ─────────────────────────────────────────────────────────────────

static void push_promos(ML& ml, uint8_t from, uint8_t to, uint8_t flags) {
    for (int8_t p : {W_QUEEN, W_ROOK, W_BISHOP, W_KNIGHT})
        ml.push_back({from, to, p, flags});
}

static void gen_pawns(const Board& b, ML& ml) {
    const int    dir    = b.wtm ?  8 : -8;
    const int    start  = b.wtm ?  1 :  6;  // starting rank index
    const int    prank  = b.wtm ?  6 :  1;  // rank before promotion
    const int8_t pawn   = b.wtm ? W_PAWN : B_PAWN;

    for (int s = 0; s < 64; s++) {
        if (b.sq[s] != pawn) continue;
        const int r = rank_of(s), f = file_of(s);
        const int fwd = s + dir;

        // Forward push
        if (b.sq[fwd] == EMPTY) {
            if (r == prank)
                push_promos(ml, s, fwd, 0);
            else
                ml.push_back({(uint8_t)s, (uint8_t)fwd, 0, 0});

            // Double push from starting rank
            if (r == start && b.sq[fwd + dir] == EMPTY)
                ml.push_back({(uint8_t)s, (uint8_t)(fwd + dir), 0, Move::DBL_PUSH});
        }

        // Diagonal captures and en passant
        for (int df : {-1, 1}) {
            int nf = f + df;
            if (nf < 0 || nf >= 8) continue;
            int cap = fwd + df;

            if (cap == (int)b.ep) {
                ml.push_back({(uint8_t)s, (uint8_t)cap, 0, Move::EN_PASS | Move::CAPTURE});
            } else if (b.sq[cap] != EMPTY && ((b.sq[cap] > 0) != b.wtm)) {
                // Enemy piece
                if (r == prank)
                    push_promos(ml, s, cap, Move::CAPTURE);
                else
                    ml.push_back({(uint8_t)s, (uint8_t)cap, 0, Move::CAPTURE});
            }
        }
    }
}

// ── Knights ───────────────────────────────────────────────────────────────

static void gen_knights(const Board& b, ML& ml) {
    static const int8_t DR[8] = {-2,-2,-1,-1, 1, 1, 2, 2};
    static const int8_t DF[8] = {-1, 1,-2, 2,-2, 2,-1, 1};
    const int8_t knight = b.wtm ? W_KNIGHT : B_KNIGHT;

    for (int s = 0; s < 64; s++) {
        if (b.sq[s] != knight) continue;
        int r = rank_of(s), f = file_of(s);
        for (int i = 0; i < 8; i++) {
            int nr = r + DR[i], nf = f + DF[i];
            if (nr < 0 || nr >= 8 || nf < 0 || nf >= 8) continue;
            int ns = make_sq(nr, nf);
            if (b.sq[ns] != EMPTY && ((b.sq[ns] > 0) == b.wtm)) continue; // friendly
            uint8_t flags = (b.sq[ns] != EMPTY) ? Move::CAPTURE : 0;
            ml.push_back({(uint8_t)s, (uint8_t)ns, 0, flags});
        }
    }
}

// ── Sliding pieces ────────────────────────────────────────────────────────

static void slide(const Board& b, ML& ml, int s, const int8_t* dr, const int8_t* df, int ndirs) {
    int r = rank_of(s), f = file_of(s);
    for (int d = 0; d < ndirs; d++) {
        int cr = r + dr[d], cf = f + df[d];
        while (cr >= 0 && cr < 8 && cf >= 0 && cf < 8) {
            int ns = make_sq(cr, cf);
            if (b.sq[ns] != EMPTY) {
                if ((b.sq[ns] > 0) != b.wtm) // enemy
                    ml.push_back({(uint8_t)s, (uint8_t)ns, 0, Move::CAPTURE});
                break;
            }
            ml.push_back({(uint8_t)s, (uint8_t)ns, 0, 0});
            cr += dr[d]; cf += df[d];
        }
    }
}

static void gen_bishops(const Board& b, ML& ml) {
    static const int8_t DR[4] = { 1, 1,-1,-1};
    static const int8_t DF[4] = { 1,-1, 1,-1};
    const int8_t bishop = b.wtm ? W_BISHOP : B_BISHOP;
    for (int s = 0; s < 64; s++)
        if (b.sq[s] == bishop) slide(b, ml, s, DR, DF, 4);
}

static void gen_rooks(const Board& b, ML& ml) {
    static const int8_t DR[4] = { 1,-1, 0, 0};
    static const int8_t DF[4] = { 0, 0, 1,-1};
    const int8_t rook = b.wtm ? W_ROOK : B_ROOK;
    for (int s = 0; s < 64; s++)
        if (b.sq[s] == rook) slide(b, ml, s, DR, DF, 4);
}

static void gen_queens(const Board& b, ML& ml) {
    static const int8_t DR[8] = { 1,-1, 0, 0, 1, 1,-1,-1};
    static const int8_t DF[8] = { 0, 0, 1,-1, 1,-1, 1,-1};
    const int8_t queen = b.wtm ? W_QUEEN : B_QUEEN;
    for (int s = 0; s < 64; s++)
        if (b.sq[s] == queen) slide(b, ml, s, DR, DF, 8);
}

// ── King ──────────────────────────────────────────────────────────────────

static void gen_king(const Board& b, ML& ml) {
    const int8_t king = b.wtm ? W_KING : B_KING;

    for (int s = 0; s < 64; s++) {
        if (b.sq[s] != king) continue;
        int r = rank_of(s), f = file_of(s);

        // Normal king moves
        for (int dr = -1; dr <= 1; dr++)
        for (int df = -1; df <= 1; df++) {
            if (!dr && !df) continue;
            int nr = r + dr, nf = f + df;
            if (nr < 0 || nr >= 8 || nf < 0 || nf >= 8) continue;
            int ns = make_sq(nr, nf);
            if (b.sq[ns] != EMPTY && ((b.sq[ns] > 0) == b.wtm)) continue; // friendly
            uint8_t flags = (b.sq[ns] != EMPTY) ? Move::CAPTURE : 0;
            ml.push_back({(uint8_t)s, (uint8_t)ns, 0, flags});
        }

        // Castling (also checks that king doesn't pass through check)
        bool opp = !b.wtm;
        if (b.wtm) {
            if ((b.castle & CR_WK) && b.sq[F1] == EMPTY && b.sq[G1] == EMPTY
                && !b.square_attacked(E1, opp)
                && !b.square_attacked(F1, opp)
                && !b.square_attacked(G1, opp))
                ml.push_back({E1, G1, 0, Move::CASTLE});

            if ((b.castle & CR_WQ) && b.sq[D1] == EMPTY && b.sq[C1] == EMPTY && b.sq[B1] == EMPTY
                && !b.square_attacked(E1, opp)
                && !b.square_attacked(D1, opp)
                && !b.square_attacked(C1, opp))
                ml.push_back({E1, C1, 0, Move::CASTLE});
        } else {
            if ((b.castle & CR_BK) && b.sq[F8] == EMPTY && b.sq[G8] == EMPTY
                && !b.square_attacked(E8, opp)
                && !b.square_attacked(F8, opp)
                && !b.square_attacked(G8, opp))
                ml.push_back({E8, G8, 0, Move::CASTLE});

            if ((b.castle & CR_BQ) && b.sq[D8] == EMPTY && b.sq[C8] == EMPTY && b.sq[B8] == EMPTY
                && !b.square_attacked(E8, opp)
                && !b.square_attacked(D8, opp)
                && !b.square_attacked(C8, opp))
                ml.push_back({E8, C8, 0, Move::CASTLE});
        }
        break; // only one king
    }
}

// ── Public API ────────────────────────────────────────────────────────────

std::vector<Move> generate_pseudo_legal(const Board& b) {
    ML ml;
    ml.reserve(48);
    gen_pawns(b, ml);
    gen_knights(b, ml);
    gen_bishops(b, ml);
    gen_rooks(b, ml);
    gen_queens(b, ml);
    gen_king(b, ml);
    return ml;
}

std::vector<Move> generate_legal_moves(Board& b) {
    auto pseudo = generate_pseudo_legal(b);
    ML legal;
    legal.reserve(pseudo.size());
    for (const auto& m : pseudo) {
        auto u = b.make_move(m);
        // After make_move, b.wtm is flipped; check king of the side that just moved
        if (!b.square_attacked(b.king_sq(!b.wtm), b.wtm))
            legal.push_back(m);
        b.undo_move(m, u);
    }
    return legal;
}

std::string move_to_str(const Move& m) {
    std::string s;
    s += char('a' + file_of(m.from));
    s += char('1' + rank_of(m.from));
    s += char('a' + file_of(m.to));
    s += char('1' + rank_of(m.to));
    if (m.promo) {
        const char* pchars = "  nbrq";
        s += pchars[m.promo]; // promo is 2=N,3=B,4=R,5=Q
    }
    return s;
}
