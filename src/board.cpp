#include "board.h"
#include <cctype>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <algorithm>

void Board::reset() {
    std::fill(sq, sq + 64, (int8_t)EMPTY);
    wtm    = true;
    castle = 0;
    ep     = -1;
    half   = 0;
    full   = 1;
}

void Board::set_startpos() {
    load_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

bool Board::load_fen(const std::string& fen) {
    reset();
    std::istringstream ss(fen);
    std::string pieces, side, castle_str, ep_str;
    int hm = 0, fm = 1;

    if (!(ss >> pieces >> side >> castle_str >> ep_str >> hm >> fm))
        return false;

    // Piece placement: FEN starts from rank 8 (r=7) down to rank 1 (r=0)
    int r = 7, f = 0;
    for (char c : pieces) {
        if (c == '/') { r--; f = 0; }
        else if (std::isdigit(c)) { f += c - '0'; }
        else {
            int8_t p = EMPTY;
            switch (std::toupper(c)) {
                case 'P': p = W_PAWN;   break;
                case 'N': p = W_KNIGHT; break;
                case 'B': p = W_BISHOP; break;
                case 'R': p = W_ROOK;   break;
                case 'Q': p = W_QUEEN;  break;
                case 'K': p = W_KING;   break;
            }
            if (std::islower(c)) p = -p;
            sq[make_sq(r, f++)] = p;
        }
    }

    wtm = (side == "w");

    castle = 0;
    for (char c : castle_str) {
        switch (c) {
            case 'K': castle |= CR_WK; break;
            case 'Q': castle |= CR_WQ; break;
            case 'k': castle |= CR_BK; break;
            case 'q': castle |= CR_BQ; break;
        }
    }

    ep = -1;
    if (ep_str != "-")
        ep = make_sq(ep_str[1] - '1', ep_str[0] - 'a');

    half = hm;
    full = fm;
    return true;
}

std::string Board::to_fen() const {
    std::string fen;
    for (int r = 7; r >= 0; r--) {
        int empty = 0;
        for (int f = 0; f < 8; f++) {
            int8_t p = sq[make_sq(r, f)];
            if (p == EMPTY) { empty++; continue; }
            if (empty) { fen += char('0' + empty); empty = 0; }
            int8_t t = piece_type(p);
            char c = "PNBRQK"[t - 1];
            fen += (p > 0) ? c : char(std::tolower(c));
        }
        if (empty) fen += char('0' + empty);
        if (r > 0)  fen += '/';
    }
    fen += wtm ? " w " : " b ";
    std::string cr;
    if (castle & CR_WK) cr += 'K';
    if (castle & CR_WQ) cr += 'Q';
    if (castle & CR_BK) cr += 'k';
    if (castle & CR_BQ) cr += 'q';
    fen += cr.empty() ? "-" : cr;
    fen += ' ';
    if (ep < 0) fen += '-';
    else { fen += char('a' + file_of(ep)); fen += char('1' + rank_of(ep)); }
    fen += ' '; fen += std::to_string(half);
    fen += ' '; fen += std::to_string(full);
    return fen;
}

void Board::print() const {
    printf("\n");
    for (int r = 7; r >= 0; r--) {
        printf(" %d | ", r + 1);
        for (int f = 0; f < 8; f++) {
            int8_t p = sq[make_sq(r, f)];
            char c;
            if (p == 0) c = '.';
            else if (p > 0) c = "PNBRQK"[p - 1];
            else             c = "pnbrqk"[-p - 1];
            printf("%c ", c);
        }
        printf("\n");
    }
    printf("     ----------------\n");
    printf("     a b c d e f g h\n\n");
    char ep_str[3] = "-";
    if (ep >= 0) { ep_str[0] = 'a' + file_of(ep); ep_str[1] = '1' + rank_of(ep); ep_str[2] = '\0'; }
    printf("%s to move | castle: %c%c%c%c | ep: %s | half: %d | full: %d\n",
           wtm ? "White" : "Black",
           (castle & CR_WK) ? 'K' : '-',
           (castle & CR_WQ) ? 'Q' : '-',
           (castle & CR_BK) ? 'k' : '-',
           (castle & CR_BQ) ? 'q' : '-',
           ep_str, half, full);
}

// Revoke castling rights when pieces move to/from key squares
static void revoke_castle(uint8_t& castle, int sq_idx) {
    switch (sq_idx) {
        case E1: castle &= ~(CR_WK | CR_WQ); break;
        case E8: castle &= ~(CR_BK | CR_BQ); break;
        case A1: castle &= ~CR_WQ; break;
        case H1: castle &= ~CR_WK; break;
        case A8: castle &= ~CR_BQ; break;
        case H8: castle &= ~CR_BK; break;
        default: break;
    }
}

UndoInfo Board::make_move(const Move& m) {
    UndoInfo u{sq[m.to], castle, ep, half};
    int8_t moving = sq[m.from];

    ep = -1;

    // Halfmove clock: reset on pawn move or capture
    if (piece_type(moving) == 1 || (m.flags & Move::CAPTURE))
        half = 0;
    else
        half++;

    // En passant capture: captured pawn is not on m.to
    if (m.flags & Move::EN_PASS) {
        int cap_sq = m.to + (wtm ? -8 : 8);
        u.captured = sq[cap_sq];
        sq[cap_sq] = EMPTY;
    }

    // Move piece (handle promotion)
    if (m.promo)
        sq[m.to] = wtm ? m.promo : -m.promo;
    else
        sq[m.to] = moving;
    sq[m.from] = EMPTY;

    // Move rook when castling
    if (m.flags & Move::CASTLE) {
        switch (m.to) {
            case G1: sq[F1] = W_ROOK; sq[H1] = EMPTY; break;
            case C1: sq[D1] = W_ROOK; sq[A1] = EMPTY; break;
            case G8: sq[F8] = B_ROOK; sq[H8] = EMPTY; break;
            case C8: sq[D8] = B_ROOK; sq[A8] = EMPTY; break;
        }
    }

    // Set en passant square for double pawn push
    if (m.flags & Move::DBL_PUSH)
        ep = m.to + (wtm ? -8 : 8);

    // Update castling rights based on squares touched
    revoke_castle(castle, m.from);
    revoke_castle(castle, m.to);

    if (!wtm) full++;
    wtm = !wtm;
    return u;
}

void Board::undo_move(const Move& m, const UndoInfo& u) {
    wtm = !wtm;
    if (!wtm) full--;

    // The piece on m.to might be a promoted piece — restore original pawn
    int8_t moving = m.promo ? (wtm ? W_PAWN : B_PAWN) : sq[m.to];

    sq[m.from] = moving;

    // Restore destination (may be EMPTY for EP, otherwise captured piece)
    sq[m.to] = (m.flags & Move::EN_PASS) ? EMPTY : u.captured;

    // Restore en passant captured pawn
    if (m.flags & Move::EN_PASS) {
        int cap_sq = m.to + (wtm ? -8 : 8);
        sq[cap_sq] = u.captured;
    }

    // Undo castling rook move
    if (m.flags & Move::CASTLE) {
        switch (m.to) {
            case G1: sq[H1] = W_ROOK; sq[F1] = EMPTY; break;
            case C1: sq[A1] = W_ROOK; sq[D1] = EMPTY; break;
            case G8: sq[H8] = B_ROOK; sq[F8] = EMPTY; break;
            case C8: sq[A8] = B_ROOK; sq[D8] = EMPTY; break;
        }
    }

    castle = u.castle;
    ep     = u.ep;
    half   = u.half;
}

int Board::king_sq(bool white) const {
    int8_t king = white ? W_KING : B_KING;
    for (int s = 0; s < 64; s++)
        if (sq[s] == king) return s;
    return -1;
}

bool Board::square_attacked(int s, bool by_white) const {
    const int8_t sign = by_white ? 1 : -1;
    const int r = rank_of(s), f = file_of(s);

    // Pawn attacks: check if a pawn of the attacking side attacks s
    // A white pawn on (r-1, f±1) attacks s
    {
        int pr = by_white ? r - 1 : r + 1;
        if (pr >= 0 && pr < 8) {
            for (int df : {-1, 1}) {
                int pf = f + df;
                if (pf >= 0 && pf < 8 && sq[make_sq(pr, pf)] == sign * W_PAWN)
                    return true;
            }
        }
    }

    // Knight attacks
    static const int8_t KNR[8] = {-2,-2,-1,-1, 1, 1, 2, 2};
    static const int8_t KNF[8] = {-1, 1,-2, 2,-2, 2,-1, 1};
    for (int i = 0; i < 8; i++) {
        int nr = r + KNR[i], nf = f + KNF[i];
        if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8)
            if (sq[make_sq(nr, nf)] == sign * W_KNIGHT) return true;
    }

    // Diagonal sliders (bishop / queen)
    static const int8_t DR[4] = { 1, 1,-1,-1};
    static const int8_t DF[4] = { 1,-1, 1,-1};
    for (int d = 0; d < 4; d++) {
        int cr = r + DR[d], cf = f + DF[d];
        while (cr >= 0 && cr < 8 && cf >= 0 && cf < 8) {
            int8_t p = sq[make_sq(cr, cf)];
            if (p != EMPTY) {
                if (p == sign * W_BISHOP || p == sign * W_QUEEN) return true;
                break;
            }
            cr += DR[d]; cf += DF[d];
        }
    }

    // Straight sliders (rook / queen)
    static const int8_t SR[4] = { 1,-1, 0, 0};
    static const int8_t SF[4] = { 0, 0, 1,-1};
    for (int d = 0; d < 4; d++) {
        int cr = r + SR[d], cf = f + SF[d];
        while (cr >= 0 && cr < 8 && cf >= 0 && cf < 8) {
            int8_t p = sq[make_sq(cr, cf)];
            if (p != EMPTY) {
                if (p == sign * W_ROOK || p == sign * W_QUEEN) return true;
                break;
            }
            cr += SR[d]; cf += SF[d];
        }
    }

    // King attacks
    for (int dr = -1; dr <= 1; dr++)
    for (int df = -1; df <= 1; df++) {
        if (!dr && !df) continue;
        int nr = r + dr, nf = f + df;
        if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8)
            if (sq[make_sq(nr, nf)] == sign * W_KING) return true;
    }

    return false;
}

bool Board::in_check() const {
    return square_attacked(king_sq(wtm), !wtm);
}
