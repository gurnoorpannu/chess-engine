#pragma once
#include <cstdint>
#include <string>

// Square layout: A1=0, B1=1, ..., H1=7, A2=8, ..., H8=63
//   rank 8 → squares 56-63 (top of board)
//   rank 1 → squares  0- 7 (bottom of board)
inline constexpr int rank_of(int s) { return s >> 3; }
inline constexpr int file_of(int s) { return s & 7; }
inline constexpr int make_sq(int r, int f) { return (r << 3) | f; }

enum : int {
    A1=0, B1, C1, D1, E1, F1, G1, H1,
    A2=8, B2, C2, D2, E2, F2, G2, H2,
    A3=16,B3, C3, D3, E3, F3, G3, H3,
    A4=24,B4, C4, D4, E4, F4, G4, H4,
    A5=32,B5, C5, D5, E5, F5, G5, H5,
    A6=40,B6, C6, D6, E6, F6, G6, H6,
    A7=48,B7, C7, D7, E7, F7, G7, H7,
    A8=56,B8, C8, D8, E8, F8, G8, H8,
};

// Piece encoding: positive=white, negative=black, 0=empty
// |piece|: 1=pawn 2=knight 3=bishop 4=rook 5=queen 6=king
enum Piece : int8_t {
    B_KING=-6, B_QUEEN=-5, B_ROOK=-4, B_BISHOP=-3, B_KNIGHT=-2, B_PAWN=-1,
    EMPTY=0,
    W_PAWN=1, W_KNIGHT=2, W_BISHOP=3, W_ROOK=4, W_QUEEN=5, W_KING=6
};

inline int8_t piece_type(int8_t p) { return p < 0 ? -p : p; }

// Castling rights bit flags
constexpr uint8_t CR_WK = 1, CR_WQ = 2, CR_BK = 4, CR_BQ = 8;

struct Move {
    uint8_t from;
    uint8_t to;
    int8_t  promo;  // 0 or piece type (always positive: KNIGHT/BISHOP/ROOK/QUEEN)
    uint8_t flags;

    enum Flags : uint8_t {
        CAPTURE  = 0x01,
        EN_PASS  = 0x02,
        CASTLE   = 0x04,
        DBL_PUSH = 0x08,
    };

    bool is_null() const { return from == 0 && to == 0 && promo == 0 && flags == 0; }
    static Move null() { return {0, 0, 0, 0}; }
};

struct UndoInfo {
    int8_t  captured;
    uint8_t castle;
    int8_t  ep;
    int     half;
};

struct Board {
    int8_t  sq[64]{};
    bool    wtm{true};   // white to move
    uint8_t castle{0};   // castling rights
    int8_t  ep{-1};      // en passant target square, -1 = none
    int     half{0};     // halfmove clock (50-move rule)
    int     full{1};     // fullmove number

    void        reset();
    void        set_startpos();
    bool        load_fen(const std::string& fen);
    std::string to_fen() const;
    void        print() const;

    UndoInfo make_move(const Move& m);
    void     undo_move(const Move& m, const UndoInfo& u);

    int  king_sq(bool white) const;
    bool square_attacked(int s, bool by_white) const;
    bool in_check() const;
};
