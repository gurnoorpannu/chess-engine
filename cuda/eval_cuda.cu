// eval_cuda.cu — GPU batch evaluation of chess positions
//
// Architecture:
//   Each CUDA thread evaluates one board position using material + PST scores.
//   The CPU hands off a flat array of board states; the GPU evaluates them all
//   in parallel and returns an array of integer scores.
//
// This file is compiled with nvcc; everything else with g++/clang.

#include "eval_cuda.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t _e = (call); \
        if (_e != cudaSuccess) \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_e)); \
    } while (0)

// ── Device-side constants ─────────────────────────────────────────────────
// Stored in fast __constant__ memory (broadcast to all threads).

__constant__ int d_MAT[7] = {0, 100, 320, 330, 500, 900, 20000};

// PST[piece_type][square], A1=0 ... H8=63, white's perspective.
// For a black piece at square s, index with (s ^ 56) to mirror vertically.
__constant__ int d_PST[7][64] = {
    // [0] unused
    {},
    // [1] Pawn
    {
         0,  0,  0,  0,  0,  0,  0,  0,
         5, 10, 10,-20,-20, 10, 10,  5,
         5, -5,-10,  0,  0,-10, -5,  5,
         0,  0,  0, 20, 20,  0,  0,  0,
         5,  5, 10, 25, 25, 10,  5,  5,
        10, 10, 20, 30, 30, 20, 10, 10,
        50, 50, 50, 50, 50, 50, 50, 50,
         0,  0,  0,  0,  0,  0,  0,  0,
    },
    // [2] Knight
    {
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50,
    },
    // [3] Bishop
    {
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -20,-10,-10,-10,-10,-10,-10,-20,
    },
    // [4] Rook
    {
          0,  0,  0,  5,  5,  0,  0,  0,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
          5, 10, 10, 10, 10, 10, 10,  5,
          0,  0,  0,  0,  0,  0,  0,  0,
    },
    // [5] Queen
    {
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -10,  5,  5,  5,  5,  5,  0,-10,
          0,  0,  5,  5,  5,  5,  0, -5,
         -5,  0,  5,  5,  5,  5,  0, -5,
        -10,  0,  5,  5,  5,  5,  0,-10,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20,
    },
    // [6] King (middlegame)
    {
         20, 30, 10,  0,  0, 10, 30, 20,
         20, 20,  0,  0,  0,  0, 20, 20,
        -10,-20,-20,-20,-20,-20,-20,-10,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
    },
};

// ── GPU evaluation kernel ─────────────────────────────────────────────────
//
// Each thread evaluates ONE board position.
// boards:   flat array of (n * 64) int8_t — board states packed back-to-back
// wtm_arr:  n int8_t — side to move (1=white, -1=black)
// scores:   n int    — output scores (current side's perspective)
// n:        number of positions
__global__ void eval_kernel(
    const int8_t* __restrict__ boards,
    const int8_t* __restrict__ wtm_arr,
    int*          __restrict__ scores,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const int8_t* board = boards + idx * 64;
    int score = 0;

    for (int s = 0; s < 64; s++) {
        int8_t p = board[s];
        if (p == 0) continue;

        int sign   = (p > 0) ? 1 : -1;
        int pt     = sign * p;            // piece type: 1-6
        int pst_sq = (p > 0) ? s : (s ^ 56); // mirror for black

        score += sign * (d_MAT[pt] + d_PST[pt][pst_sq]);
    }

    // Convert to current-side perspective
    scores[idx] = (wtm_arr[idx] > 0) ? score : -score;
}

// ── Host API ──────────────────────────────────────────────────────────────

void cuda_init() {
    // Warm up the CUDA runtime (first call to any CUDA function is slow)
    cudaFree(nullptr);
}

void cuda_cleanup() {
    cudaDeviceReset();
}

std::vector<int> gpu_evaluate_batch(const std::vector<BoardData>& positions) {
    if (positions.empty()) return {};

    const int n = (int)positions.size();

    // Flatten into contiguous arrays for easy cudaMemcpy
    std::vector<int8_t> h_boards(n * 64);
    std::vector<int8_t> h_wtm(n);
    for (int i = 0; i < n; i++) {
        std::memcpy(h_boards.data() + i * 64, positions[i].sq, 64);
        h_wtm[i] = positions[i].wtm;
    }

    // Allocate device buffers
    int8_t *d_boards = nullptr, *d_wtm = nullptr;
    int    *d_scores = nullptr;

    CUDA_CHECK(cudaMalloc(&d_boards, n * 64 * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_wtm,    n      * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_scores, n      * sizeof(int)));

    // Transfer input to GPU
    CUDA_CHECK(cudaMemcpy(d_boards, h_boards.data(), n * 64, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wtm,    h_wtm.data(),    n,      cudaMemcpyHostToDevice));

    // Launch: 256 threads per block, ceil(n/256) blocks
    constexpr int BLOCK = 256;
    int grid = (n + BLOCK - 1) / BLOCK;
    eval_kernel<<<grid, BLOCK>>>(d_boards, d_wtm, d_scores, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Retrieve results
    std::vector<int> scores(n);
    CUDA_CHECK(cudaMemcpy(scores.data(), d_scores, n * sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_boards);
    cudaFree(d_wtm);
    cudaFree(d_scores);

    return scores;
}
