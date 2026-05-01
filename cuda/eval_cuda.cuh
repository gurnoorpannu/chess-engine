#pragma once
#include <cstdint>
#include <vector>

// Compact board state for GPU transfer (64 bytes + 1 byte side-to-move)
struct BoardData {
    int8_t sq[64];
    int8_t wtm; // 1 = white to move, -1 = black to move
};

// Evaluate a batch of board positions on the GPU.
// Returns a vector of scores (centipawns) from each position's current
// side-to-move perspective (positive = good for the side to move).
std::vector<int> gpu_evaluate_batch(const std::vector<BoardData>& positions);

// Call once at program start / end if CUDA is available
void cuda_init();
void cuda_cleanup();
