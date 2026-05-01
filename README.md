# Chess Engine Benchmark

This project is a compact chess engine benchmark with three search paths:

- `serial`: single-threaded alpha-beta search
- `openmp`: root-parallel CPU search
- `cuda`: hybrid CPU search with GPU batch evaluation at the leaf frontier

It also includes [`gen_notebook.py`](/Users/gurnoor/Projects/chess/gen_notebook.py), which generates a Colab notebook that rebuilds the project from source, verifies perft counts, and runs a benchmark sweep.

## Project Layout

- [`src/main.cpp`](/Users/gurnoor/Projects/chess/src/main.cpp): CLI entry point, perft runner, benchmark report, HTML export, and minimal UCI loop
- [`src/search_serial.cpp`](/Users/gurnoor/Projects/chess/src/search_serial.cpp): serial alpha-beta and quiescence search
- [`src/search_omp.cpp`](/Users/gurnoor/Projects/chess/src/search_omp.cpp): OpenMP root-parallel search
- [`src/search_cuda.cpp`](/Users/gurnoor/Projects/chess/src/search_cuda.cpp): CUDA-assisted search
- [`cuda/eval_cuda.cu`](/Users/gurnoor/Projects/chess/cuda/eval_cuda.cu): GPU batch evaluator

## Build Locally

CPU-only:

```bash
cmake -S . -B build -DUSE_OMP=ON -DUSE_CUDA=OFF
cmake --build build -j
```

With CUDA:

```bash
cmake -S . -B build -DUSE_OMP=ON -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build -j
```

On Apple Silicon, `USE_OMP=ON` expects Homebrew `libomp` in `/opt/homebrew/opt/libomp`.

## Run

Benchmark one position:

```bash
./build/chess bench --depth 6
```

Sweep depths and write an HTML report:

```bash
./build/chess report --maxdepth 8 --out perf_report.html
```

Verify move generation:

```bash
./build/chess perft --depth 5 --startpos
./build/chess perft --depth 4 --fen "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
```

Minimal UCI mode:

```bash
./build/chess uci
```

## Colab Workflow

1. Run `python3 gen_notebook.py`.
2. Open `chess_cuda_benchmark.ipynb`.
3. Upload the notebook to Colab.
4. Switch the runtime to GPU.
5. Run the notebook top to bottom.

The generated notebook detects the GPU compute capability, writes the embedded source tree into `/content/chess`, builds with CMake, verifies perft counts, and saves benchmark artifacts including:

- `/content/chess/benchmark_output.txt`
- `/content/chess/benchmark_summary.csv`
- `/content/chess/chess_benchmark.png`
- `/content/chess/build/perf_report.html`
