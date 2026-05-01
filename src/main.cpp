// main.cpp — Chess engine driver
//
// Modes:
//   bench  [--fen FEN] [--depth N]  — benchmark serial / OpenMP / CUDA on one position
//   report [--fen FEN] [--maxdepth N] — sweep depths, print table, write perf_report.html
//   perft  [--fen FEN] [--depth N]  — move-generation correctness test (node counts)
//   play   [--fen FEN] [--depth N]  — engine plays from position (prints best move)
//   uci                             — minimal UCI loop

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "board.h"
#include "eval.h"
#include "movegen.h"
#include "search.h"

#ifdef USE_OMP
#ifdef __APPLE__
#include "/opt/homebrew/opt/libomp/include/omp.h"
#else
#include <omp.h>
#endif
#endif

#ifdef USE_CUDA
#include "../cuda/eval_cuda.cuh"
#endif

using Clock = std::chrono::high_resolution_clock;
inline double elapsed_ms(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ── Perft ─────────────────────────────────────────────────────────────────

static long long perft(Board& b, int depth) {
    if (depth == 0) return 1;
    auto moves = generate_legal_moves(b);
    if (depth == 1) return (long long)moves.size();
    long long total = 0;
    for (const auto& m : moves) {
        auto u = b.make_move(m);
        total += perft(b, depth - 1);
        b.undo_move(m, u);
    }
    return total;
}

static void run_perft(const std::string& fen, int depth) {
    Board b;
    b.load_fen(fen);
    b.print();
    printf("\nPerft depth %d:\n", depth);
    auto  moves = generate_legal_moves(b);
    long long total = 0;
    auto t0 = Clock::now();
    for (const auto& m : moves) {
        auto u   = b.make_move(m);
        long long cnt = perft(b, depth - 1);
        b.undo_move(m, u);
        printf("  %s: %lld\n", move_to_str(m).c_str(), cnt);
        total += cnt;
    }
    double ms = elapsed_ms(t0);
    printf("\nTotal: %lld  Time: %.1f ms  NPS: %.0f\n", total, ms, total / (ms / 1000.0));
}

// ── Single-depth benchmark ────────────────────────────────────────────────

static void print_result(const char* tag, const SearchResult& r, double ms) {
    printf("  [%-6s]  move: %-6s  score: %+6d  nodes: %8lld  time: %8.1f ms  knps: %6.0f\n",
           tag, move_to_str(r.best).c_str(), r.score, r.nodes, ms, r.nodes / ms);
}

static void run_bench(const std::string& fen, int depth) {
    Board b;
    b.load_fen(fen);
    b.print();
    printf("\nSearching depth %d ...\n\n", depth);

    { Board l = b; auto t = Clock::now(); auto r = search_serial(l, depth); print_result("serial", r, elapsed_ms(t)); }
#ifdef USE_OMP
    { Board l = b; auto t = Clock::now(); auto r = search_omp(l, depth);    print_result("openmp", r, elapsed_ms(t)); }
#endif
#ifdef USE_CUDA
    { cuda_init(); Board l = b; auto t = Clock::now(); auto r = search_cuda(l, depth); print_result("cuda", r, elapsed_ms(t)); cuda_cleanup(); }
#endif
}

// ── Multi-depth report ────────────────────────────────────────────────────

struct DepthRow {
    int       depth;
    double    serial_ms;
    long long serial_nodes;
    double    omp_ms;
    long long omp_nodes;
    double    cuda_ms;
    long long cuda_nodes;
};

static void write_html_report(const std::vector<DepthRow>& rows,
                               const std::string& fen,
                               const std::string& outfile) {
    const bool has_cuda = std::any_of(
        rows.begin(), rows.end(), [](const DepthRow& row) { return row.cuda_ms > 0.0; }
    );

    // Build JS arrays
    std::string labels, s_ms, o_ms, c_ms, speedup_omp, speedup_cuda, s_knps, o_knps, c_knps;
    for (int i = 0; i < (int)rows.size(); i++) {
        const auto& r = rows[i];
        if (i) {
            labels += ",";
            s_ms += ",";
            o_ms += ",";
            c_ms += ",";
            speedup_omp += ",";
            speedup_cuda += ",";
            s_knps += ",";
            o_knps += ",";
            c_knps += ",";
        }
        labels       += std::to_string(r.depth);
        s_ms         += std::to_string((long long)r.serial_ms);
        o_ms         += std::to_string((long long)r.omp_ms);
        c_ms         += r.cuda_ms > 0 ? std::to_string((long long)r.cuda_ms) : "null";
        speedup_omp  += std::to_string(r.serial_ms / r.omp_ms).substr(0, 5);
        speedup_cuda += r.cuda_ms > 0 ? std::to_string(r.serial_ms / r.cuda_ms).substr(0, 5) : "null";
        s_knps       += std::to_string((long long)(r.serial_nodes / r.serial_ms));
        o_knps       += std::to_string((long long)(r.omp_nodes / r.omp_ms));
        c_knps       += r.cuda_ms > 0 ? std::to_string((long long)(r.cuda_nodes / r.cuda_ms)) : "null";
    }

    // Build ASCII table rows for the HTML table
    std::string table_rows;
    for (const auto& r : rows) {
        double sup_omp  = r.serial_ms / r.omp_ms;
        double sup_cuda = r.cuda_ms > 0 ? r.serial_ms / r.cuda_ms : 0.0;
        table_rows += "<tr>";
        table_rows += "<td>" + std::to_string(r.depth) + "</td>";
        table_rows += "<td>" + std::to_string(r.serial_ms).substr(0, std::to_string(r.serial_ms).find('.') + 2) + "</td>";
        table_rows += "<td>" + std::to_string(r.serial_nodes) + "</td>";
        table_rows += "<td>" + std::to_string((long long)(r.serial_nodes / r.serial_ms)) + "</td>";
        table_rows += "<td>" + std::to_string(r.omp_ms).substr(0, std::to_string(r.omp_ms).find('.') + 2) + "</td>";
        table_rows += "<td>" + std::to_string(r.omp_nodes) + "</td>";
        table_rows += "<td>" + std::to_string(sup_omp).substr(0, std::to_string(sup_omp).find('.') + 3) + "×</td>";
        if (has_cuda) {
            if (sup_cuda > 0) {
                table_rows += "<td>" + std::to_string(r.cuda_ms).substr(0, std::to_string(r.cuda_ms).find('.') + 2) + "</td>";
                table_rows += "<td>" + std::to_string(r.cuda_nodes) + "</td>";
                table_rows += "<td>" + std::to_string(sup_cuda).substr(0, std::to_string(sup_cuda).find('.') + 3) + "×</td>";
            } else {
                table_rows += "<td>n/a</td><td>n/a</td><td>n/a</td>";
            }
        }
        table_rows += "</tr>\n";
    }

    std::ofstream f(outfile);
    f << R"(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Chess Engine Performance Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  body { font-family: 'Segoe UI', Arial, sans-serif; background: #0d1117; color: #e6edf3; margin: 0; padding: 32px; }
  h1   { font-size: 1.8rem; margin-bottom: 4px; color: #58a6ff; }
  .sub { color: #8b949e; font-size: .9rem; margin-bottom: 36px; font-family: monospace; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 28px; margin-bottom: 36px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 24px; }
  .card h2 { margin: 0 0 16px; font-size: 1rem; color: #8b949e; text-transform: uppercase; letter-spacing: .08em; }
  table { width: 100%; border-collapse: collapse; font-size: .88rem; }
  th    { background: #21262d; color: #8b949e; padding: 10px 14px; text-align: right; font-weight: 600; white-space: nowrap; }
  th:first-child { text-align: center; }
  td    { padding: 9px 14px; text-align: right; border-bottom: 1px solid #21262d; }
  td:first-child { text-align: center; font-weight: 700; color: #58a6ff; }
  tr:last-child td { border-bottom: none; }
  .speedup { color: #3fb950; font-weight: 700; }
  .badge   { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: .75rem; font-weight: 700; margin-left: 8px; }
  .serial-badge { background: #1f3a5f; color: #79c0ff; }
  .omp-badge    { background: #1f3a2f; color: #56d364; }
  .cuda-badge   { background: #3a1f5f; color: #d2a8ff; }
</style>
</head>
<body>
<h1>Chess Engine Performance Report</h1>
<p class="sub">Position: )";
    f << fen << "</p>\n";

    f << R"(
<div class="grid">
  <div class="card">
    <h2>Search Time (ms) — lower is better</h2>
    <canvas id="timeChart"></canvas>
  </div>
  <div class="card">
    <h2>Speedup vs Serial — higher is better</h2>
    <canvas id="speedupChart"></canvas>
  </div>
  <div class="card">
    <h2>Throughput (kilo-nodes/ms) — higher is better</h2>
    <canvas id="knpsChart"></canvas>
  </div>
  <div class="card">
    <h2>Search Time — log scale</h2>
    <canvas id="logChart"></canvas>
  </div>
</div>

<div class="card" style="margin-bottom:36px">
  <h2>Raw Data</h2>
  <table>
    <thead>
      <tr>
        <th rowspan="2">Depth</th>
        <th colspan="3"><span class="badge serial-badge">SERIAL</span></th>
        <th colspan="3"><span class="badge omp-badge">OPENMP</span></th>
)";
    if (has_cuda) {
        f << R"(
        <th colspan="3"><span class="badge cuda-badge">CUDA</span></th>
)";
    }
    f << R"(
      </tr>
      <tr>
        <th>Time (ms)</th><th>Nodes</th><th>kn/ms</th>
        <th>Time (ms)</th><th>Nodes</th><th>Speedup</th>)";
    if (has_cuda) {
        f << R"(<th>Time (ms)</th><th>Nodes</th><th>Speedup</th>)";
    }
    f << R"(
      </tr>
    </thead>
    <tbody>
)";
    f << table_rows;
    f << R"(    </tbody>
  </table>
</div>

<script>
const labels = [)" << labels << R"(];
const cfg = (type, datasets, extra={}) => ({
  type, data: { labels, datasets },
  options: {
    responsive: true,
    plugins: { legend: { labels: { color: '#8b949e' } } },
    scales: {
      x: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' }, title: { display: true, text: 'Search Depth', color: '#8b949e' } },
      y: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' }, ...extra }
    }
  }
});

// Time chart (bar)
new Chart(document.getElementById('timeChart'), cfg('bar', [
  { label: 'Serial', data: [)" << s_ms << R"(], backgroundColor: 'rgba(88,166,255,0.7)', borderColor: '#58a6ff', borderWidth: 1 },
  { label: 'OpenMP', data: [)" << o_ms << R"(], backgroundColor: 'rgba(63,185,80,0.7)',  borderColor: '#3fb950', borderWidth: 1 })";
    if (has_cuda) {
        f << R"(,
  { label: 'CUDA',   data: [)" << c_ms << R"(], backgroundColor: 'rgba(188,140,255,0.7)',borderColor: '#bc8cff', borderWidth: 1 })";
    }
    f << R"(
]));

// Speedup chart (line)
new Chart(document.getElementById('speedupChart'), cfg('line', [
  { label: 'OpenMP speedup', data: [)" << speedup_omp << R"(], borderColor: '#3fb950', backgroundColor: 'rgba(63,185,80,0.15)', tension: 0.3, fill: true })";
    if (has_cuda) {
        f << R"(,
  { label: 'CUDA speedup',   data: [)" << speedup_cuda << R"(], borderColor: '#bc8cff', backgroundColor: 'rgba(188,140,255,0.15)', tension: 0.3, fill: true })";
    }
    f << R"(
], { title: { display: true, text: '× faster than serial', color: '#8b949e' } }));

// knps chart
new Chart(document.getElementById('knpsChart'), cfg('bar', [
  { label: 'Serial',  data: [)" << s_knps << R"(], backgroundColor: 'rgba(88,166,255,0.7)',  borderColor: '#58a6ff', borderWidth: 1 },
  { label: 'OpenMP',  data: [)" << o_knps << R"(], backgroundColor: 'rgba(63,185,80,0.7)',   borderColor: '#3fb950', borderWidth: 1 },
)";
    if (has_cuda) {
        f << R"(
  { label: 'CUDA',    data: [)" << c_knps << R"(], backgroundColor: 'rgba(188,140,255,0.7)', borderColor: '#bc8cff', borderWidth: 1 },
)";
    }
    f << R"(
]));

// Log-scale time chart
new Chart(document.getElementById('logChart'), cfg('line', [
  { label: 'Serial', data: [)" << s_ms << R"(], borderColor: '#58a6ff', tension: 0.3, fill: false },
  { label: 'OpenMP', data: [)" << o_ms << R"(], borderColor: '#3fb950', tension: 0.3, fill: false })";
    if (has_cuda) {
        f << R"(,
  { label: 'CUDA',   data: [)" << c_ms << R"(], borderColor: '#bc8cff', tension: 0.3, fill: false })";
    }
    f << R"(
], { type: 'logarithmic', title: { display: true, text: 'ms (log)', color: '#8b949e' } }));
</script>
</body>
</html>
)";
    f.close();
    printf("\nReport written to: %s\n", outfile.c_str());
}

static void run_report(const std::string& fen, int maxdepth, const std::string& outfile) {
    Board b;
    b.load_fen(fen);
    b.print();

    int ncores = 1;
#ifdef USE_OMP
    #pragma omp parallel
    { ncores = omp_get_num_threads(); }
#endif

    printf("\nSweeping depths 3..%d | position: %s\n", maxdepth, fen.c_str());
    printf("CPU threads: %d\n\n", ncores);

    // Header
    printf("%-5s  %-28s  %-28s  %s\n",
           "Depth",
           "──── Serial ────────────────",
           "──── OpenMP ───────────────",
           "Speedup");
    printf("%-5s  %-10s %-10s %-6s  %-10s %-10s %-6s  %s\n",
           "", "Time(ms)", "Nodes", "kn/ms",
               "Time(ms)", "Nodes", "kn/ms", "OMP");
    printf("%s\n", std::string(80, '-').c_str());

    std::vector<DepthRow> rows;

    for (int d = 3; d <= maxdepth; d++) {
        DepthRow row{};
        row.depth = d;

        // Serial
        {
            Board l = b;
            auto  t = Clock::now();
            auto  r = search_serial(l, d);
            row.serial_ms    = elapsed_ms(t);
            row.serial_nodes = r.nodes;
        }

#ifdef USE_OMP
        // OpenMP
        {
            Board l = b;
            auto  t = Clock::now();
            auto  r = search_omp(l, d);
            row.omp_ms    = elapsed_ms(t);
            row.omp_nodes = r.nodes;
        }
#else
        row.omp_ms = row.serial_ms; row.omp_nodes = row.serial_nodes;
#endif

#ifdef USE_CUDA
        {
            cuda_init();
            Board l = b;
            auto  t = Clock::now();
            auto  r = search_cuda(l, d);
            row.cuda_ms    = elapsed_ms(t);
            row.cuda_nodes = r.nodes;
            cuda_cleanup();
        }
#endif

        double sup_omp  = row.serial_ms / row.omp_ms;
        double sup_cuda = row.cuda_ms > 0 ? row.serial_ms / row.cuda_ms : 0.0;

        printf("%-5d  %10.1f %10lld %6lld  %10.1f %10lld %6lld  %5.2f×",
               d,
               row.serial_ms, row.serial_nodes, (long long)(row.serial_nodes / row.serial_ms),
               row.omp_ms,    row.omp_nodes,    (long long)(row.omp_nodes / row.omp_ms),
               sup_omp);
        if (sup_cuda > 0) printf("  CUDA %.2f×", sup_cuda);
        printf("\n");
        fflush(stdout);

        rows.push_back(row);
    }

    printf("%s\n", std::string(80, '-').c_str());
    write_html_report(rows, fen, outfile);
}

// ── Minimal UCI loop ──────────────────────────────────────────────────────

static void run_uci(int default_depth) {
    printf("id name CudaChess\nid author gurnoor\nuciok\n");
    Board board; board.set_startpos();
    int depth = default_depth;
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream ss(line); std::string tok; ss >> tok;
        if (tok == "quit") break;
        if (tok == "isready") { printf("readyok\n"); }
        else if (tok == "ucinewgame") { board.set_startpos(); }
        else if (tok == "position") {
            ss >> tok;
            if (tok == "startpos") { board.set_startpos(); ss >> tok; }
            else if (tok == "fen") {
                std::string fen;
                for (int i = 0; i < 6 && ss >> tok; i++) fen += (i ? " " : "") + tok;
                board.load_fen(fen); ss >> tok;
            }
            while (ss >> tok) {
                auto moves = generate_legal_moves(board);
                for (const auto& m : moves) if (move_to_str(m) == tok) { board.make_move(m); break; }
            }
        } else if (tok == "go") {
            std::string opt;
            while (ss >> opt) if (opt == "depth") ss >> depth;
            auto res = search_serial(board, depth);
            printf("bestmove %s\n", move_to_str(res.best).c_str()); fflush(stdout);
        } else if (tok == "d") { board.print(); }
    }
}

// ── Entry point ───────────────────────────────────────────────────────────

static const char* STARTFEN  = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
static const char* BENCH_FEN = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

int main(int argc, char** argv) {
    std::string mode     = "bench";
    std::string fen      = BENCH_FEN;
    std::string outfile  = "perf_report.html";
    int         depth    = 6;
    int         maxdepth = 8;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "bench" || a == "perft" || a == "play" || a == "uci" || a == "report") {
            mode = a;
        } else if (a == "--fen"      && i+1 < argc) { fen      = argv[++i]; }
          else if (a == "--depth"    && i+1 < argc) { depth    = std::atoi(argv[++i]); }
          else if (a == "--maxdepth" && i+1 < argc) { maxdepth = std::atoi(argv[++i]); }
          else if (a == "--out"      && i+1 < argc) { outfile  = argv[++i]; }
          else if (a == "--startpos")               { fen      = STARTFEN; }
    }

    if      (mode == "uci")    { run_uci(depth); }
    else if (mode == "perft")  { run_perft(fen, depth); }
    else if (mode == "play")   {
        Board b; b.load_fen(fen); b.print();
        auto r = search_serial(b, depth);
        printf("\nBest: %s  Score: %+d  Nodes: %lld\n",
               move_to_str(r.best).c_str(), r.score, r.nodes);
    }
    else if (mode == "report") { run_report(fen, maxdepth, outfile); }
    else                       { run_bench(fen, depth); }

    return 0;
}
