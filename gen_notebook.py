#!/usr/bin/env python3
"""Generate a Colab notebook for building and benchmarking the chess engine."""

from __future__ import annotations

import json
import os
from pathlib import Path


BASE = Path(__file__).resolve().parent
NOTEBOOK_PATH = BASE / "chess_cuda_benchmark.ipynb"

FILE_LIST = [
    "src/board.h",
    "src/board.cpp",
    "src/movegen.h",
    "src/movegen.cpp",
    "src/eval.h",
    "src/eval.cpp",
    "src/search.h",
    "src/search_serial.cpp",
    "src/search_omp.cpp",
    "src/search_cuda.cpp",
    "src/main.cpp",
    "cuda/eval_cuda.cuh",
    "cuda/eval_cuda.cu",
    "CMakeLists.txt",
]

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
BENCH_FEN = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"


def markdown(*lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": list(lines)}


def code(src: str) -> dict:
    lines = src.splitlines(keepends=True)
    if not lines:
        lines = [""]
    return {
        "cell_type": "code",
        "metadata": {},
        "source": lines,
        "outputs": [],
        "execution_count": None,
    }


def load_sources() -> dict[str, str]:
    return {rel: (BASE / rel).read_text() for rel in FILE_LIST}


def build_write_sources_cell(file_contents: dict[str, str]) -> str:
    lines = [
        "from pathlib import Path",
        "",
        "project_dir = Path('/content/chess')",
        "project_dir.mkdir(parents=True, exist_ok=True)",
        "(project_dir / 'src').mkdir(exist_ok=True)",
        "(project_dir / 'cuda').mkdir(exist_ok=True)",
        "",
        "files = {}",
        "",
    ]

    for rel in FILE_LIST:
        lines.append(f"files[{rel!r}] = {file_contents[rel]!r}")
        lines.append("")

    lines.extend(
        [
            "for rel, content in files.items():",
            "    path = project_dir / rel",
            "    path.parent.mkdir(parents=True, exist_ok=True)",
            "    path.write_text(content)",
            "    print(f'wrote {path} ({len(content)} bytes)')",
            "",
            "print(f'\\nWrote {len(files)} files into {project_dir}')",
        ]
    )
    return "\n".join(lines)


def build_notebook() -> dict:
    file_contents = load_sources()

    detect_gpu_cell = """import subprocess

query_cmd = [
    'nvidia-smi',
    '--query-gpu=name,memory.total,driver_version,compute_cap',
    '--format=csv,noheader',
]
result = subprocess.run(query_cmd, capture_output=True, text=True)
if result.returncode != 0:
    raise RuntimeError(
        'No NVIDIA GPU runtime detected. In Colab, switch to Runtime > Change runtime type > GPU.'
    )

gpu_rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
for idx, row in enumerate(gpu_rows, start=1):
    print(f'GPU {idx}: {row}')

first_gpu_cols = [part.strip() for part in gpu_rows[0].split(',')]
compute_cap = first_gpu_cols[-1] if first_gpu_cols else '7.5'

digits = ''.join(ch for ch in compute_cap if ch.isdigit())
cuda_arch = digits or '75'
print(f'Using CMAKE_CUDA_ARCHITECTURES={cuda_arch}')
"""

    build_cell = """import os
import subprocess

cmake_configure = [
    'cmake',
    '-S', '.',
    '-B', 'build',
    '-DUSE_OMP=ON',
    '-DUSE_CUDA=ON',
    f'-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}',
]
cmake_build = ['cmake', '--build', 'build', '--verbose', f'-j{min(os.cpu_count() or 2, 2)}']

for cmd in (cmake_configure, cmake_build):
    print('$', ' '.join(cmd))
    completed = subprocess.run(cmd, cwd=project_dir, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f'Command failed: {" ".join(cmd)}')

print('\\nBuild complete.')
"""

    perft_cell = f"""import subprocess

checks = [
    (
        'Starting position depth 5',
        ['./chess', 'perft', '--depth', '5', '--startpos'],
        '4865609',
    ),
    (
        'Kiwipete depth 4',
        ['./chess', 'perft', '--depth', '4', '--fen', {BENCH_FEN!r}],
        '4085603',
    ),
]

for label, cmd, expected in checks:
    print(f'=== {{label}} ===')
    result = subprocess.run(cmd, cwd=project_dir / 'build', capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f'Perft command failed: {{cmd}}')
    if f'Total: {{expected}}' not in result.stdout:
        raise AssertionError(f'Expected total {{expected}} was not found in output for {{label}}')
"""

    bench_cell = """import subprocess
from pathlib import Path

report_cmd = ['./chess', 'report', '--fen', BENCH_FEN, '--maxdepth', '8', '--out', 'perf_report.html']
result = subprocess.run(report_cmd, cwd=project_dir / 'build', capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print(result.stderr)
    raise RuntimeError('Benchmark run failed')

benchmark_output = result.stdout
(project_dir / 'benchmark_output.txt').write_text(benchmark_output)
print('\\nSaved benchmark output to', project_dir / 'benchmark_output.txt')
print('Saved HTML report to', project_dir / 'build' / 'perf_report.html')
"""

    parse_cell = """import math
import re

row_pattern = re.compile(
    r'^\\s*(\\d+)\\s+'
    r'([\\d.]+)\\s+(\\d+)\\s+(\\d+)\\s+'
    r'([\\d.]+)\\s+(\\d+)\\s+(\\d+)\\s+'
    r'([\\d.]+)×'
    r'(?:\\s+CUDA\\s+([\\d.]+)×)?\\s*$'
)

rows = []
for line in benchmark_output.splitlines():
    match = row_pattern.match(line)
    if not match:
        continue
    rows.append(
        {
            'depth': int(match.group(1)),
            'serial_ms': float(match.group(2)),
            'serial_nodes': int(match.group(3)),
            'serial_knps': int(match.group(4)),
            'omp_ms': float(match.group(5)),
            'omp_nodes': int(match.group(6)),
            'omp_knps': int(match.group(7)),
            'omp_speedup': float(match.group(8)),
            'cuda_speedup': float(match.group(9)) if match.group(9) else math.nan,
        }
    )

if not rows:
    raise RuntimeError('Could not parse benchmark table from report output.')

has_cuda = any(not math.isnan(row['cuda_speedup']) for row in rows)
print(f'Parsed {len(rows)} benchmark rows. CUDA data present: {has_cuda}')
"""

    plot_cell = """import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

depths = [row['depth'] for row in rows]
serial_ms = [row['serial_ms'] for row in rows]
omp_ms = [row['omp_ms'] for row in rows]
serial_knps = [row['serial_knps'] for row in rows]
omp_knps = [row['omp_knps'] for row in rows]
omp_speedup = [row['omp_speedup'] for row in rows]
cuda_speedup = [row['cuda_speedup'] for row in rows]

plt.style.use('dark_background')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor('#0d1117')
fig.suptitle('Chess Engine Performance: Serial vs OpenMP vs CUDA', fontsize=18, y=0.98)

serial_color = '#58a6ff'
omp_color = '#3fb950'
cuda_color = '#bc8cff'
x = np.arange(len(depths))
width = 0.35

axes[0, 0].bar(x - width / 2, serial_ms, width, label='Serial', color=serial_color, alpha=0.85)
axes[0, 0].bar(x + width / 2, omp_ms, width, label='OpenMP', color=omp_color, alpha=0.85)
axes[0, 0].set_title('Search Time (ms)')
axes[0, 0].set_xlabel('Depth')
axes[0, 0].set_ylabel('Milliseconds')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(depths)
axes[0, 0].legend()

axes[0, 1].plot(depths, omp_speedup, 'o-', color=omp_color, linewidth=2, label='OpenMP')
if has_cuda:
    cuda_depths = [row['depth'] for row in rows if not math.isnan(row['cuda_speedup'])]
    cuda_values = [row['cuda_speedup'] for row in rows if not math.isnan(row['cuda_speedup'])]
    axes[0, 1].plot(cuda_depths, cuda_values, 's-', color=cuda_color, linewidth=2, label='CUDA')
axes[0, 1].axhline(1.0, color='#8b949e', linestyle='--', alpha=0.6)
axes[0, 1].set_title('Speedup vs Serial')
axes[0, 1].set_xlabel('Depth')
axes[0, 1].set_ylabel('Speedup (x)')
axes[0, 1].legend()

axes[1, 0].bar(x - width / 2, serial_knps, width, label='Serial', color=serial_color, alpha=0.85)
axes[1, 0].bar(x + width / 2, omp_knps, width, label='OpenMP', color=omp_color, alpha=0.85)
axes[1, 0].set_title('Throughput (kn/ms)')
axes[1, 0].set_xlabel('Depth')
axes[1, 0].set_ylabel('kilo-nodes per ms')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(depths)
axes[1, 0].legend()

axes[1, 1].semilogy(depths, serial_ms, 'o-', color=serial_color, linewidth=2, label='Serial')
axes[1, 1].semilogy(depths, omp_ms, 's-', color=omp_color, linewidth=2, label='OpenMP')
axes[1, 1].set_title('Search Time (log scale)')
axes[1, 1].set_xlabel('Depth')
axes[1, 1].set_ylabel('Milliseconds')
axes[1, 1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f'{value:,.0f}'))
axes[1, 1].legend()

for ax in axes.flat:
    ax.set_facecolor('#161b22')
    ax.grid(color='#30363d', alpha=0.25)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plot_path = project_dir / 'chess_benchmark.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print('Saved plot to', plot_path)
"""

    summary_cell = """import pandas as pd

df = pd.DataFrame(rows)
df = df.rename(
    columns={
        'depth': 'Depth',
        'serial_ms': 'Serial (ms)',
        'serial_nodes': 'Serial Nodes',
        'serial_knps': 'Serial kn/ms',
        'omp_ms': 'OpenMP (ms)',
        'omp_nodes': 'OpenMP Nodes',
        'omp_knps': 'OpenMP kn/ms',
        'omp_speedup': 'OpenMP Speedup',
        'cuda_speedup': 'CUDA Speedup',
    }
)

if not has_cuda:
    df = df.drop(columns=['CUDA Speedup'])

styled = (
    df.style
      .format(
          {
              'Serial (ms)': '{:.1f}',
              'OpenMP (ms)': '{:.1f}',
              'OpenMP Speedup': '{:.2f}x',
              **({'CUDA Speedup': '{:.2f}x'} if 'CUDA Speedup' in df.columns else {}),
          },
          na_rep='n/a',
      )
      .format('{:,}', subset=['Serial Nodes', 'OpenMP Nodes'])
      .set_properties(**{'text-align': 'right'})
      .set_table_styles(
          [
              {'selector': 'th', 'props': [('background-color', '#21262d'), ('color', '#8b949e')]},
              {'selector': 'td', 'props': [('border-bottom', '1px solid #30363d')]},
          ]
      )
)

display(styled)
csv_path = project_dir / 'benchmark_summary.csv'
df.to_csv(csv_path, index=False)
print('Saved summary CSV to', csv_path)
"""

    cells = [
        markdown(
            "# CUDA Chess Engine Colab Benchmark\n",
            "\n",
            "This notebook rebuilds the project from embedded source, verifies move generation, and benchmarks serial, OpenMP, and CUDA search paths.\n",
            "\n",
            "Before running the build cell, switch Colab to a GPU runtime.\n",
        ),
        markdown("## 1. Detect GPU Runtime\n"),
        code(detect_gpu_cell),
        markdown(
            "## 2. Write Project Files\n",
            "\n",
            f"The notebook writes all `{len(FILE_LIST)}` tracked source files into `/content/chess`.\n",
        ),
        code(build_write_sources_cell(file_contents)),
        markdown("## 3. Build With CMake\n"),
        code(build_cell),
        markdown(
            "## 4. Verify Move Generation\n",
            "\n",
            "These checks confirm the engine still matches known perft totals before benchmarking.\n",
        ),
        code(perft_cell),
        markdown("## 5. Run Benchmark Sweep\n"),
        code(f"BENCH_FEN = {BENCH_FEN!r}\n\n" + bench_cell),
        markdown("## 6. Parse Report Output\n"),
        code(parse_cell),
        markdown("## 7. Plot Results\n"),
        code(plot_cell),
        markdown("## 8. Summary Table\n"),
        code(summary_cell),
    ]

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": cells,
    }


def main() -> None:
    notebook = build_notebook()
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1))
    print(f"Notebook written: {NOTEBOOK_PATH}")
    print(f"Cells: {len(notebook['cells'])}")


if __name__ == "__main__":
    main()
