"""Generate compact two-panel PNG figures for the README from benchmark records."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator
from plot_fonts import load_geist

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(__file__).resolve().parent
RESULTS = ROOT / "benchmarks/results"
COLORS = {"tonari": "#087F8C", "vesin": "#D46A39", "ase": "#7560A8"}
MARKERS = {"tonari": "o", "vesin": "s", "ase": "^"}


def record(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def figure(titles: list[str], subtitles: list[str]):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.7))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.77, wspace=0.32)
    for ax, title, subtitle in zip(axes, titles, subtitles, strict=True):
        left = ax.get_position().x0
        fig.text(left, 0.94, title, fontsize=18, weight="bold")
        fig.text(left, 0.88, subtitle, fontsize=11, color="#536174")
        ax.grid(axis="y", color="#E0E5EB", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.minorticks_off()
    return fig, axes


def line(ax, xs, ys, backend: str, label: str, label_below: bool = False):
    color = COLORS[backend]
    ax.plot(
        xs,
        ys,
        color=color,
        marker=MARKERS[backend],
        linewidth=2.2,
        markersize=5,
        label=label,
    )
    precision = 2 if backend == "tonari" else 1
    ax.annotate(
        f"{ys[-1]:.{precision}f} ms",
        (xs[-1], ys[-1]),
        xytext=(-5, -24 if label_below else 12),
        textcoords="offset points",
        ha="right",
        fontsize=11,
        weight="bold",
        color=color,
    )


def finish(fig, axes, name: str):
    for ax in axes:
        ax.legend(
            loc="lower left",
            bbox_to_anchor=(-0.015, 1.025),
            frameon=False,
            fontsize=11,
            labelspacing=0.25,
            borderaxespad=0,
            ncol=3,
            columnspacing=1.0,
            handlelength=1.7,
        )
    fig.savefig(OUTPUT / name, dpi=200, facecolor="white")
    plt.close(fig)


def main() -> None:
    load_geist()
    plt.rcParams.update(
        {
            "font.family": "Geist",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#ABB4BF",
            "text.color": "#182638",
            "axes.labelcolor": "#182638",
            "xtick.color": "#536174",
            "ytick.color": "#536174",
        }
    )
    batch = record("rtx-pro-6000-blackwell-batch-scaling-20260914.json")["workloads"]
    batch += record("rtx-pro-6000-blackwell-batch-scaling-bs16-20260914.json")[
        "workloads"
    ]
    fig, axes = figure(
        ["QMugs · CUDA", "Matbench · CUDA"],
        ["Avg 55.3 atoms per structure", "Avg 49.0 atoms per structure"],
    )
    for ax, dataset in zip(axes, ["qmugs_population", "matbench"], strict=True):
        ws = sorted(
            [w for w in batch if w["dataset"] == dataset and w["batch_size"] <= 512],
            key=lambda w: w["batch_size"],
        )
        xs = [w["batch_size"] for w in ws]
        for backend, key, label in [
            ("tonari", "production_cuda", "tonari"),
            ("vesin", "vesin_gpu_per_structure", "vesin-torch"),
        ]:
            line(ax, xs, [w["backends"][key]["median_ms"] for w in ws], backend, label)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlim(6.5, 650)
        ax.set_ylim(0.065, 350)
        ax.xaxis.set_major_locator(FixedLocator(xs))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.yaxis.set_major_locator(FixedLocator([0.1, 1, 10, 100]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        ax.minorticks_off()
        ax.set_xlabel("Batch Size", labelpad=8)
        ax.set_ylabel("Time per batch (ms, log)", labelpad=8)
    finish(fig, axes, "readme-cuda-batches.png")

    cuda = [
        w
        for w in record("rtx-pro-6000-blackwell.json")["workloads"]
        if w["name"].startswith("matbench_supercell_")
    ]
    cpu = record("threadripper-pro-9975wx-single-structure-20260914.json")["workloads"]
    fig, axes = figure(
        ["Large structure · CUDA", "Large structure · CPU"],
        ["RTX PRO 6000 Blackwell", "Threadripper PRO 9975WX · 1 thread"],
    )
    for ax, ws, device in zip(axes, [cuda, cpu], ["cuda", "cpu"], strict=True):
        xs = [
            w["backends"]["production_cuda"]["atoms"]
            if device == "cuda"
            else w["atoms"]
            for w in ws
        ]
        pairs = (
            [
                ("tonari", "production_cuda", "tonari"),
                ("vesin", "vesin_gpu_per_structure", "vesin-torch"),
            ]
            if device == "cuda"
            else [
                ("tonari", "tonari", "tonari"),
                ("vesin", "vesin", "vesin"),
                ("ase", "ase", "ASE"),
            ]
        )
        for backend, key, label in pairs:
            line(
                ax,
                xs,
                [w["backends"][key]["median_ms"] for w in ws],
                backend,
                label,
                label_below=(device == "cpu" and backend == "vesin"),
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlim(45, 45000)
        ax.set_ylim((0.05, 1.0) if device == "cuda" else (0.02, 1600))
        ax.xaxis.set_major_locator(FixedLocator([64, 512, 4096, 32768]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.yaxis.set_major_locator(
            FixedLocator(
                [0.05, 0.1, 0.2, 0.5, 1]
                if device == "cuda"
                else [0.1, 1, 10, 100, 1000]
            )
        )
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        ax.minorticks_off()
        ax.set_xlabel("Number of Atoms", labelpad=8)
        ax.set_ylabel("Time per structure (ms, log)", labelpad=8)
    finish(fig, axes, "readme-large-structures.png")

    threads = record("threadripper-pro-9975wx-cpu-thread-scaling-numpy-20260914.json")
    assert threads["method"]["array_api"].startswith("NumPy")
    fig, axes = figure(
        ["QMugs · CPU threads", "Matbench · CPU threads"],
        ["Fixed batch: 4,096 molecules", "Fixed batch: 1,536 crystals"],
    )
    for ax, prefix in zip(axes, ["qmugs", "matbench_1536"], strict=True):
        w = next(w for w in threads["workloads"] if w["name"].startswith(prefix))
        xs = [1, 2, 4, 8]
        maximum = 0.0
        for backend in ["tonari", "vesin"]:
            ys = [w["measurements"][str(n)][backend]["median_ms"] for n in xs]
            maximum = max(maximum, max(ys))
            line(ax, xs, ys, backend, backend, label_below=(backend == "tonari"))
        ax.set_xlim(0.6, 8.4)
        ax.set_ylim(0, maximum * 1.25)
        ax.xaxis.set_major_locator(FixedLocator(xs))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_xlabel("CPU Threads", labelpad=8)
        ax.set_ylabel("Time per batch (ms)", labelpad=8)
    finish(fig, axes, "readme-cpu-threads.png")


if __name__ == "__main__":
    main()
