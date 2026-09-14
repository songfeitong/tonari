"""Render single-structure CUDA scaling from committed supercell measurements."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter
from plot_fonts import load_geist

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks/results/rtx-pro-6000-blackwell.json"
OUTPUT = Path(__file__).resolve().parent


def main() -> None:
    report = json.loads(RESULTS.read_text())
    load_geist()
    plt.rcParams.update(
        {
            "font.family": "Geist",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#ABB4BF",
            "text.color": "#182638",
            "axes.labelcolor": "#182638",
            "xtick.color": "#536174",
            "ytick.color": "#536174",
            "pdf.fonttype": 42,
        }
    )
    title = "Large Periodic structure"
    filename = "single-structure-cuda-latency.pdf"
    workloads = sorted(
        (w for w in report["workloads"] if w["name"].startswith("matbench_supercell_")),
        key=lambda w: w["backends"]["production_cuda"]["atoms"],
    )
    sizes = np.array([w["backends"]["production_cuda"]["atoms"] for w in workloads])
    assert sizes.tolist() == [64, 512, 1728, 4096, 13824, 32768]
    for w in workloads:
        native = w["backends"]["production_cuda"]
        reference = w["backends"]["vesin_gpu_per_structure"]
        assert native["structures"] == reference["structures"] == 1
        assert native["atoms"] == reference["atoms"]
        assert native["pairs"] == reference["pairs"]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    fig.subplots_adjust(left=0.115, right=0.96, bottom=0.14, top=0.725)
    fig.text(0.115, 0.945, f"{title} · CUDA", fontsize=19, weight="bold")
    fig.text(
        0.115,
        0.892,
        "Cutoff 5 Å · RTX PRO 6000 Blackwell",
        fontsize=11,
        color="#536174",
    )
    for backend, label, color, marker in (
        ("production_cuda", "tonari", "#087F8C", "o"),
        (
            "vesin_gpu_per_structure",
            "vesin-torch",
            "#D46A39",
            "s",
        ),
    ):
        values = [w["backends"][backend] for w in workloads]
        medians = np.array([v["median_ms"] for v in values])
        ax.plot(
            sizes,
            medians,
            color=color,
            marker=marker,
            markersize=5,
            linewidth=2,
            label=label,
        )
        precision = 2 if backend == "production_cuda" else 1
        for size, median in zip(sizes, medians, strict=True):
            if size not in (512, 4096, 32768):
                continue
            ax.annotate(
                f"{median:.{precision}f} ms",
                (size, median),
                xytext=(-9, 11) if size == 32768 else (0, 11),
                textcoords="offset points",
                ha="right" if size == 32768 else "center",
                color=color,
                fontsize=10,
                weight="bold",
            )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(45, 45000)
    ax.set_ylim(0.05, 1.0)
    ax.xaxis.set_major_locator(FixedLocator([64, 512, 4096, 32768]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.yaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.2, 0.5, 1]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    ax.minorticks_off()
    ax.set_xlabel("Number of Atoms", labelpad=9)
    ax.set_ylabel("Time per structure (ms, log scale)", labelpad=9)
    ax.grid(axis="y", color="#E0E5EB", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(-0.01, 1.03),
        labelspacing=0.25,
        frameon=False,
        fontsize=12,
        borderaxespad=0,
    )
    fig.savefig(
        OUTPUT / filename,
        metadata={
            "Title": f"{title}: CUDA single-structure latency",
            "Author": "tonari",
            "Subject": "64–32,768 atoms; historical synchronized latency medians",
        },
    )
    fig.savefig(OUTPUT / filename.replace(".pdf", ".png"), dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
