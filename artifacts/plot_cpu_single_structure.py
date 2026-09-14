"""Render single-structure CPU scaling from committed supercell measurements."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.ticker import FixedLocator, FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
RESULTS = (
    ROOT / "benchmarks/results/threadripper-pro-9975wx-single-structure-20260914.json"
)
OUTPUT = Path(__file__).resolve().parent


def main() -> None:
    report = json.loads(RESULTS.read_text())
    for filename in ("Geist-Regular.ttf", "Geist-Bold.ttf"):
        font_manager.fontManager.addfont(OUTPUT / "fonts/geist" / filename)
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
    filename = "single-structure-cpu-latency.pdf"
    workloads = sorted(
        (w for w in report["workloads"] if w["name"].startswith("matbench_supercell_")),
        key=lambda w: w["atoms"],
    )
    sizes = np.array([w["atoms"] for w in workloads])
    assert sizes.tolist() == [64, 512, 1728, 4096, 13824, 32768]
    assert all(w["validation"]["exact_key_match"] for w in workloads)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    fig.subplots_adjust(left=0.115, right=0.96, bottom=0.13, top=0.68)
    fig.text(0.115, 0.945, f"{title} · CPU", fontsize=19, weight="bold")
    fig.text(
        0.115,
        0.892,
        "Cutoff 5 Å · Threadripper PRO 9975WX · 1 thread",
        fontsize=11,
        color="#536174",
    )
    for backend, label, color, marker in (
        ("tonari", "tonari", "#087F8C", "o"),
        (
            "vesin",
            "vesin",
            "#D46A39",
            "s",
        ),
        ("ase", "ASE", "#7560A8", "^"),
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
        precision = 2 if backend == "tonari" else 1
        for size, median in zip(sizes, medians, strict=True):
            if size not in (512, 4096, 32768):
                continue
            ax.annotate(
                f"{median:.{precision}f} ms",
                (size, median),
                xytext=(-9 if size == 32768 else 0, -28 if backend == "vesin" else 11),
                textcoords="offset points",
                ha="right" if size == 32768 else "center",
                color=color,
                fontsize=10,
                weight="bold",
            )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(45, 45000)
    ax.set_ylim(0.02, 1200)
    ax.xaxis.set_major_locator(FixedLocator([64, 512, 4096, 32768]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.yaxis.set_major_locator(FixedLocator([0.1, 1, 10, 100, 1000]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    ax.minorticks_off()
    ax.set_xlabel("Number of Atoms", labelpad=9)
    ax.set_ylabel("Time per structure (ms, log scale)", labelpad=9)
    ax.grid(axis="y", color="#E0E5EB", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(-0.01, 1.03),
        labelspacing=0.6,
        frameon=False,
        fontsize=12,
        borderaxespad=0,
    )
    fig.savefig(
        OUTPUT / filename,
        metadata={
            "Title": f"{title}: CPU single-structure latency",
            "Author": "tonari",
            "Subject": "64–32,768 atoms; NumPy single-thread latency medians",
        },
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
