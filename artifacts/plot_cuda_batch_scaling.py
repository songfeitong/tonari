"""Render the CUDA batch-latency figures from committed measurements."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks/results/rtx-pro-6000-blackwell-batch-scaling-20260914.json"
OUTPUT = Path(__file__).resolve().parent


def main() -> None:
    report = json.loads(RESULTS.read_text())
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
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
    for dataset, title, filename in (
        ("qmugs_population", "QMugs molecules", "qmugs-cuda-batch-latency.pdf"),
        ("matbench", "Matbench crystals", "matbench-cuda-batch-latency.pdf"),
    ):
        workloads = sorted(
            (
                w
                for w in report["workloads"]
                if w["dataset"] == dataset and w["batch_size"] <= 512
            ),
            key=lambda w: w["batch_size"],
        )
        sizes = np.array([w["batch_size"] for w in workloads])
        assert sizes.tolist() == [8, 32, 64, 128, 256, 512]
        fig, ax = plt.subplots(figsize=(7.2, 4.9))
        fig.subplots_adjust(left=0.115, right=0.96, bottom=0.285, top=0.78)
        fig.text(0.115, 0.935, title, fontsize=19, weight="bold")
        fig.text(
            0.115,
            0.885,
            "CUDA neighbor-list construction · single-batch latency",
            fontsize=11,
            color="#536174",
        )
        for backend, label, color, marker in (
            ("production_cuda", "tonari · native batch", "#087F8C", "o"),
            (
                "vesin_gpu_per_structure",
                "Vesin CUDA · per-structure + concat",
                "#D46A39",
                "s",
            ),
        ):
            values = [w["backends"][backend] for w in workloads]
            medians = np.array([v["median_ms"] for v in values])
            low = np.array([v["p10_ms"] for v in values])
            high = np.array([v["p90_ms"] for v in values])
            ax.fill_between(sizes, low, high, color=color, alpha=0.16, linewidth=0)
            ax.plot(
                sizes,
                medians,
                color=color,
                marker=marker,
                markersize=5,
                linewidth=2,
                label=label,
            )
            ax.annotate(
                f"{medians[-1]:.3f} ms",
                (sizes[-1], medians[-1]),
                xytext=(-9, 11),
                textcoords="offset points",
                ha="right",
                color=color,
                fontsize=10,
                weight="bold",
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlim(6.5, 650)
        ax.set_ylim(0.065, 300)
        ax.xaxis.set_major_locator(FixedLocator(sizes))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.yaxis.set_major_locator(FixedLocator([0.1, 1, 10, 100]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        ax.minorticks_off()
        ax.set_xlabel("Structures per batch", labelpad=9)
        ax.set_ylabel("Time per batch (ms, log scale)", labelpad=9)
        ax.grid(axis="y", color="#E0E5EB", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(
            loc="lower left",
            bbox_to_anchor=(-0.01, 1.015),
            frameon=False,
            fontsize=9,
            borderaxespad=0,
        )
        fig.text(
            0.115,
            0.13,
            "Median of 16 batch medians; shaded bands: P10–P90. 7 timed calls per batch.",
            fontsize=8.2,
            color="#536174",
        )
        fig.text(
            0.115,
            0.093,
            "RTX PRO 6000 Blackwell · float32 · 5 Å cutoff · full pair indices + cell shifts",
            fontsize=8.2,
            color="#536174",
        )
        fig.text(
            0.115,
            0.056,
            "Synchronized wall time; excludes data loading and H2D. Measured 2026-09-14.",
            fontsize=8.2,
            color="#536174",
        )
        fig.savefig(
            OUTPUT / filename,
            metadata={
                "Title": f"{title}: CUDA batch latency",
                "Author": "tonari",
                "Subject": "Batch sizes 8–512; synchronized latency with P10–P90 across batch medians",
            },
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
