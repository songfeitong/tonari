"""Plot CPU thread scaling for a large structure and two fixed real-data batches."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, MaxNLocator
from plot_fonts import load_geist

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(__file__).resolve().parent
RESULTS = (
    ROOT
    / "benchmarks/results/threadripper-pro-9975wx-cpu-thread-scaling-numpy-20260914.json"
)


def main() -> None:
    report = json.loads(RESULTS.read_text())
    assert report["method"]["array_api"].startswith("NumPy")
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
    cases = {
        "matbench_1536_structure_batch": (
            "Matbench batch",
            "1,536 structures",
            "matbench-cpu-thread-scaling.pdf",
            "Time per batch (ms)",
        ),
        "qmugs_population_4096_structure_batch": (
            "QMugs batch",
            "4,096 structures",
            "qmugs-cpu-thread-scaling.pdf",
            "Time per batch (ms)",
        ),
        "matbench_32768_atom_supercell": (
            "Large Periodic structure",
            "32,768 atoms",
            "single-structure-cpu-thread-scaling.pdf",
            "Time per structure (ms)",
        ),
    }
    for workload in report["workloads"]:
        title, size, filename, ylabel = cases[workload["name"]]
        counts = np.array(sorted(map(int, workload["measurements"])))
        assert counts.tolist() == [1, 2, 4, 8]
        assert workload["validation"]["exact_key_match"]
        assert all(workload["validation"]["thread_count_key_match"].values())
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        fig.subplots_adjust(left=0.115, right=0.96, bottom=0.14, top=0.725)
        fig.text(0.115, 0.945, f"{title} · CPU", fontsize=19, weight="bold")
        fig.text(
            0.115,
            0.892,
            f"{size} · Cutoff 5 Å · Threadripper PRO 9975WX",
            fontsize=11,
            color="#536174",
        )
        maximum = 0.0
        for backend, label, color, marker in (
            ("tonari", "tonari", "#087F8C", "o"),
            ("vesin", "vesin", "#D46A39", "s"),
        ):
            medians = np.array(
                [workload["measurements"][str(n)][backend]["median_ms"] for n in counts]
            )
            maximum = max(maximum, float(medians.max()))
            ax.plot(
                counts,
                medians,
                label=label,
                color=color,
                marker=marker,
                markersize=5,
                linewidth=2,
            )
            precision = 2 if backend == "tonari" else 1
            for n, median in zip(counts, medians, strict=True):
                if n not in (1, 4, 8):
                    continue
                # Separate labels even when the two series start close together.
                upper = (backend == "tonari") == (
                    workload["name"] == "matbench_32768_atom_supercell"
                )
                if n == 1:
                    upper = True
                    if workload["name"].startswith("qmugs"):
                        other = "vesin" if backend == "tonari" else "tonari"
                        upper = (
                            median > workload["measurements"]["1"][other]["median_ms"]
                        )
                ax.annotate(
                    f"{median:.{precision}f} ms",
                    (n, median),
                    xytext=(
                        -8 if n == 8 else 0,
                        12 if upper else -34 if n == 1 else -22,
                    ),
                    textcoords="offset points",
                    ha="right" if n == 8 else "center",
                    fontsize=10,
                    weight="bold",
                    color=color,
                )
        ax.set_xlim(0.6, 8.4)
        ax.set_ylim(0, maximum * 1.25)
        ax.xaxis.set_major_locator(FixedLocator(counts))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_xlabel("CPU Threads", labelpad=9)
        ax.set_ylabel(ylabel, labelpad=9)
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
                "Title": f"{title}: CPU thread scaling",
                "Author": "tonari",
                "Subject": "1, 2, 4, 8 threads; fixed workload; median wall times",
            },
        )
        fig.savefig(
            OUTPUT / filename.replace(".pdf", ".png"), dpi=180, facecolor="white"
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
