"""Render the CUDA batch-latency figures from committed measurements."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter
from plot_fonts import load_geist

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks/results/rtx-pro-6000-blackwell-batch-scaling-20260914.json"
SUPPLEMENT = (
    ROOT / "benchmarks/results/rtx-pro-6000-blackwell-batch-scaling-bs16-20260914.json"
)
OUTPUT = Path(__file__).resolve().parent


def main() -> None:
    report = json.loads(RESULTS.read_text())
    report["workloads"].extend(json.loads(SUPPLEMENT.read_text())["workloads"])
    with (ROOT / "benchmarks/data/qmugs_sample_structures.csv").open(
        newline=""
    ) as handle:
        molecule_sizes = [
            int(row["n_atoms"])
            for row in csv.DictReader(handle)
            if row["workload"] == "population"
        ]
    crystal_manifest = json.loads(
        (ROOT / "benchmarks/data/matbench_mp_e_form_sample.json").read_text()
    )
    crystal_sizes = [row["n_atoms"] for row in crystal_manifest["structures"]]
    subtitles = {
        "qmugs_population": f"Avg {np.mean(molecule_sizes):.1f} atoms per structure · Cutoff 5 Å · RTX PRO 6000 Blackwell",
        "matbench": f"Avg {np.mean(crystal_sizes):.1f} atoms per structure · Cutoff 5 Å · RTX PRO 6000 Blackwell",
    }
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
        assert sizes.tolist() == [8, 16, 32, 64, 128, 256, 512]
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        fig.subplots_adjust(left=0.115, right=0.96, bottom=0.14, top=0.725)
        fig.text(0.115, 0.945, f"{title} · CUDA", fontsize=19, weight="bold")
        fig.text(
            0.115,
            0.892,
            subtitles[dataset],
            fontsize=11,
            color="#536174",
        )
        for backend, label, color, marker in (
            ("production_cuda", "tonari · native batch", "#087F8C", "o"),
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
                if size not in (32, 128, 512):
                    continue
                ax.annotate(
                    f"{median:.{precision}f} ms",
                    (size, median),
                    xytext=(-9, 11) if size == 512 else (0, 11),
                    textcoords="offset points",
                    ha="right" if size == 512 else "center",
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
        ax.set_xlabel("Batch Size", labelpad=9)
        ax.set_ylabel("Time per batch (ms, log scale)", labelpad=9)
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
                "Title": f"{title}: CUDA batch latency",
                "Author": "tonari",
                "Subject": "Batch sizes 8–512; synchronized latency medians",
            },
        )
        fig.savefig(
            OUTPUT / filename.replace(".pdf", ".png"), dpi=180, facecolor="white"
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
