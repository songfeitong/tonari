# CUDA batch-latency figures

All figures use bundled [Geist Regular and Bold](fonts/geist/README.md), embedded in the vector PDFs.

English vector PDF figures for QMugs population and Matbench, showing batch sizes 8, 16, 32, 64, 128, 256, and 512. Both figures use identical logarithmic axes. Lines show the median across 16 per-batch latency medians. Subtitles report the mean atom count in the source benchmark samples: 226,648 / 4,096 = 55.3 atoms per QMugs molecule and 75,238 / 1,536 = 49.0 atoms per Matbench crystal (rounded to one decimal). The plotting script computes these from the committed dataset manifests. The GPU name in the subtitle is shortened to RTX PRO 6000 Blackwell; the measured card is the NVIDIA RTX PRO 6000 Blackwell Workstation Edition. The Vesin CUDA baseline includes per-structure calls and output concatenation.

- [QMugs](qmugs-cuda-batch-latency.pdf)
- [Matbench](matbench-cuda-batch-latency.pdf)

Source: [2026-09-14 measurements](../benchmarks/results/rtx-pro-6000-blackwell-batch-scaling-20260914.json). The figures also load the [batch-size-16 supplement](../benchmarks/results/rtx-pro-6000-blackwell-batch-scaling-bs16-20260914.json), measured with the same binary, seeds, and timing protocol. The complete record also includes batch size 1024. See [benchmark methodology](../docs/benchmark.md) for sampling and timing details.

Regenerate from the repository root with Matplotlib and NumPy installed:

```bash
python artifacts/plot_cuda_batch_scaling.py
```

## Single-structure scaling

[Single periodic structure PDF](single-structure-cuda-latency.pdf) uses the same visual style, plotting tonari and vesin-torch at 64, 512, 1,728, 4,096, 13,824, and 32,768 atoms. Labels highlight 512, 4,096, and 32,768 atoms. These are integer supercells of one sampled 64-atom Matbench crystal, not different independent crystals. The cutoff is 5 Å and geometry is float32 on the RTX PRO 6000 Blackwell Workstation Edition.

Source: [historical CUDA measurements](../benchmarks/results/rtx-pro-6000-blackwell.json), revision `09c968610a54412b6a0b665861c5adfa8632bdb2`. Each point is the median of 12 synchronized single-structure calls. This figure does not represent a new run at the current HEAD; the batch figures use separate newer measurements.

```bash
python artifacts/plot_cuda_single_structure.py
```

## CPU single-structure scaling

[CPU PDF](single-structure-cpu-latency.pdf) compares tonari, Vesin, and ASE on the same six periodic supercells using NumPy inputs and outputs. Measurements use float64, a 5 Å cutoff, full lists without zero-shift self pairs, and one pinned CPU core on the Threadripper PRO 9975WX. ASE uses `primitive_neighbor_list`; Vesin reuses its configuration but rebuilds neighbors each call. All three produce identical canonical pair keys at every size.

Source: [2026-09-14 CPU measurements](../benchmarks/results/threadripper-pro-9975wx-single-structure-20260914.json). These are new NumPy measurements and are separate from the historical Torch CPU adapter results. Each point is the median of 11 timed calls. The CPU legend uses `vesin`, since this benchmark does not use Torch input/output.

```bash
python artifacts/plot_cpu_single_structure.py
```
