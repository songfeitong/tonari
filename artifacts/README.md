# Benchmark figures

All figures use the locally installed [Geist v1.7.2 font](https://github.com/vercel/geist-font/releases/tag/v1.7.2). The repository contains no font binaries; PDF files embed the glyphs they use. Regeneration requires Matplotlib, NumPy, and system-installed Geist Regular and Bold. On this workstation, the fonts and their SIL Open Font License are installed in `/usr/local/share/fonts/geist/`; `fc-match Geist` confirms discovery. The scripts fail if Geist is unavailable rather than silently using a different font.

The main README uses three compact two-panel PNGs: [CUDA batches](readme-cuda-batches.png), [large periodic structures](readme-large-structures.png), and [CPU threads](readme-cpu-threads.png). Each single-panel figure is also exported as PNG and vector PDF. README layouts use shorter titles and endpoint labels; the individual PDFs retain the full annotations.

```bash
python artifacts/plot_readme.py
```

## Downloads

| Figure | PNG | PDF |
| --- | --- | --- |
| QMugs CUDA batch | [PNG](qmugs-cuda-batch-latency.png) | [PDF](qmugs-cuda-batch-latency.pdf) |
| Matbench CUDA batch | [PNG](matbench-cuda-batch-latency.png) | [PDF](matbench-cuda-batch-latency.pdf) |
| Large periodic structure CUDA | [PNG](single-structure-cuda-latency.png) | [PDF](single-structure-cuda-latency.pdf) |
| Large periodic structure CPU | [PNG](single-structure-cpu-latency.png) | [PDF](single-structure-cpu-latency.pdf) |
| QMugs CPU threads | [PNG](qmugs-cpu-thread-scaling.png) | [PDF](qmugs-cpu-thread-scaling.pdf) |
| Matbench CPU threads | [PNG](matbench-cpu-thread-scaling.png) | [PDF](matbench-cpu-thread-scaling.pdf) |
| Large periodic structure CPU threads | [PNG](single-structure-cpu-thread-scaling.png) | [PDF](single-structure-cpu-thread-scaling.pdf) |

## CUDA batch latency

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

## CPU thread scaling

Three fixed-workload figures use 1, 2, 4, and 8 threads on eight physical CPU cores. Curves show median wall time from 11 repeats, with linear axes and the same Geist style:

- [Single 32,768-atom periodic structure](single-structure-cpu-thread-scaling.pdf)
- [QMugs: fixed batch of 4,096 molecules](qmugs-cpu-thread-scaling.pdf)
- [Matbench: fixed batch of 1,536 crystals](matbench-cpu-thread-scaling.pdf)

Source: [2026-09-14 NumPy thread-scaling results](../benchmarks/results/threadripper-pro-9975wx-cpu-thread-scaling-numpy-20260914.json). Both backends use NumPy inputs and outputs, float64, a 5 Å cutoff, and complete `ijS` output. Torch is used only for data preparation outside timing. Tonari processes each workload as one native batch call. Vesin processes structures sequentially using the specified internal thread count; global index offsets and NumPy concatenation are included for batches. Single-structure calls return directly, without concatenation. There is no outer structure parallelism in the Vesin adapter. Outputs are released outside timing. Both backends at all four thread counts exactly match canonical pair keys. Earlier Torch thread-scaling measurements have been superseded for these figures. The NumPy single-thread three-way comparison uses core 31, while these measurements use affinity 0–7. See [benchmark methodology](../docs/benchmark.md) for details.

```bash
python artifacts/plot_cpu_thread_scaling.py
```
