# CUDA batch-latency figures

English vector PDF figures for QMugs population and Matbench, showing batch sizes 8, 16, 32, 64, 128, 256, and 512. Both figures use identical logarithmic axes. Lines show the median across 16 per-batch latency medians. Subtitles report the mean atom count in the source benchmark samples: 226,648 / 4,096 = 55.3 atoms per QMugs molecule and 75,238 / 1,536 = 49.0 atoms per Matbench crystal (rounded to one decimal). The plotting script computes these from the committed dataset manifests. The GPU name in the subtitle is shortened to RTX PRO 6000 Blackwell; the measured card is the NVIDIA RTX PRO 6000 Blackwell Workstation Edition. The Vesin CUDA baseline includes per-structure calls and output concatenation.

- [QMugs](qmugs-cuda-batch-latency.pdf)
- [Matbench](matbench-cuda-batch-latency.pdf)

Source: [2026-09-14 measurements](../benchmarks/results/rtx-pro-6000-blackwell-batch-scaling-20260914.json). The figures also load the [batch-size-16 supplement](../benchmarks/results/rtx-pro-6000-blackwell-batch-scaling-bs16-20260914.json), measured with the same binary, seeds, and timing protocol. The complete record also includes batch size 1024. See [benchmark methodology](../docs/benchmark.md) for sampling and timing details.

Regenerate from the repository root with Matplotlib and NumPy installed:

```bash
python artifacts/plot_cuda_batch_scaling.py
```
