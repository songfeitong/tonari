# Real-material benchmark

## Dataset and sampling

The primary source is ColabFit `matbench_mp_e_form`, dataset ID `DS_5drebe4tktiu_0`. The source page currently reports 132,741 configurations, whereas an earlier task description stated 132,752; this repository records the observed source value rather than silently normalizing the discrepancy. The pinned Parquet object is `colabfit/Matbench_mp_e_form` revision `9880d5b9b62877ec5aa14d1a4c2a9ff4ee870b8d`, path `co/co_0.parquet`, 128,655,162 bytes, SHA-256 `4b815791cc31862895b23cda7339d96217c37815c8f183949dc59b3035ee2afd`.

`scripts/prepare_matbench.py` deterministically selects 1,536 structures with seed `20260809`. It computes fixed strata over atom count, cell-vector length anisotropy, maximum absolute inter-vector cosine, and element count, orders each stratum by `sha256(f"{seed}:{configuration_id}")`, and round-robins across sorted strata. The sample spans 948 occupied strata and 1,343 unique reduced formulas. Atom-count quantiles at 0/10/25/50/75/90/95/99/100% are 1, 4, 9, 24, 58, 134, 176, 322.6, and 444; cell anisotropy spans 1.0–52.27 and cell skew spans 0.0–0.9966. All source crystals are full-PBC. The committed 992 KiB manifest contains source configuration IDs, Matbench names when present, compositions, atom counts, cell metrics, strata, source revision, and selection method; the 123 MiB raw Parquet and 1.5 MiB tensor cache remain ignored.

The DataLoader uses `batch_size=32`, deterministic shuffle with the same seed, a standard map-style `Dataset`, a custom collate into concatenated tensors plus `ptr`, pinned CPU memory, and one transfer of each complete batch to CUDA. Timed graph construction traverses the preloaded CUDA batches, so DataLoader and host-to-device transfer are deliberately excluded and reported as such. Each timing has warmup, device synchronization, 7 repeats for the full epoch and at least 12 repeats for single-batch/scaling cases; the table reports medians.

## Baselines and validation

The Vesin baseline uses Vesin 0.6.1 `NeighborList(cutoff, full_list=True)` once per structure on CUDA tensors, concatenates the results, and remaps its direction to this API's `(source, target, S)` convention. The dense baseline independently implements the Equiformer/FairChem pattern of materializing batch-local `N^2` atom pairs and a batch-wide padded image range in PyTorch; its cutoff and onsite semantics are adjusted to the exact contract under test. This is a style-equivalent exact-semantics baseline, not a claim that unmodified upstream Equiformer produces identical edge keys, because upstream uses an inclusive cutoff and additional near-zero filtering.

Before timing, the production CUDA output matched Vesin exactly for all 48 batches: 1,536 structures and 2,780,158 complete five-component keys. The median-density batch also matched the independent dense baseline exactly for 43,842 edges. Output ordering was not compared. The Equiformer V3 reference checkout was revision `a7300c58df683dc99cb48027d5bfd4c887486c48`; the benchmarked Vesin version was 0.6.1.

## Results

Hardware was one NVIDIA RTX PRO 6000 Blackwell Workstation Edition, compute capability 12.0, with PyTorch 2.12.1+cu130, Python 3.12.3, and float32 geometry at a 5.0 Å cutoff. Both GPUs were idle immediately before the run; GPU 1 was selected. The complete machine-readable record is `benchmarks/results/rtx-pro-6000-blackwell.json`, whose implementation revision is `c9b5dfa3300bec5cee8a75fb6cc06fc8aa6b5de9`.

| Workload | Atoms | New batch CUDA | Vesin GPU/structure | Vesin / new | Dense PyTorch | Dense / new |
| --- | --: | --: | --: | --: | --: | --: |
| 1,536-structure DataLoader epoch | 75,238 | 34.127 ms | 489.831 ms | 14.35× | skipped | — |
| Median 32-structure batch | 1,126 | 0.821 ms | 9.299 ms | 11.33× | 42.757 ms | 52.10× |
| Real structure, 1×1×1 | 64 | 0.223 ms | 0.378 ms | 1.69× | 0.699 ms | 3.14× |
| Derived supercell, 2×2×2 | 512 | 0.294 ms | 0.702 ms | 2.39× | 6.672 ms | 22.66× |
| Derived supercell, 3×3×3 | 1,728 | 0.344 ms | 0.942 ms | 2.73× | 73.211 ms | 212.56× |
| Derived supercell, 4×4×4 | 4,096 | 0.346 ms | 0.920 ms | 2.66× | skipped | — |
| Derived supercell, 6×6×6 | 13,824 | 0.357 ms | 1.108 ms | 3.11× | skipped | — |
| Derived supercell, 8×8×8 | 32,768 | 0.409 ms | 1.635 ms | 3.99× | skipped | — |

The scaling source is the real 64-atom configuration `CO_8661596785617876616983344`; only integer supercell repetition is synthetic. Its edge count scales from 744 to 380,928. The dense candidate estimate grows from 110,592 at 1×1×1 to 28,991,029,248 at 8×8×8, so runs above the configured 150 million-candidate safety limit were skipped instead of risking avoidable out-of-memory failure. At the median batch the dense baseline used 6,915,438,080 bytes of additional PyTorch allocator memory versus 1,494,528 bytes for the new path; at 3×3×3 it used 11,710,352,384 bytes. Vesin's reported allocator peak does not include all native temporary allocations and must not be interpreted as its complete memory footprint.

## Profiling and interpretation

Nsight Systems on the 32,768-atom derived workload showed about 0.126 ms of CUDA kernels per call. The largest kernels were fused representative wrapping plus Cartesian bounds at 48.3 µs, cell-list edge writing at 35.5 µs, and edge counting at 28.1 µs. A trial replacement of the fused atomic bounds with a separate CUB block reduction increased the bounds work to a combined 106.4 µs and increased the 20-call NVTX range from 9.256 ms to 10.179 ms, so it was reverted. CUDA query kernels were therefore not obscuring an unexamined larger device hotspot. The machine-readable comparison is `benchmarks/results/nsys-matbench-32768-summary.json`; raw profiler traces remain ignored under `runs/`.

The remaining one-shot overhead is primarily metadata and exact-size allocation synchronization. A reusable metadata API could save a measured roughly 0.177 ms when `ptr/cells/pbc/cutoff` are unchanged, but safe behavior under tensor mutation requires an explicit ownership/invalidation design; it was not added as a hidden identity-based cache. Output count synchronization could be reduced with over-allocation or allocator/workspace contracts, but those change memory behavior and the public API. These are the main measured future directions, rather than speculative kernel micro-optimizations.

These numbers are workstation evidence, not portable performance guarantees or test thresholds. Crossovers depend on atom-count distribution, density, cutoff, cell geometry, dtype, GPU, CUDA/PyTorch versions, and whether metadata can be reused. The new path remains faster in every reported workload, but its smallest single-structure advantage is only 1.69×; the strongest benefit is heterogeneous batch parallelism that removes thousands of per-structure launches.
