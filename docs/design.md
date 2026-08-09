# Current design

## Geometry contract

For an edge key `(source, target, S)`, the returned Cartesian vector is `r_source - r_target + S @ cell`, with the three cell vectors stored as rows. Only `pbc[batch, axis]` determines whether a cell row is periodic. Active rows must be linearly independent; inactive rows may be zero, nonzero, or make the complete `3 x 3` cell rank deficient. The graph is fully directed, uses a strict cutoff, excludes the zero-shift onsite edge, and retains all other periodic self-images and multiple images.

Search representatives are wrapped only along active periodic axes. If the source and target representatives have integer wraps `q_source` and `q_target`, a search-image shift `T` is returned as `S = T - q_source + q_target`. This preserves the exact vector in the original input representatives and makes representative translations relabel shifts rather than change physical edges.

For the active-row matrix `A`, metadata construction forms the dual `A.T @ inv(A @ A.T)`. Its column norms are reciprocal face-height factors, and each active image range is `ceil(cutoff * norm(dual_axis))`; inactive ranges are zero. This supports finite geometries, rank-1 wires, rank-2 slabs, triclinic cells, and full periodic cells without completing or inverting the full cell.

## Hybrid CUDA search

The small-structure path maps the complete `(source, target, image)` candidate space directly onto CUDA blocks without materializing candidate tensors. A count pass uses one global atomic per block, the host allocates exact-size outputs, and a write pass uses a block scan plus one atomic reservation per block. This path avoids cell-list setup overhead and is selected when every structure has fewer than 256 atoms.

The large-structure path wraps all representatives, fuses per-structure Cartesian bounds into that kernel, and inserts relevant periodic source images into cutoff-sized Cartesian bins backed by a dense head array and linked node array. One warp owns each target, its first 27 lanes traverse the neighboring `3 x 3 x 3` bins, and two query passes first count per-target edges and then write into offsets produced by a device prefix sum. Because bin size equals the cutoff, these 27 bins are sufficient independent of cell shape; periodic geometry is already represented by the inserted images.

The 256-atom crossover is a provisional, named performance choice measured on the target Blackwell GPU. The cell-list path falls back to the fused exhaustive path when its Cartesian bounding box would require more than `2^28` dense bins or more than 64 bins per inserted node, avoiding pathological memory use for extremely sparse finite coordinates. Cell-list nodes and returned shifts use int32 storage; atom indices and output offsets use int64.

## PyTorch boundary

The Python boundary validates shapes, dtypes, devices, `ptr`, cutoff, and independence of active cell rows, then builds the small per-structure search metadata. Floating inputs are detached only for discrete topology construction. The extension guards the input device, launches on PyTorch's current CUDA stream, and returns `int64 [2, E]` edges plus `int32 [E, 3]` shifts. Continuous vectors are intentionally reconstructed with normal PyTorch operations from the original inputs, so no custom autograd function is needed.

Exact-size allocation currently requires device-to-host synchronization after count or prefix-sum passes. Search metadata also moves the small `ptr/cells/pbc` tensors to the CPU for robust batched rank and dual calculations. On a 32,768-atom Matbench-derived supercell, isolated median wall times were about 0.177 ms for metadata and 0.323 ms for the extension with synchronization; the public one-shot API measured 0.409 ms in the formal benchmark. Caching is potentially valuable for static geometry metadata, but it is not a low-risk internal optimization because tensor mutation and cache invalidation need an explicit API contract.

## Correctness strategy

`reference_radius_graph_pbc` is an independent exhaustive PyTorch implementation and does not call either CUDA path. Unit tests compare complete `(source, target, Sx, Sy, Sz)` key sets rather than edge order, and cover finite, partial and full PBC, rank-deficient cells, mixed batches, unwrapped representatives, multiple images, periodic self-images, exact and nextafter cutoff boundaries, float32/float64, empty members, non-default streams, and continuous-geometry backward. ASE supplies an external triclinic partial-PBC check, and Vesin 0.6.1 supplies an external exact check over all 1,536 sampled real structures. CUDA memcheck completed representative dense and cell-list cases with zero errors.
