# 当前设计

## 几何约定

对于 edge key `(source, target, S)`，返回的 Cartesian vector 是 `r_source - r_target + S @ cell`，三个 cell vectors 按行保存。只有 `pbc[batch, axis]` 决定某个 cell row 是否具有周期性。Active rows 必须线性独立；inactive rows 可以为零或非零，完整 `3 x 3` cell 也可以 rank deficient。Graph 是完整有向图，使用严格 cutoff，只排除 zero-shift onsite edge，并保留其他 periodic self-images 和 multiple images。

搜索时只沿 active periodic axes wrap representatives。若 source 和 target representatives 的整数 wraps 分别为 `q_source` 和 `q_target`，search-image shift `T` 会按 `S = T - q_source + q_target` 返回。这保证使用原始输入 representatives 重建的 vector 完全一致，使 representative translation 只 relabel shifts，而不改变物理 edges。每个 `q` 必须落在 int32 范围内，最终 `S` 也必须落在 int32 范围内；任一条件不满足都会显式报错。

对于 active-row matrix `A`，metadata 构造使用 dual `A.T @ inv(A @ A.T)`。它的 column norms 是 reciprocal face-height factors，每个 active image range 为 `ceil(cutoff * norm(dual_axis))`，inactive range 为零。该处理无需补齐或求逆完整 cell，统一支持 finite Geometry、rank-1 wire、rank-2 slab、triclinic cell 和 full periodic cell。

## Hybrid CUDA 搜索

小结构路径先用一个 O(N) kernel 验证 finite positions 并为每个 atom 预计算 int32 representative wraps，再把完整 `(source, target, image)` candidate space 直接映射到 CUDA blocks，不 materialize candidate tensors。Count pass 每个 block 只执行一次 global atomic；host 精确分配输出后，write pass 使用 block scan，并由每个 block 一次性预留输出区间。该路径避免在每个 candidate 内重复 dual dot/floor，也避免 cell-list setup overhead；当所有 structure 都少于 256 atoms 时启用。

大结构路径先 wrap 所有 representatives，并在同一 kernel 中融合 per-structure Cartesian bounds；随后把相关 periodic source images 插入以 cutoff 为边长的 Cartesian bins，底层使用 dense head array 和 linked node array。每个 warp 负责一个 target，由前 27 个 lanes 遍历相邻 `3 x 3 x 3` bins；两次 query pass 先统计 per-target edges，再写入 device prefix sum 生成的 offsets。因为 bin size 等于 cutoff，不论 cell shape 如何，这 27 个 bins 都足够；periodic geometry 已由插入的 images 表达。

256-atom crossover 是在目标 Blackwell GPU 上实测后命名保存的 provisional performance choice。若 Cartesian bounding box 需要超过 `2^28` 个 dense bins，或每个 inserted node 平均对应超过 64 个 bins，cell-list 路径会回退到融合 exhaustive path，以避免极端稀疏 finite coordinates 造成病态内存开销。这两个数是明确的工程安全阈值，不是物理 tolerance。Cell-list nodes 和返回 shifts 使用 int32 storage；atom indices 和 output offsets 使用 int64。

## PyTorch 边界

Python boundary 验证 shapes、dtypes、devices、`ptr`、cutoff、finite cells 和 active cell rows 的独立性，再构造较小的 per-structure search metadata。Positions 的 finite/range flags 融合进现有 CUDA prepare/wrap pass，并在本来就需要的 count synchronization 处检查，不增加额外 D2H sync。浮点输入只在离散 topology 构造中 detach。Extension guard 输入 device，使用 PyTorch current CUDA stream launch，并返回 `int64 [2, E]` edges 与 `int32 [E, 3]` shifts。连续 vectors 由原始输入通过普通 PyTorch operations 重建，因此无需 custom autograd function。

精确大小分配目前需要在 count 或 prefix-sum pass 后执行 device-to-host synchronization。Search metadata 也会把很小的 `ptr/cells/pbc` tensors 移到 CPU，以稳健地执行 batched rank 和 dual 计算。在一个 32,768-atom 的 Matbench-derived supercell 上，单独测得 metadata median wall time 约 0.177 ms，extension 连同同步约 0.323 ms，正式 benchmark 的公开 one-shot API 为 0.409 ms。对于静态 geometry metadata，cache 可能有价值；但 tensor mutation 与 cache invalidation 需要明确 API contract，因此它不是一个低风险的内部优化。

## 正确性策略

`reference_radius_graph_pbc` 是独立的 exhaustive PyTorch 实现，不调用任一 CUDA path。26 项 unit tests 比较完整 `(source, target, Sx, Sy, Sz)` key sets，而不冻结 edge order；覆盖 finite/partial/full PBC、rank-deficient cell、mixed batch、unwrapped representatives、int32 range rejection、nonfinite rejection、multiple images、periodic self-images、tiny-cell empty structure、exact 与 nextafter cutoff boundary、float32/float64、non-default stream，以及 continuous-geometry backward。ASE 提供一个外部 triclinic partial-PBC 检查，Vesin 0.6.1 则对全部 1,536 个真实抽样结构执行 external exact check。CUDA memcheck 在代表性的 exhaustive 和 cell-list cases 上报告零错误。
