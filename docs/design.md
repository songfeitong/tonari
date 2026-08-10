# tonari 当前设计

## 公共边界

项目与 Python package 名为 `tonari`。Public surface 只有：

```python
pair_indices, cell_shifts = find_neighbors(
    positions,
    cells,
    pbc,
    cutoff,
    offsets=None,
)
```

不保留旧 API alias。函数名不强调 radius 或 PBC：它执行通用 scalar-cutoff neighbor search，periodicity 完全由 `pbc` 参数决定。

单结构输入为 `positions: (N, 3)`、`cells: (3, 3)`、`pbc: (3,)`，`offsets=None` 等价于 `[0, N]`。Batch 输入为拼接的 `positions: (N_total, 3)`、逐结构 `cells: (B, 3, 3)`、`pbc: (B, 3)` 和 `offsets: (B + 1,)`。`offsets` 使用 int64、从零开始、非递减且最后一个值等于 `N_total`；empty structures 由相邻相等 boundaries 表达。

Torch positions/cells 接受 float32 或 float64，所有 Torch arrays 必须同 device，pbc 为 bool，offsets 为 int64。CPU Tensor 走 native CPU backend，CUDA Tensor 走 native CUDA backend。NumPy 接受同样的 float/bool/int dtypes 与 single/batch shapes，只走 CPU backend并返回 NumPy arrays。所有 array 参数必须属于同一生态；frontend 在 native dispatch 前拒绝 NumPy/Torch 混用。

NumPy frontend 不是第二套搜索实现。Writeable、aligned、nonnegative-stride arrays 尽量通过 `torch.from_numpy` 零复制进入共享 CPU path；不能安全建立 Tensor view 的 arrays 会在 frontend 复制，非 contiguous layout 会在 native boundary 做必要 packing。Native output Tensor 的 CPU storage 直接导出为 NumPy arrays。

## Pair 方向与几何约定

`pair_indices` 为 int64 `[2, P]`，并固定 `source, target = pair_indices`。`cell_shifts` 为 int32 `[P, 3]`，每个 row 是施加在 source image 上的整数晶胞平移。Cell vectors 按行保存，因此 structure `b` 中 pair `k` 的 Cartesian displacement 是：

```python
positions[source[k]] - positions[target[k]] + cell_shifts[k] @ cells[b]
```

结果包含 squared distance 严格小于 `cutoff**2` 的全部有向 atom-image pairs。只排除 zero-shift onsite `(i, i, [0, 0, 0])`；保留 periodic self-images、同一 atom pair 的 multiple images 与 reverse pair。Inactive PBC axes 的 shift 必须为零。Batch members 之间绝不产生 pair。Output order 没有接口保证，correctness 使用完整五元 key set 比较。

函数不绑定 Å、Bohr 或其他长度单位。`positions`、`cells` 与 `cutoff` 必须使用同一单位；`pair_indices` 和 `cell_shifts` 无量纲。一致缩放三种长度输入不改变 pair identity。

## Representative wrapping

搜索只沿 active periodic axes wrap representatives。若 source 与 target 的整数 wraps 分别为 `q_source` 和 `q_target`，search image shift 为 `T`，则返回 `S = T - q_source + q_target`。这保证用原始输入 positions 与 output shifts 重建的 displacement 完全一致；把某个 representative 平移整数 cell 只会 relabel shifts，不改变物理 displacement multiset。

每个 representative wrap 和最终 output shift 都必须落在 int32 range。实现先验证 wraps，再以 int64 intermediate 计算差并检查 output；不接受静默截断。总 atom indexing 也限制在 int32-compatible implementation range。Positions/cells 必须 finite；active cell rows 必须线性独立，inactive rows 可为零或非零，完整 `3×3` cell 可以 rank deficient。

## 公共 periodic geometry

Python boundary 验证 ecosystem、shapes、dtypes、devices、cutoff 与 offsets，再把很小的 offsets/cells/pbc metadata 复制到 CPU。`_C_cpu.build_periodic_metadata_cpu` 在一次 native call 中完成 finite/rank check、active duals、image ranges 与拼接 image shifts；CPU 与 CUDA 复用同一结果。Empty structure 在 rank/repeat/image enumeration 前短路，其 image count 为零。

设 active-row matrix 为 `A`。Metadata 直接在 `A` 上执行 long-double one-sided Jacobi SVD，并据此判秩与构造 pseudoinverse；它刻意不形成 `A Aᵀ`，因为 normal equations 会平方 condition number。Dual column norms 给出 reciprocal face-height factors，每个 active image range 为 `ceil(cutoff * norm(dual_axis))`，inactive range 为零。该算法不需要补齐或求逆完整 cell，因此统一支持 rank-1 wire、rank-2 slab 与 full periodic triclinic cell。

非空 structures 的 batched image shifts 总数有 checked `2^24` resource guard，防止极小合法 cell 在 host Cartesian product 中不可控 OOM。这个 guard 是 implementation resource limit，不改变其范围内的物理 predicate。

## CPU backend

CPU 对 batch members 顺序处理，每个 structure 独立选择 exhaustive 或 Cartesian cell list。Native call 使用 `py::gil_scoped_release`，但内部不启动 thread pool；调用方可以在 DataLoader workers、进程池或 DDP 层决定并行度，避免 workers 与 backend threads 乘法 oversubscription。

### Exhaustive path

候选工作量为 `N² × image_count`。不超过 16,384 时，直接遍历 target、source 与 search image，只构造三维 displacement 并执行 strict `distance² < cutoff²`。它不建立 bins、hash table或 candidate tensors，适合常见小结构。16,384 来自固定 Matbench workload 的 crossover sweep，是性能参数而非 public contract。

### Cell-list path

大候选空间先 wrap representatives，以 target positions 的 Cartesian AABB 建立 cutoff-sized dense bins。只有落在 AABB 外扩 search cutoff 范围内的 periodic source images 才插入 linked nodes；每个 target 扫描相邻 27 bins 并筛选 candidate。Node 保存 int32 source、image index 与 next。

Search cutoff 只用于 conservative broad phase。实现根据 positions、wraps、image shifts 与 cells 的最大操作尺度构造浮点误差带；明显位于带内侧的 candidate 可使用 wrapped distance，边界壳按原始 positions 与 output shift 重算 public predicate。若保守 padding 不再小于 cutoff，则回退 exhaustive。最终 pair identity 始终由 public displacement formula 定义。

Dense bin grid 少于 `2^26` entries，并限制相对 possible source images 的空网格膨胀；超限时回退 exhaustive，避免极端稀疏 finite coordinates 分配巨大空 grid。固定 density、cutoff 与平均 pair 数时，该路径接近 `O(N + P)`，但仍必须写出全部 `P` 个 outputs。

## CUDA backend

CUDA 接受整个 heterogeneous batch。`CudaSearchSchedule` 将每个 structure 的 atom/image tasks 编入全局 block/node offsets，kernels 通过 segment lookup 找到所属 structure；所有生成 pair 都留在相应 offsets 区间。

Batch 内最大 structure 少于 256 atoms 时使用 fused exhaustive：`(source, target, image)` candidates 直接映射到 threads，block reduction 计数、block scan 写出，不 materialize dense candidates。达到 crossover 后使用 batched Cartesian cell list：prepare kernel 融合 representative wrapping 与 per-structure bounds；source images 插入 bins；每个 warp 负责一个 target，由前 27 lanes 遍历相邻 bins；count/prefix-sum 后精确分配并 write。

CUDA bin count 对每个 structure 在超过 `2^28` allocation limit 时饱和为 `limit + 1`，避免多个仍在 int64 内的巨大 sparse counts 先在 batched cumsum 中相加溢出。Host 观察到超限后回退 exhaustive，并只在此时检查 exhaustive block-grid bound。Cell-list nodes 与 shifts 受 int32 range 保护。

CUDA cell-list 的 wrapped predicate 只在 representative wraps 全为零时与 public original-position formula 具有相同浮点运算。Prepare kernel 把 nonzero wrap 编码进已有 bin-count status；已有 host read 同时返回 bins 与 status，命中后 whole call 复用 fused exhaustive canonical predicate。正常 well-wrapped path 不增加同步；unwrapped 大体系可能退化为 `O(N² × images)` 并受 exhaustive `< 2^31` blocks 限制。

Exact-size allocation 需要 count/prefix-sum 后的 device-to-host synchronization。Nonfinite input、wrap overflow 与 output-shift overflow 被编码进已有状态位置，与必要 read 一次返回，不为 validation 新增同步。所有 launches 使用 PyTorch current CUDA stream，并在目标 device guard 下运行。

## Autograd

Neighbor identity 是离散 topology，production 在搜索边界 detach 浮点 tensors，返回整数 arrays，无 custom backward。调用方必须使用原始 `positions`、`cells` 与返回的 `cell_shifts` 重建 `displacements`；identity 固定时，PyTorch gradients 正常流向连续几何。NumPy 路径自然不涉及 autograd。

## Reference 与测试

内部 `_reference.find_neighbors_reference` 与 public Torch shapes/signature 一致，独立执行 exhaustive PyTorch enumeration，不调用 production native search。它按原始 positions/output shifts 重建 displacement，并共享 int32/image-resource contract；只用于开发期 correctness，不属于 public surface。

68 项 tests 覆盖 public surface、NumPy/Torch ecosystem、single/batch shapes、单位一致缩放、finite/partial/full PBC、rank-deficient 与近共线 active rows、mixed batch、ordinary/large unwrapped representatives、int32 rejection、nonfinite rejection、multiple images、periodic self-images、empty tiny cell、image/bin resource limits、exact 与 nextafter cutoff、float32/float64、randomized differential、rotation/reflection covariance、CUDA current stream 与 continuous-geometry backward。Tests 比较完整 `(source, target, Sx, Sy, Sz)` sets，不冻结 output order。

## ELFES 只读需求核对

ELFES 当前 `group_atom_image_pairs` 接受单个 NumPy `Geometry`，用统一 `2 * max(atom_cutoffs)` broad cutoff 调 Vesin half list，再按 species-dependent cutoff 严格过滤并显式加入 onsite。它与 `tonari` 的共同需求是 NumPy CPU、单结构、scalar broad cutoff、partial/full PBC、strict boundary、multiple images、int32 shifts 与 unwrapped representative gauge。

`tonari` 返回 full directed pairs 并排除 onsite，因此未来 adapter 仍需定义 half-list canonicalization、species post-filter 与 onsite insertion。当前任务只确认能力覆盖，不修改 ELFES，也不把该 adapter 反向污染核心 API。

## 暂不支持

当前没有 pair sorting、neighbor cap、per-species cutoff、Verlet skin、prepared metadata/workspace cache、CUDA Graph capture、`torch.compile`/export contract 或 GNN/PyG adapter。未来 adapter 只在边界把 `pair_indices` 映射为 `edge_index`、把 `displacements` 映射为 `edge_vectors`；核心 neighbor-search API 不引入图专属术语。
