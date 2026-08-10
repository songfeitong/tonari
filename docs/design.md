# 当前设计

## 设计目标

Public surface 只有一个 `radius_graph_pbc(positions, ptr, cells, pbc, cutoff)`。CPU 与 CUDA 接受相同 batched tensor shapes、返回相同 dtypes，并共享完整 graph identity；device 只决定 execution backend，不改变物理语义。CPU backend 是一等实现而非 CUDA fallback，CUDA backend 也不需要迁就 CPU 的逐 structure 执行方式。

Native build 分成始终构建的 `_C_cpu` 与可选的 `_C_cuda`。`_C_cpu` 同时包含公共 periodic metadata 与 CPU search；只有 PyTorch 本身支持 CUDA 且检测到 CUDA toolkit 时才构建 `_C_cuda`。这允许 CPU-only environment 完整安装和运行，又避免把本质上 host-side 的 1–3 维几何准备复制到两个 extensions。

## 几何约定

对于 edge key `(source, target, S)`，返回的 Cartesian vector 是 `r_source - r_target + S @ cell`，三个 cell vectors 按行保存。只有 `pbc[batch, axis]` 决定某个 cell row 是否具有周期性。Active rows 必须线性独立；inactive rows 可以为零或非零，完整 `3 x 3` cell 也可以 rank deficient。Graph 是完整有向图，使用严格 cutoff，只排除 zero-shift onsite edge，并保留其他 periodic self-images 和 multiple images。

搜索时只沿 active periodic axes wrap representatives。若 source 和 target representatives 的整数 wraps 分别为 `q_source` 和 `q_target`，search-image shift `T` 会按 `S = T - q_source + q_target` 返回。这保证使用原始输入 representatives 重建的 vector 完全一致，使 representative translation 只 relabel shifts，而不改变物理 edges。每个 `q` 和最终 `S` 都必须落在 int32 范围内；任一条件不满足都会显式报错。

对于 active-row matrix `A`，metadata 使用其 pseudoinverse transpose 作为 dual；满行秩时与 `A.T @ inv(A @ A.T)` 等价。Dual column norms 是 reciprocal face-height factors，每个 active image range 为 `ceil(cutoff * norm(dual_axis))`，inactive range 为零。Native implementation 直接在 active rows 上执行 long-double one-sided Jacobi SVD，同时完成 rank check 与 pseudoinverse；它刻意不先形成 Gram matrix，因为正规方程会把条件数平方并把合法的近共线 active rows误判为退化。该处理无需补齐或求逆完整 cell，统一支持 finite Geometry、rank-1 wire、rank-2 slab、triclinic cell 和 full periodic cell。

## 公共 metadata 与 device schedule

Python boundary 先验证 shapes、dtypes、devices 和 cutoff，再把很小的 `ptr/cells/pbc` 复制到 CPU。`_C_cpu.build_periodic_metadata_cpu` 一次返回 duals、拼接的 int32 image shifts 和 image pointers；empty structure 的 image count 直接为零，因此 tiny periodic cell 不会为零 atoms 枚举 images。Cells 的 finite 与 active-rank 验证也在这个负责边界完成。

公共 `SearchMetadata` 只包含 CPU/CUDA 都需要的信息：duals、image shifts、image pointers、atom counts、image counts 与 maximum atoms。非空 batch 的累计 periodic image shifts 设有 `2^24` resource limit，并在 Cartesian product 分配和枚举前以 checked multiplication 验证；这把极小 cell 的不可控 host OOM 变成确定错误。CUDA-only block pointers、node pointers、total blocks 和 total nodes 位于独立 `CudaSearchSchedule`。这个拆分消除了旧设计中“搜索 metadata 天生等于 CUDA launch metadata”的偶然耦合。

## Hybrid CPU 搜索

CPU public batch 在一个释放 GIL 的 native C++ call 中按 structure 顺序处理。每个非空 structure 先用 dual 把 representatives wrap 到 active periodic fundamental directions，验证 finite positions 和 int32 wraps，并预计算每个 image shift 的 Cartesian translation。浮点计算保留输入的 float32/float64 dtype；metadata geometry 在 float64 中建立后转换为输入 dtype。

### Exhaustive path

若 `N² × image_count <= 16,384`，直接遍历 target、source 和 search image。每个 candidate 只构造三维 displacement、比较严格 `distance² < cutoff²`，命中后转换为原始 representative 对应的 output shift。该路径没有 bins、hash table 或 candidate tensors，适合小 structure 与 finite molecules。

16,384 是真实 Matbench threshold sweep 的 provisional performance constant。它按实际候选数而非单纯 atom count 决策，因此 small-cell multiple-image structure 会比相同 `N` 的 finite structure 更早转入 cell list。该值只影响性能，不影响输出，不写进 correctness tests。

### Cell-list path

大候选空间先以 wrapped target representatives 的 Cartesian AABB 建立 dense bins。对每个 source atom 与 periodic image，只把落在 AABB 外扩 search cutoff 范围内的 image 插入；node 保存 int32 source、image index 和 linked-list next。每个 target 固定检查自身 bin 周围 `3 x 3 x 3` stencil，再对 nodes 做距离筛选。曾实现的 target-to-bin corner pruning 在严格 cutoff 边界会受 bin-coordinate 舍入影响而破坏双向语义，收益又很小，因此 production 明确不使用它。

Search cutoff 不是物理 tolerance，而是只用于 broad phase 的保守上界。Wrapped search formula 与使用原始 representatives/output shifts 的 public formula 在实数上等价，但对很大的未 wrap coordinates 具有不同浮点消去；实现根据 position、wrap、image shift 与 cell 的最大操作尺度扩大 broad-phase cutoff。明显处于误差带内侧的 candidate 可直接接受 wrapped distance，边界壳必须用 public formula 重算严格 `distance² < cutoff²`；若所需 padding 超过 cutoff，则不冒险使用 cell list，回退 exhaustive。最终 graph 始终由 public formula 定义。

固定 density、cutoff 和邻居数时，该路径接近 `O(N + E)`；但它仍必须写出全部 `E` 条有向 edges。Dense bin grid 限制为少于 `2^26` entries，并要求平均每个 possible source image 不超过 64 bins，否则回退 exhaustive，避免极端稀疏 finite coordinates 分配巨大空 grid。回退可能在人工超稀疏大体系上退化为 `O(N²)`，这是明确记录的安全取舍。

CPU backend 当前内部单线程。pybind binding 使用 `gil_scoped_release`，允许 Python runtime 继续调度其他 threads，但不隐式启动 OpenMP/thread pool。对 DDP、multiprocessing DataLoader 或多个 independent structures，推荐由上层选择进程级并行度，避免 backend nested parallelism。

## Hybrid CUDA 搜索

CUDA 小结构路径先用 O(N) kernel 验证 finite positions 并预计算 int32 representative wraps，再把完整 `(source, target, image)` candidate space 直接映射到 CUDA blocks，不 materialize candidate tensors。Count pass 每个 block 只执行一次 global atomic；host 精确分配输出后，write pass 使用 block scan，并由每个 block 一次性预留输出区间。当 batch 内所有 structure 都少于 256 atoms 时启用。

大结构路径先 wrap representatives，并在同一 kernel 中融合 per-structure Cartesian bounds；随后把相关 periodic source images 插入 cutoff-sized Cartesian bins。每个 warp 负责一个 target，由前 27 个 lanes 遍历相邻 bins；两次 query pass 先统计 per-target edges，再写入 device prefix sum 生成的 offsets。每个 structure 的 bin count 一旦超过 `2^28` allocation limit 就在 device 上饱和为 `limit + 1`，因此 batched cumsum 不会先被无用的巨大精确计数溢出；host 看到超过 limit 后回退 fused exhaustive，并在此时检查 exhaustive block-grid bound。

CUDA cell-list 的 wrapped predicate 只在所有 representative wraps 都为零时与 public original-position formula 具有相同浮点运算。Prepare kernel 因此把“发现任一 nonzero wrap”编码进原有 bin-count status word，已有 cumsum host read 同时返回 bins 与 status；命中后 whole call 直接复用 `radius_graph_pbc_cuda` 的 original-position/output-shift predicate，不维护第三份容易漂移的 strict-cutoff 实现。该设计不增加正常 well-wrapped cell-list 的 host synchronization，但 unwrapped 大体系会退化为 exhaustive，并受 `< 2^31` thread blocks 限制。

256-atom crossover 是目标 Blackwell GPU 上的 provisional heuristic。它与 CPU 的 16,384-candidate threshold 相互独立：CUDA 的固定成本、parallel occupancy 与 batch composition 不适合套用 CPU 的决策规则。

## PyTorch、autograd 与错误边界

浮点输入只在离散 topology 构造中 detach。Extension 返回 `int64 [2, E]` edges 与 `int32 [E, 3]` shifts；连续 vectors 必须由调用方使用原始 `positions/cells` 重建，因此无需 custom autograd function，且 topology 固定时 gradients 正常流向两者。

CPU positions 的 finite/range validation 在 native per-atom preparation 中执行。CUDA 对应 flags 融合进已有 prepare/count synchronization，避免额外 D2H sync。Cells finite 和 active-rank validation 位于共享 host metadata 边界。总 atoms、representative wraps、returned shifts 和 cell-list nodes 均受明确 int32 indexing contract 约束；不提供“内部算成 int64、最后悄悄截断”的伪支持。

CUDA exact-size output allocation 需要 count/prefix-sum 后的 device-to-host synchronization；CPU 使用 native vectors 收集后一次精确分配和 memcpy。One-shot API 每次重建 metadata，当前没有 identity-based implicit cache，因为原地修改 cells/PBC 会产生 stale graph，而 hash/copy 可能吃掉收益。未来 prepared API 必须先定义 ownership、mutation invalidation 与 workspace lifetime。

## ELFES 需求适配评估

ELFES 当前 `group_atom_image_pairs` 接受一个 NumPy `Geometry`，构造 Vesin `NeighborList(cutoff=2 * max(atom_cutoffs), full_list=False, n_threads=1)`，然后按 species-dependent pair cutoff 二次过滤，并显式加入 onsite。它与本 backend 的共同需求是：单 structure、scalar broad cutoff、partial/full PBC、strict cutoff、multiple images、int32 shifts 和 unwrapped representative gauge。

因此 native CPU search 足以承担 ELFES broad-phase neighbor enumeration，并可用 `torch.from_numpy` 零拷贝包装 positions/cell；但它不是当前函数的 drop-in replacement。差异包括 Torch API、full directed output、zero onsite exclusion，以及 ELFES 需要 deterministic Hermitian half-list 和 explicit onsite。未来优雅接入应在清晰 adapter 或更底层 native search policy 中表达这些差异，而不是在 ELFES 内随手拼接一串临时转换。本任务只确认可行性，没有修改 ELFES 或删除其 Vesin dependency。

## 正确性策略

`reference_radius_graph_pbc` 是独立 exhaustive PyTorch implementation，不调用 CPU/CUDA native search，并按原始 positions 与 output shifts 逐轴执行 public vector formula。53 项 unit tests 比较完整 `(source, target, Sx, Sy, Sz)` key sets，不冻结 edge order；覆盖 finite/partial/full PBC、rank-deficient 与近共线但满秩 active rows、mixed batch、普通和约一亿 cell translation 的 unwrapped representatives、int32 range rejection、nonfinite rejection、multiple images、periodic self-images、tiny-cell empty structure、production/reference image-count resource limit、sparse batched bin-count saturation、exact 与 nextafter cutoff boundary、float32/float64、CPU randomized differential、rotation/reflection covariance、CUDA non-default stream，以及 continuous-geometry backward。

ASE 提供 triclinic partial-PBC external case。正式 CPU benchmark 还对全部 1,536 个 Matbench structures、2,780,158 个 keys 使用 Vesin 0.6.1 做 exact comparison；CUDA 既有正式记录使用同一 corpus，并另外与独立 Equiformer/FairChem-style dense semantics baseline 对比。
