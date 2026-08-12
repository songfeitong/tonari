# 设计文档

本文是当前公共契约和 backend 行为的技术说明。它描述必须保持的语义，不记录开发过程中已经撤销的实现、临时 benchmark 或逐次审查历史。

## 公共接口

```python
results = neighbor_list(
    quantities,
    positions,
    cell,
    pbc,
    cutoff,
    batch_ptr=None,
    *,
    algorithm="auto",
    num_threads=1,
    sorted=False,
    half_list=False,
    include_self=False,
)
```

单结构输入：

- `positions`: `(N, 3)`
- `cell`: `(3, 3)`
- `pbc`: `(3,)`
- `batch_ptr=None`

Batch 输入：

- `positions`: `(N_total, 3)`，按 structure 拼接
- `cell`: `(B, 3, 3)`
- `pbc`: `(B, 3)`
- `batch_ptr`: `(B + 1,)`

`batch_ptr` 从零开始、非递减，最后一个值等于 `N_total`。相邻的相同值表示 empty structure；`batch_ptr=None` 等价于单结构边界 `[0, N]`。

NumPy 与 Torch 使用同一个函数，所有 array 参数必须属于同一生态。NumPy 只走 CPU；Torch 根据 `positions.device` 选择 CPU 或 CUDA。`positions` 和 `cell` 必须使用相同的 `float32` 或 `float64` dtype，`pbc` 为 bool，`batch_ptr` 为 int64；Torch arrays 还必须位于同一 device。

函数与具体长度单位无关，但 `positions`、`cell` 和 `cutoff` 必须使用同一单位。`cutoff` 必须有限且为正数。

`algorithm` 接受 `"auto"`、`"brute_force"` 或 `"cell_list"`。它只决定如何寻找 candidates，不改变 pair identity；精确选择规则和 fallback 语义见[算法选择](algorithm-selection.md)。

`num_threads` 是包括调用线程在内的正整数 CPU thread count，默认值为 `1`。NumPy 与 Torch CPU 都服从该参数；Torch CUDA 不使用 CPU search pool，因此只接受默认值。选择建议与并行生命周期见 [CPU 多线程](cpu-multithreading.md)。

## Quantities、输出与方向

`quantities` 是必填字符串。每个字符选择一个返回 array，tuple 中的顺序与字符串一致；重复字符会产生重复位置，空字符串返回空 tuple，未知字符报错。设找到 `E` 个 pairs：

| 字符 | dtype | shape | 含义 |
| --- | --- | --- | --- |
| `i` | int64 | `(E,)` | source atom indices |
| `j` | int64 | `(E,)` | target atom indices |
| `P` | int64 | `(E, 2)` | 每行一个 `(source, target)` pair |
| `S` | int32 | `(E, 3)` | 施加在 target image 上的整数 cell shifts |
| `d` | input float | `(E,)` | distances |
| `D` | input float | `(E, 3)` | source 指向 shifted target 的 displacement vectors |

所有返回值都属于输入的 array 生态；Torch 结果还保持输入 device。`P` 采用 edge-first 布局，使它与 `S`、`d`、`D` 在第一维逐 pair 对齐。PyG/GNN adapter 应使用 `edge_index = P.T.contiguous()`，核心 API 不返回 `(2, E)` graph layout。

Cell vectors 按行存储。单结构中 pair `k` 的 displacement 为：

```text
positions[target[k]] - positions[source[k]] + S[k] @ cell
```

Batch 中先由 `batch_ptr` 确定 pair 所属 structure `b`，再使用 `cell[b]`。Pairs 不会跨 structure 产生，inactive PBC axes 上的 shift 必须为零。

结果只包含 squared distance 严格小于 `cutoff**2` 的 atom-image pairs。默认输出顺序没有保证。`sorted=True` 只保证 source indices 非递减；相同 source 内的 target indices 和 cell shifts 顺序仍未指定。

## Full、half 与 self

默认 `half_list=False, include_self=False` 返回 full directed list，并排除 zero-shift self `(i, i, [0, 0, 0])`。Pair `(source, target, S)` 的 reverse 是 `(target, source, -S)`。

`half_list=True` 比较五元 key `(source, target, Sx, Sy, Sz)` 与其 reverse key，只保留 lexicographically smaller 的一侧。它只去除 reverse redundancy，不执行 minimum-image reduction，因此同一 atom pair 的不同 periodic images 仍会分别保留。

`include_self=True` 为每个原子加入且只加入一个 zero-shift self。Periodic self-images `(i, i, S != 0)` 始终作为普通 cutoff pairs 处理；half list 会在 `S` 与 `-S` 中保留 canonical 一侧。

## Periodic geometry

`pbc` 是 periodicity 的唯一依据。非空 structure 的 active cell rows 必须线性独立；inactive rows 可以为零或非零，完整 cell 可以 rank deficient。Finite system 即使带有非零 box，只要 `pbc` 全 false，就不会产生 periodic shifts。

搜索需要保留 cutoff 内所有 images，包括同一 `(source, target)` 对应多个 shifts，以及小晶胞中的 periodic self-images。实现不能擅自应用 minimum-image convention。

输入 positions 不要求 wrap。内部搜索可以把 representatives 移入 active periodic cell，但返回 shift 必须补偿这次变换，使原始 positions 按公共 displacement 公式得到相同的物理 vectors。Representative wraps 与最终 output shifts 必须可由 int32 表示，差值计算使用更宽的 intermediate 并在输出前检查。

Positions 和 cell 必须有限。非空 structure 的 active rows、periodic image 数、atom indexing 和 backend allocations 还必须落在当前实现可安全表示的范围内；超限必须报错或选择安全 fallback，不能静默截断。

## 共享 geometry 与 pair policy

Framework-neutral C++ core 根据 cell、pbc、atom counts 和 cutoff 建立 active duals 与必要的 periodic image ranges。它不要求求逆完整 cell，因此 rank-1、rank-2 和 full-periodic geometry 使用同一条路径。Empty structure 在几何准备阶段直接短路。

Pair 方向、zero-shift self 和 canonical half rule 属于共享 policy。CPU 与 CUDA 可以采用不同的 broad phase 和数据布局，但最终 acceptance 必须服从同一个公共 displacement、strict cutoff 和 pair identity。

## CPU backend

CPU 在一个 native call 内处理 batch。`"auto"` 为每个 structure 独立选择 brute-force path 或 Cartesian cell list；显式指定算法时，每个非空 structure 都使用该路径。

Brute-force path 直接枚举 source、target 和相关 cell translations。Cell-list path wrap representatives、建立 cutoff-sized bins、插入可能相关的 periodic target images，再让每个 source 查询邻近 bins。Broad phase 必须保守；接近浮点边界的 candidate 会按原始 positions 与最终 shift 重算公共 strict predicate。

CPU core 在 `num_threads>1` 时使用进程级复用 worker pool，并在 native search 期间释放 Python GIL。Structures 和大型 structure 的 source ranges 进入同一个动态 task schedule；task-local outputs 按 source-major 顺序直接合并到 framework array。默认单线程用于避免与 DataLoader workers、DDP 和其他 threaded runtimes 隐式叠加。

## CUDA backend

CUDA provider 一次接收完整 batch。`"auto"` 为整个调用选择 fused brute-force path 或 batched cell list；显式指定算法时直接使用相应路径。

所有 CUDA tasks 都携带 structure segmentation，geometry、image insertion、query 和 output 只在所属 structure 内执行。Kernels 使用 PyTorch current stream 和当前 device；不同 cell、PBC patterns 与 atom counts 可以存在于同一 batch。

正常 well-wrapped input 可以使用 batched cell list。`"auto"` 遇到不适合 cell list 的 unwrapped representatives 或 bin layout 时回退 brute force；显式 `"cell_list"` 则报错。因此这类输入不保证 cell-list complexity。

## NumPy、Torch 与内存

NumPy CPU 与 Torch CPU 通过不同的 binding 调用同一个 C++ search core。NumPy 路径不导入或链接 LibTorch；Torch CUDA 使用独立 provider。Frontend 只在 dtype、device 或 contiguous-memory 要求需要时整理输入，不实现第二套 search。

Native provider 生成 edge-first `P` 与 `S`；frontend 按 `quantities` 选择 `i/j/P/S`，并在需要时用原始 positions 与 cell 计算 `d/D`。Neighbor identity 是离散结果，不参与 autograd；Torch 的 `d/D` 在固定 neighbor identity 下保持对原始浮点 tensors 的梯度。

## Reference 与验证原则

内部 reference 使用独立 brute-force enumeration，只用于 tests 和 differential validation，不属于 public surface。Production tests 以公开 pair keys 和 displacement 公式为准，不冻结 output order 或 `"auto"` 的内部选择。

外部 Vesin 与 ASE 使用相同的 pair 方向和 target-shift convention；验证只需统一 half/self policy 后进行 exact key comparison。性能 benchmark 与 correctness reference 是两个角色：慢但独立的实现仍可作为 reference，快的 baseline 也必须先证明语义一致。

## 暂不支持

当前不提供 neighbor cap、species-dependent cutoff、Verlet skin、prepared metadata/workspace、CUDA Graph capture、`torch.compile`/export contract 或 GNN/PyG adapter。未来 adapter 应在边界把 `P` 转置为 `edge_index`，把 `D` 映射为 `edge_vectors`，而不是把 graph 专属术语引入核心 API。
