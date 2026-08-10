# tonari

`tonari` 是一个同时面向 NumPy 与 PyTorch、CPU 与 CUDA 的周期性 neighbor-search 实验项目。它用一个公共函数返回 strict cutoff 内全部有向 atom-image pairs；periodicity 只由 `pbc` 表达，核心接口不依赖 GNN/PyG 术语。Production implementation 不依赖 Vesin；Vesin 只用于外部 correctness reference 与公平性能 baseline。

第一次阅读建议从[算法总览](docs/algorithm-overview.md)开始；精确契约与内部结构见[设计文档](docs/design.md)，真实材料测量见[性能方法与结果](docs/benchmark.md)，开发取舍见[工作记录](notes/work-log.md)，独立审查见[终审记录](docs/review.md)。

## 公共 API

```python
pair_indices, cell_shifts = find_neighbors(
    positions,
    cells,
    pbc,
    cutoff,
    offsets=None,
)
```

单结构输入使用 `positions: (N, 3)`、`cells: (3, 3)`、`pbc: (3,)`，无需构造 `offsets`。Batch 输入把原子坐标拼接成 `positions: (N_total, 3)`，并传入 `cells: (B, 3, 3)`、`pbc: (B, 3)` 与 `offsets: (B + 1,)`；`offsets` 从零开始、非递减，最后一个元素等于 `N_total`。

Torch 输入返回 Torch tensors，可位于 CPU 或 CUDA；NumPy 输入返回 NumPy arrays，并复用同一 native CPU backend。所有 array 参数必须属于同一生态，NumPy/Torch 混用会被明确拒绝。`positions` 与 `cells` 使用相同的 `float32` 或 `float64` dtype，Torch arrays 还必须位于同一 device；`pbc` 为 bool，`offsets` 为 int64。函数与具体长度单位无关，但 `positions`、`cells` 和 `cutoff` 必须使用同一单位。

```python
import torch

from tonari import find_neighbors

positions = torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]])
cells = torch.eye(3) * 4.0
pbc = torch.tensor([False, False, False])

pair_indices, cell_shifts = find_neighbors(positions, cells, pbc, cutoff=1.0)
source, target = pair_indices
displacements = (
    positions[source] - positions[target] + cell_shifts.to(positions.dtype) @ cells
)
distances = torch.linalg.vector_norm(displacements, dim=1)
```

`pair_indices` 为 int64、形状 `[2, num_pairs]`；`cell_shifts` 为 int32、形状 `[num_pairs, 3]`。Cell vectors 按行保存，shift 施加在 source image，因此 displacement 是 `positions[source] - positions[target] + cell_shifts @ cell`。结果包含距离严格小于 `cutoff` 的全部有向 pairs，只排除 `(i, i, [0, 0, 0])`，保留 periodic self-images 与 multiple images，不产生跨 structure pairs，并保证 inactive PBC axes 上的 shift 为零。输出顺序没有接口保证。

NumPy 调用使用完全相同的函数与形状规则：

```python
import numpy as np

from tonari import find_neighbors

positions = np.array([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]])
cells = np.eye(3) * 4.0
pbc = np.array([False, False, False])
pair_indices, cell_shifts = find_neighbors(positions, cells, pbc, 1.0)
```

Pair identity 是离散结果，不参与 autograd。Torch `positions` 与 `cells` 可以设置 `requires_grad=True`；按上例从原始浮点 tensors 重算 `displacements`，即可在 neighbor identity 固定时对连续几何求导。完整的 Torch-style `Args:`、`Returns:`、`Raises:`、`Note:` 与 `Example:` 文档位于 `find_neighbors.__doc__`。

## 构建与验证

CPU build 需要 Python 3.12、PyTorch 2.12.1、C++20 compiler 和 Ninja。如果 PyTorch 与本机 CUDA toolkit 都可用，`setup.py` 同时构建 `_C_cpu` 与 `_C_cuda`；否则只构建始终可用的 CPU extension。NumPy 是明确的运行时依赖。

本机执行复用 ELFES 已有环境，没有重新下载 Python、PyTorch 或 PyG：

```bash
cd /home/ftsong/projects/elfes-workspace/tonari
/home/ftsong/projects/elfes-workspace/elfes/.venv/bin/python setup.py build_ext --inplace

PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 \
  /home/ftsong/projects/elfes-workspace/elfes/.venv/bin/python -m pytest -q
```

系统 CUDA toolkit 为 13.2，而 PyTorch wheel 使用 CUDA 13.0 构建，因此 extension build 会出现 minor-version warning；当前机器上的编译、导入、74 项测试与 benchmark 均成功。正式发布工具链仍应优先让 toolkit minor version 与 PyTorch wheel 对齐。

## 真实材料证据

主要 workload 是从 `matbench_mp_e_form` 确定性抽取的 1,536 个真实晶体，覆盖 1–444 atoms、1,343 个不同化学式及多样 cell shapes。原始 Parquet 与派生 cache 位于 Git ignored `cache/`；仓库只保存固定数据 revision、SHA-256、可重复脚本和 sample manifest。没有下载 OMat24，也没有保留本任务不需要的 energy/force labels。

在 AMD Ryzen Threadripper PRO 9975WX 的单个固定 core 上，`DataLoader(batch_size=1)` 的完整 epoch 中，`tonari` 为 143.80 ms，复用同一个单线程 Vesin `NeighborList` 为 248.08 ms，前者快 1.73×。单个 64-atom 真实结构为 0.0417 ms 对 0.0453 ms；512-atom real-derived supercell 已接近交叉点，之后 Vesin 的成熟 CPU cell list 更快。

在 NVIDIA RTX PRO 6000 Blackwell 上，`DataLoader(batch_size=32)` 的完整 epoch 中，`tonari` 为 12.11 ms，逐 structure Vesin GPU 为 494.39 ms。代表性 32-structure batch 中，`tonari` 为 0.225 ms、Vesin 为 9.29 ms、独立 Equiformer/FairChem-style dense baseline 为 42.78 ms，三者得到完全相同的 43,842 个 pair keys。32,768-atom real-derived supercell 中，`tonari` 为 0.254 ms、Vesin 为 1.491 ms。

CPU 与 CUDA 都在全部 1,536 个结构、2,780,158 个 `(source, target, Sx, Sy, Sz)` keys 上与 Vesin 精确一致。正式 JSON 记录 clean implementation revision、data/cache/extension SHA 与全部 timing samples；Nsight summary 与 CSV 保存 kernel、memory、API 和 NVTX 证据。

## 支持范围与边界

当前支持 finite、partial/full PBC、triclinic 与 rank-deficient inactive cell rows、empty structures、未 wrap representatives、mixed-structure batches、CUDA current stream，以及 float32/float64。Active periodic rows 必须线性独立；positions/cells 必须有限。Representative periodic wraps、返回 cell shifts、atom indexing 与内部离散索引必须落在已文档化的整数范围内，越界会报错而不是截断。

CPU backend 当前单线程并在 batch 内顺序处理 structures；CUDA 对正常 well-wrapped batch 使用 fused exhaustive 或 batched cell list。大规模未 wrap CUDA input 为保持 public displacement formula 的浮点语义，可能回退 exhaustive 并失去 cell-list 复杂度。One-shot API 每次重建 metadata；当前不提供排序、neighbor cap、species-dependent cutoff、Verlet skin、prepared metadata cache、CUDA Graph capture 或 `torch.compile`/export contract。

ELFES 仍保持只读。现有 two-center 路径需要单结构 NumPy、统一 broad cutoff、partial/full PBC、multiple images 和 strict filtering，`tonari` 已覆盖这些基础能力；未来接入仍需明确把 full directed pairs 转成 half list、做 species cutoff post-filter 并补 onsite，本任务没有提前冻结或实现该 adapter。
