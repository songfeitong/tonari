# 算法选择

`neighbor_list` 提供 brute force 和 cell list 两种搜索方式。它们遵守相同的 cutoff、周期边界和 pair 方向约定；区别只在于如何找到需要检查的原子组合。

## 两种搜索方式

Brute force 穷举结构中的 source atom、target atom，以及周期边界下需要考虑的相邻晶胞。这种方法没有建表开销，通常适合小结构。

Cell list 把原子所在空间划分成许多小格子。每个 source atom 只需要检查自己附近格子中的 target atoms，而不必遍历整个结构。建立这些格子需要额外工作和内存，但结构较大时通常更合适。

## 公共选项

| `algorithm`     | 行为                                                  |
| --------------- | ----------------------------------------------------- |
| `"auto"`        | 根据 backend 和输入规模自动选择；必要时允许安全回退。 |
| `"brute_force"` | 强制使用穷举。                                        |
| `"cell_list"`   | 强制使用 cell list；如果当前输入无法安全执行则报错。  |

显式选择算法主要用于性能诊断、benchmark 和特殊 workload。普通调用应使用默认的 `"auto"`。不同算法可能产生不同的输出顺序，但返回的 pairs 必须相同。

## CPU 的自动选择

CPU 对 Batch 中的每个 structure 独立选择算法。它先估算 brute force 需要做多少次原子间比较：每个 source atom 要与每个 target atom 比较；如果启用了周期边界，还要为 cutoff 可能触及的相邻晶胞副本重复这些比较。这里的“晶胞副本”是把原晶胞沿 periodic cell vectors 平移整数次后得到的搜索位置。

当前估算不超过 16,384 次比较时使用 brute force，超过时先尝试 cell list。这个 crossover 来自真实结构的 benchmark；它是当前 CPU 实现的调优参数，而不是永久的公共契约。

## CUDA 的自动选择

CUDA 以整个调用中的 Batch 为执行单位，因此一次调用只选择一种主路径。当前规则查看 Batch 中最大的 structure：所有 structure 都少于 256 atoms 时使用 fused brute force；只要有一个 structure 达到 256 atoms，就先尝试 batched cell list。

这一规则的重点不是认为 256 atoms 具有特殊物理意义，而是在当前 CUDA 实现中，小结构的 brute force 更容易充分利用 GPU，也省去了建立 cell list 的固定成本。阈值同样属于可随实现和测量更新的内部调优参数。

## 安全回退

Cell list 需要为搜索区域建立有限数量的格子。极端稀疏的坐标可能在少量原子之间产生大量空格子；某些远离原晶胞的 representatives 也可能让 cell-list path 无法在保持公共浮点语义的同时安全处理。

使用 `"auto"` 时，这些情况会回退到 brute force。显式使用 `"cell_list"` 时不会悄悄换算法，而是给出错误并建议改用 `"auto"` 或 `"brute_force"`。资源上限与这种 fallback 判断负责防止无界分配，不参与正常 workload 的性能调优。
