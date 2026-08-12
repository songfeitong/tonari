# CPU 多线程

`neighbor_list(..., num_threads=N)` 显式设置一次 CPU 调用使用的线程数，调用线程计入其中。默认值是 `1`，因此升级 Tonari 不会让已有的 DataLoader、DDP 或 BLAS workload 意外多占 CPU。NumPy 和 Torch CPU 通过同一个 native implementation 执行；CUDA 不使用这个 CPU thread pool，只接受默认值 `1`。

## 如何选择线程数

先从 `num_threads=1` 开始，只在一次 `neighbor_list` 调用本身是 CPU 瓶颈时增加到 `2`、`4` 或 `8` 并测量完整 workload。大量小 structures 组成一个 `batch_ptr` batch 时，多个 structures 会并行；只有一个或少数大 structure 时，source atoms 会切成多个连续区间并行。两类任务进入同一个动态调度器，不需要用户选择不同模式。

线程数不是越大越好。输出 pair 的分配、内存带宽、单个 structure 的 cell-list 准备，以及 Python frontend 对 `d`/`D` 的计算都会限制 scaling。小到只有几十微秒的调用通常保持单线程更合适；数千到数万 atoms 的单体系，或能合并成一个较大 native batch 的 workload，才更容易获得稳定收益。

## 与 DataLoader、DDP 和 BLAS 组合

总并行度应按进程和 worker 一起计算，而不是为每一层都使用全部 cores。粗略预算为 `DDP 进程数 × 每进程 DataLoader workers × num_threads`；若 neighbor search 在主训练进程而不是 worker 内执行，就只计算实际调用它的进程。通常让一种层级承担主要并行度：已有多个 DataLoader workers 或每个 GPU 一个 DDP 进程时保持 `num_threads=1`；单进程、`num_workers=0` 且 CPU neighbor search 明显占时，再提高 `num_threads`。

Tonari 的线程数不控制 PyTorch、NumPy、MKL 或 OpenBLAS 的线程。若调用还请求 `d`/`D` 或随后执行 dense linear algebra，应同时检查 `torch.set_num_threads`、`OMP_NUM_THREADS`、`MKL_NUM_THREADS` 等外部配置，避免嵌套并行抢占同一组 cores。在 NUMA 或共享节点上，进程 affinity 和调度配额也应计入可用 CPU 数量。

## 执行模型与生命周期

CPU core 先准备每个 structure 的 periodic geometry 和可选 cell list，再把结构查询组织成 source-major tasks。小 structures 通常是一项任务；大型 brute-force 或 cell-list structure 会按连续 source ranges 拆分。Workers 动态领取任务，每项任务写入自己的 output chunk，最后按原始 structure/source 顺序直接复制到返回数组，因此 hot loop 没有共享 push-back 或粗粒度输出锁，`sorted=True` 的 source 非递减语义也保持不变。

Worker pool 在进程内惰性创建并复用于后续调用，调用线程自身也参与工作；`num_threads=1` 不会创建 workers。并发 Python threads 可以同时发起调用，每次调用仍只使用自己请求的线程额度。`fork()` 后子进程不会复用父进程中已经消失的 worker threads，而会在首次多线程调用时建立自己的 pool。即便如此，更稳妥的 DataLoader 配置仍是先 fork workers、再在各 worker 内首次使用 native thread pool，或在适合的平台选择 `spawn`。

Worker 中的第一个 native 异常会取消尚未领取的任务，在所有参与线程退出本次 job 后回传给 Python；后续调用可以继续使用 pool。异常不保证已经运行的 task 提前中断，但它们的局部输出不会作为部分结果返回。
