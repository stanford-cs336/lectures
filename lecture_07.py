import torch
import time
import math
import sys
import os
from inspect import isfunction
from typing import Callable
from torch import nn, tensor
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from edtrace import text, image, link
from gpu_util import cuda_if_available
from lecture_util import article_link, bilingual_text, bilingual_verbatim

if not torch.cuda.is_available():
    torch.cuda.synchronize = lambda: None  # No-op if CUDA is not available

def main():
    bilingual_text("# Lecture 7: parallelism", '# 第 7 讲：并行')
    bilingual_text("Last week: parallelism within a single GPU", '上周：单个 GPU 内部的并行。')
    bilingual_text("This week: parallelism across multiple GPUs", '本周：跨多个 GPU 的并行。')
    image("images/gpu-node-overview.png", width=700)

    bilingual_text("In both cases, **compute** (arithmetic logic units) is far from inputs/outputs (**data**).", '在这两种情况下，**计算**（算术逻辑单元）都离输入/输出（**数据**）很远。')
    bilingual_text("Unifying theme: orchestrate computation to avoid data transfer bottlenecks", '统一主题：组织计算，以避免数据传输瓶颈。')

    bilingual_text("Generalized hierarchy:", '广义层次结构：')
    bilingual_text("- Single node, single GPU: L1 cache / shared memory (fastest)", '- 单节点、单 GPU：L1 缓存 / 共享内存（最快）。')
    bilingual_text("- Single node, single GPU: HBM", '- 单节点、单 GPU：HBM。')
    bilingual_text("- Single node, multi-GPU: NVLink/NVSwitch", '- 单节点、多 GPU：NVLink/NVSwitch。')
    bilingual_text("- Multi-node, multi-GPU: Infiniband/Ethernet (slowest)", '- 多节点、多 GPU：Infiniband/Ethernet（最慢）。')

    bilingual_text("Last week: reduce memory accesses via fusion/tiling", '上周：通过融合/分块减少内存访问。')
    bilingual_text("This week: reduce communication across GPUs/nodes via replication/sharding", '本周：通过复制/分片减少 GPU/节点之间的通信。')

    bilingual_text("Why do multi-GPU?", '为什么要使用多 GPU？')
    bilingual_text("1. Your parameters (optimizer state + gradients + activations) don't fit on a single GPU.", '1. 你的参数（优化器状态 + 梯度 + 激活值）放不进单个 GPU。')
    bilingual_text("2. You want to use more GPUs (more FLOPs) to train faster.", '2. 你想使用更多 GPU（更多 FLOPs）来更快训练。')

    # When you execute this lecture directly (python lecture_07.py), it uses multiprocessing, which produces output from each process (below).
    # However, when you trace this lecture (python -m edtrace.execute -m lecture_07), we turn off multiprocessing.
    link(title="stdout for this lecture", url="var/traces/lecture_07_stdout.txt")

    bilingual_text("### Part 1: building blocks of distributed communication/computation", '### 第 1 部分：分布式通信/计算的构建块')
    collective_operations()    # Programming model
    hardware()                 # Hardware: how GPUs are connected
    torch_distributed()        # How this is implemented in NCCL/PyTorch
    benchmarking()             # Measure actual NCCL bandwidth

    bilingual_text("### Part 2: distributed training", '### 第 2 部分：分布式训练')
    bilingual_text("Walk through bare-bones implementations of each strategy on deep MLPs.", '在深层 MLP 上逐步讲解每种策略的最小实现。')
    bilingual_text("Recall that MLPs are the compute bottleneck in Transformers, so this is representative.", '回忆一下，MLP 是 Transformer 中的计算瓶颈，因此这个例子具有代表性。')

    data_parallelism()         # Cut up along the batch dimension
    tensor_parallelism()       # Cut up along the width dimension
    pipeline_parallelism()     # Cut up along the depth dimension

    bilingual_text("What's missing?", '还缺什么？')
    bilingual_text("- Communication/computation overlap", '- 通信与计算重叠。')
    bilingual_text("- More general models (with attention, etc.)", '- 更通用的模型（带注意力等）。')
    bilingual_text("- Other forms of parallelism (e.g., sequence parallelism, expert parallelism, combinations)", '- 其他形式的并行（例如序列并行、专家并行以及组合）。')
    bilingual_text("- Jax/TPUs: just define the model, the sharding strategy, and the Jax compiler handles the rest ", '- Jax/TPU：只需定义模型和分片策略，其余由 Jax 编译器处理。'), link(title="levanter", url="https://crfm.stanford.edu/2023/06/16/levanter-1_0-release.html")
    bilingual_text("- But we're doing PyTorch so you can see how one builds up from the primitives", '- 但我们使用 PyTorch，这样你可以看到如何从原语逐步搭建。')

    bilingual_text("### Summary", '### 总结')
    bilingual_text("- Many ways to parallelize: data (batch), tensor/expert (width), pipeline (depth), sequence (length)", '- 并行方式很多：数据（批次）、张量/专家（宽度）、流水线（深度）、序列（长度）。')
    bilingual_text("- Data parallelism: DDP (all-reduce), FSDP/ZeRO (all-gather + reduce-scatter)", '- 数据并行：DDP（all-reduce）、FSDP/ZeRO（all-gather + reduce-scatter）。')
    bilingual_text("- Tensor parallelism: requires very fast interconnects (e.g., NVLink)", '- 张量并行：需要非常快的互连（例如 NVLink）。')
    bilingual_text("- Pipeline parallelism: can work with slow interconnects, but need to work to reduce pipeline bubbles", '- 流水线并行：可在较慢互连上工作，但需要努力减少流水线气泡。')
    bilingual_text("- Can **re-compute** or store in **memory** or store in another GPUs memory and **communicate**", '- 可以**重算**，也可以存入**内存**，或者存入另一块 GPU 的内存并进行**通信**。')
    bilingual_text("- Hardware is getting faster, but will always want bigger models, so will have this hierarchical structure", '- 硬件会越来越快，但我们总想要更大的模型，因此这种层次结构会一直存在。')


def collective_operations():
    bilingual_text("**Collective operations** are the conceptual primitives used for distributed programming ", '**集合通信操作**是分布式编程使用的概念原语。'), article_link("https://en.wikipedia.org/wiki/Collective_operation")
    bilingual_text("- These are classic in the parallel programming literature from the 1980s.", '- 这些是 1980 年代并行编程文献中的经典概念。')
    bilingual_text("- *Collective* means that you specify a general communication pattern across many devices.", '- *集合通信*意味着你指定跨多个设备的一般通信模式。')
    bilingual_text("- This can be better/faster than managing point-to-point communication yourself.", '- 这可能比自己管理点对点通信更好、更快。')

    bilingual_text("**Setup**:", '**设置**：')
    image("images/ranks.png", width=500)
    bilingual_text("- **Rank**: a particular device/GPU (e.g., 0, 1, 2, 3)", '- **Rank**：某个具体设备/GPU（例如 0、1、2、3）。')
    bilingual_text("- **World size**: total number of devices (e.g., 4)", '- **World size**：设备总数（例如 4）。')

    bilingual_text("Operations:", '操作：')
    bilingual_text("- Broadcast, scatter, gather, reduce (foundations)", '- broadcast、scatter、gather、reduce（基础）。')
    bilingual_text("- All-gather, reduce-scatter, all-reduce (workhorse)", '- all-gather、reduce-scatter、all-reduce（主力）。')
    bilingual_text("- All-to-all (for MoEs)", '- all-to-all（用于 MoE）。')

    bilingual_text("**Broadcast**: copy from rank 0 to all ranks", '**Broadcast**：从 rank 0 复制到所有 rank。')
    # Input
    rank0 = tensor([0., 1, 2, 3])

    # Output
    rank0 = tensor([0., 1, 2, 3])
    rank1 = tensor([0., 1, 2, 3])
    rank2 = tensor([0., 1, 2, 3])
    rank3 = tensor([0., 1, 2, 3])

    bilingual_text("Minor use case: rank 0 loads initial checkpoint and broadcasts to all ranks", '小用例：rank 0 加载初始 checkpoint，并广播到所有 rank。')

    bilingual_text("**Scatter** tensor on rank 0 to all ranks", '**Scatter**：把 rank 0 上的张量分发到所有 rank。')
    # Input
    rank0 = tensor([0., 1, 2, 3])

    # Output
    rank0 = tensor([0.])
    rank1 = tensor([1.])
    rank2 = tensor([2.])
    rank3 = tensor([3.])

    bilingual_text("Note: stepping stone to understanding reduce-scatter", '注意：这是理解 reduce-scatter 的垫脚石。')

    bilingual_text("**Gather** pieces from all ranks to rank 0 (opposite of scatter)", '**Gather**：把所有 rank 的片段收集到 rank 0（scatter 的反向操作）。')
    # Input
    rank0 = tensor([0.])
    rank1 = tensor([1.])
    rank2 = tensor([2.])
    rank3 = tensor([3.])

    # Output
    rank0 = tensor([0., 1, 2, 3])

    bilingual_text("Note: stepping stone to understanding all-gather", '注意：这是理解 all-gather 的垫脚石。')

    bilingual_text("**Reduce** pieces from all ranks to rank 0, applying some operation (e.g., sum, min, max)", '**Reduce**：把所有 rank 的片段聚合到 rank 0，并应用某个操作（例如 sum、min、max）。')
    # Input
    rank0 = tensor([0.])
    rank1 = tensor([1.])
    rank2 = tensor([2.])
    rank3 = tensor([3.])

    # Output
    rank0 = tensor([6.])  # Sum of all ranks (0 + 1 + 2 + 3)

    bilingual_text("Note: stepping stone to understanding all-reduce", '注意：这是理解 all-reduce 的垫脚石。')

    bilingual_text("**All-gather**: perform gather to all ranks, not just rank 0", '**All-gather**：对所有 rank 执行 gather，而不仅是 rank 0。')
    # Input
    rank0 = tensor([0.])
    rank1 = tensor([1.])
    rank2 = tensor([2.])
    rank3 = tensor([3.])

    # Output
    rank0 = tensor([0., 1, 2, 3])
    rank1 = tensor([0., 1, 2, 3])
    rank2 = tensor([0., 1, 2, 3])
    rank3 = tensor([0., 1, 2, 3])

    bilingual_text("Use case: each rank holds parameter shard, gather to get full parameters for forward pass", '用例：每个 rank 持有参数分片，前向传播时 gather 得到完整参数。')

    bilingual_text("**Reduce-scatter**: perform reduce on each dimension, scatter results", '**Reduce-scatter**：在每个维度上执行 reduce，再把结果 scatter。')
    # Input
    rank0 = tensor([0., 1, 2, 3])
    rank1 = tensor([1., 2, 3, 4])
    rank2 = tensor([2., 3, 4, 5])
    rank3 = tensor([3., 4, 5, 6])

    # Output
    rank0 = tensor([6.])  # Sum along dim 0 (0 + 1 + 2 + 3)
    rank1 = tensor([10.]) # Sum along dim 1 (1 + 2 + 3 + 4)
    rank2 = tensor([14.]) # Sum along dim 2 (2 + 3 + 4 + 5)
    rank3 = tensor([18.]) # Sum along dim 3 (3 + 4 + 5 + 6)

    bilingual_text("Use case: after backward pass, sum gradients from different data shards, but distribute storage", '用例：反向传播后，对不同数据分片上的梯度求和，但分布式存储结果。')

    bilingual_text("**All-reduce** = reduce-scatter + all-gather", '**All-reduce（全规约）** = reduce-scatter + all-gather。')
    # Input
    rank0 = tensor([0., 1, 2, 3])
    rank1 = tensor([1., 2, 3, 4])
    rank2 = tensor([2., 3, 4, 5])
    rank3 = tensor([3., 4, 5, 6])

    # Output
    rank0 = tensor([6., 10, 14, 18])
    rank1 = tensor([6., 10, 14, 18])
    rank2 = tensor([6., 10, 14, 18])
    rank3 = tensor([6., 10, 14, 18])

    bilingual_text("Use case: after backward pass, sum gradients from different data shards, but replicate full parameters", '用例：反向传播后，对不同数据分片上的梯度求和，但复制完整参数。')
    bilingual_text("Breaking all-reduce into reduce-scatter + all-gather allows for flexibility (e.g., ZeRO/FSDP)", '把 all-reduce 拆成 reduce-scatter + all-gather 可以带来灵活性（例如 ZeRO/FSDP）。')

    bilingual_text("**All-to-all**: each rank sends each other rank some tensor (most general)", '**All-to-all**：每个 rank 都向其他每个 rank 发送一些张量（最一般的形式）。')
    # Input
    rank0 = tensor([0., 1, 2, 3])      # send  0 to rank 0,  1 to rank 1,  2 to rank 2,  3 to rank 3
    rank1 = tensor([4., 5, 6, 7])      # send  4 to rank 0,  5 to rank 1,  6 to rank 2,  7 to rank 3
    rank2 = tensor([8., 9, 10, 11])    # send  8 to rank 0,  9 to rank 1, 10 to rank 2, 11 to rank 3
    rank3 = tensor([12., 13, 14, 15])  # send 12 to rank 0, 13 to rank 1, 14 to rank 2, 15 to rank 3

    # Output
    rank0 = tensor([0, 4, 8, 12])
    rank1 = tensor([1, 5, 9, 13])
    rank2 = tensor([2, 6, 10, 14])
    rank3 = tensor([3, 7, 11, 15])

    bilingual_text("Notes:", '说明：')
    bilingual_text("- Useful for MoEs: each rank has split of data and subset of experts; need to route data to experts", '- 对 MoE 很有用：每个 rank 拥有一部分数据和一部分专家，需要把数据路由到专家。')
    bilingual_text("- For balanced splits, all-to-all looks like transpose", '- 对均衡切分来说，all-to-all 看起来像转置。')
    bilingual_text("- Also handles unbalanced splits (but want splits to be as balanced as possible)", '- 也能处理不均衡切分（但我们希望切分尽可能均衡）。')

    bilingual_text("Way to remember the terminology:", '记住这些术语的方法：')
    bilingual_text("- Reduce: performs some associative/commutative operation (sum, min, max)", '- Reduce：执行某种满足结合律/交换律的操作（sum、min、max）。')
    bilingual_text("- Scatter is inverse of gather", '- Scatter 是 gather 的反向操作。')
    bilingual_text("- All: means destination is all devices", '- All：表示目标是所有设备。')


def hardware():
    bilingual_text("Classic (in the home):", '经典情况（家用环境）：')
    image("https://media.springernature.com/lw685/springer-static/image/art%3A10.1186%2Fs42774-021-00098-3/MediaObjects/42774_2021_98_Fig1_HTML.png?as=webp", width=500)
    bilingual_text("- GPUs on same node communicate via a PCI(e) bus (v7.0, 16 lanes => 242 GB/s) ", '- 同一节点上的 GPU 通过 PCI(e) 总线通信（v7.0，16 lanes => 242 GB/s）。'), article_link("https://en.wikipedia.org/wiki/PCI_Express")
    bilingual_text("- GPUs on different nodes communicate via Ethernet (~200 MB/s)", '- 不同节点上的 GPU 通过 Ethernet 通信（约 200 MB/s）。')
    
    bilingual_text("Modern (in the data center):", '现代情况（数据中心）：')
    image("images/gpu-node-overview.png", width=700)

    bilingual_text("Typical setup:", '典型配置：')
    bilingual_text("- 8 GPUs per node, connected by NVLink to an NVSwitch (B200s' NVLink 5.0 gets 1.8 TB/s; HBM was 8 TB/s)", '- 每个节点 8 块 GPU，通过 NVLink 连接到 NVSwitch（B200 的 NVLink 5.0 达到 1.8 TB/s；HBM 是 8 TB/s）。')
    bilingual_text("- 256 nodes per pod, connected by Infiniband (via PCIe -> HCA / Infiniband NIC -> Infiniband cable) (~0.05 TB/s)", '- 每个 pod 256 个节点，通过 Infiniband 连接（PCIe -> HCA / Infiniband NIC -> Infiniband cable）（约 0.05 TB/s）。')
    bilingual_text("- N pods per cluster / datacenter, connected by Ethernet (via PCIe -> CPU)", '- 每个集群/数据中心有 N 个 pod，通过 Ethernet 连接（经 PCIe -> CPU）。')

    bilingual_text("Bypassing the CPU:", '绕过 CPU：')
    bilingual_text("- Ethernet requires passing through the CPU (copying data to kernel socket buffer, build TCP packets, copy to NIC ring buffer)", '- Ethernet 需要经过 CPU（把数据复制到内核 socket buffer、构造 TCP 包、复制到 NIC ring buffer）。')
    bilingual_text("- Remote Direct Memory Access (RDMA): allows one GPU to directly read/write another GPU's memory without involving the CPU", '- 远程直接内存访问（RDMA）：允许一个 GPU 不经过 CPU，直接读写另一个 GPU 的内存。')
    bilingual_text("- Infiniband supports RDMA, but standard Ethernet does not", '- Infiniband 支持 RDMA，但标准 Ethernet 不支持。')

    bilingual_text("Advancements:", '进展：')
    bilingual_text("- GB200/GB300 NVL72: 8 GPUs per tray, 9 trays per rack -> 72 GPUs in one NVLink domain", '- GB200/GB300 NVL72：每个 tray 8 块 GPU，每个机架 9 个 tray -> 一个 NVLink domain 中有 72 块 GPU。')
    bilingual_text("- RDMA over Converged Ethernet (RoCE): Ethernet bypasses CPU, similar but cheaper/weaker than Infiniband, used by Meta", '- RDMA over Converged Ethernet（RoCE）：Ethernet 绕过 CPU，类似 Infiniband 但更便宜也更弱，Meta 使用这种方案。')

    bilingual_text("### NVIDIA Collective Communication Library (NCCL)", '### NVIDIA 集合通信库（NCCL）')
    bilingual_text("NCCL translates collective operations into low-level packets that are sent between GPUs. ", 'NCCL 把集合通信操作转换为 GPU 之间发送的低层数据包。'), link(title="talk", url="https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31880/")
    bilingual_text("- Detects topology of hardware (e.g., number of nodes, switches, NVLink/PCIe)", '- 检测硬件拓扑（例如节点数量、交换机、NVLink/PCIe）。')
    bilingual_text("- Optimizes the path between GPUs", '- 优化 GPU 之间的路径。')
    bilingual_text("- Launches GPU kernels to send/receive data", '- 启动 GPU 内核来发送/接收数据。')


def torch_distributed():
    bilingual_text("PyTorch distributed library (`torch.distributed`) ", 'PyTorch 分布式库（`torch.distributed`）。'), link(title="documentation", url="https://pytorch.org/docs/stable/distributed.html")
    bilingual_text("- Provides clean interface for collective operations (e.g., `all_gather_into_tensor`)", '- 为集合通信操作提供清晰接口（例如 `all_gather_into_tensor`）。')
    bilingual_text("- Supports multiple backends for different hardware: gloo (CPU), nccl (GPU)", '- 支持面向不同硬件的多个后端：gloo（CPU）、nccl（GPU）。')
    bilingual_text("- Also supports higher-level algorithms (e.g., `FullyShardedDataParallel`) [not used in this course]", '- 也支持更高层算法（例如 `FullyShardedDataParallel`）[本课程不使用]。')

    bilingual_text("Let's walk through some examples.", '我们来看几个例子。')
    spawn(collective_operations_main, world_size=4)


def collective_operations_main(rank: int, world_size: int):  # @inspect rank world_size
    """This function is running asynchronously for each process (rank = 0, ..., world_size - 1)."""
    setup(rank, world_size)

    ### All-reduce (dist = torch.distributed)
    dist.barrier()  # Waits for all processes to get to this point (in this case, for print statements)

    data = tensor([0., 1, 2, 3], device=cuda_if_available(rank)) + rank  # Both input and output

    print(f"Rank {rank} [before all-reduce]: {data}", flush=True)
    dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)  # Modifies tensor in place
    print(f"Rank {rank} [after all-reduce]: {data}", flush=True)

    ### Reduce-scatter
    dist.barrier()

    input = torch.arange(world_size, dtype=torch.float32, device=cuda_if_available(rank)) + rank  # Input
    output = torch.empty(1, device=cuda_if_available(rank))  # Allocate output

    print(f"Rank {rank} [before reduce-scatter]: input = {input}, output = {output}", flush=True)
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    print(f"Rank {rank} [after reduce-scatter]: input = {input}, output = {output}", flush=True)

    ### All-gather
    dist.barrier()

    input = output  # Input is the output of reduce-scatter
    output = torch.empty(world_size, device=cuda_if_available(rank))  # Allocate output

    print(f"Rank {rank} [before all-gather]: input = {input}, output = {output}", flush=True)
    dist.all_gather_into_tensor(output_tensor=output, input_tensor=input, async_op=False)
    print(f"Rank {rank} [after all-gather]: input = {input}, output = {output}", flush=True)

    bilingual_text("Indeed, all-reduce = reduce-scatter + all-gather!", '确实，all-reduce = reduce-scatter + all-gather！')

    cleanup()


def benchmarking():
    bilingual_text("How fast does communication happen?", '通信发生得有多快？')

    # All-reduce
    spawn(all_reduce, world_size=4, num_elements=100 * 1024**2)

    # Reduce-scatter
    spawn(reduce_scatter, world_size=4, num_elements=100 * 1024**2)

    bilingual_text("References:", '参考资料：')
    link(title="How to reason about collective operations", url="https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#allreduce")
    link(title="Sample benchmarking code", url="https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/all_reduce_bench.py")


def all_reduce(rank: int, world_size: int, num_elements: int):
    setup(rank, world_size)  # @stepover

    # Create tensor
    data = torch.randn(num_elements, device=cuda_if_available(rank))

    # Warmup
    dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)
    torch.cuda.synchronize()  # Wait for CUDA kernels to finish
    dist.barrier()            # Wait for all the processes to get here

    # Perform all-reduce
    start_time = time.time()
    dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)
    torch.cuda.synchronize()  # Wait for CUDA kernels to finish
    dist.barrier()            # Wait for all the processes to get here
    end_time = time.time()

    duration = end_time - start_time
    print(f"[all_reduce] Rank {rank}: all_reduce(world_size={world_size}, num_elements={num_elements}) took {render_duration(duration)}", flush=True)  # @stepover

    # Measure the effective bandwidth
    dist.barrier()
    size_bytes = data.element_size() * data.numel()
    sent_bytes = size_bytes * 2 * (world_size - 1)  # 2x because send + receive, world_size-1 steps in all-reduce
    total_duration = world_size * duration
    bandwidth = sent_bytes / total_duration
    print(f"[all_reduce] Rank {rank}: all_reduce measured bandwidth = {round(bandwidth / 1024**3)} GB/s", flush=True)

    # Notes:
    # - Effective bandwidth ~ 2 * size_bytes / total_duration
    # - Independent of world_size
    # - Independent of topology (ring or tree)

    cleanup()  # @stepover


def reduce_scatter(rank: int, world_size: int, num_elements: int):
    setup(rank, world_size)  # @stepover

    # Create input and outputs
    input = torch.randn(world_size, num_elements, device=cuda_if_available(rank))  # Each rank has a matrix
    output = torch.empty(num_elements, device=cuda_if_available(rank))

    # Warmup
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    torch.cuda.synchronize()  # Wait for CUDA kernels to finish
    dist.barrier()            # Wait for all the processes to get here

    # Perform reduce-scatter
    start_time = time.time()
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    torch.cuda.synchronize()  # Wait for CUDA kernels to finish
    dist.barrier()            # Wait for all the processes to get here
    end_time = time.time()

    duration = end_time - start_time
    print(f"[reduce_scatter] Rank {rank}: reduce_scatter(world_size={world_size}, num_elements={num_elements}) took {render_duration(duration)}", flush=True)  # @stepover

    # Measure the effective bandwidth
    dist.barrier()
    data_bytes = input.element_size() * input.numel()  # How much data in the input
    sent_bytes = data_bytes * (world_size - 1)  # How much needs to be sent (no 2x here)
    total_duration = world_size * duration  # Total time for transmission
    bandwidth = sent_bytes / total_duration
    print(f"[reduce_scatter] Rank {rank}: reduce_scatter measured bandwidth = {round(bandwidth / 1024**3)} GB/s", flush=True)

    # Notes:
    # - all-reduce = reduce-scatter + all-gather
    # - all-reduce moves 2x the data in 2x the time compared to reduce-scatter, so similar bandwidth

    cleanup()  # @stepover


def data_parallelism():
    image("images/data-parallelism.png", width=300)
    bilingual_text("Sharding strategy: each rank gets a slice of the data", '分片策略：每个 rank 获得数据的一部分。')

    data = generate_sample_data()
    spawn(data_parallelism_main, world_size=4, data=data, num_layers=4, num_steps=1)

    bilingual_text("Notes:", '说明：')
    bilingual_text("- Losses are different across ranks (computed on local data)", '- 各 rank 上的损失不同（在本地数据上计算）。')
    bilingual_text("- Gradients are all-reduced to be the same across ranks", '- 梯度通过 all-reduce 变得在各 rank 上相同。')
    bilingual_text("- Therefore, parameters remain the same across ranks", '- 因此，各 rank 上的参数保持相同。')

    bilingual_text("Next time: FSDP/ZeRO: use all-gather and reduce-scatter to avoid holding all parameters in memory", '下次：FSDP/ZeRO：使用 all-gather 和 reduce-scatter，避免在内存中保存所有参数。')


def generate_sample_data():
    batch_size = 128
    num_dim = 1024
    data = torch.randn(batch_size, num_dim)
    return data


def data_parallelism_main(rank: int, world_size: int, data: tensor, num_layers: int, num_steps: int):
    setup(rank, world_size)  # @stepover

    # Get the slice of data for this rank (in practice, each rank should load only its own data)
    # --- B0 ---
    # --- B1 ---
    # --- B2 ---
    # --- B3 ---
    batch_size = data.size(0)  # @inspect batch_size
    num_dim = data.size(1)  # @inspect num_dim
    local_batch_size = int_divide(batch_size, world_size)  # @inspect local_batch_size @stepover
    start_index = rank * local_batch_size  # @inspect start_index
    end_index = start_index + local_batch_size  # @inspect end_index
    data = data[start_index:end_index].to(cuda_if_available(rank))

    # Create MLP parameters params[0], ..., params[num_layers - 1] (each rank has all parameters)
    params = [get_init_params(num_dim, num_dim, rank) for layer in range(num_layers)]
    optimizer = torch.optim.AdamW(params, lr=1e-3)  # Each rank has own optimizer state

    for step in range(num_steps):
        # Forward pass
        x = data
        for param in params:
            x = x @ param
            x = F.gelu(x)
        loss = x.square().mean()  # Loss function is average squared magnitude

        # Backward pass
        loss.backward()

        # Sync gradients across workers (ONLY difference between standard training and DDP)
        for param in params:
            dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)

        # Update parameters
        optimizer.step()

        print(f"[data_parallelism] Rank {rank}: step = {step}, loss = {loss.item()}, params = {[summarize_tensor(params[layer]) for layer in range(num_layers)]}", flush=True)  # @stepover

    cleanup()  # @stepover


def tensor_parallelism():
    image("images/tensor-parallelism.png", width=300)
    bilingual_text("Sharding strategy: each rank gets part of each layer, transfer all data/activations", '分片策略：每个 rank 获得每一层的一部分，并传输所有数据/激活值。')

    data = generate_sample_data()
    spawn(tensor_parallelism_main, world_size=4, data=data, num_layers=4)


def tensor_parallelism_main(rank: int, world_size: int, data: tensor, num_layers: int):
    setup(rank, world_size)  # @stepover

    data = data.to(cuda_if_available(rank))  # All ranks get the data (batch_size x num_dim)
    batch_size = data.size(0)  # @inspect batch_size
    num_dim = data.size(1)  # @inspect num_dim
    local_num_dim = int_divide(num_dim, world_size)  # Shard `num_dim`  @inspect local_num_dim @stepover

    # Create model (each rank gets 1/world_size of the parameters)
    #  |  |  |  |
    # W0 W1 W2 W3
    #  |  |  |  |
    params = [get_init_params(num_dim, local_num_dim, rank) for layer in range(num_layers)]

    # Forward pass
    x = data
    for layer in range(num_layers):
        # Compute activations (batch_size x local_num_dim)
        x = x @ params[layer]  # Note: this is only on a slice of the parameters
        x = F.gelu(x)

        # Allocate memory for activations (world_size x batch_size x local_num_dim)
        activations = [torch.empty(batch_size, local_num_dim, device=cuda_if_available(rank)) for _ in range(world_size)]

        # Send activations via all gather
        dist.all_gather(tensor_list=activations, tensor=x, async_op=False)

        # Concatenate them to get batch_size x num_dim
        x = torch.cat(activations, dim=1)

    print(f"[tensor_parallelism] Rank {rank}: forward pass produced activations {summarize_tensor(x)}", flush=True)  # @stepover

    # Backward pass: homework exercise

    cleanup()  # @stepover


def pipeline_parallelism():
    image("images/pipeline-parallelism.png", width=300)
    bilingual_text("Sharding strategy: each rank gets subset of layers, transfer all data/activations", '分片策略：每个 rank 获得一部分层，并传输所有数据/激活值。')

    data = generate_sample_data()
    spawn(pipeline_parallelism_main, world_size=2, data=data, num_layers=4, num_micro_batches=4)


def pipeline_parallelism_main(rank: int, world_size: int, data: tensor, num_layers: int, num_micro_batches: int):
    setup(rank, world_size)  # @stepover

    # Use all the data
    data = data.to(cuda_if_available(rank))
    batch_size = data.size(0)  # @inspect batch_size
    num_dim = data.size(1)  # @inspect num_dim

    # Split up layers
    local_num_layers = int_divide(num_layers, world_size)  # @inspect local_num_layers @stepover

    # Each rank gets a subset of layers
    local_params = [get_init_params(num_dim, num_dim, rank) for layer in range(local_num_layers)]  # @stepover

    # Forward pass

    # Break up into micro batches to minimize the bubble
    micro_batch_size = int_divide(batch_size, num_micro_batches)  # @inspect micro_batch_size @stepover
    if rank == 0:
        # The data
        micro_batches = data.chunk(chunks=num_micro_batches, dim=0)
    else:
        # Allocate memory for activations
        micro_batches = [torch.empty(micro_batch_size, num_dim, device=cuda_if_available(rank)) for _ in range(num_micro_batches)]

    for x in micro_batches:
        # Get activations from previous rank
        if rank - 1 >= 0:
            dist.recv(tensor=x, src=rank - 1)

        # Compute layers assigned to this rank
        for param in local_params:
            x = x @ param
            x = F.gelu(x)

        # Send to the next rank
        if rank + 1 < world_size:
            print(f"[pipeline_parallelism] Rank {rank}: sending {summarize_tensor(x)} to rank {rank + 1}", flush=True)  # @stepover
            dist.send(tensor=x, dst=rank + 1)

    bilingual_text("Not handled: overlapping communication/computation to eliminate pipeline bubbles", '未处理：通过通信/计算重叠来消除流水线气泡。')

    # Backward pass: homework exercise

    cleanup()  # @stepover

############################################################

def setup(rank: int, world_size: int):
    """Initializes the distributed environment (called at start of process)."""
    # Specify where master lives (rank 0), used to coordinate (actual data goes through NCCL)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "15623"

    if torch.cuda.is_available():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    """Cleans up the distributed environment (called at end of process)."""
    torch.distributed.destroy_process_group()


class DisableDistributed:
    """
    Context manager that temporarily disables distributed functions (replaces with no-ops).
    This is for when we're tracing the lecture, since we can't trace through
    multiprocessing, so we just want to run the function directly without
    distributed communication.
    """
    def __enter__(self):
        self.old_functions = {}
        for name in dir(dist):
            value = getattr(dist, name, None)
            if isfunction(value):
                self.old_functions[name] = value
                setattr(dist, name, lambda *args, **kwargs: None)

    def __exit__(self, exc_type, exc_value, traceback):
        for name in self.old_functions:
            setattr(dist, name, self.old_functions[name])


def spawn(func: Callable, world_size: int, *args, **kwargs):
    """
    Launches `world_size` processes that each calls `func` on world_size, args, kwargs.
    Note: if we are being traced (inside edtrace), we just run the function directly without multiprocessing and disable distributed functions.
    """
    # Note: assume kwargs are in the same order as what main needs
    if not sys.gettrace():
        # This is the normal code path for multiprocessing
        args = (world_size,) + args + tuple(kwargs.values())
        mp.spawn(func, args=args, nprocs=world_size, join=True)
    else:
        # If we're being traced (inside edtrace), just run the function directly.
        with DisableDistributed():  # @stepover
            args = (0, world_size,) + args + tuple(kwargs.values())
            func(*args)


def get_init_params(num_inputs: int, num_outputs: int, rank: int) -> nn.Parameter:
    """Create parameters and put them on the `rank`-th GPU."""
    torch.random.manual_seed(0)  # For reproducibility
    return nn.Parameter(torch.randn(num_inputs, num_outputs, device=cuda_if_available(rank)) / math.sqrt(num_outputs))


def int_divide(a: int, b: int):
    """Return a / b and throw an error if there's a remainder."""
    assert a % b == 0
    return a // b


def summarize_tensor(tensor: tensor) -> str:
    return "x".join(map(str, tensor.shape)) + "[" + str(round(tensor.view(-1)[0].item(), 4)) + "...]"


def render_duration(duration: float) -> str:
    if duration < 1e-3:
        return f"{duration * 1e6:.2f}us"
    if duration < 1:
        return f"{duration * 1e3:.2f}ms"
    return f"{duration:.2f}s"


if __name__ == "__main__":
    main()
