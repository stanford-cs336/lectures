import os
import time
from typing import Callable
import torch
from torch.profiler import ProfilerActivity
import triton
import triton.language as tl
from edtrace import text, link, image
from lecture_util import get_local_url, bilingual_text, bilingual_verbatim
from gpu_util import cuda_if_available


def main():
    bilingual_text("Last lecture: high-level overview of GPUs and performance", '上节课：GPU 和性能的高层概览。')
    bilingual_text("This lecture: benchmarking/profiling + writing kernels", '本节课：基准测试/性能分析，以及编写内核。')

    review_of_gpus()
    benchmarking_and_profiling()           # Where are the bottlenecks?
    naive_vs_builtin_vs_compiled_gelu()    # Apply it to the GeLU example

    # Write Triton kernels
    triton_introduction()
    triton_gelu_example()      # Elementwise operation
    triton_softmax_example()   # Reduction (row fits in a block)
    triton_row_sum_example()   # Reduction (row doesn't fit in block)
    triton_matmul_relu_example()    # Tiling: use shared memory

    bilingual_text("Summary:", '总结：')
    bilingual_text("- Know the programming model (PyTorch, Triton, PTX) to give you correctness", '- 了解编程模型（PyTorch、Triton、PTX），以保证正确性。')
    bilingual_text("- Understand the hardware (SMs, warps, occupancy, bank conflicts, etc.) to optimize performance", '- 理解硬件（SM、warp、占用率、bank 冲突等），以优化性能。')
    bilingual_text("- Benchmark to understand scaling", '- 通过基准测试理解扩展行为。')
    bilingual_text("- Profile to see what's being executed for how long", '- 通过性能分析查看执行了什么以及耗时多久。')
    bilingual_text("- Triton: think in terms of thread blocks (read to shared memory, do stuff (fusion), write back HBM)", '- Triton：以线程块为单位思考（读入共享内存、执行操作/融合、写回 HBM）。')
    bilingual_text("- Examples: GeLU (elementwise), softmax (row-wise), row sum (baby tiling), matmul (tiling)", '- 示例：GeLU（逐元素）、softmax（逐行）、行求和（入门分块）、矩阵乘法（分块）。')

    bilingual_text("Next time: more than one GPU!", '下次：不止一块 GPU！')


def review_of_gpus():
    bilingual_text("## Hardware", '## 硬件')
    image("images/gpu-hardware.png", width=800)
    bilingual_verbatim("| Accelerator                        | A100      | H100      | B200      |", '上方等宽内容是硬件规格表或矩阵示意的一行；英文、数值和列对齐保持原样。', verbatim=True)
    bilingual_verbatim("+------------------------------------+-----------+-----------+-----------+", '上方等宽内容是硬件规格表或矩阵示意的一行；英文、数值和列对齐保持原样。', verbatim=True)
    bilingual_verbatim("| # SMs                              |       108 |       132 |       148 |", '上方等宽内容是硬件规格表或矩阵示意的一行；英文、数值和列对齐保持原样。', verbatim=True)
    bilingual_verbatim("+------------------------------------+-----------+-----------+-----------+", '上方等宽内容是硬件规格表或矩阵示意的一行；英文、数值和列对齐保持原样。', verbatim=True)
    bilingual_verbatim("| Register size (per SM)             |    256 KB |    256 KB |    256 KB |", '上方等宽内容是硬件规格表或矩阵示意的一行；英文、数值和列对齐保持原样。', verbatim=True)
    bilingual_verbatim("| L1 cache + shared memory (per SM)  |    192 KB |    256 KB |    256 KB |", '上方等宽内容是硬件规格表或矩阵示意的一行；英文、数值和列对齐保持原样。', verbatim=True)
    bilingual_verbatim("| L2 cache size                      |     40 MB |     50 MB | 96-126 MB |", '上方等宽内容是硬件规格表或矩阵示意的一行；英文、数值和列对齐保持原样。', verbatim=True)
    bilingual_verbatim("| HBM size                           |     80 GB |     80 GB |    192 GB |", '上方等宽内容是硬件规格表或矩阵示意的一行；英文、数值和列对齐保持原样。', verbatim=True)
    bilingual_verbatim("+------------------------------------+-----------+-----------+-----------+", '上方等宽内容是硬件规格表或矩阵示意的一行；英文、数值和列对齐保持原样。', verbatim=True)
    bilingual_verbatim("| Register bandwidth                 | ~116 TB/s | ~401 TB/s | ~447 TB/s |", '上方等宽内容是硬件规格表或矩阵示意的一行；英文、数值和列对齐保持原样。', verbatim=True)
    bilingual_verbatim("| L1 cache + shared memory bandwidth |  ~19 TB/s |  ~33 TB/s |  ~19 TB/s |", '上方等宽内容是硬件规格表或矩阵示意的一行；英文、数值和列对齐保持原样。', verbatim=True)
    bilingual_verbatim("| L2 cache bandwidth                 | ~5-8 TB/s |  ~12 TB/s |   ~9 TB/s |", '上方等宽内容是硬件规格表或矩阵示意的一行；英文、数值和列对齐保持原样。', verbatim=True)
    bilingual_verbatim("| HBM bandwidth                      |    2 TB/s | 3.35 TB/s |    8 TB/s |", '上方等宽内容是硬件规格表或矩阵示意的一行；英文、数值和列对齐保持原样。', verbatim=True)

    bilingual_text("(B200s also have tensor memory (TMEM) for tensor cores (between registers and shared memory) that are invisible to programmer.)", 'B200 还为张量核心提供张量内存（TMEM），位于寄存器和共享内存之间，但对程序员不可见。')

    bilingual_text("## Programming model", '## 编程模型')
    image("https://docs.nvidia.com/cuda/parallel-thread-execution/_images/grid-with-CTAs.png", width=600)
    bilingual_text("- *Thread*: executes code on a small part of the data", '- *线程*：在数据的一小部分上执行代码。')
    bilingual_text("- *Thread block* or concurrent thread array (CTA): a group of threads", '- *线程块*或并发线程数组（CTA）：一组线程。')
    bilingual_text("- *Grid*: collection of thread blocks", '- *网格*：线程块的集合。')

    bilingual_text("(H100s and B200s also have thread block clusters that enable distributed shared memory.)", 'H100 和 B200 还具有线程块集群，可以启用分布式共享内存。')

    bilingual_text("Why thread blocks?", '为什么需要线程块？')
    bilingual_text("For elementwise operations (e.g., GeLU), threads are most natural: each thread processes one element.", '对于逐元素操作（例如 GeLU），线程是最自然的抽象：每个线程处理一个元素。')
    bilingual_text("- f(i) for i = 0, ..., N-1", '- f(i)，其中 i = 0, ..., N-1。')
    bilingual_text("However, for non-elementwise operations like softmax or matrix multiplication, threads need to communicate.", '但是，对于 softmax 或矩阵乘法这样的非逐元素操作，线程之间需要通信。')
    bilingual_text("Reading/writing from HBM is slow, so use shared memory (local to SM).", '从 HBM 读写很慢，因此要使用共享内存（位于 SM 本地）。')
    bilingual_text("Thread block: a collection of threads that access the same shared memory.", '线程块：一组访问同一块共享内存的线程。')
    bilingual_text("Consequently, a thread block is scheduled on one SM.", '因此，一个线程块会被调度到一个 SM 上。')
    bilingual_text("In Triton, think natively in terms of thread blocks (later).", '在 Triton 中，要自然地按线程块来思考（后面会看到）。')

    bilingual_text("## Interaction between programming model and hardware", '## 编程模型与硬件的相互作用')
    bilingual_text("Programming model provides an abstraction of the hardware.", '编程模型提供了对硬件的抽象。')
    bilingual_text("In principle, don't need to think about anything else (for correctness).", '原则上，为了正确性你不需要考虑其他细节。')
    bilingual_text("In practice, performance is very sensitive to the hardware, so need to understand it to obtain high performance.", '实践中，性能对硬件非常敏感，因此要获得高性能就必须理解硬件。')

    bilingual_text("Let's go over some considerations.", '让我们看一些需要考虑的因素。')

    bilingual_text("**Warps**:", '**Warp（线程束）**：')
    bilingual_text("- Within a thread block, threads are grouped into warps (32 threads per warp).", '- 在线程块内部，线程会被分组成 warp（每个 warp 32 个线程）。')
    bilingual_text("- Example: thread block has 64 threads => it has 2 warps.", '- 示例：一个线程块有 64 个线程 => 它有 2 个 warp。')
    bilingual_verbatim("| TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT | TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT |", '上方等宽内容是硬件规格表或矩阵示意的一行；英文、数值和列对齐保持原样。', verbatim=True)
    bilingual_text("- All threads within a warp must execute same instructions in lockstep on an SM.", '- 同一个 warp 内的所有线程必须在 SM 上锁步执行相同指令。')
    bilingual_text("- Control divergence: if different threads in a warp need to execute different instructions (if A, else B), must be done sequentially (bad)", '- 控制流分歧：如果同一 warp 中不同线程需要执行不同指令（if A, else B），就必须顺序执行（不好）。')
    bilingual_verbatim("| AAAAAAAAA....................... |", '上方等宽内容是硬件规格表或矩阵示意的一行；英文、数值和列对齐保持原样。', verbatim=True)
    bilingual_verbatim("| .........BBBBBBBBBBBBBBBBBBBBBBB |", '上方等宽内容是硬件规格表或矩阵示意的一行；英文、数值和列对齐保持原样。', verbatim=True)
    bilingual_text("- SM runs multiple warps and switches between them (e.g., when one warp is blocked on HBM reads/writes) with zero cost.", '- SM 会运行多个 warp，并在它们之间零成本切换（例如某个 warp 因 HBM 读写而阻塞时）。')

    bilingual_text("**(Warp) occupancy**:", '**（Warp）占用率**：')
    bilingual_text("- Each thread can use between 0 and 255 registers.", '- 每个线程可以使用 0 到 255 个寄存器。')
    bilingual_text("- The more registers threads use, the fewer threads can be scheduled on an SM (low occupancy).", '- 每个线程使用的寄存器越多，一个 SM 上可调度的线程越少（占用率低）。')
    bilingual_text("- Low occupancy isn't necessarily bad if each thread is doing more work.", '- 如果每个线程做了更多工作，低占用率不一定是坏事。')
    bilingual_text("- Example: thread coarsening (each thread processes multiple elements).", '- 示例：线程粗化（每个线程处理多个元素）。')
    bilingual_text("- Example: thread block has 64 threads, each using 160 registers, SM has 65536 registers", '- 示例：线程块有 64 个线程，每个线程使用 160 个寄存器，SM 有 65536 个寄存器。')
    
    # What we want to run
    num_threads_per_block = 128
    num_registers_per_thread = 160

    # What hardware offers
    max_registers = 65536  # Registers allowed per SM
    max_warps = 64         # Concurrent warps allowed per SM

    # What we can run at once
    assert num_registers_per_thread <= 255
    num_registers_per_block = num_threads_per_block * num_registers_per_thread  # @inspect num_registers_per_block
    num_blocks = max_registers // num_registers_per_block  # Limited by registers @inspect num_blocks
    num_warps = num_blocks * num_threads_per_block / 32  # @inspect num_warps
    occupancy = num_warps / max_warps  # @inspect occupancy

    bilingual_text("**Bank conflicts** (shared memory):", '**Bank 冲突**（共享内存）：')
    bilingual_text("- Shared memory is divided into 32 banks, each 4 bytes wide.", '- 共享内存被划分为 32 个 bank，每个 bank 宽 4 字节。')
    bilingual_verbatim("B00 B01 B02 B03 B04 B05 B06 B07 B08 B09 B10 B11 B12 B13 B14 B15 B16 B17 B18 B19 B20 B21 B22 B23 B24 B25 B26 B27 B28 B29 B30 B31", '上方等宽内容是访问模式示意；字符布局保持原样。', verbatim=True)
    bilingual_verbatim("... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...", '上方等宽内容是访问模式示意；字符布局保持原样。', verbatim=True)
    bilingual_verbatim("... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...", '上方等宽内容是访问模式示意；字符布局保持原样。', verbatim=True)
    bilingual_verbatim("... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...", '上方等宽内容是访问模式示意；字符布局保持原样。', verbatim=True)
    bilingual_text("- Each cycle, each bank can only be accessed by one thread (if not the same exact location).", '- 每个周期中，每个 bank 只能被一个线程访问（除非访问的是完全相同的位置）。')
    bilingual_text("- If multiple threads access the same bank, accesses serialized (bank conflict).", '- 如果多个线程访问同一个 bank，访问会串行化（bank 冲突）。')
    bilingual_text("- Worst case example: matrix where each row spans all banks; 32 threads accessing first column results in 32-way bank conflict!", '- 最坏示例：矩阵每一行跨越所有 bank；32 个线程访问第一列会导致 32 路 bank 冲突！')
    bilingual_text("- Unavoidable: when doing matmul A @ B, access rows of A and columns of B", '- 难以避免：做矩阵乘法 A @ B 时，会访问 A 的行和 B 的列。')
    bilingual_text("- Solution: swizzling rearranges shared memory (e.g., row xor col) to avoid bank conflicts", '- 解决方案：swizzling 重新排列共享内存（例如 row xor col）以避免 bank 冲突。')

    bilingual_text("**Memory coalescing** (HBM):", '**内存合并访问**（HBM）：')
    bilingual_text("- When the 32 threads in a warp access HBM, memory accesses combined into transactions of 128 bytes (cache lines).", '- 当一个 warp 中的 32 个线程访问 HBM 时，内存访问会合并为 128 字节的事务（缓存行）。')
    bilingual_verbatim("M00 M01 M02 M03 M04 M05 M06 M07 M08 M09 M10 M11 M12 M13 M14 M15 M16 M17 M18 M19 M20 M21 M22 M23 M24 M25 M26 M27 M28 M29 M30 M31", '上方等宽内容是访问模式示意；字符布局保持原样。', verbatim=True)
    bilingual_verbatim("M32 M33 M34 M35 M36 M37 M38 M39 M40 M41 M42 M43 M44 M45 M46 M47 M48 M49 M50 M51 M52 M53 M54 M55 M56 M57 M58 M59 M60 M61 M62 M63", '上方等宽内容是访问模式示意；字符布局保持原样。', verbatim=True)
    bilingual_text("- Best case: full coalescing, all threads access the same cache line (32 threads x 4 bytes = 128 bytes).", '- 最好情况：完全合并访问，所有线程访问同一条缓存行（32 个线程 x 4 字节 = 128 字节）。')

    bilingual_text("**Block occupancy**:", '**块占用率**：')
    image("https://developer-blogs.nvidia.com/wp-content/uploads/2019/06/pasted-image-0.png", width=400)
    bilingual_text("- Thread blocks scheduled onto SMs in waves.", '- 线程块会以一波一波的方式调度到 SM 上。')
    bilingual_text("- B200 has 148 SMs, if we launch 160 thread blocks, first wave has 148 blocks, second wave has 12 blocks.", '- B200 有 148 个 SM；如果启动 160 个线程块，第一波有 148 个块，第二波只有 12 个块。')
    bilingual_text("- Wave quantization problem: last wave has fewer thread blocks, leaving some SMs idle (low block occupancy).", '- 波量化问题：最后一波线程块较少，会让一些 SM 空闲（块占用率低）。')
    bilingual_text("- Solution: make number of thread blocks divide # SMs.", '- 解决方案：让线程块数量能整除 SM 数量。')

    bilingual_text("Summary:", '总结：')
    bilingual_text("- Programming model: grid (HBM) -> thread block (shared memory) -> thread (registers)", '- 编程模型: grid (HBM) -> 线程块 (共享内存) -> 线程 (寄存器)')
    bilingual_text("- Details of hardware (warps, bank conflicts, memory coalescing, occupancy) determine performance", '- Details of 硬件 (warp, bank 冲突, 内存合并访问, 占用率) determine 性能')


def benchmarking_and_profiling():
    bilingual_text("Recipe for success:", '成功配方：')
    bilingual_text("1. Benchmark and profile your code", '1. 对代码做基准测试和性能分析。')
    bilingual_text("2. Make changes", '2. 做出修改。')
    bilingual_text("3. Benchmark and profile your code again", '3. 再次做基准测试和性能分析。')

    benchmarking()   # How long does it take?
    profiling()      # Where time is being spent?

    bilingual_text("Benchmark and profile your code!", '对你的代码做基准测试和性能分析！')


def benchmarking():
    bilingual_text("Benchmarking measures the wall-clock time of performing some operation.", '基准测试衡量执行某个操作的墙钟时间。')
    bilingual_text("It only gives you end-to-end time, not where time is spent (profiling).", '它只给出端到端时间，不告诉你时间花在哪里（这需要性能分析）。')

    bilingual_text("It is still useful for:", '它仍然有用，因为可以：')
    bilingual_text("- comparing different implementations (which is faster?), and", '- 比较不同实现（哪个更快？），以及')
    bilingual_text("- understanding how performance scales (e.g., with dimension).", '- 理解性能如何随规模变化（例如随维度变化）。')

    bilingual_text("You can use [`torch.utils.benchmark`](https://pytorch.org/tutorials/recipes/recipes/benchmark.html).", 'You can use [torch.utils.基准测试](https://pytorch.org/tutorials/recipes/recipes/基准测试.html).')
    bilingual_text("We will roll our own to make benchmarking more transparent.", '我们会自己实现一个基准测试工具，让过程更透明。')

    # Benchmark matrix multiplication
    matmul = run_operation2(dim=1024, operation=lambda a, b: a @ b)
    result = benchmark(matmul)  # @inspect result

    # See how timing scales with dimension
    results = {}
    for dim in [256, 512, 1024, 2048, 4096, 8192]:
         results[dim] = benchmark(run_operation2(dim=dim, operation=lambda a, b: a @ b))  # @inspect results @stepover

    bilingual_text("Note: time is roughly constant when dimension is small, then cubic scaling.", '注意：维度较小时耗时大致恒定，之后呈三次方扩展。')


def benchmark(run: Callable, num_warmups: int = 1, num_trials: int = 3) -> float:
    """Benchmark `func` by running it `num_trials`.  Return the average time."""
    # Warmup: first times might be slower due to compilation, etc.
    # Since we will run the kernel multiple times, the timing that matters is steady state.
    for _ in range(num_warmups):
        run()
    torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

    # Time it for real now!
    times: list[float] = [] # @inspect times
    for trial in range(num_trials):  # Do it multiple times to capture variance
        # Use CUDA events for accurate GPU timing (avoid capturing CPU overhead)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()  # Start timing
        run()  # Actually perform computation
        end_event.record()  # End timing

        torch.cuda.synchronize()  # Wait for CUDA threads to finish

        times.append((start_event.elapsed_time(end_event)))  # @inspect times

    mean_time = mean(times)   # @inspect mean_time @stepover
    return mean_time


def profiling():
    bilingual_text("While benchmarking looks at end-to-end time, profiling looks at where time is spent.", '基准测试看端到端时间，而性能分析看时间花在了哪里。')
    bilingual_text("Independent of time, profiling also helps you understand what's going under the hood.", '除了时间以外，性能分析还能帮助你理解底层实际发生了什么。')

    bilingual_text("PyTorch has a built-in [profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html).", 'PyTorch 内置了 [profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)。')
    bilingual_text("In your assignment, you will use nsight to get more details.", '在作业中，你会使用 nsight 获得更多细节。')

    bilingual_text("## add(dim=2048)", '## 说明：add(dim=2048)')
    add_profile = profile(run_operation2(dim=2048, operation=lambda a, b: a + b))
    text(add_profile, verbatim=True)

    bilingual_text("## matmul(dim=2048)", '## 矩阵乘法(dim=2048)')
    matmul_profile = profile(run_operation2(dim=2048, operation=lambda a, b: a @ b)) # @stepover
    text(matmul_profile, verbatim=True)

    bilingual_text("## matmul(dim=128)", '## 矩阵乘法(dim=128)')
    matmul_profile = profile(run_operation2(dim=128, operation=lambda a, b: a @ b)) # @stepover
    text(matmul_profile, verbatim=True)

    bilingual_text("Observations:", '观察：')
    bilingual_text("- You can see which CUDA kernels are actually being called (the long names).", '- 你可以看到实际调用了哪些 CUDA 内核（那些很长的名字）。')
    bilingual_text("- Different CUDA kernels are invoked depending on the tensor dimensions.", '- 根据张量维度不同，会调用不同的 CUDA 内核。')

    bilingual_text("Name of CUDA kernel tells us something about the implementation.", 'CUDA 内核名称会透露一些实现信息。')
    bilingual_text("Example: cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x64x16_1x1x1_3_nnn_align1_bi...", '示例：cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x64x16_1x1x1_3_nnn_align1_bi...')
    bilingual_text("- cutlass: NVIDIA's CUDA library for linear algebra", '- cutlass：NVIDIA 用于线性代数的 CUDA 库。')
    bilingual_text("- sm100: corresponds to the NVIDIA Blackwell architecture (B200)", '- sm100：对应 NVIDIA Blackwell 架构（B200）。')
    bilingual_text("- f32: float32", '- f32：float32，即 32 位浮点数。')
    bilingual_text("- 64x64x16: tile shape (more on this later)", '- 64x64x16：分块形状（后面会进一步说明）。')


def profile(run: Callable, num_warmups: int = 1):
    # Warmup
    for _ in range(num_warmups):
        run()
    torch.cuda.synchronize()

    # Run the code with the profiler
    with torch.profiler.profile(activities=[ProfilerActivity.CUDA],
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)) as prof:
        run()
        torch.cuda.synchronize()

    # Print out table
    table = prof.key_averages().table(sort_by="cuda_time_total",
                                      max_name_column_width=100,
                                      row_limit=10)

    # Append to profiles.txt
    with open("var/profiles.txt", "a") as f:
        f.write(f"Profile at {time.ctime()}:\n")
        f.write(table)
        f.write("\n\n")
    return table


def naive_vs_builtin_vs_compiled_gelu():
    bilingual_text("Let's benchmark and profile the [GeLU activation function](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html).", "Let's 基准测试 and 性能分析 the [GeLU activation function](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html).")

    x = torch.tensor([1.])  # @inspect x

    # 1. Implementation naively from scratch in PyTorch (non-fused)
    y1 = naive_gelu(x)  # @inspect y1

    # 2. Built-in PyTorch implementation (fused)
    y2 = builtin_gelu(x)  # @inspect y2
    check_equal_1d(naive_gelu, builtin_gelu)  # Check it works

    # 3. Use PyTorch compiler on the naive implementation
    compiled_gelu = torch.compile(naive_gelu)  # @stepover
    y3 = compiled_gelu(x)  # @inspect y3 @stepover
    check_equal_1d(naive_gelu, compiled_gelu)  # Check it works (compilation shouldn't change semantics) @stepover

    # Benchmarking
    naive_time = benchmark(run_operation1(dim=16384, operation=naive_gelu)) # @inspect naive_time @stepover
    builtin_time = benchmark(run_operation1(dim=16384, operation=builtin_gelu)) # @inspect builtin_time @stepover
    compiled_time = benchmark(run_operation1(dim=16384, operation=compiled_gelu)) # @inspect compiled_time @stepover
    bilingual_text("The builtin and compiled versions are significantly faster!", '内置版本和编译版本明显更快！')

    bilingual_text("To understand why, let's look at the profiler to see where time is being spent.", '为了理解原因，我们查看 profiler，看时间花在哪里。')

    bilingual_text("## naive_gelu", '## 说明：naive_gelu')
    naive_gelu_profile = profile(run_operation1(dim=16384, operation=naive_gelu))  # @stepover
    text(naive_gelu_profile, verbatim=True)

    bilingual_text("## builtin_gelu", '## 说明：builtin_gelu')
    builtin_gelu_profile = profile(run_operation1(dim=16384, operation=builtin_gelu))  # @stepover
    text(builtin_gelu_profile, verbatim=True)

    bilingual_text("## compiled_gelu", '## 说明：compiled_gelu')
    compiled_gelu_profile = profile(run_operation1(dim=16384, operation=compiled_gelu))  # @stepover
    text(compiled_gelu_profile, verbatim=True)

    bilingual_text("Notes:", '说明：')
    bilingual_text("- Naive implementation: multiple kernels, requires many reads/writes from/to HBM (**no fusion**).", '- 朴素实现：多个内核，需要多次从 HBM 读取/写入 HBM（**没有融合**）。')
    bilingual_text("- Builtin and compiled versions: one kernel (**kernel fusion**), one read from HBM, one write to HBM.", '- 内置和编译版本：一个内核（**内核融合**），一次从 HBM 读取，一次写回 HBM。')
    bilingual_text("- The compiled kernel is a Triton kernel.", '- 编译后的内核是 Triton 内核。')


def triton_introduction():
    image("https://docs.nvidia.com/cuda/parallel-thread-execution/_images/grid-with-CTAs.png", width=600)

    bilingual_text("In CUDA (developed by NVIDIA), specify what each thread does.", '在 CUDA（由 NVIDIA 开发）中，需要指定每个线程做什么。')
    bilingual_text("- Pros: fine-grained control", '- 优点：细粒度控制。')
    bilingual_text("- Cons: need to manage more things (e.g., shared memory)", '- 缺点：需要管理更多东西（例如共享内存）。')

    bilingual_text("In Triton (developed by OpenAI), specify what each thread block does.", '在 Triton（由 OpenAI 开发）中，需要指定每个线程块做什么。')
    bilingual_text("- Generally powerful enough (especially when getting started)", '- 通常已经足够强大（尤其是入门时）。')
    bilingual_text("- Conceptual framework: load data into shared memory, operate on it, write back to global memory", '- 概念框架：把数据加载到共享内存，对其操作，再写回全局内存。')


def triton_gelu_example():
    bilingual_text("Let's write the Triton kernel for GeLU.", '让我们为 GeLU 编写 Triton 内核。')

    x = torch.randn(8192, device=cuda_if_available())
    y = triton_gelu(x)

    check_equal_1d(triton_gelu, naive_gelu)  # Check for correctness @stepover

    bilingual_text("Triton compiles down to PTX (parallel thread execution), an assembly language for GPUs.", 'Triton 会编译成 PTX（parallel thread execution），这是 GPU 的汇编语言。')

    bilingual_text("We can see the PTX code generated by Triton.", '我们可以看到 Triton 生成的 PTX 代码。')
    link(get_local_url("var/triton_gelu-ptx.txt"))

    bilingual_text("Observations:", '观察：')
    bilingual_text("- ld.global.* and st.global.* reads and writes from global memory", '- ld.global.* 和 st.global.* 表示从全局内存读写。')
    bilingual_text("- %ctaid.x is block index, %tid.x is thread index", '- %ctaid.x 是块索引，%tid.x 是线程索引。')
    bilingual_text("- %f* are floating point registers, %r* are integer registers", '- %f* 是浮点寄存器，%r* 是整数寄存器。')
    bilingual_text("- One thread processes 8 elements at the same time (thread coarsening)", '- 一个线程同时处理 8 个元素（线程粗化）。')


def triton_gelu(x: torch.Tensor):
    # Check input
    assert x.is_cuda
    assert x.is_contiguous()

    # Allocate output tensor
    y = torch.empty_like(x)

    # Determine grid (elements divided into blocks)
    # | T T T T T T T T | T T T T T T T T | T T T T T T T T | T T T T T T T T |
    # |    Block 0      |    Block 1      |     Block 2      |    Block 3     |
    num_elements = x.numel()  # @inspect num_elements
    BLOCK_SIZE = 1024  # Number of threads
    num_blocks = triton.cdiv(num_elements, BLOCK_SIZE)  # @inspect num_blocks

    # Launch the kernel
    kernel = triton_gelu_kernel[(num_blocks,)](x, y, num_elements, BLOCK_SIZE=BLOCK_SIZE)

    # Write out PTX (look at this later)
    output_ptx("triton_gelu", kernel)  # @stepover

    return y


@triton.jit
def triton_gelu_kernel(x_ptr, y_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    # Input starts at `x_ptr`
    # Output starts at `y_ptr`

    # | T T T T T T T T | T T T T T T T T | T T T T T T T T | T T T T T T T T |
    # |    Block 0      |    Block 1      |     Block 2      |    Block 3     |

    pid = tl.program_id(axis=0)      # Identifies the block
    start = pid * BLOCK_SIZE         # Starting index of this block

    # Indices where this thread block should operate
    offsets = start + tl.arange(0, BLOCK_SIZE)

    # Don't read/write past the end of the tensor
    mask = offsets < num_elements

    # Read
    x = tl.load(x_ptr + offsets, mask=mask)

    # Approx gelu is 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Compute (tl.tanh doesn't exist, use tanh(a) = (exp(2a) - 1) / (exp(2a) + 1)
    a = 0.79788456 * (x + 0.044715 * x * x * x)
    exp = tl.exp(2 * a)
    tanh = (exp - 1) / (exp + 1)
    y = 0.5 * x * (1 + tanh)

    # Store
    tl.store(y_ptr + offsets, y, mask=mask)


def triton_softmax_example():
    bilingual_text("So far, we've looked at elementwise operations in Triton (e.g., GeLU).", '到目前为止，我们看了 Triton 中的逐元素操作（例如 GeLU）。')
    bilingual_text("Now let us look at operations that aggregate over multiple values.", '现在来看会聚合多个值的操作。')

    bilingual_text("We will roughly follow the Triton fused softmax tutorial: ", '说明：We will roughly follow the Triton fused softmax tutorial:'), link("https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html")

    bilingual_text("Recall the softmax operation is used in attention and generating probabilities.", '回忆一下，softmax 用于注意力和生成概率。')
    bilingual_text("Exponentiate and normalize each row of a matrix:", '对矩阵的每一行做指数化并归一化：')
    bilingual_verbatim("[0 0 0]      =>   [1/3 1/3 1/3]", '上方等宽内容保持原样，用于展示代码、性能输出或矩阵布局。', verbatim=True)
    bilingual_verbatim("[1 1 -inf]        [1/2 1/2 0  ]", '上方等宽内容保持原样，用于展示代码、性能输出或矩阵布局。', verbatim=True)

    bilingual_text("Let's first start with the naive implementation and keep track of reads/writes.", '先从朴素实现开始，并跟踪读写次数。')
    x = torch.tensor([
        [5., 5, 5],
        [0, 0, 100],
    ], device=cuda_if_available())
    y1 = naive_softmax(x) # @inspect y1

    bilingual_text("Now let us write the Triton kernel.", '现在来编写 Triton 内核。')
    image("images/triton-softmax.png", width=600)
    y2 = triton_softmax(x)  # @inspect y2

    # Check our implementations are correct
    check_equal_2d(pytorch_softmax, naive_softmax) # @stepover
    check_equal_2d(pytorch_softmax, triton_softmax) # @stepover


def naive_softmax(x: torch.Tensor):
    # M: number of rows, N: number of columns
    M, N = x.shape

    # Compute the max of each row (MN reads, M writes)
    x_max = x.max(dim=1)[0]

    # Subtract off the max (MN + M reads, MN writes)
    x = x - x_max[:, None]

    # Exponentiate (MN reads, MN writes)
    numerator = torch.exp(x)

    # Compute normalization constant (MN reads, M writes)
    denominator = numerator.sum(dim=1)

    # Normalize (MN reads, MN writes)
    y = numerator / denominator[:, None]

    # Total: 5MN + M reads, 3MN + 2M writes
    # In principle, should have MN reads, MN writes (speedup of 4x!)
    return y


def triton_softmax(x: torch.Tensor):
    # Allocate output tensor
    y = torch.empty_like(x)

    # Determine grid
    M, N = x.shape                          # Number of rows x number of columns
    block_size = triton.next_power_of_2(N)  # Each block contains all the columns
    num_blocks = M                          # Each block is a row

    # Launch kernel
    triton_softmax_kernel[(M,)](
        x_ptr=x, y_ptr=y,
        x_row_stride=x.stride(0), y_row_stride=y.stride(0),
        num_cols=N, BLOCK_SIZE=block_size
    )

    return y


@triton.jit
def triton_softmax_kernel(x_ptr, y_ptr, x_row_stride, y_row_stride, num_cols, BLOCK_SIZE: tl.constexpr):
    assert num_cols <= BLOCK_SIZE

    # Process each row independently
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # Read from global memory
    x_start_ptr = x_ptr + row_idx * x_row_stride
    x_ptrs = x_start_ptr + col_offsets
    x_row = tl.load(x_ptrs, mask=col_offsets < num_cols, other=float("-inf"))

    # Compute
    x_row = x_row - tl.max(x_row, axis=0)
    numerator = tl.exp(x_row)
    denominator = tl.sum(numerator, axis=0)
    y_row = numerator / denominator

    # Write back to global memory
    y_start_ptr = y_ptr + row_idx * y_row_stride
    y_ptrs = y_start_ptr + col_offsets
    tl.store(y_ptrs, y_row, mask=col_offsets < num_cols)


def triton_row_sum_example():
    bilingual_text("In the softmax example, an entire row fits in a block, so the reduction happens within a block (handled by Triton).", '在 softmax 示例中，整行可以放进一个块，所以规约发生在块内部（由 Triton 处理）。')
    bilingual_text("What if the row doesn't fit in a block?", '如果一行放不进一个块怎么办？')
    bilingual_text("Example: 4096 columns, but block size is 1024...", '示例：有 4096 列，但块大小是 1024……')

    bilingual_text("Strategy:", '策略：')
    bilingual_text("- Break up row into tiles (4 in the example above)", '- 把一行拆成多个分块（上例中为 4 个）。')
    bilingual_text("- Each thread iterates over tiles and accumulates a sum", '- 每个线程遍历分块并累加求和。')
    bilingual_text("- Do final reduction (sum) over accumulators of each thread (shared memory or warp shuffles)", '- 对每个线程的累加器做最终规约（求和）（使用共享内存或 warp shuffle）。')

    bilingual_text("Consider the simpler example (row sum instead of softmax):", '考虑一个更简单的例子（行求和，而不是 softmax）：')
    x = torch.tensor([[1., 2, 3, 4], [5, 6, 7, 8]], device=cuda_if_available())  # @inspect x
    y1 = builtin_row_sum(x)  # @inspect y1

    image("images/triton-row-sum.png", width=600)

    y2 = triton_row_sum(x)  # @inspect y2


def builtin_row_sum(x: torch.Tensor):
    return x.sum(dim=1)


def triton_row_sum(x: torch.Tensor, BLOCK_SIZE: int = 1024) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty(M, device=x.device, dtype=x.dtype)
    row_sum_kernel[(M,)](x, y, N, BLOCK_SIZE=BLOCK_SIZE)
    return y


@triton.jit
def row_sum_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)  # Which row are we processing?

    # Accumulator for each thread
    # One row: T1 T2 T3 T4 | T1 T2 T3 T4 | T1 T2 T3 T4 (N = 12, BLOCK_SIZE = 4)
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Loop over tiles
    for start in range(0, N, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0)
        acc += x

    # Final reduction from BLOCK_SIZE (all threads) to a scalar
    result = tl.sum(acc, axis=0)

    tl.store(out_ptr + row, result)


def triton_matmul_relu_example():
    bilingual_text("Matrix multiplication is the bread and butter of deep learning.", '矩阵乘法是深度学习的核心基础。')
    a = torch.randn(1024, 1024, device=cuda_if_available())
    b = torch.randn(1024, 1024, device=cuda_if_available())
    c = naive_matmul_relu(a, b)

    bilingual_text("How should we build a matmul kernel?", '我们应该如何构建矩阵乘法内核？')

    bilingual_verbatim("|        k                  n                     ", '上方等宽内容是硬件规格表或矩阵示意的一行；英文、数值和列对齐保持原样。', verbatim=True)
    bilingual_verbatim("|   [ A1 A2 A3 ]       [ B1 B2 B3 ]   [ C1 C2 C3 ]", '上方等宽内容是硬件规格表或矩阵示意的一行；英文、数值和列对齐保持原样。', verbatim=True)
    bilingual_verbatim("| m [ A4 A5 A6 ]  *  k [ B4 B5 B6 ] = [ C4 C5 C6 ]", '上方等宽内容是硬件规格表或矩阵示意的一行；英文、数值和列对齐保持原样。', verbatim=True)
    bilingual_verbatim("|   [ A7 A8 A9 ]       [ B7 B8 B9 ]   [ C7 C8 C9 ]", '上方等宽内容是硬件规格表或矩阵示意的一行；英文、数值和列对齐保持原样。', verbatim=True)

    bilingual_text("**Naive approach:**", '**朴素方法：**')
    bilingual_text("Fix any (m, n).", '固定任意一个 (m, n)。')
    bilingual_text("For each k:", '对每个 k：')
    bilingual_text("- Read A[m, k] and B[k, n] from HBM.", '- 从 HBM 读取 A[m, k] 和 B[k, n]。')
    bilingual_text("- Multiply and accumulate.", '- 相乘并累加。')
    bilingual_text("Write result to C[m, n] in HBM.", '把结果写入 HBM 中的 C[m, n]。')

    bilingual_text("Bottleneck: M K N reads, M N writes", '瓶颈：M K N 次读取，M N 次写入。')
    bilingual_text("Arithmetic intensity: O(1)", '算术强度：O(1)。')

    bilingual_text("Computing C4 and C5 both need A4, A5, A6.", '计算 C4 和 C5 都需要 A4、A5、A6。')
    bilingual_text("Can we read A4, A5, A6 from HBM once to compute both?", '能否只从 HBM 读取一次 A4、A5、A6，就同时计算二者？')
    bilingual_text("Answer: yes, using shared memory!", '答案是可以，使用共享内存！')

    bilingual_text("**Idealized approach:**", '**理想化方法：**')
    bilingual_text("- Load all of A and B into shared memory, then compute C.", '- 把 A 和 B 全部加载到共享内存，然后计算 C。')
    bilingual_text("- Now we get M K + K N reads and M N writes.", '- 现在得到 M K + K N 次读取和 M N 次写入。')
    bilingual_text("- This yields the idealized O(N) arithmetic intensity from before.", '- 这得到前面理想化的 O(N) 算术强度。')
    bilingual_text("- However, A and B are usually too large to fit in shared memory.", '- 但是，A 和 B 通常太大，无法放入共享内存。')

    bilingual_text("**Tiling:**", '**分块：**')

    image("images/gemm_tiled.png", width=600)
    bilingual_text("Key idea: divide the matrix C into output tiles (thread blocks).", '关键思想：把矩阵 C 划分为输出分块（线程块）。')
    bilingual_text("Fix an output tile in C.", '固定 C 中的一个输出分块。')
    bilingual_text("For each pair of (row tile of A, column tile of B):", '对每一对（A 的行分块，B 的列分块）：')
    bilingual_text("- Load the corresponding A tile and B tile from HBM into shared memory.", '- 从 HBM 把对应的 A 分块和 B 分块加载到共享内存。')
    bilingual_text("- Perform matrix multiplication on the tiles.", '- 在这些分块上执行矩阵乘法。')
    bilingual_text("- Accumulate into the partial sum (in shared memory).", '- 累加到部分和中（位于共享内存）。')
    bilingual_text("Write output tile to HBM.", '把输出分块写回 HBM。')

    bilingual_text("Arithmetic intensity: O(tile_size).", '算术强度：O(tile_size)。')

    bilingual_text("Bonus:", '额外收益：')
    bilingual_text("- Often, you want to apply an elementwise activation function.", '- 通常，你还想应用逐元素激活函数。')
    bilingual_text("- Example: GeLU(A @ B)", '- 示例：GeLU(A @ B)。')
    bilingual_text("- Solution: kernel fusion!", '- 解决方案：内核融合！')

    bilingual_text("**Implementation.**", '**实现。**')

    bilingual_text("Review: each matrix is linearized in memory", '回顾：每个矩阵都会在线性内存中展开。')
    x = torch.tensor([[0., 1, 2, 3], [4, 5, 6, 7]])  # @inspect x
    stride_row, stride_col = x.stride()  # @inspect stride_row stride_col
    row = 1
    col = 2
    index = row * stride_row + col * stride_col  # @inspect index

    # Compute c = a @ b
    c = triton_matmul_relu(a, b)


def naive_matmul_relu(x: torch.Tensor, y: torch.Tensor):
    # Matmul followed by ReLU
    return torch.nn.functional.relu(x @ y)


def triton_matmul_relu(a: torch.Tensor, b: torch.Tensor):
    assert a.is_cuda and b.is_cuda
    assert a.is_contiguous() and b.is_contiguous()
    assert a.shape[1] == b.shape[0]

    # A is M x K, B is K x N
    M, K = a.shape
    K, N = b.shape

    # Allocate output tensor
    c = torch.empty((M, N), device=a.device)

    # Determine grid
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_relu_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
    )

    return c


@triton.jit
def matmul_relu_kernel(
    a_ptr, b_ptr, c_ptr,    # Compute c = a @ b
    M, N, K,                # a is M x K, b is K x N, c is M x N
    stride_am, stride_ak,   # How to navigate a
    stride_bk, stride_bn,   # How to navigate b
    stride_cm, stride_cn,   # How to navigate c
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # We are working on the (m, n)-th tile
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Indices
    indices_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # Row indices of a [BLOCK_M]
    indices_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # Column indices of b [BLOCK_N]
    indices_k = tl.arange(0, BLOCK_K)                    # Row indices of a = column indices of b [BLOCK_K]

    # Initial matrix of pointers of a and b
    a_ptrs = a_ptr + indices_m[:, None] * stride_am + indices_k[None, :] * stride_ak  # [BLOCK_M, BLOCK_K]
    b_ptrs = b_ptr + indices_k[:, None] * stride_bk + indices_n[None, :] * stride_bn  # [BLOCK_K, BLOCK_N]

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Move along row tiles of a, column tiles of b
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(indices_m[:, None] < M) & (indices_k[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(indices_k[:, None] + k < K) & (indices_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak  # Advance to the next row tile of a
        b_ptrs += BLOCK_K * stride_bk  # Advance to the next column tile of b

    # Apply activation function (e.g., ReLU)
    acc = tl.maximum(acc, 0.0)

    # Write output tile
    c_ptrs = c_ptr + indices_m[:, None] * stride_cm + indices_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(indices_m[:, None] < M) & (indices_n[None, :] < N))


############################################################

def run_operation1(dim: int, operation: Callable) -> Callable:
    # Setup: create one random dim x dim matrices
    x = torch.randn(dim, dim, device=cuda_if_available())
    # Return a function to perform the operation
    return lambda : operation(x)


def run_operation2(dim: int, operation: Callable) -> Callable:
    # Setup: create two random dim x dim matrices
    x = torch.randn(dim, dim, device=cuda_if_available())
    y = torch.randn(dim, dim, device=cuda_if_available())
    # Return a function to perform the operation
    return lambda : operation(x, y)


def naive_gelu(x: torch.Tensor):
    # tanh approximation to the gelu activation function
    # https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html
    return 0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * x * x * x)))


def builtin_gelu(x: torch.Tensor):
    # PyTorch's built-in GeLU with the tanh approximation
    return torch.nn.functional.gelu(x, approximate="tanh")


def pytorch_softmax(x: torch.Tensor):
    return torch.nn.functional.softmax(x, dim=-1)


def check_equal_1d(f1, f2):
    x = torch.randn(2048, device=cuda_if_available())
    y1 = f1(x)  # @stepover
    y2 = f2(x)  # @stepover
    assert torch.allclose(y1, y2, atol=1e-6)


def check_equal_2d(f1, f2):
    x = torch.randn(2048, 2048, device=cuda_if_available())
    y1 = f1(x)
    y2 = f2(x)
    assert torch.allclose(y1, y2, atol=1e-6)


def check_equal_2d_2d(f1, f2):
    x1 = torch.randn(2048, 2048, device=cuda_if_available())
    x2 = torch.randn(2048, 2048, device=cuda_if_available())
    y1 = f1(x1, x2)
    y2 = f2(x1, x2)
    assert torch.allclose(y1, y2, atol=1e-6)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)

    
def output_ptx(name: str, kernel):
    """Print out the PTX code generated by Triton for the given `kernel`."""
    ptx_path = f"var/{name}-ptx.txt"
    with open(ptx_path, "w") as f:
        ptx = kernel.asm["ptx"]
        f.write(ptx)


if __name__ == "__main__":
    main()
