import math
import torch.nn.functional as F
import timeit
from typing import Iterable
import torch
from torch import nn
from einops import rearrange, einsum, reduce

from edtrace import text, image, link
from lecture_util import article_link, bilingual_text
from gpu_util import cuda_if_available, get_max_memory_usage
from facts import h100_flop_per_sec, h100_bytes_per_sec
from references import deepseek_v3_2_2025, adagrad_2011, nemotron_3_super_2026


def main():
    bilingual_text("Announcements:", '通知：')
    bilingual_text("- Join the CS336 slack", '- 加入 CS336 Slack。')
    bilingual_text("- Sign up on Modal with your **Stanford** email", '- 使用你的 **Stanford** 邮箱注册 Modal。')
    bilingual_text("- Read the [AI policy guide](https://docs.google.com/document/d/1SZAlExB1qAc9izHt54gwunNpjKE6wXb8Y7yA_e-baK8/edit?tab=t.0)", '- 阅读 [AI 使用政策指南](https://docs.google.com/document/d/1SZAlExB1qAc9izHt54gwunNpjKE6wXb8Y7yA_e-baK8/edit?tab=t.0)。')
    bilingual_text("- Read the [cluster guide](https://docs.google.com/document/d/1cHE0iKVyXLJ3XpIs2XuXTmZ-HMmPk2hIPeCvy-AydMg/edit?tab=t.otis27tacaef)", '- 阅读 [集群使用指南](https://docs.google.com/document/d/1cHE0iKVyXLJ3XpIs2XuXTmZ-HMmPk2hIPeCvy-AydMg/edit?tab=t.otis27tacaef)。')
    
    bilingual_text("Marin 1e23 FLOPs run finished and [matched forecasts](https://x.com/WilliamBarrHeld/status/2039373983632814318)!", 'Marin 的 1e23 FLOPs 训练运行已经完成，并且[符合预测](https://x.com/WilliamBarrHeld/status/2039373983632814318)！')
    image("https://pbs.twimg.com/media/HE1P1HmaUAAjLXF?format=jpg&name=medium", width=800)

    bilingual_text("Last lecture: overview, tokenization", '上节课：概览、分词。')
    bilingual_text("Today: resource accounting (systems)", '今天：资源核算（系统）。')

    bilingual_text("Recall: what's the best model one can train given fixed resources (compute, memory)?", '回顾：在固定资源（计算量、内存）下，可以训练出的最佳模型是什么？')
    bilingual_text("In other words: maximize (computational) **efficiency**.", '换句话说：最大化（计算）**效率**。')
    bilingual_text("Prerequisite: understand the resources (compute, memory) for a given computation.", '前提：理解一次给定计算所需的资源（计算量、内存）。')

    motivating_questions()

    bilingual_text("What knowledge to take away from this lecture:", '这节课希望你带走的知识：')
    bilingual_text("- Mechanics: straightforward (PyTorch semantics)", '- 机制：相对直接（PyTorch 语义）。')
    bilingual_text("- Mindset: resource accounting (remember to do it)", '- 思维方式：资源核算（记得要做）。')
    bilingual_text("- Intuitions: get a sense of how resources are spent, no ML magic today", '- 直觉：理解资源花在了哪里，今天没有机器学习魔法。')

    # Memory accounting
    tensors_basics()
    tensors_memory()
    tensors_on_gpus()

    # Compute accounting
    tensor_einops()
    tensor_operations_flops()

    arithmetic_intensity()

    # Memory and compute accounting for training
    deep_network()
    gradients_basics()
    gradients_flops()
    optimizer()
    train_loop()
    
    # More memory optimizations
    gradient_accumulation()
    activation_checkpointing()

    bilingual_text("Summary:", '总结：')
    bilingual_text("- Everything is operations on tensors (parameters, gradients, activations, optimizer states, data)", '- 一切都是对张量的操作（参数、梯度、激活值、优化器状态、数据）。')
    bilingual_text("- einops: better way to think about tensor operations", '- einops：理解张量操作的更好方式。')
    bilingual_text("- 6 (# data points) (# parameters) FLOPs per training step", '- 每个训练步骤约为 6 ×（数据点数）×（参数数）FLOPs。')
    bilingual_text("- Arithmetic intensity / roofline analysis: compute-bound or memory-bound?", '- 算术强度 / 屋顶线分析：受计算限制还是受内存带宽限制？')
    bilingual_text("- Matrix multiplications are compute-bound, elementwise operations are memory-bound", '- 矩阵乘法通常受计算限制，逐元素操作通常受内存带宽限制。')
    bilingual_text("- Gradient accumulation, activation checkpointing: reduce memory to use bigger batch sizes", '- 梯度累积、激活检查点：减少内存占用，以使用更大的批大小。')


def motivating_questions():
    bilingual_text("**Question**: How long would it take to train a 70B parameter model on 15T tokens on 1024 H100s?", '**问题**：在 1024 块 H100 上，用 15T token 训练一个 70B 参数模型需要多久？')
    total_flops = 6 * 70e9 * 15e12  # @inspect total_flops
    h100_flop_per_sec = 1979e12 / 2
    mfu = 0.5
    flops_per_day = h100_flop_per_sec * mfu * 1024 * 60 * 60 * 24  # @inspect flops_per_day
    days = total_flops / flops_per_day  # @inspect days

    bilingual_text("**Question**: What's the largest model that can you can train on 8 H100s using AdamW?", '**问题**：使用 AdamW，在 8 块 H100 上最多能训练多大的模型？')
    h100_bytes = 80e9  # @inspect h100_bytes
    bytes_per_parameter = 2 + 2 + (4 + 4)  # parameters (2), gradients (2), optimizer state (4 + 4) @inspect bytes_per_parameter
    num_parameters = (h100_bytes * 8) / bytes_per_parameter  # @inspect num_parameters
    bilingual_text("Caveat: activations are not accounted for (depends on batch size and sequence length), so this is an upper bound.", '注意：这里没有计入激活值（它取决于批大小和序列长度），因此这是一个上界。')

    bilingual_text("This is a rough back-of-the-envelope calculation.", '这是一个粗略的信封背面估算。')
    bilingual_text("But it gives you the flavor of napkin math one can quickly do to get a sense of resources.", '但它展示了如何快速做纸巾估算，来获得对资源需求的直觉。')


def tensors_basics():
    bilingual_text("Tensors are the basic building block for storing everything:", '张量是存储一切内容的基本构件：')
    bilingual_text("- data", '- 数据。')
    bilingual_text("- parameters", '- 参数。')
    bilingual_text("- gradients", '- 梯度。')
    bilingual_text("- optimizer state", '- 优化器状态。')
    bilingual_text("- activations", '- 激活值。')

    bilingual_text("Example: parameters of the DeepSeek v3.2 model ", '示例：DeepSeek v3.2 模型的参数。'), link(deepseek_v3_2_2025)
    link(title="DeepSeek v3.2 model on Hugging Face", url="https://huggingface.co/deepseek-ai/DeepSeek-V3.2?show_file_info=model.safetensors.index.json")

    bilingual_text("Each tensor has a rank, which is the number of dimensions.", '每个张量都有一个秩，也就是维度数量。')
    x = torch.zeros(4)        # rank 1 tensor (vector) @inspect x
    x = torch.zeros(4, 8)     # rank 2 tensor (matrix) @inspect x
    x = torch.zeros(4, 8, 2)  # rank 3 tensor @inspect x

    bilingual_text("In Transformers, will see tensors of rank 4:", '在 Transformer 中，我们会看到 4 阶张量：')
    B = 32   # Batch size
    S = 16   # Sequence length
    H = 16   # Number of heads
    D = 64   # Hidden dimension per head
    x = torch.zeros(B, S, H, D)


def tensors_memory():
    bilingual_text("Elements of tensors are generally floating point numbers.", '张量中的元素通常是浮点数。')

    bilingual_text("## fp32", '## fp32（单精度浮点）')
    link(title="Wikipedia", url="https://en.wikipedia.org/wiki/Single-precision_floating-point_format")
    image("images/fp32.png", width=700)
    bilingual_text("The fp32 data type (also known as float32 or single precision) is the default.", 'fp32 数据类型（也称为 float32 或单精度）是默认类型。')
    bilingual_text("Traditionally, in scientific computing, fp32 is the baseline; you could use double precision (fp64) in some cases.", '传统上，在科学计算中 fp32 是基线；某些情况下也可以使用双精度（fp64）。')
    bilingual_text("In deep learning, you can be a lot sloppier.", '在深度学习中，数值精度通常可以更“粗糙”一些。')

    bilingual_text("Let's examine memory usage of these tensors.", '让我们检查这些张量的内存使用。')
    bilingual_text("Memory is determined by the (i) number of values and (ii) data type of each value.", '内存由两件事决定：（i）数值个数；（ii）每个数值的数据类型。')
    x = torch.zeros(4, 8)  # @inspect x
    assert x.dtype == torch.float32  # Default type
    assert x.numel() == 4 * 8
    assert x.element_size() == 4  # Float is 4 bytes
    assert get_memory_usage(x) == 4 * 8 * 4  # 128 bytes

    bilingual_text("One matrix in the feedforward layer of GPT-3:", 'GPT-3 前馈层中的一个矩阵：')
    assert get_memory_usage(torch.empty(12288 * 4, 12288)) == 2304 * 1024 * 1024  # 2.3 GB @stepover

    bilingual_text("## fp16", '## fp16（半精度浮点）')
    link(title="Wikipedia", url="https://en.wikipedia.org/wiki/Half-precision_floating-point_format")
    image("images/fp16.png", width=400)
    bilingual_text("The fp16 data type (also known as float16 or half precision) cuts down the memory.", 'fp16 数据类型（也称为 float16 或半精度）可以减少内存。')
    x = torch.zeros(4, 8, dtype=torch.float16)  # @inspect x
    assert x.element_size() == 2
    bilingual_text("However, the dynamic range (especially for small numbers) isn't great.", '不过，它的动态范围（尤其对小数）并不好。')
    x = torch.tensor([1e-8], dtype=torch.float16)  # @inspect x
    assert x == 0  # Underflow!
    bilingual_text("If this happens when you train, you can get instability.", '如果训练时发生这种情况，可能会导致不稳定。')

    bilingual_text("## bf16", '## bf16（脑浮点，brain floating point）')
    link(title="Wikipedia", url="https://en.wikipedia.org/wiki/Bfloat16_floating-point_format")
    image("images/bf16.png", width=400)
    bilingual_text("Google Brain developed brain floating point (bf16) in 2018 to address this issue.", 'Google Brain 在 2018 年开发了 brain floating point（bf16）来解决这个问题。')
    bilingual_text("bf16 uses the same memory as fp16 but has the same dynamic range as fp32!", 'bf16 的内存占用与 fp16 相同，但动态范围与 fp32 相同！')
    bilingual_text("The only catch is that the resolution is worse, but this matters less for deep learning.", '唯一的代价是精度分辨率更低，但这对深度学习影响较小。')
    x = torch.tensor([1e-8], dtype=torch.bfloat16)  # @inspect x
    assert x != 0  # No underflow!

    bilingual_text("## Mixed precision", '## 混合精度')
    bilingual_text("Implications on training:", '对训练的影响：')
    bilingual_text("- Training with fp32 works, but requires lots of memory.", '- 使用 fp32 训练可行，但需要大量内存。')
    bilingual_text("- Training with fp16 and even bf16 is risky, and you can get instability.", '- 使用 fp16 甚至 bf16 训练有风险，可能出现不稳定。')

    bilingual_text("Solution: mixed precision training ", '解决方案：混合精度训练。'), link("https://arxiv.org/pdf/1710.03740.pdf")
    bilingual_text("- Use bf16 for parameters, activations, and gradients", '- 对参数、激活值和梯度使用 bf16。')
    bilingual_text("- Use fp32 for optimizer states", '- 对优化器状态使用 fp32。')

    bilingual_text("Pytorch has an automatic mixed precision (AMP) library. ", 'PyTorch 提供了自动混合精度（AMP）库。'), link(title="docs", url="https://pytorch.org/docs/stable/amp.html")
    bilingual_text("Tries to cast things into bf16 when safe (matmuls, not exp).", '它会在安全时把计算转换为 bf16（例如矩阵乘法，但不包括 exp）。')
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        x = torch.zeros(4, 8)  # @inspect x

    bilingual_text("## fp8", '## fp8（8 位浮点）')
    bilingual_text("In 2022, fp8 was standardized, motivated by machine learning workloads [primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html).", '2022 年，受机器学习工作负载推动，fp8 被标准化；可参考这篇[入门资料](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)。')
    image("https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/_images/fp8_formats.png", width=600)
    bilingual_text("H100s support two variants of FP8: E4M3 (range [-448, 448]) and E5M2 ([-57344, 57344]).", 'H100 支持两种 FP8 变体：E4M3（范围 [-448, 448]）和 E5M2（范围 [-57344, 57344]）。')
    bilingual_text("Reference: ", '参考资料：'), link("https://arxiv.org/pdf/2209.05433.pdf")

    bilingual_text("## fp4", '## fp4（4 位浮点）')
    bilingual_text("In 2025, NVIDIA developed [nvfp4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)", '2025 年，NVIDIA 开发了 [nvfp4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)。')
    bilingual_text("Only 4 bits per value!", '每个数值只用 4 bit！')
    bilingual_text("Values: -6, -4, -3, -2, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2, 3, 4, 6", '可表示的值：-6, -4, -3, -2, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2, 3, 4, 6。')
    bilingual_text("Use a separate scale factor per block, so actually get more dynamic range (but just can't vary freely from neighbors).", '每个块使用单独的缩放因子，因此实际获得更大的动态范围（但相邻值之间不能完全自由变化）。')
    bilingual_text("Nemotron 3 Super was trained in NVFP4 ", 'Nemotron 3 Super 使用 NVFP4 训练。'), link(nemotron_3_super_2026)

    bilingual_text("Some of this is done in NVIDIA libraries outside of user control.", '其中一些由 NVIDIA 库在用户控制之外完成。')


def tensors_on_gpus():
    bilingual_text("By default, tensors are stored in CPU memory.", '默认情况下，张量存储在 CPU 内存中。')
    x = torch.zeros(32, 32)
    assert x.device == torch.device("cpu")

    bilingual_text("However, what about GPUs?", '那么 GPU 呢？')
    image("images/cpu-gpu.png", width=600)
    device = cuda_if_available()  # @inspect device

    bilingual_text("In order to take advantage of the massive parallelism of GPUs, we need to move them to GPU memory.", '为了利用 GPU 的大规模并行能力，我们需要把张量移到 GPU 内存中。')
    x = x.to(device)

    bilingual_text("Or create the tensor directly on the GPU:", '也可以直接在 GPU 上创建张量：')
    with torch.device(device):
        x = torch.zeros(32, 32)
        assert x.device == device


def tensor_einops():
    einops_motivation()

    bilingual_text("Einops is a library for manipulating tensors where dimensions are named.", 'Einops 是一个操作张量的库，它给维度命名。')
    bilingual_text("It is inspired by Einstein summation notation (Einstein, 1916).", '它受到爱因斯坦求和记号（Einstein, 1916）的启发。')
    link(title="Einops tutorial", url="https://einops.rocks/1-einops-basics/")

    einops_einsum()
    einops_reduce()
    einops_rearrange()
    

def einops_motivation():
    bilingual_text("Traditional PyTorch code:", '传统 PyTorch 代码：')
    x = torch.ones(2, 2, 3)      # batch seq hidden  @inspect x
    y = torch.ones(2, 2, 3)      # batch seq hidden  @inspect y
    z = x @ y.transpose(-2, -1)  # batch seq seq  @inspect z
    bilingual_text("Easy to mess up the dimensions (what is -2, -1?)...", '维度很容易写错（-2、-1 到底是什么？）……')


def einops_einsum():
    bilingual_text("Einsum is generalized matrix multiplication with good bookkeeping.", 'Einsum 是带有良好记账方式的广义矩阵乘法。')

    x = torch.ones(3, 4)  # seq1 hidden @inspect x
    y = torch.ones(4, 3)  # hidden seq2 @inspect y

    # Old way
    z = x @ y   # seq1 seq2 @inspect z

    # New (einops) way
    z = einsum(x, y, "seq1 hidden, hidden seq2 -> seq1 seq2")  # @inspect z

    bilingual_text("Let's try a more complex example...", '让我们试一个更复杂的例子……')  # @clear x y z

    x = torch.ones(2, 3, 4)  # batch seq1 hidden @inspect x
    y = torch.ones(2, 3, 4)  # batch seq2 hidden @inspect y

    # Old way
    z = x @ y.transpose(-2, -1)  # batch seq1 seq2  @inspect z

    # New (einops) way
    z = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")  # @inspect z
    bilingual_text("Dimensions that are not named in the output are summed over.", '没有出现在输出中的维度会被求和消去。')

    # Or can use `...` to represent broadcasting over any number of dimensions
    z = einsum(x, y, "... seq1 hidden, ... seq2 hidden -> ... seq1 seq2")  # @inspect z


def einops_reduce():
    bilingual_text("You can reduce a single tensor via some operation (e.g., sum, mean, max, min).", '你可以用某种操作规约单个张量（例如 sum、mean、max、min）。')
    x = torch.ones(2, 3, 4)  # batch seq hidden @inspect x

    # Old way
    y = x.sum(dim=-1)  # @inspect y

    # New (einops) way
    y = reduce(x, "... hidden -> ...", "sum")  # @inspect y


def einops_rearrange():
    bilingual_text("Sometimes, a dimension represents two dimensions", '有时，一个维度其实代表两个维度。')
    bilingual_text("...and you want to operate on one of them.", '……而你想只对其中一个维度进行操作。')

    x = torch.ones(3, 8)  # seq total_hidden @inspect x
    bilingual_text("...where `total_hidden` is a flattened representation of `heads * hidden1`", '……其中 `total_hidden` 是 `heads * hidden1` 展平后的表示。')
    w = torch.ones(4, 4)  # hidden1 hidden2 @inspect w

    # Break up `total_hidden` into two dimensions (`heads` and `hidden1`
    x = rearrange(x, "... (heads hidden1) -> ... heads hidden1", heads=2)  # @inspect x

    # Perform the transformation by `w`
    x = einsum(x, w, "... hidden1, hidden1 hidden2 -> ... hidden2")  # @inspect x

    # Combine `heads` and `hidden2` back together
    x = rearrange(x, "... heads hidden2 -> ... (heads hidden2)")  # @inspect x


def tensor_operations_flops():
    bilingual_text("Having gone through all the operations, let us examine their computational cost.", '看过这些操作之后，我们来检查它们的计算成本。')

    bilingual_text("A floating-point operation (FLOP) is a basic operation like addition (x + y) or multiplication (x y).", '浮点运算（FLOP）是加法（x + y）或乘法（x y）这样的基本操作。')

    bilingual_text("Two terribly confusing acronyms (pronounced the same!):", '两个非常容易混淆、读音相同的缩写：')
    bilingual_text("- FLOPs: floating-point operations (measure of computation done)", '- FLOPs：浮点运算次数（衡量完成了多少计算）。')
    bilingual_text("- FLOP/s: floating-point operations per second (also written as FLOPS), which is used to measure the speed of hardware.", '- FLOP/s：每秒浮点运算次数（也写作 FLOPS），用于衡量硬件速度。')

    bilingual_text("## Intuitions", '## 直觉')
    bilingual_text("Training GPT-3 (2020) took 3.14e23 FLOPs. ", '训练 GPT-3（2020）用了 3.14e23 FLOPs。'), article_link("https://lambdalabs.com/blog/demystifying-gpt-3")
    bilingual_text("Training GPT-4 (2023) is speculated to take 2e25 FLOPs. ", '据推测训练 GPT-4（2023）用了 2e25 FLOPs。'), article_link("https://patmcguinness.substack.com/p/gpt-4-details-revealed")

    bilingual_text("H100 has a peak performance of 1979 teraFLOP/s with sparsity, 50% without ", 'H100 在利用稀疏性时峰值性能为 1979 teraFLOP/s，不利用稀疏性时约为其 50%。'), link(title="spec", url="https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet")
    h100_flop_per_sec = 1979e12 / 2

    bilingual_text("8 H100s for 2 weeks:", '8 块 H100 运行 2 周：')
    total_flops = 8 * 2 * (60 * 60 * 24 * 7) * h100_flop_per_sec  # @inspect total_flops

    bilingual_text("## Linear model", '## 线性模型')
    if torch.cuda.is_available():
        B = 16384  # Number of points
        D = 32768  # Dimension of each point
        K = 8192   # Number of outputs
    else:
        B = 1024
        D = 256
        K = 64

    x = torch.ones(B, D, device=cuda_if_available())
    w = torch.randn(D, K, device=cuda_if_available())
    y = x @ w

    bilingual_text("How many FLOPs is this matmul?", '这次矩阵乘法需要多少 FLOPs？')
    bilingual_text("We have one multiplication (x[i][j] * w[j][k]) and one addition per (i, j, k) triple.", '对每个 (i, j, k) 三元组，都有一次乘法（x[i][j] * w[j][k]）和一次加法。')
    actual_num_flops = 2 * B * D * K  # @inspect actual_num_flops

    bilingual_text("We can also time this operation to see how long it takes.", '我们也可以给这个操作计时，看看它需要多久。')
    actual_time = benchmark(lambda: x @ w)  # @inspect actual_time

    bilingual_text("The actual FLOP/s of this operation:", '这次操作实际达到的 FLOP/s：') 
    actual_flop_per_sec = actual_num_flops / actual_time  # @inspect actual_flop_per_sec

    bilingual_text("Each GPU has a specification sheet that provides the peak performance.", '每个 GPU 都有规格表，给出其峰值性能。')
    bilingual_text("- Example: ", '- 示例：'), link(title="H100 spec", url="https://resources.nvidia.com/en-us-gpu-resources/h100-datasheet-24306")
    bilingual_text("Note that the FLOP/s depends heavily on the data type!", '注意，FLOP/s 很大程度取决于数据类型！')
    promised_flop_per_sec = get_promised_flop_per_sec(x.dtype)  # @inspect promised_flop_per_sec

    bilingual_text("## Model FLOPs utilization (MFU)", '## 模型 FLOPs 利用率（MFU）')

    bilingual_text("Definition: MFU = (actual FLOP/s) / (promised FLOP/s) [ignore communication/overhead]", '定义：MFU =（实际 FLOP/s）/（标称 FLOP/s）[忽略通信和其他开销]。')
    mfu = actual_flop_per_sec / promised_flop_per_sec if promised_flop_per_sec else None  # @inspect mfu

    bilingual_text("Usually, MFU of ≥ 0.5 is quite good!", '通常，MFU ≥ 0.5 已经相当不错！')
    bilingual_text("But why is MFU not closer to 1?", '但为什么 MFU 不更接近 1 呢？')
    bilingual_text("To answer this question, we need to look more closely at how computations are done on GPUs...", '要回答这个问题，我们需要更仔细地看 GPU 上的计算是如何完成的……')


def arithmetic_intensity():
    image("images/compute-memory.png", width=300)
    bilingual_text("How to compute a thing:", '如何完成一次计算：')
    bilingual_text("1. Send inputs from memory to accelerator", '1. 把输入从内存发送到加速器。')
    bilingual_text("2. Perform computation", '2. 执行计算。')
    bilingual_text("3. Send outputs from accelerator to memory", '3. 把输出从加速器发送回内存。')

    bilingual_text("How long does this take?", '这需要多长时间？')

    bilingual_text("Depends on two things:", '取决于两件事：')
    bilingual_text("1. Accelerator speed (FLOP/s)", '1. 加速器速度（FLOP/s）。')
    bilingual_text("2. Memory bandwidth (bytes/s)", '2. 内存带宽（bytes/s）。')
    assert h100_flop_per_sec == 1979e12 / 2  # Half without sparsity
    assert h100_bytes_per_sec == 3.35e12

    arithmetic_intensity_relu()
    arithmetic_intensity_gelu()
    arithmetic_intensity_dot_product()
    arithmetic_intensity_matrix_vector_product()
    arithmetic_intensity_matmul()

    # Let's visualize it
    roofline_plots()


def arithmetic_intensity_relu():
    n = 1024 * 1024
    x = torch.ones(n, dtype=torch.bfloat16, device=cuda_if_available())
    y = torch.relu(x)

    bytes = (2 * n) + (2 * n)  # Read x, write y (bf16 is 2 bytes/float)
    flops = n  # n comparisons

    communication_time = bytes / h100_bytes_per_sec  # @inspect communication_time
    computation_time = flops / h100_flop_per_sec  # @inspect computation_time

    bilingual_text("Assume we can overlap communication and computation perfectly.", '假设通信和计算可以完全重叠。')
    total_time = max(communication_time, computation_time)  # @inspect total_time

    bilingual_text("What is the bottleneck?", '瓶颈是什么？')
    bilingual_text("- Memory-bound: communication time > computation time", '- 受内存带宽限制：通信时间 > 计算时间。')
    bilingual_text("- Compute-bound: computation time > communication time", '- 受计算限制：计算时间 > 通信时间。')

    bilingual_text("In this case, ReLU is memory-bound.", '在这个例子中，ReLU 受内存带宽限制。')

    bilingual_text("Alternative way to see this:", '另一种看法：')
    bilingual_text("Accelerator intensity: how much work can the accelerator do per byte transferred?", '加速器强度：每传输 1 字节，加速器能完成多少工作？')
    h100_accelerator_intensity = h100_flop_per_sec / h100_bytes_per_sec  # @inspect h100_accelerator_intensity

    bilingual_text("Arithmetic intensity: how much actual work per byte for this workload?", '算术强度：这个工作负载每字节实际完成多少工作？')
    arithmetic_intensity = flops / bytes  # ~1/4 @inspect arithmetic_intensity

    bilingual_text("What is the bottleneck?", '瓶颈是什么？')
    bilingual_text("- Memory-bound: arithmetic intensity < accelerator intensity", '- 受内存带宽限制：算术强度 < 加速器强度。')
    bilingual_text("- Compute-bound: arithmetic intensity > accelerator intensity", '- 受计算限制：算术强度 > 加速器强度。')

    assert arithmetic_intensity < h100_accelerator_intensity

    bilingual_text("In general, we'll find ourselves memory bound.", '一般来说，我们会发现自己受内存带宽限制。')
    bilingual_text("Can we increase arithmetic intensity?", '我们能提高算术强度吗？')


def arithmetic_intensity_gelu():
    n = 1024 * 1024
    x = torch.ones(n, dtype=torch.bfloat16, device=cuda_if_available())
    y = F.gelu(x)  # GELU(x) = 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))

    bytes = (2 * n) + (2 * n)  # Read x, write y (bf16 is 2 bytes/float)
    flops = 20 * n  # tanh can be approximated in various ways (e.g., polynomials)

    arithmetic_intensity = flops / bytes  # @inspect arithmetic_intensity

    h100_accelerator_intensity = h100_flop_per_sec / h100_bytes_per_sec  # @inspect h100_accelerator_intensity
    assert arithmetic_intensity < h100_accelerator_intensity

    bilingual_text("Note that GeLU does more work than ReLU per byte moved, so it has higher arithmetic intensity.", '注意，每移动 1 字节，GeLU 比 ReLU 做更多计算，因此算术强度更高。')
    bilingual_text("But still memory-bound!", '但它仍然受内存带宽限制！')
    bilingual_text("In other words, ReLU is not faster than GeLU (when doing things in an isolated way).", '换句话说，在孤立执行这些操作时，ReLU 并不比 GeLU 更快。')


def arithmetic_intensity_dot_product():
    n = 1024 * 1024
    x = torch.ones(n, dtype=torch.bfloat16, device=cuda_if_available())
    w = torch.ones(n, dtype=torch.bfloat16, device=cuda_if_available())
    y = x @ w

    bytes = (2 * n) + (2 * n) + 2  # Read x, read w, write y
    flops = 2 * n - 1  # n multiplications, n-1 additions

    arithmetic_intensity = flops / bytes  # ~1/2 @inspect arithmetic_intensity

    h100_accelerator_intensity = h100_flop_per_sec / h100_bytes_per_sec  # @inspect h100_accelerator_intensity
    assert arithmetic_intensity < h100_accelerator_intensity
    bilingual_text("Memory-bound!", '受内存带宽限制！')


def arithmetic_intensity_matrix_vector_product():
    n = 1024
    x = torch.ones(n, dtype=torch.bfloat16, device=cuda_if_available())
    w = torch.ones(n, n, dtype=torch.bfloat16, device=cuda_if_available())
    y = x @ w

    bytes = (2 * n) + (2 * n * n) + (2 * n)  # Read x, read w, write y
    flops = n * (2 * n - 1)  # n dot-products

    arithmetic_intensity = flops / bytes  # ~1 @inspect arithmetic_intensity

    h100_accelerator_intensity = h100_flop_per_sec / h100_bytes_per_sec  # @inspect h100_accelerator_intensity
    assert arithmetic_intensity < h100_accelerator_intensity
    bilingual_text("Memory-bound!", '受内存带宽限制！')

def arithmetic_intensity_matmul():
    n = 1024
    x = torch.ones(n, n, dtype=torch.bfloat16, device=cuda_if_available())
    w = torch.ones(n, n, dtype=torch.bfloat16, device=cuda_if_available())
    y = x @ w

    bytes = (2 * n * n) + (2 * n * n) + (2 * n * n)  # Read x, read w, write y
    flops = n * n * (2 * n - 1)  # n^2 dot products

    arithmetic_intensity = flops / bytes  # ~n/3 @inspect arithmetic_intensity

    h100_accelerator_intensity = h100_flop_per_sec / h100_bytes_per_sec  # @inspect h100_accelerator_intensity
    assert arithmetic_intensity > h100_accelerator_intensity
    bilingual_text("Finally, compute-bound!", '终于受计算限制了！')

    bilingual_text("As long as we have large matrices, we're compute-bound (saturating the accelerator).", '只要矩阵足够大，就会受计算限制（能够打满加速器）。')
    bilingual_text("Training Transformers involves big matrix multiplications.", '训练 Transformer 涉及大型矩阵乘法。')
    bilingual_text("Matrix-vector product is what happens during inference, which is why inference is memory-bound.", '推理时发生的是矩阵-向量乘法，这就是推理受内存带宽限制的原因。')

    bilingual_text("Note: arithmetic/accelerator intensity also depends on the precision (bf16 versus fp32).", '注意：算术强度/加速器强度也取决于精度（bf16 与 fp32）。')


def roofline_plots():
    bilingual_text("We can visualize the relationship between arithmetic intensity and performance using roofline plots.", '我们可以用屋顶线图可视化算术强度和性能之间的关系。')
    image("https://jax-ml.github.io/scaling-book/assets/img/roofline-improved-1400.webp", width=600)
    bilingual_text("- Each slice on the x-axis is a particular computation (with some arithmetic intensity)", '- x 轴上的每一段代表某个具体计算（具有一定算术强度）。')
    bilingual_text("- Each piecewise linear function corresponds to a particular hardware", '- 每条分段线性函数对应一种特定硬件。')
    bilingual_text("- Kink is the accelerator intensity (transition from memory-bound to compute-bound)", '- 拐点是加速器强度（从受内存带宽限制转为受计算限制）。')

    bilingual_text("We can now relate this back to MFU:", '现在可以把它和 MFU 联系起来：')
    bilingual_text("MFU = min(1, arithmetic-intensity / accelerator-intensity)", 'MFU = min(1, 算术强度 / 加速器强度)。')

    link(title="reference", url="https://jax-ml.github.io/scaling-book/roofline/")


def gradients_basics():
    bilingual_text("So far, we've constructed tensors and passed them through operations (forward).", '到目前为止，我们构造了张量，并让它们通过操作向前传播。')
    bilingual_text("Now, we're going to compute the gradient (backward).", '现在，我们要计算梯度（反向传播）。')

    bilingual_text("As a simple example, let's consider the simple linear model:", '作为一个简单例子，考虑这个简单线性模型：')
    bilingual_text("y = 0.5 (x * w - 5)^2", '公式保持不变：y = 0.5 (x * w - 5)^2。')

    bilingual_text("Forward pass: compute loss", '前向传播：计算损失。')
    x = torch.tensor([1., 2, 3])
    w = torch.tensor([1., 1, 1], requires_grad=True)  # Want gradient
    pred_y = x @ w
    loss = 0.5 * (pred_y - 5).pow(2)

    bilingual_text("Backward pass: compute gradients", '反向传播：计算梯度。')
    loss.backward()
    assert torch.equal(w.grad, torch.tensor([1, 2, 3]))  # @inspect w.grad


def gradients_flops():
    bilingual_text("Let us count the FLOPs for computing gradients.", '让我们统计计算梯度需要的 FLOPs。')

    image("images/deep-network.png", width=800)

    B = 1024  # Number of points
    D = 256   # Dimension

    bilingual_text("Define a simplified model (2-layer linear network):", '定义一个简化模型（两层线性网络）：')
    x = torch.ones(B, D, device=cuda_if_available())
    w1 = torch.randn(D, D, device=cuda_if_available(), requires_grad=True)
    w2 = torch.randn(D, D, device=cuda_if_available(), requires_grad=True)

    # Forward pass
    h1 = einsum(x, w1, "batch in, in out -> batch out")  # x @ w1
    h2 = einsum(h1, w2, "batch in, in out -> batch out")  # h1 @ w2
    loss = (h2.mean() - 0)**2  # Regress everything to 0 (arbitrary)

    # Backward pass
    h1.retain_grad()  # For debugging
    h2.retain_grad()  # For debugging
    loss.backward()

    bilingual_text("## Zoom in on one layer", '## 放大观察其中一层')
    bilingual_text("Let's focus on the second layer (h2 = h1 @ w2)", '我们关注第二层（h2 = h1 @ w2）。')

    bilingual_text("**Forward pass**: Recall the number of forward FLOPs: ", '**前向传播**：回忆前向 FLOPs 数量：')
    num_forward_flops = 2 * B * D * D   # @inspect num_forward_flops

    bilingual_text("**Backward pass**: How many FLOPs is running the backward pass?", '**反向传播**：运行反向传播需要多少 FLOPs？')

    bilingual_text("We need to compute:", '我们需要计算：')
    bilingual_text("- h1.grad = d loss / d h1", '- h1.grad 表示损失对 h1 的梯度：d loss / d h1。')
    bilingual_text("- w2.grad = d loss / d w2", '- w2.grad 表示损失对 w2 的梯度：d loss / d w2。')

    h1_grad = einsum(h2.grad, w2, "batch out, in out -> batch in")
    assert torch.allclose(h1.grad, h1_grad)

    w2_grad = einsum(h2.grad, h1, "batch out, batch in -> in out")
    assert torch.allclose(w2.grad, w2_grad)

    num_backward_flops = (2 * B * D * D) + (2 * B * D * D)  # @inspect num_backward_flops

    bilingual_text("Note that the backward pass is 2x more expensive than the forward pass.", '注意，反向传播的计算量是前向传播的 2 倍。')

    bilingual_text("## Consider all layers", '## 考虑所有层')
    bilingual_text("This was just for w2, need to apply it to all parameters in the network.", '刚才只分析了 w2，需要把它应用到网络中的所有参数。')

    bilingual_text("Putting it together:", '合在一起：')
    bilingual_text("- Forward pass: 2 (# data points) (# parameters) FLOPs", '- 前向传播：2 ×（数据点数）×（参数数）FLOPs。')
    bilingual_text("- Backward pass: 4 (# data points) (# parameters) FLOPs", '- 反向传播：4 ×（数据点数）×（参数数）FLOPs。')
    bilingual_text("- Total: 6 (# data points) (# parameters) FLOPs", '- 总计：6 ×（数据点数）×（参数数）FLOPs。')

    bilingual_text("This is for multilayer perceptrons (MLPs)", '这是针对多层感知机（MLP）的结论。')
    bilingual_text("...but it turns out to be a good approximation for Transformers for short context lengths as well.", '……但事实证明，对短上下文长度的 Transformer 来说，这也是很好的近似。')


def deep_network():
    image("images/deep-network.png", width=800)
    bilingual_text("Consider a deep network with L layers and D-dimensional inputs, activations, and outputs.", '考虑一个深度网络，它有 L 层，输入、激活值和输出都是 D 维。')

    # Define the network
    D = 8  # Dimensionality of input, activations, and output
    L = 3  # Number of layers
    model = DeepNetwork(dim=D, num_layers=L).to(cuda_if_available())

    num_parameters = get_num_parameters(model)  # @inspect num_parameters @stepover
    assert num_parameters == (D * D) * L

    # Run the model on a batch of data
    B = 4  # Batch size
    x = torch.randn(B, D, device=cuda_if_available())  # @inspect x
    y = model(x)  # @inspect y


class Block(nn.Module):
    """Simple block that applies a linear transformation followed by a ReLU nonlinearity."""
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, dim) / math.sqrt(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x @ self.weight  # Linear
        x = F.relu(x)        # Activation
        return x


class DeepNetwork(nn.Module):
    """Map `dim`-vector to a `dim`-vector."""
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([Block(dim) for i in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply all the layers sequentially
        for layer in self.layers:
            x = layer(x)  # @stepover
        return x


def optimizer():
    bilingual_text("Recall our deep network.", '回忆我们的深度网络。')
    B = 2  # Batch size
    D = 4  # Dimensionality of input, activations, and output
    L = 3  # Number of layers
    model = DeepNetwork(dim=D, num_layers=L).to(cuda_if_available())  # @stepover

    bilingual_text("Let's define the AdaGrad optimizer", '让我们定义 AdaGrad 优化器。')
    bilingual_text("- momentum = SGD + exponential averaging of grad", '- 动量 = SGD + 梯度的指数平均。')
    bilingual_text("- AdaGrad = SGD + averaging by grad^2", '- AdaGrad = SGD + 按 grad^2 做平均。')
    bilingual_text("- RMSProp = AdaGrad but with exponential averaging of grad^2", '- RMSProp = AdaGrad，但对 grad^2 使用指数平均。')
    bilingual_text("- Adam = RMSProp + momentum", '- Adam = RMSProp + 动量。')

    bilingual_text("AdaGrad ", '说明：AdaGrad'), link(adagrad_2011)
    optimizer = AdaGrad(model.parameters(), lr=0.01)  # @stepover
    state = model.state_dict()  # @inspect state

    # Compute gradients
    x = torch.randn(B, D, device=cuda_if_available())
    y = torch.tensor([4., 5.], device=cuda_if_available())
    pred_y = model(x).mean()  # @stepover
    loss = F.mse_loss(input=pred_y, target=y)
    loss.backward()

    # Take a step
    optimizer.step()
    optimizer_state = {i: dict(p_state) for i, (p, p_state) in enumerate(optimizer.state.items())}  # @inspect optimizer_state

    # Free up the memory
    optimizer.zero_grad(set_to_none=True)

    bilingual_text("## Memory", '## 内存')

    num_parameters = D * D * L
    parameter_memory = 2 * num_parameters  # (2 bytes for bf16) @inspect parameter_memory
    gradient_memory = 2 * num_parameters  # (2 bytes for bf16) @inspect gradient_memory
    optimizer_state_memory = 4 * num_parameters  # (4 bytes for fp32) @inspect optimizer_state_memory
    activation_memory = 2 * (B * D * L)  # (2 bytes for bf16) @inspect activation_memory
    bilingual_text("It is customary to use fp32 for stability (accumulating averages over powers over many steps).", '为了稳定性，通常使用 fp32（在许多步骤上累积幂的平均值）。')
    bilingual_text("Optimizer state memory:", '优化器状态内存：')
    bilingual_text("- AdaGrad: 4 bytes/parameter for storing second moments", '- AdaGrad：每个参数 4 字节，用于存储二阶矩。')
    bilingual_text("- Adam: 8 bytes/parameter for storing first and second moments", '- Adam：每个参数 8 字节，用于存储一阶矩和二阶矩。')

    # Putting it all together
    total_memory = parameter_memory + activation_memory + gradient_memory + optimizer_state_memory  # @inspect total_memory

    bilingual_text("## Compute (for one training step)", '## 计算量（一次训练步骤）')
    num_parameters = D * D * L
    flops = 6 * B * num_parameters  # @inspect flops

    bilingual_text("## Transformers", '## Transformer（变换器）')
    bilingual_text("The accounting for a Transformer is more complicated, but the same idea.", 'Transformer 的核算更复杂，但思路相同。')
    bilingual_text("Assignment 1 will ask you to do that.", '作业 1 会要求你完成这件事。')

    bilingual_text("Blog post describing memory usage for Transformer training ", 'Blog post describing 内存 usage for Transformer 训练'), article_link("https://erees.dev/transformer-memory/")
    bilingual_text("Blog post describing FLOPs for a Transformer: ", '说明：Blog post describing FLOPs for a Transformer:'), article_link("https://www.adamcasson.com/posts/transformer-flops")


class AdaGrad(torch.optim.Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 0.01):
        super(AdaGrad, self).__init__(params, dict(lr=lr))

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                # Optimizer state
                state = self.state[p]
                grad = p.grad.data

                # Get squared gradients g2 = sum_{i<t} g_i^2
                g2 = state.get("g2", torch.zeros_like(grad))

                # Update optimizer state
                g2 += torch.square(grad)
                state["g2"] = g2

                # Update parameters
                p.data -= lr * grad / torch.sqrt(g2 + 1e-5)


def train_loop():
    # True linear function with weights (0, 1, 2, ..., D-1)
    D = 16  # Dimensionality
    true_w = torch.arange(D, dtype=torch.float32, device=cuda_if_available())

    # Data loader that generates (x, y) pairs
    B = 4  # Batch size
    def get_batch() -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(B, D).to(cuda_if_available())
        true_y = x @ true_w
        return (x, true_y)

    # Define the model and optimizer
    L = 2  # Number of layers
    model = DeepNetwork(dim=D, num_layers=L).to(cuda_if_available()) # @stepover
    optimizer = AdaGrad(model.parameters(), lr=0.01) # @stepover

    # Train!
    num_train_steps = 3
    for t in range(num_train_steps):
        # Get data
        x, y = get_batch()

        # Forward (compute loss)
        pred_y = model(x).mean()  # @stepover
        loss = F.mse_loss(pred_y, y)

        # Backward (compute gradients)
        loss.backward()

        # Update parameters
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


def gradient_accumulation():
    bilingual_text("Large batch sizes: improve training stability", '大批大小：提升训练稳定性。')
    bilingual_text("However, activation memory scales with batch size, so might run out.", '但是激活值内存会随批大小增长，因此可能耗尽。')
    B = 64     # Batch size
    D = 1024   # Dimensionality
    L = 16     # Number of layers
    activation_memory = 2 * B * D * L  # (2 bytes for bf16) @inspect activation_memory
    bilingual_text("Gradient accumulation:", '梯度累积：')
    bilingual_text("- Compute gradient on micro batches", '- 在微批次上计算梯度。')
    bilingual_text("- Accumulate the gradients (don't zero it out)", '- 累积梯度（不要清零）。')
    bilingual_text("- Every batch_size / micro_batch_size steps, update the parameters and zero out the gradients", '- 每经过 batch_size / micro_batch_size 步，更新参数并清零梯度。')
    micro_batch_size = B / 4
    activation_memory = 2 * micro_batch_size * D * L  # (2 bytes for bf16) @inspect activation_memory


def activation_checkpointing():
    bilingual_text("For training, we need to store the activations of all layers", '训练时，我们需要存储所有层的激活值。')
    bilingual_text("For inference, we don't compute gradients, so we only need to store the current layer's activations.", '推理时不计算梯度，所以只需要存储当前层的激活值。')

    image("images/deep-network.png", width=800)
    bilingual_text("The memory usage is", '内存使用量为：')
    B = 64     # Batch size
    D = 1024   # Dimensionality
    L = 16     # Number of layers

    x = torch.randn(B, D, device=cuda_if_available(), requires_grad=True)
    activation_memory = 2 * B * D * L  # @inspect activation_memory

    model = DeepNetwork(dim=D, num_layers=L).to(cuda_if_available())  # @stepover
    memory = get_max_memory_usage(lambda: model(x).sum().backward())  # @inspect memory @stepover

    bilingual_text("Can we reduce this?", '我们能减少它吗？')

    bilingual_text("Activation checkpointing = gradient checkpointing = rematerialization", '激活检查点 = 梯度检查点 = 重物化。')
    bilingual_text("Key idea:", '关键思想：')
    bilingual_text("- Forward pass: keep only activations at subset of layers", '- 前向传播：只保留一部分层的激活值。')
    bilingual_text("- Backward pass: recompute the missing activations from the last checkpoint", '- 反向传播：从最近的检查点重新计算缺失的激活值。')
    bilingual_text("Philosophy: tradeoff memory for compute", '理念：用更多计算换取更少内存。')

    # Store all activations:    x g1 h1 g2 h2 g3 h3 g4 h4
    # Activation checkpointing: x    h1    h2    h3    h4

    # Define the model with checkpointing
    model = DeepNetworkCheckpointed(dim=D, num_layers=L).to(cuda_if_available())  # @stepover
    checkpointed_memory = get_max_memory_usage(lambda: model(x).sum().backward())  # @inspect checkpointed_memory @stepover

    bilingual_text("Can we reduce this even more, especially for deep networks (large L)?", '我们还能进一步减少它吗，尤其是对深层网络（大的 L）？')

    # Store all layers:   | h1 h2 h3 h4 h5 h6 h7 h8 h9 |
    # Store no layers:    |                            |
    # Store some layers:  |    h3       h6          h9 |

    bilingual_text("How frequently to checkpoint?", '应该多频繁地设置检查点？')
    bilingual_text("- If store each layer's activations, then activation memory is O(L) and no recomputation.", '- 如果存储每层激活值，则激活内存是 O(L)，不需要重算。')
    bilingual_text("- If store no activations, then activation memory is O(1) and compute is O(L^2) (recompute from the start for each layer).", '- 如果不存储激活值，则激活内存是 O(1)，计算量是 O(L^2)（每层都从头重算）。')
    bilingual_text("- If store every sqrt(L) layers, then activation memory is O(sqrt(L)) and O(L) recomputation.", '- 如果每 sqrt(L) 层存一次，则激活内存是 O(sqrt(L))，重算量是 O(L)。')


class DeepNetworkCheckpointed(nn.Module):
    """Same as DeepNetwork, but with activation checkpointing."""
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([Block(dim) for i in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply all the layers sequentially
        for layer in self.layers:
            # KEY: only store activations at checkpoints, recompute the rest
            x = torch.utils.checkpoint.checkpoint(layer, x)  # @stepover
        return x

############################################################

def get_memory_usage(x: torch.Tensor):
    return x.numel() * x.element_size()


def get_promised_flop_per_sec(dtype: torch.dtype) -> float:
    """Return the peak FLOP/s for `device` operating on `dtype`."""
    if not torch.cuda.is_available():
        # No CUDA device available, so can't get FLOP/s
        return 1
    properties = torch.cuda.get_device_properties(cuda_if_available())  # @inspect properties.name

    if "A100" in properties.name:
        # https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
        if dtype == torch.float32:
            return 19.5e12
        if dtype in (torch.bfloat16, torch.float16):
            return 312e12
        raise ValueError(f"Unknown dtype: {dtype}")

    if "H100" in properties.name:
        # https://www.nvidia.com/en-us/data-center/h100/
        if dtype == torch.float32:
            return 67.5e12
        if dtype in (torch.bfloat16, torch.float16):
            return 1979e12 / 2  # 1979 is for sparse, dense is half of that
        raise ValueError(f"Unknown dtype: {dtype}")

    if "B200" in properties.name:
        # https://www.primeline-solutions.com/media/categories/server/nach-gpu/nvidia-hgx-h200/nvidia-blackwell-b200-datasheet.pdf
        if dtype == torch.float32:
            return 75e12
        if dtype in (torch.bfloat16, torch.float16):
            return 4.5e15 / 2  # 4.5e15 is for sparse, dense is half of that
        raise ValueError(f"Unknown dtype: {dtype}")

    # Unknown GPU: return None so caller can handle gracefully
    return None


def benchmark(func, num_trials: int = 5) -> float:
    """Return the number of seconds required to perform `func`."""

    # Wait until previous CUDA threads are done
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    def run():
        # Perform the operation
        func()

        # Wait until CUDA threads are done
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Time the operation `num_trials` times
    total_time = timeit.timeit(run, number=num_trials)

    return total_time / num_trials


def get_num_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


if __name__ == "__main__":
    main()
