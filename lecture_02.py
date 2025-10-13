from execute_util import text, link, image
from facts import a100_flop_per_sec, h100_flop_per_sec
import torch.nn.functional as F
import timeit
import torch
from typing import Iterable
from torch import nn
import numpy as np
from lecture_util import article_link
from jaxtyping import Float
from einops import rearrange, einsum, reduce
from references import zero_2019


def main():
    text("上节课：概览，分词")

    text("本节课概览:")
    text("- 我们将讨论训练模型所需的所有**基本元素**。")
    text("- 我们将从张量到模型再到优化器再到训练循环自下而上地讲解。")
    text("- 我们将密切关注效率（**资源**的使用）。")

    text("特别是，我们将考虑两种类型的资源:")
    text("- 内存 (GB)")
    text("- 计算 (FLOPs)")

    motivating_questions()

    text("我们不会讲解 Transformer。")
    text("有很好的讲解资料:")
    link(title="Assignment 1 handout", url="https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_spring2025_assignment1_basics.pdf")
    link(title="数学描述", url="https://johnthickstun.com/docs/transformers.pdf")
    link(title="图解 Transformer", url="http://jalammar.github.io/illustrated-transformer/")
    link(title="图解 GPT-2", url="https://jalammar.github.io/illustrated-gpt2/")
    text("相反，我们将使用更简单的模型。")

    text("要掌握的知识:")
    text("- 机制：简单直接（就是 PyTorch）")
    text("- 思维方式：资源核算（记住要做）")
    text("- 直觉：大体了解（不涉及大模型）")

    text("## 内存核算")
    tensors_basics()
    tensors_memory()

    text("## 计算核算")
    tensors_on_gpus()
    tensor_operations()
    tensor_einops()
    tensor_operations_flops()
    gradients_basics()
    gradients_flops()

    text("## 模型")
    module_parameters()
    custom_model()

    text("训练循环和最佳实践")
    note_about_randomness()
    data_loading()

    optimizer()
    train_loop()
    checkpointing()
    mixed_precision_training()


def motivating_questions():
    text("让我们做一些估算。")

    text("**问题**: 在1024个H100上用15T个标记训练一个70B参数的模型需要多长时间？")
    total_flops = 6 * 70e9 * 15e12  # @inspect total_flops
    assert h100_flop_per_sec == 1979e12 / 2
    mfu = 0.5
    flops_per_day = h100_flop_per_sec * mfu * 1024 * 60 * 60 * 24  # @inspect flops_per_day
    days = total_flops / flops_per_day  # @inspect days

    text("**问题**: 在8个H100上使用AdamW（天真地）可以训练的最大模型有多大？")
    h100_bytes = 80e9  # @inspect h100_bytes
    bytes_per_parameter = 4 + 4 + (4 + 4)  # 参数、梯度、优化器状态  @inspect bytes_per_parameter
    num_parameters = (h100_bytes * 8) / bytes_per_parameter  # @inspect num_parameters
    text("注意1：我们天真地对参数和梯度使用float32。我们也可以对参数和梯度使用bf16（2 + 2）并保留一个额外的参数float32副本（4）。这不会节省内存，但是更快。"), link(zero_2019)
    text("注意2：没有计算激活值（取决于批处理大小和序列长度）。")

    text("这是一个粗略的估算。")


def tensors_basics():
    text("张量是存储所有内容的基本构建块：参数、梯度、优化器状态、数据、激活值。")
    link(title="[PyTorch 张量文档]", url="https://pytorch.org/docs/stable/tensors.html")

    text("您可以通过多种方式创建张量:")
    x = torch.tensor([[1., 2, 3], [4, 5, 6]])  # @inspect x
    x = torch.zeros(4, 8)  # 全零的4x8矩阵 @inspect x
    x = torch.ones(4, 8)  # 全一的4x8矩阵 @inspect x
    x = torch.randn(4, 8)  # 4x8矩阵，包含独立同分布的正态分布(0, 1)样本 @inspect x

    text("分配但不初始化值:")
    x = torch.empty(4, 8)  # 4x8矩阵，包含未初始化的值 @inspect x
    text("...因为您想稍后使用一些自定义逻辑来设置值")
    nn.init.trunc_normal_(x, mean=0, std=1, a=-2, b=2)  # @inspect x


def tensors_memory():
    text("几乎所有内容（参数、梯度、激活值、优化器状态）都以浮点数形式存储。")

    text("## float32")
    link(title="[Wikipedia]", url="https://en.wikipedia.org/wiki/Single-precision_floating-point_format")
    image("images/fp32.png", width=600)
    text("float32数据类型（也称为fp32或单精度）是默认值。")
    text("传统上，在科学计算中，float32是基准；在某些情况下您可以使用双精度（float64）。")
    text("在深度学习中，您可以更随意一些。")

    text("让我们检查这些张量的内存使用情况。")
    text("内存由（i）值的数量和（ii）每个值的数据类型确定。")
    x = torch.zeros(4, 8)  # @inspect x
    assert x.dtype == torch.float32  # 默认类型
    assert x.numel() == 4 * 8
    assert x.element_size() == 4  # Float 是 4 字节
    assert get_memory_usage(x) == 4 * 8 * 4  # 128 字节

    text("GPT-3前馈层中的一个矩阵:")
    assert get_memory_usage(torch.empty(12288 * 4, 12288)) == 2304 * 1024 * 1024  # 2.3 GB
    text("...这是很多的内存！")

    text("## float16")
    link(title="[Wikipedia]", url="https://en.wikipedia.org/wiki/Half-precision_floating-point_format")
    image("images/fp16.png", width=400)
    text("float16数据类型（也称为fp16或半精度）减少了内存。")
    x = torch.zeros(4, 8, dtype=torch.float16)  # @inspect x
    assert x.element_size() == 2
    text("然而，动态范围（特别是对于小数）不是很好。")
    x = torch.tensor([1e-8], dtype=torch.float16)  # @inspect x
    assert x == 0  # 下溢！
    text("如果在训练时发生这种情况，可能会导致不稳定。")

    text("## bfloat16")
    link(title="[Wikipedia]", url="https://en.wikipedia.org/wiki/Bfloat16_floating-point_format")
    image("images/bf16.png", width=400)
    text("Google Brain 在2018年开发了 bfloat（brain floating point）来解决这个问题。")
    text("bfloat16 使用与 float16 相同的内存，但具有与 float32 相同的动态范围！")
    text("唯一的问题是分辨率较差，但这在深度学习中不太重要。")
    x = torch.tensor([1e-8], dtype=torch.bfloat16)  # @inspect x
    assert x != 0  # 没有下溢！

    text("让我们比较不同数据类型的动态范围和内存使用情况:")
    float32_info = torch.finfo(torch.float32)  # @inspect float32_info
    float16_info = torch.finfo(torch.float16)  # @inspect float16_info
    bfloat16_info = torch.finfo(torch.bfloat16)  # @inspect bfloat16_info

    text("## fp8")
    text("2022年，FP8被标准化，由机器学习工作负载推动。")
    link("https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html")
    image("https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/_images/fp8_formats.png", width=400)
    text("H100支持两种FP8变体：E4M3（范围[-448, 448]）和E5M2（[-57344, 57344]）。")
    text("参考: "), link("https://arxiv.org/pdf/2209.05433.pdf")

    text("对训练的影响:")
    text("- 使用float32训练有效，但需要大量内存。")
    text("- 使用fp8、float16甚至bfloat16训练存在风险，可能会出现不稳定。")
    text("- 解决方案（稍后）：使用混合精度训练，参见 "), link(mixed_precision_training)


def tensors_on_gpus():
    text("默认情况下，张量存储在CPU内存中。")
    x = torch.zeros(32, 32)
    assert x.device == torch.device("cpu")

    text("但是，为了利用GPU的巨大并行性，我们需要将它们移到GPU内存中。")
    image("images/cpu-gpu.png", width=400)

    text("首先让我们看看是否有任何GPU。")
    if not torch.cuda.is_available():
        return

    num_gpus = torch.cuda.device_count()  # @inspect num_gpus
    for i in range(num_gpus):
        properties = torch.cuda.get_device_properties(i)  # @inspect properties

    memory_allocated = torch.cuda.memory_allocated()  # @inspect memory_allocated

    text("将张量移动到GPU内存（设备0）。")
    y = x.to("cuda:0")
    assert y.device == torch.device("cuda", 0)

    text("或者直接在GPU上创建张量:")
    z = torch.zeros(32, 32, device="cuda:0")

    new_memory_allocated = torch.cuda.memory_allocated()  # @inspect new_memory_allocated
    memory_used = new_memory_allocated - memory_allocated  # @inspect memory_used
    assert memory_used == 2 * (32 * 32 * 4)  # 2个4字节浮点数的32x32矩阵



def tensor_operations():
    text("大多数张量是通过对其他张量执行操作而创建的。")
    text("每个操作都有一些内存和计算后果。")

    tensor_storage()
    tensor_slicing()
    tensor_elementwise()
    tensor_matmul()


def tensor_storage():
    text("PyTorch中的张量是什么?")
    text("PyTorch张量是指向已分配内存的指针")
    text("...带有描述如何获取张量的任何元素的元数据。")
    image("https://martinlwx.github.io/img/2D_tensor_strides.png", width=400)
    link(title="[PyTorch 文档]", url="https://pytorch.org/docs/stable/generated/torch.Tensor.stride.html")
    x = torch.tensor([
        [0., 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
    ])

    text("要跳到下一行（维度0），在存储中跳过4个元素。")
    assert x.stride(0) == 4

    text("要跳到下一列（维度1），在存储中跳过1个元素。")
    assert x.stride(1) == 1

    text("要找到一个元素:")
    r, c = 1, 2
    index = r * x.stride(0) + c * x.stride(1)  # @inspect index
    assert index == 6


def tensor_slicing():
    x = torch.tensor([[1., 2, 3], [4, 5, 6]])  # @inspect x

    text("许多操作只是提供张量的不同**视图**。")
    text("这不会创建副本，因此一个张量的改变会影响另一个。")

    text("获取第0行:")
    y = x[0]  # @inspect y
    assert torch.equal(y, torch.tensor([1., 2, 3]))
    assert same_storage(x, y)

    text("获取第1列:")
    y = x[:, 1]  # @inspect y
    assert torch.equal(y, torch.tensor([2, 5]))
    assert same_storage(x, y)

    text("将2x3矩阵视为3x2矩阵:")
    y = x.view(3, 2)  # @inspect y
    assert torch.equal(y, torch.tensor([[1, 2], [3, 4], [5, 6]]))
    assert same_storage(x, y)

    text("转置矩阵:")
    y = x.transpose(1, 0)  # @inspect y
    assert torch.equal(y, torch.tensor([[1, 4], [2, 5], [3, 6]]))
    assert same_storage(x, y)

    text("检查改变x也会改变y。")
    x[0][0] = 100  # @inspect x, @inspect y
    assert y[0][0] == 100

    text("注意，某些视图是非连续的条目，这意味着无法进行进一步的视图操作。")
    x = torch.tensor([[1., 2, 3], [4, 5, 6]])  # @inspect x
    y = x.transpose(1, 0)  # @inspect y
    assert not y.is_contiguous()
    try:
        y.view(2, 3)
        assert False
    except RuntimeError as e:
        assert "view size is not compatible with input tensor's size and stride" in str(e)

    text("可以先强制张量连续:")
    y = x.transpose(1, 0).contiguous().view(2, 3)  # @inspect y
    assert not same_storage(x, y)
    text("视图是免费的，复制既占用额外的内存又占用计算资源。")


def tensor_elementwise():
    text("这些操作将某些操作应用于张量的每个元素")
    text("...并返回相同形状的（新）张量。")

    x = torch.tensor([1, 4, 9])
    assert torch.equal(x.pow(2), torch.tensor([1, 16, 81]))
    assert torch.equal(x.sqrt(), torch.tensor([1, 2, 3]))
    assert torch.equal(x.rsqrt(), torch.tensor([1, 1 / 2, 1 / 3]))  # i -> 1/sqrt(x_i)

    assert torch.equal(x + x, torch.tensor([2, 8, 18]))
    assert torch.equal(x * 2, torch.tensor([2, 8, 18]))
    assert torch.equal(x / 0.5, torch.tensor([2, 8, 18]))

    text("`triu` 取矩阵的上三角部分。")
    x = torch.ones(3, 3).triu()  # @inspect x
    assert torch.equal(x, torch.tensor([
        [1, 1, 1],
        [0, 1, 1],
        [0, 0, 1]],
    ))
    text("这对于计算因果注意力掩码很有用，其中 M[i, j] 是 i 对 j 的贡献。")


def tensor_matmul():
    text("最后，深度学习的核心：矩阵乘法。")
    x = torch.ones(16, 32)
    w = torch.ones(32, 2)
    y = x @ w
    assert y.size() == torch.Size([16, 2])

    text("通常，我们对批次中的每个示例和序列中的每个标记执行操作。")
    image("images/batch-sequence.png", width=400)
    x = torch.ones(4, 8, 16, 32)
    w = torch.ones(32, 2)
    y = x @ w
    assert y.size() == torch.Size([4, 8, 16, 2])
    text("在这种情况下，我们迭代 `x` 的前2个维度的值并乘以 `w`。")


def tensor_einops():
    einops_motivation()

    text("Einops is a library for manipulating tensors where dimensions are named.")
    text("It is inspired by Einstein summation notation (Einstein, 1916).")
    link(title="[Einops tutorial]", url="https://einops.rocks/1-einops-basics/")

    jaxtyping_basics()
    einops_einsum()
    einops_reduce()
    einops_rearrange()
    

def einops_motivation():
    text("Traditional PyTorch code:")
    x = torch.ones(2, 2, 3)  # batch, sequence, hidden  @inspect x
    y = torch.ones(2, 2, 3)  # batch, sequence, hidden  @inspect y
    z = x @ y.transpose(-2, -1)  # batch, sequence, sequence  @inspect z
    text("Easy to mess up the dimensions (what is -2, -1?)...")


def jaxtyping_basics():
    text("如何跟踪张量维度?")

    text("老方法:")
    x = torch.ones(2, 2, 1, 3)  # 批次 序列 头 隐藏  @inspect x

    text("新 (jaxtyping) 方法:")
    x: Float[torch.Tensor, "batch seq heads hidden"] = torch.ones(2, 2, 1, 3)  # @inspect x
    text("注意: 这只是文档（没有强制执行）。")


def einops_einsum():
    text("Einsum 是带有良好记录的广义矩阵乘法。")

    text("定义两个张量:")
    x: Float[torch.Tensor, "batch seq1 hidden"] = torch.ones(2, 3, 4)  # @inspect x
    y: Float[torch.Tensor, "batch seq2 hidden"] = torch.ones(2, 3, 4)  # @inspect y

    text("老方法:")
    z = x @ y.transpose(-2, -1)  # batch, sequence, sequence  @inspect z

    text("新 (einops) 方法:")
    z = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")  # @inspect z
    text("未在输出中命名的维度会被求和。")

    text("或者可以使用 `...` 来表示在任意数量的维度上进行广播:")
    z = einsum(x, y, "... seq1 hidden, ... seq2 hidden -> ... seq1 seq2")  # @inspect z


def einops_reduce():
    text("您可以通过某些操作（如求和、平均、最大值、最小值）来约简单个张量。")
    x: Float[torch.Tensor, "batch seq hidden"] = torch.ones(2, 3, 4)  # @inspect x

    text("老方法:")
    y = x.mean(dim=-1)  # @inspect y

    text("新 (einops) 方法:")
    y = reduce(x, "... hidden -> ...", "sum")  # @inspect y


def einops_rearrange():
    text("有时，一个维度表示两个维度")
    text("...您想对其中一个进行操作。")

    x: Float[torch.Tensor, "batch seq total_hidden"] = torch.ones(2, 3, 8)  # @inspect x
    text("...其中 `total_hidden` 是 `heads * hidden1` 的扁平化表示")
    w: Float[torch.Tensor, "hidden1 hidden2"] = torch.ones(4, 4)

    text("将 `total_hidden` 分成两个维度 (`heads` 和 `hidden1`):")
    x = rearrange(x, "... (heads hidden1) -> ... heads hidden1", heads=2)  # @inspect x

    text("通过 `w` 执行变换:")
    x = einsum(x, w, "... hidden1, hidden1 hidden2 -> ... hidden2")  # @inspect x

    text("将 `heads` 和 `hidden2` 合并回来:")
    x = rearrange(x, "... heads hidden2 -> ... (heads hidden2)")  # @inspect x


def tensor_operations_flops():
    text("经历了所有操作后，让我们检查它们的计算成本。")

    text("浮点运算 (FLOP) 是基本运算，如加法 (x + y) 或乘法 (x y)。")

    text("两个非常令人困惑的缩写（发音相同!）:")
    text("- FLOPs: 浮点运算（计算量的度量）")
    text("- FLOP/s: 每秒浮点运算次数（也写作FLOPS），用于衡量硬件速度。")

    text("## 直觉")
    text("训练 GPT-3 (2020) 耗费了 3.14e23 FLOPs。"), article_link("https://lambdalabs.com/blog/demystifying-gpt-3")
    text("训练 GPT-4 (2023) 据推测需要 2e25 FLOPs "), article_link("https://patmcguinness.substack.com/p/gpt-4-details-revealed")
    text("美国行政命令：任何使用 >= 1e26 FLOPs 训练的基础模型必须向政府报告（2025年撤销）")

    text("A100 的峰值性能为 312 teraFLOP/s "), link(title="[规格]", url="https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf")
    assert a100_flop_per_sec == 312e12

    text("H100 的峰值性能为 1979 teraFLOP/s（带稀疏性），无稀疏性为 50% "), link(title="[规格]", url="https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet")
    assert h100_flop_per_sec == 1979e12 / 2

    text("8个H100运行2周:")
    total_flops = 8 * (60 * 60 * 24 * 7) * h100_flop_per_sec  # @inspect total_flops

    text("## 线性模型")
    text("作为动机，假设您有一个线性模型。")
    text("- 我们有 n 个点")
    text("- 每个点是 d 维的")
    text("- 线性模型将每个 d 维向量映射到 k 个输出")

    if torch.cuda.is_available():
        B = 16384  # 点数
        D = 32768  # 维度
        K = 8192   # 输出数
    else:
        B = 1024
        D = 256
        K = 64

    device = get_device()
    x = torch.ones(B, D, device=device)
    w = torch.randn(D, K, device=device)
    y = x @ w
    text("对于每个 (i, j, k) 三元组，有一个乘法 (x[i][j] * w[j][k]) 和一个加法。")
    actual_num_flops = 2 * B * D * K  # @inspect actual_num_flops

    text("## 其他操作的 FLOPs")
    text("- 对 m x n 矩阵的逐元素操作需要 O(m n) FLOPs。")
    text("- 两个 m x n 矩阵的加法需要 m n FLOPs。")
    text("一般来说，对于足够大的矩阵，您在深度学习中遇到的没有其他操作比矩阵乘法更耗时。")

    text("解释:")
    text("- B 是数据点的数量")
    text("- (D K) 是参数数量")
    text("- 前向传播的 FLOPs 是 2 (# tokens) (# parameters)")
    text("事实证明，这可以推广到 Transformers（一阶近似）。")

    text("我们的 FLOPs 计算如何转化为实际时间（秒）？")
    text("让我们来计时!")
    actual_time = time_matmul(x, w)  # @inspect actual_time
    actual_flop_per_sec = actual_num_flops / actual_time  # @inspect actual_flop_per_sec

    text("每个GPU都有一个规格表，报告峰值性能。")
    text("- A100 "), link(title="[规格]", url="https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf")
    text("- H100 "), link(title="[规格]", url="https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet")
    text("注意，FLOP/s 高度依赖于数据类型！")
    promised_flop_per_sec = get_promised_flop_per_sec(device, x.dtype)  # @inspect promised_flop_per_sec

    text("## 模型 FLOPs 利用率 (MFU)")

    text("定义: (实际 FLOP/s) / (承诺 FLOP/s) [忽略通信/开销]")
    mfu = actual_flop_per_sec / promised_flop_per_sec  # @inspect mfu
    text("通常，MFU >= 0.5 是相当不错的（如果矩阵乘法占主导地位，则会更高）")

    text("让我们用 bfloat16 来做:")
    x = x.to(torch.bfloat16)
    w = w.to(torch.bfloat16)
    bf16_actual_time = time_matmul(x, w)  # @inspect bf16_actual_time
    bf16_actual_flop_per_sec = actual_num_flops / bf16_actual_time  # @inspect bf16_actual_flop_per_sec
    bf16_promised_flop_per_sec = get_promised_flop_per_sec(device, x.dtype)  # @inspect bf16_promised_flop_per_sec
    bf16_mfu = bf16_actual_flop_per_sec / bf16_promised_flop_per_sec  # @inspect bf16_mfu
    text("注意：比较 bfloat16 与 float32，实际 FLOP/s 更高。")
    text("这里的 MFU 相当低，可能是因为承诺的 FLOPs 有些乐观。")

    text("## 总结")
    text("- 矩阵乘法占主导地位: (2 m n p) FLOPs")
    text("- FLOP/s 依赖于硬件 (H100 >> A100) 和数据类型 (bfloat16 >> float32)")
    text("- 模型 FLOPs 利用率 (MFU): (实际 FLOP/s) / (承诺 FLOP/s)")


def gradients_basics():
    text("到目前为止，我们已经构建了张量（对应于参数或数据）并将它们通过操作传递（前向传播）。")
    text("现在，我们要计算梯度（反向传播）。")

    text("举一个简单的例子，让我们考虑简单的线性模型:")
    text("y = 0.5 (x * w - 5)^2")

    text("前向传播: 计算损失")
    x = torch.tensor([1., 2, 3])
    w = torch.tensor([1., 1, 1], requires_grad=True)  # 需要梯度
    pred_y = x @ w
    loss = 0.5 * (pred_y - 5).pow(2)

    text("反向传播: 计算梯度")
    loss.backward()
    assert loss.grad is None
    assert pred_y.grad is None
    assert x.grad is None
    assert torch.equal(w.grad, torch.tensor([1, 2, 3]))


def gradients_flops():
    text("让我们计算计算梯度的 FLOPs。")

    text("回顾我们的线性模型")
    if torch.cuda.is_available():
        B = 16384  # 点数
        D = 32768  # 维度
        K = 8192   # 输出数
    else:
        B = 1024
        D = 256
        K = 64

    device = get_device()
    x = torch.ones(B, D, device=device)
    w1 = torch.randn(D, D, device=device, requires_grad=True)
    w2 = torch.randn(D, K, device=device, requires_grad=True)

    text("模型: x --w1--> h1 --w2--> h2 -> loss")
    h1 = x @ w1
    h2 = h1 @ w2
    loss = h2.pow(2).mean()

    text("回忆前向 FLOPs 的数量: "), link(tensor_operations_flops)
    text("- 乘以 x[i][j] * w1[j][k]")
    text("- 加到 h1[i][k]")
    text("- 乘以 h1[i][j] * w2[j][k]")
    text("- 加到 h2[i][k]")
    num_forward_flops = (2 * B * D * D) + (2 * B * D * K)  # @inspect num_forward_flops

    text("运行反向传播需要多少 FLOPs？")
    h1.retain_grad()  # 用于调试
    h2.retain_grad()  # 用于调试
    loss.backward()

    text("回忆模型: x --w1--> h1 --w2--> h2 -> loss")

    text("- h1.grad = d loss / d h1")
    text("- h2.grad = d loss / d h2")
    text("- w1.grad = d loss / d w1")
    text("- w2.grad = d loss / d w2")

    text("关注参数 w2。")
    text("应用链式法则。")

    num_backward_flops = 0  # @inspect num_backward_flops

    text("w2.grad[j,k] = sum_i h1[i,j] * h2.grad[i,k]")
    assert w2.grad.size() == torch.Size([D, K])
    assert h1.size() == torch.Size([B, D])
    assert h2.grad.size() == torch.Size([B, K])
    text("对于每个 (i, j, k)，乘法和加法。")
    num_backward_flops += 2 * B * D * K  # @inspect num_backward_flops

    text("h1.grad[i,j] = sum_k w2[j,k] * h2.grad[i,k]")
    assert h1.grad.size() == torch.Size([B, D])
    assert w2.size() == torch.Size([D, K])
    assert h2.grad.size() == torch.Size([B, K])
    text("对于每个 (i, j, k)，乘法和加法。")
    num_backward_flops += 2 * B * D * K  # @inspect num_backward_flops

    text("这只是针对 w2 (D*K 个参数)。")
    text("同样可以对 w1 (D*D 个参数) 进行操作（尽管不需要 x.grad）。")
    num_backward_flops += (2 + 2) * B * D * D  # @inspect num_backward_flops

    text("一个很好的图形可视化: "), article_link("https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4")
    image("https://miro.medium.com/v2/resize:fit:1400/format:webp/1*VC9y_dHhCKFPXj90Qshj3w.gif", width=500)

    text("总结:")
    text("- 前向传播: 2 (# data points) (# parameters) FLOPs")
    text("- 反向传播: 4 (# data points) (# parameters) FLOPs")
    text("- 总计: 6 (# data points) (# parameters) FLOPs")


def module_parameters():
    input_dim = 16384
    output_dim = 32

    text("模型参数在 PyTorch 中存储为 `nn.Parameter` 对象。")
    w = nn.Parameter(torch.randn(input_dim, output_dim))
    assert isinstance(w, torch.Tensor)  # 行为像张量
    assert type(w.data) == torch.Tensor  # 访问基础张量

    text("## 参数初始化")

    text("让我们看看发生了什么。")
    x = nn.Parameter(torch.randn(input_dim))
    output = x @ w  # @inspect output
    assert output.size() == torch.Size([output_dim])
    text(f"注意，`output` 的每个元素都按 sqrt(input_dim) 缩放: {output[0]}。")
    text("大值会导致梯度爆炸并导致训练不稳定。")

    text("我们想要一个对 `input_dim` 不变的初始化。")
    text("为此，我们只需按 1/sqrt(input_dim) 进行缩放")
    w = nn.Parameter(torch.randn(input_dim, output_dim) / np.sqrt(input_dim))
    output = x @ w  # @inspect output
    text(f"现在 `output` 的每个元素都是常数: {output[0]}。")

    text("最多是一个常数，这是 Xavier 初始化。"), link(title="[论文]", url="https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf"), link(title="[stackexchange]", url="https://ai.stackexchange.com/questions/30491/is-there-a-proper-initialization-technique-for-the-weight-matrices-in-multi-head")

    text("为了更安全，我们将正态分布截断为 [-3, 3] 以避免任何异常值。")
    w = nn.Parameter(nn.init.trunc_normal_(torch.empty(input_dim, output_dim), std=1 / np.sqrt(input_dim), a=-3, b=3))


def custom_model():
    text("让我们使用 `nn.Parameter` 构建一个简单的深度线性模型。")

    D = 64  # 维度
    num_layers = 2
    model = Cruncher(dim=D, num_layers=num_layers)

    param_sizes = [
        (name, param.numel())
        for name, param in model.state_dict().items()
    ]
    assert param_sizes == [
        ("layers.0.weight", D * D),
        ("layers.1.weight", D * D),
        ("final.weight", D),
    ]
    num_parameters = get_num_parameters(model)
    assert num_parameters == (D * D) + (D * D) + D

    text("记得将模型移到 GPU。")
    device = get_device()
    model = model.to(device)

    text("在一些数据上运行模型。")
    B = 8  # 批处理大小
    x = torch.randn(B, D, device=device)
    y = model(x)
    assert y.size() == torch.Size([B])


class Linear(nn.Module):
    """Simple linear layer."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim) / np.sqrt(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight


class Cruncher(nn.Module):
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            Linear(dim, dim)
            for i in range(num_layers)
        ])
        self.final = Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply linear layers
        B, D = x.size()
        for layer in self.layers:
            x = layer(x)

        # Apply final head
        x = self.final(x)
        assert x.size() == torch.Size([B, 1])

        # Remove the last dimension
        x = x.squeeze(-1)
        assert x.size() == torch.Size([B])

        return x


def get_batch(data: np.array, batch_size: int, sequence_length: int, device: str) -> torch.Tensor:
    text("在 `data` 中采样 `batch_size` 个随机位置。")
    start_indices = torch.randint(len(data) - sequence_length, (batch_size,))
    assert start_indices.size() == torch.Size([batch_size])

    text("索引到数据中。")
    x = torch.tensor([data[start:start + sequence_length] for start in start_indices])
    assert x.size() == torch.Size([batch_size, sequence_length])

    text("## 锁页内存")

    text("默认情况下，CPU 张量在分页内存中。我们可以显式锁定。")
    if torch.cuda.is_available():
        x = x.pin_memory()

    text("这允许我们异步地将 `x` 从 CPU 复制到 GPU。")
    x = x.to(device, non_blocking=True)

    text("这允许我们并行做两件事（此处未完成）:")
    text("- 将下一批数据获取到 CPU")
    text("- 在 GPU 上处理 `x`。")

    article_link("https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/")
    article_link("https://gist.github.com/ZijiaLewisLu/eabdca955110833c0ce984d34eb7ff39?permalink_comment_id=3417135")

    return x


def note_about_randomness():
    text("随机性出现在许多地方：参数初始化、dropout、数据排序等。")
    text("为了可重复性，我们建议您在每次使用随机性时都传入不同的随机种子。")
    text("确定性在调试时特别有用，因此您可以找到错误。")

    text("有三个地方需要设置随机种子，您应该一次性全部设置以确保安全。")

    # Torch
    seed = 0
    torch.manual_seed(seed)

    # NumPy
    import numpy as np
    np.random.seed(seed)

    # Python
    import random
    random.seed(seed)


def data_loading():
    text("在语言建模中，数据是整数序列（由分词器输出）。")

    text("将它们序列化为 numpy 数组很方便（由分词器完成）。")
    orig_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32)
    orig_data.tofile("data.npy")

    text("您可以将它们作为 numpy 数组加载回来。")
    text("不想一次将所有数据加载到内存中（LLaMA 数据为 2.8TB）。")
    text("使用 memmap 惰性地仅将访问的部分加载到内存中。")
    data = np.memmap("data.npy", dtype=np.int32)
    assert np.array_equal(data, orig_data)

    text("一个 *数据加载器* 为训练生成一批序列。")
    B = 2  # 批处理大小
    L = 4  # 序列长度
    x = get_batch(data, batch_size=B, sequence_length=L, device=get_device())
    assert x.size() == torch.Size([B, L])


class SGD(torch.optim.Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 0.01):
        super(SGD, self).__init__(params, dict(lr=lr))

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                grad = p.grad.data
                p.data -= lr * grad


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


def optimizer():
    text("回忆我们的深度线性模型。")
    B = 2
    D = 4
    num_layers = 2
    model = Cruncher(dim=D, num_layers=num_layers).to(get_device())

    text("让我们定义 AdaGrad 优化器")
    text("- momentum = SGD + 梯度指数平均")
    text("- AdaGrad = SGD + 梯度平方平均")
    text("- RMSProp = AdaGrad + 梯度平方的指数平均")
    text("- Adam = RMSProp + momentum")

    text("AdaGrad: "), link("https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf")
    optimizer = AdaGrad(model.parameters(), lr=0.01)
    state = model.state_dict()  # @inspect state

    text("计算梯度")
    x = torch.randn(B, D, device=get_device())
    y = torch.tensor([4., 5.], device=get_device())
    pred_y = model(x)
    loss = F.mse_loss(input=pred_y, target=y)
    loss.backward()

    text("执行一步")
    optimizer.step()
    state = model.state_dict()  # @inspect state

    text("释放内存（可选）")
    optimizer.zero_grad(set_to_none=True)

    text("## 内存")

    # 参数
    num_parameters = (D * D * num_layers) + D  # @inspect num_parameters
    assert num_parameters == get_num_parameters(model)

    # 激活值
    num_activations = B * D * num_layers  # @inspect num_activations

    # 梯度
    num_gradients = num_parameters  # @inspect num_gradients

    # 优化器状态
    num_optimizer_states = num_parameters  # @inspect num_optimizer_states

    # 总结，假设使用 float32
    total_memory = 4 * (num_parameters + num_activations + num_gradients + num_optimizer_states)  # @inspect total_memory

    text("## 计算（一步）")
    flops = 6 * B * num_parameters  # @inspect flops

    text("## Transformers")

    text("Transformer 的计算更复杂，但想法相同。")
    text("Assignment 1 将要求您这样做。")

    text("描述 Transformer 训练内存使用的博客文章 "), article_link("https://erees.dev/transformer-memory/")
    text("描述 Transformer FLOPs 的博客文章: "), article_link("https://www.adamcasson.com/posts/transformer-flops")


def train_loop():
    text("从线性函数生成数据，权重为 (0, 1, 2, ..., D-1)。")
    D = 16
    true_w = torch.arange(D, dtype=torch.float32, device=get_device())
    def get_batch(B: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(B, D).to(get_device())
        true_y = x @ true_w
        return (x, true_y)

    text("让我们做一个基本运行")
    train("simple", get_batch, D=D, num_layers=0, B=4, num_train_steps=10, lr=0.01)

    text("进行一些超参数调整")
    train("simple", get_batch, D=D, num_layers=0, B=4, num_train_steps=10, lr=0.1)


def train(name: str, get_batch,
          D: int, num_layers: int,
          B: int, num_train_steps: int, lr: float):
    model = Cruncher(dim=D, num_layers=0).to(get_device())
    optimizer = SGD(model.parameters(), lr=0.01)

    for t in range(num_train_steps):
        # Get data
        x, y = get_batch(B=B)

        # Forward (compute loss)
        pred_y = model(x)
        loss = F.mse_loss(pred_y, y)

        # Backward (compute gradients)
        loss.backward()

        # Update parameters
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


def checkpointing():
    text("训练语言模型需要很长时间，而且肯定会崩溃。")
    text("您不想失去所有进度。")

    text("在训练期间，定期将模型和优化器状态保存到磁盘很有用。")

    model = Cruncher(dim=64, num_layers=3).to(get_device())
    optimizer = AdaGrad(model.parameters(), lr=0.01)

    text("保存检查点:")
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, "model_checkpoint.pt")

    text("加载检查点:")
    loaded_checkpoint = torch.load("model_checkpoint.pt")


def mixed_precision_training():
    text("数据类型选择（float32, bfloat16, fp8）有取舍。")
    text("- 更高精度：更准确/稳定，更多内存，更多计算")
    text("- 更低精度：不太准确/稳定，更少内存，更少计算")

    text("我们如何两全其美？")

    text("解决方案：默认使用 float32，但尽可能使用 {bfloat16, fp8}。")

    text("一个具体的计划:")
    text("- 在前向传播（激活值）中使用 {bfloat16, fp8}。")
    text("- 其余部分（参数、梯度）使用 float32。")

    text("- 混合精度训练 "), link("https://arxiv.org/pdf/1710.03740.pdf")

    text("Pytorch 有一个自动混合精度（AMP）库。")
    link("https://pytorch.org/docs/stable/amp.html")
    link("https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/")

    text("NVIDIA 的 Transformer Engine 支持线性层的 FP8")
    text("在整个训练中普遍使用 FP8 "), link("https://arxiv.org/pdf/2310.18313.pdf")


############################################################

def get_memory_usage(x: torch.Tensor):
    return x.numel() * x.element_size()


def get_promised_flop_per_sec(device: str, dtype: torch.dtype) -> float:
    """Return the peak FLOP/s for `device` operating on `dtype`."""
    if not torch.cuda.is_available():
        text("No CUDA device available, so can't get FLOP/s.")
        return 1
    properties = torch.cuda.get_device_properties(device)

    if "A100" in properties.name:
        # https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf")
        if dtype == torch.float32:
            return 19.5e12
        if dtype in (torch.bfloat16, torch.float16):
            return 312e12
        raise ValueError(f"Unknown dtype: {dtype}")

    if "H100" in properties.name:
        # https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet")
        if dtype == torch.float32:
            return 67.5e12
        if dtype in (torch.bfloat16, torch.float16):
            return 1979e12 / 2  # 1979 is for sparse, dense is half of that
        raise ValueError(f"Unknown dtype: {dtype}")

    raise ValueError(f"Unknown device: {device}")


def same_storage(x: torch.Tensor, y: torch.Tensor):
    return x.untyped_storage().data_ptr() == y.untyped_storage().data_ptr()


def time_matmul(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return the number of seconds required to perform `a @ b`."""

    # Wait until previous CUDA threads are done
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    def run():
        # Perform the operation
        a @ b

        # Wait until CUDA threads are done
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Time the operation `num_trials` times
    num_trials = 5
    total_time = timeit.timeit(run, number=num_trials)

    return total_time / num_trials


def get_num_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())

def get_device(index: int = 0) -> torch.device:
    """Try to use the GPU if possible, otherwise, use CPU."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index}")
    else:
        return torch.device("cpu")

if __name__ == "__main__":
    main()
