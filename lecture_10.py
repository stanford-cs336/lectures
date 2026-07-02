from dataclasses import dataclass

from sympy import symbols, oo
from edtrace import text, link, image
from lecture_util import article_link, bilingual_text
from references import Reference, gqa_2023, mla_2024, longformer_2020, sparse_transformer_2019, mistral_7b_2023, deepseek_v4_2026

# Define symbols corresponding to the shape of the Transformer model
B, S, T, D, F, N, K, H, L, V = symbols("B S T D F N K H L V", positive=True)
c = symbols("c", positive=True)  # Just a constant that helps with taking limits
memory_bandwidth = symbols("memory_bandwidth", positive=True)

scaling_book_transformers = Reference(title="Scaling book chapter on Transformers", url="https://jax-ml.github.io/scaling-book/transformers/")
scaling_book_inference = Reference(title="Scaling book chapter on inference", url="https://jax-ml.github.io/scaling-book/inference/")

def main():
    bilingual_text("## Lecture 10: inference", '## 第 10 讲：推理')
    image("images/inference-schema.png", width=600)

    bilingual_text("### Understanding the inference workload", '### 理解推理工作负载')
    landscape()
    review_transformer()
    review_of_arithmetic_intensity()
    arithmetic_intensity_of_inference()
    throughput_and_latency()

    bilingual_text("### Taking shortcuts (lossy)", '### 走捷径（有损）')
    reduce_kv_cache_size()
    quantization()
    model_pruning()

    bilingual_text("Summary: reduce inference complexity without hurting accuracy", '总结：在不损害准确率的前提下降低推理复杂度。')

    bilingual_text("From scratch recipe:", '从零开始的配方：')
    bilingual_text("1. Define faster model architecture", '1. 定义更快的模型架构。')
    bilingual_text("2. Train faster model", '2. 训练更快的模型。')

    bilingual_text("Distillation recipe:", '蒸馏配方：')
    bilingual_text("1. Define faster model architecture", '1. 定义更快的模型架构。')
    bilingual_text("2. Initialize weights using original model (which has a different architecture)", '2. 使用原始模型初始化权重（原始模型具有不同架构）。')
    bilingual_text("3. Repair faster model (distillation)", '3. 修复更快的模型（蒸馏）。')

    bilingual_text("### Use shortcuts but double check (lossless)", '### 使用捷径，但要复核（无损）')
    speculative_sampling()

    bilingual_text("### Handling dynamic workloads", '### 处理动态工作负载')
    bilingual_text("Batching over sequences in live traffic is tricky because:", '在实时流量中对序列做批处理很棘手，因为：')
    bilingual_text("1. Requests arrive at different times (waiting for batch is bad for early requests)", '1. 请求在不同时间到达（等待成批会伤害早到的请求）。')
    bilingual_text("2. Sequences have shared prefixes (e.g., system prompts, generating multiple samples)", '2. 序列可能共享前缀（例如系统提示、生成多个样本）。')
    bilingual_text("3. Sequences have different lengths (padding is inefficient)", '3. 序列长度不同（填充效率低）。')

    continuous_batching()
    paged_attention()

    bilingual_text("### Summary", '### 总结')
    bilingual_text("- Inference is important (actual use, evaluation, reinforcement learning)", '- 推理很重要（实际使用、评测、强化学习都需要）。')
    bilingual_text("- Different characteristics compared to training (memory-bound, dynamic)", '- 与训练相比特性不同（受内存带宽限制、动态）。')
    bilingual_text("- Techniques: new architectures, quantization, pruning/distillation, speculative sampling", '- 技术：新架构、量化、剪枝/蒸馏、推测采样。')
    bilingual_text("- Ideas from systems (speculative execution, paging)", '- 来自系统领域的思想（推测执行、分页）。')
    bilingual_text("- New architectures have huge potential for improvement", '- 新架构有巨大的改进潜力。')


def landscape():
    bilingual_text("Inference shows up in many places:", '推理出现在很多地方：')
    bilingual_text("- Actual use (chatbots, code completion, agents, batch data processing)", '- 实际使用（聊天机器人、代码补全、智能体、批量数据处理）。')
    bilingual_text("- Model evaluation (e.g., on instruction following)", '- 模型评测（例如指令遵循）。')
    bilingual_text("- Reinforcement learning (sample many generations, then apply score)", '- 强化学习（采样许多生成结果，然后打分）。')

    bilingual_text("Why **efficiency** matters: training is one-time cost, inference is repeated many times", '为什么**效率**重要：训练是一次性成本，而推理会重复发生很多次。')
    bilingual_text("- OpenAI processes ~8.6T tokens per day ", '- 说明：OpenAI processes ~8.6T tokens per day'), article_link("https://www.pymnts.com/artificial-intelligence-2/2025/openai-bests-google-in-race-for-consumer-ai-token-consumption/")
    bilingual_text("- For reference, DeepSeek v4 was trained on 32T tokens ", '- 说明：For reference, DeepSeek v4 was trained on 32T tokens'), link(deepseek_v4_2026)
    
    bilingual_text("Moreover:", '此外：')
    bilingual_text("- Chatbots: most tokens are meant for human consumption (humans are bottleneck)", '- 聊天机器人：大多数 token 是给人看的（人是瓶颈）。')
    bilingual_text("- Agents: query → internal trace → output for human (number of tokens generated can grow unbounded)", '- 智能体：查询 → 内部轨迹 → 给人的输出（生成 token 数可能无限增长）。')
    bilingual_text("- Tokens generated = compute spent", '- 生成的 token = 花掉的计算量。')

    bilingual_text("Companies doing inference (a big deal for anyone who has a product or platform):", 'Companies doing 推理 (a big deal for anyone who has a product or platform):')
    bilingual_text("- Providers serving closed models (OpenAI, Anthropic, Google, etc.)", '- Providers serving closed 模型s (OpenAI, Anthropic, Google, etc.)')
    bilingual_text("- Providers serving open-weight models (Together, Fireworks, Baseten, DeepInfra, Groq, Cerebras, etc.)", '- Providers serving open-weight 模型s (Together, Fireworks, Baseten, DeepInfra, Groq, Cerebras, etc.)')

    bilingual_text("Open-source packages:", '开源软件包：')
    bilingual_text("- vLLM: from Berkeley, pioneered PagedAttention, popular and good default ", '- vLLM: from Berkeley, pioneered Paged注意力, popular and good default'), link(title="GitHub", url="https://github.com/vllm-project/vllm")
    bilingual_text("- SGLang: from Berkeley, pioneered RadixAttention, good for agentic workloads ", '- SGLang: from Berkeley, pioneered Radix注意力, good for 智能体ic 工作负载s'), link(title="project", url="https://sgl-project.github.io/")
    bilingual_text("- TensorRT-LLM: from NVIDIA, highly optimized for GPUs ", '- 说明：TensorRT-LLM: from NVIDIA, highly optimized for GPUs'), article_link("https://nvidia.github.io/TensorRT-LLM/overview.html")
    bilingual_text("- llama.cpp: C++ only, supports CPU inference, runs locally ", '- llama.cpp: C++ only, supports CPU 推理, runs locally'), link(title="GitHub", url="https://github.com/ggml-org/llama.cpp")

    bilingual_text("Inference is huge. Important to make it fast.", '推理规模巨大，让它变快非常重要。')

    bilingual_text("What does \"fast\" mean (metrics)?", '“快”意味着什么（指标）？')
    bilingual_text("- Time-to-first-token (TTFT): how long user waits before any generation happens (for interactive applications)", '- 首 token 时间 (TTFT): how long user waits before any 生成 happens (for interactive applications)')
    bilingual_text("- Latency (seconds/token): how fast tokens appear for *one* query (for interactive applications)", '- 延迟 (seconds/token): how fast token appear for one query (for interactive applications)')
    bilingual_text("- Throughput (tokens/second): how fast tokens appear for *many* queries (for batch processing)", '- 吞吐量 (token/second): how fast token appear for many queries (for 批次 processing)')

    bilingual_text("What governs efficiency?", '什么决定效率？')
    bilingual_text("- Training (supervised): you see all tokens, can parallelize over sequence (matmul in Transformer)", '- 训练 (supervised): you see all token, can parallelize over 序列 (matmul in Transformer)')
    bilingual_text("- Inference: you have to generate sequentially, can't parallelize over generation, so harder to fully utilize compute", "- 推理: you have to generate sequentially, can't parallelize over 生成, so harder to fully utilize 计算量")


def review_transformer():
    link(scaling_book_transformers)
    bilingual_text("Notation (similar to einops):", '记号（类似 einops）：')
    bilingual_text("- Symbols denote dimensions (and their length): B (batch), T (sequence), D (model dim), H (head dim)", '- Symbols denote dimensions (and their length): B (批次), T (序列), D (模型 dim), H (head dim)')
    bilingual_text("- Example: BT<font color=\"red\">D</font> x <font color=\"red\">D</font>H → BTH", '- 示例：BT<font color="red">D</font> x <font color="red">D</font>H → BTH')
    bilingual_text("- <font color=\"red\">Contracting (red)</font> dimensions appear in both operands and disappear from result", '- 说明：<font color="red">Contracting (red)</font> dimensions appear in both operands and disappear from result')
    bilingual_text("- Regular (black) dimensions appear in one operand and stay in result", '- 说明：Regular (black) dimensions appear in one operand and stay in result')
    bilingual_text("- Example: <font color=\"blue\">B</font><font color=\"red\">D</font> x <font color=\"blue\">B</font><font color=\"red\">D</font> → B", '- 示例：<font color="blue">B</font><font color="red">D</font> x <font color="blue">B</font><font color="red">D</font> → B')
    bilingual_text("- <font color=\"blue\">Batching (blue)</font> dimensions appear in both operands and stay in result", '- <font color="blue">批处理 (blue)</font> dimensions appear in both operands and stay in result')

    image("https://jax-ml.github.io/scaling-book/assets/img/transformer-diagram.png", width=800)
    bilingual_text("Conventions:", '约定：')
    bilingual_text("- F = 4 D (MLP up-projects into 4x the model dimension)", '- F = 4 D (MLP up-projects into 4x the 模型 dimension)')
    bilingual_text("- D = N H (model dimension split across N heads)", '- D = N H (模型 dimension split across N heads)')
    bilingual_text("- N = K G (for GQA, number of heads split across K groups)", '- 说明：N = K G (for GQA, number of heads split across K groups)')
    bilingual_text("- S = T (during training, condition on S input tokens to predict T output tokens)", '- S = T (during 训练, condition on S input token to predict T output token)')


def review_of_arithmetic_intensity():
    bilingual_text("Setup: multiply X <font color=\"gray\">(B x D)</font> and W <font color=\"gray\">(D x F)</font> matrix", '设置：将矩阵 X <font color="gray">(B x D)</font> 与 W <font color="gray">(D x F)</font> 相乘。')
    bilingual_text("Intuition: B is batch size, D is hidden dimension, F is up-projection dimension in MLP", '直觉：B 是批大小，D 是隐藏维度，F 是 MLP 中上投影维度。')

    bilingual_text("Let's do FLOPs and memory read/write accounting for the matrix multiplication (X * W).", '让我们核算矩阵乘法（X * W）的 FLOPs 和内存读写。')
    flops = 0
    bytes_transferred = 0

    # Perform the operation
    bilingual_text("1. Read X <font color=\"gray\">(B x D)</font> from HBM", '1. 说明：Read X <font color="gray">(B x D)</font> from HBM')
    bytes_transferred += 2*B*D   # 2 bytes for bf16
    bilingual_text("2. Read W <font color=\"gray\">(D x F)</font> from HBM", '2. 说明：Read W <font color="gray">(D x F)</font> from HBM')
    bytes_transferred += 2*D*F
    bilingual_text("3. Compute Y = X <font color=\"gray\">(B x D)</font> @ W <font color=\"gray\">(D x F)</font>", '3. 计算量 Y = X <font color="gray">(B x D)</font> @ W <font color="gray">(D x F)</font>')
    flops += 2*B*D*F
    bilingual_text("4. Write Y <font color=\"gray\">(B x F)</font> to HBM", '4. 说明：Write Y <font color="gray">(B x F)</font> to HBM')
    bytes_transferred += 2*B*F

    assert flops == 2*B*D*F
    assert bytes_transferred == 2*B*D + 2*D*F + 2*B*F

    bilingual_text("Recall that **arithmetic intensity** is how much compute we do per byte transferred (want to be high).", '回忆：**算术强度**表示每传输 1 字节完成多少计算（越高越好）。')
    intensity = (flops / bytes_transferred).simplify()  # @inspect intensity

    bilingual_text("Assuming B is much less than D and F, then we can simplify:", '假设 B 远小于 D 和 F，则可以简化：')
    intensity = intensity.subs(D, c*B).subs(F, c*B).limit(c, oo).simplify()  # @inspect intensity
    assert intensity == B

    bilingual_text("Accelerator intensity of H100:", 'H100 的加速器强度：')
    flops_per_second = 989e12
    memory_bandwidth = 3.35e12
    accelerator_intensity = flops_per_second / memory_bandwidth  # @inspect accelerator_intensity
    assert round(accelerator_intensity) == 295

    bilingual_text("If computation intensity > accelerator intensity, **compute-bound** (good)", '如果计算强度 > 加速器强度，则**受计算限制**（好）。')
    bilingual_text("If computation intensity < accelerator intensity, **memory-bound** (bad)", '如果计算强度 < 加速器强度，则**受内存带宽限制**（坏）。')
    bilingual_text("Conclusion: compute-bound iff B > 295", '结论：当且仅当 B > 295 时受计算限制。')

    bilingual_text("Extreme case (B = 1, corresponding to matrix-vector product):", '极端情况（B = 1，对应矩阵-向量乘法）：')
    bilingual_text("- Arithmetic intensity: 1", '- 算术强度: 1')
    bilingual_text("- Memory-bound (read D x F matrix, perform only 2 D F FLOPs)", '- 受内存带宽限制 (read D x F matrix, perform only 2 D F FLOPs)')
    bilingual_text("- This is basically what happens with inference...", '- This is basically what happens with 推理...')


def arithmetic_intensity_of_inference():
    link(scaling_book_inference)

    image("https://jax-ml.github.io/scaling-book/assets/img/naive-inference-1400.webp", width=800)
    bilingual_text("Naive inference: to generate each token, feed history into Transformer", '朴素推理：为了生成每个 token，把历史输入 Transformer。')
    bilingual_text("Complexity: generating T tokens requires O(T^3) FLOPs (one feedforward pass is O(T^2))", '说明：Complexity: generating T tokens requires O(T^3) FLOPs (one feedforward pass is O(T^2))')

    bilingual_text("Observation: a lot of the work can be shared across prefixes", '观察：许多工作可以在前缀之间共享。')
    bilingual_text("Solution: store **KV cache** in HBM", '解决方案：在 HBM 中存储 **KV cache**。')
    image("https://jax-ml.github.io/scaling-book/assets/img/cached-inference-1400.webp", width=800)
    bilingual_text("KV cache: for every sequence (B), token (S), layer (L), head (K), store an H-dimensional vector", 'KV cache: for every 序列 (B), token (S), layer (L), head (K), store an H-dimensional vector')

    bilingual_text("Two stages of inference:", '推理的两个阶段：')
    bilingual_text("1. **Prefill**: given a prompt, encode into vectors (parallelizable like in training)", '1. 预填充: given a 提示, encode into vectors (parallelizable like in 训练)')
    bilingual_text("2. **Generation**: generate new response tokens (sequential)", '2. 生成: generate new 回答 token (sequential)')

    bilingual_text("Let's compute the FLOPs and memory IO for both the MLP and attention layers.", "Let's 计算量 the FLOPs and 内存 IO for both the MLP and 注意力 layers.")
    bilingual_text("S is the number of tokens we're conditioning on, T is the number of tokens we're generating.", "说明：S is the number of tokens we're conditioning on, T is the number of tokens we're generating.")
    bilingual_text("Later, we'll specialize to prefill (T = S) and generation (T = 1).", "Later, we'll specialize to 预填充 (T = S) and 生成 (T = 1).")

    bilingual_text("### MLP layers (only looking at the matrix multiplications)", '### MLP 层（只看矩阵乘法）')
    flops = 0
    bytes_transferred = 0
    
    # Perform the operation
    bilingual_text("1. Read X <font color=\"gray\">(B x T x D)</font> from HBM", '1. 说明：Read X <font color="gray">(B x T x D)</font> from HBM')
    bytes_transferred += 2*B*T*D
    bilingual_text("2. Read Wup <font color=\"gray\">(D x F)</font>, Wgate <font color=\"gray\">(D x F)</font>, Wdown <font color=\"gray\">(F x D)</font> from HBM", '2. 说明：Read Wup <font color="gray">(D x F)</font>, Wgate <font color="gray">(D x F)</font>, Wdown <font color="gray">(F x D)</font> from HBM')
    bytes_transferred += 3 * 2*D*F
    bilingual_text("3. Compute U = X <font color=\"gray\">(B x T x D)</font> @ Wup <font color=\"gray\">(D x F)</font>", '3. 计算量 U = X <font color="gray">(B x T x D)</font> @ Wup <font color="gray">(D x F)</font>')
    flops += 2*B*T*D*F
    bilingual_text("4. Write U <font color=\"gray\">(B x T x F)</font> to HBM", '4. 说明：Write U <font color="gray">(B x T x F)</font> to HBM')
    bytes_transferred += 2*B*T*F
    bilingual_text("5. Compute G = X <font color=\"gray\">(B x T x D)</font> @ Wgate <font color=\"gray\">(D x F)</font>", '5. 计算量 G = X <font color="gray">(B x T x D)</font> @ Wgate <font color="gray">(D x F)</font>')
    flops += 2*B*T*D*F
    bilingual_text("6. Write G <font color=\"gray\">(B x T x F)</font> to HBM", '6. 说明：Write G <font color="gray">(B x T x F)</font> to HBM')
    bytes_transferred += 2*B*T*F
    bilingual_text("7. Compute Y = GeLU(G)*U <font color=\"gray\">(B x T x F)</font> @ Wdown <font color=\"gray\">(F x D)</font>", '7. 计算量 Y = GeLU(G)U <font color="gray">(B x T x F)</font> @ Wdown <font color="gray">(F x D)</font>')
    flops += 2*B*T*D*F
    bilingual_text("8. Write Y <font color=\"gray\">(B x T x D)</font> to HBM", '8. 说明：Write Y <font color="gray">(B x T x D)</font> to HBM')
    bytes_transferred += 2*B*T*D

    assert flops == 6*B*T*D*F
    assert bytes_transferred == 4*B*T*D + 4*B*T*F + 6*D*F

    # Compute the arithmetic intensity
    intensity = (flops / bytes_transferred).simplify()  # @inspect intensity
    bilingual_text("Assume that B*T is much smaller than D and F.", '说明：Assume that BT is much smaller than D and F.')
    intensity = intensity.subs(D, c*B*T).subs(F, c*B*T).limit(c, oo).simplify()  # @inspect intensity
    assert intensity == B*T

    bilingual_text("For the two stages:", '对于两个阶段：')
    bilingual_text("1. Prefill: easy to make compute-bound (good) by making `B*T` large enough (large batches, long sequences)", '1. 预填充: easy to make 受计算限制 (good) by making BT large enough (large 批次es, long 序列)')
    bilingual_text("2. Generation: two problems", '2. 生成: two problems')
    bilingual_text("- Generating one token at a time (T = 1)", '- 说明：Generating one token at a time (T = 1)')
    bilingual_text("- B is number of concurrent requests, unpredictable for interactive applications", '- B is number of concurrent 请求, unpredictable for interactive applications')

    bilingual_text("### Attention layers (focusing on the matrix multiplications with FlashAttention)", '### 注意力层（聚焦使用 FlashAttention 的矩阵乘法）')
    bilingual_text("- S is number of previous tokens (already generated)", '- 说明：S is number of previous tokens (already generated)')
    bilingual_text("- T is number of next tokens (to generate logits for)", '- 说明：T is number of next tokens (to generate logits for)')
    flops = 0
    bytes_transferred = 0
    
    # Perform the operation
    bilingual_text("1. Read Q <font color=\"gray\">(B x T x D)</font>, K <font color=\"gray\">(B x S x D)</font>, V <font color=\"gray\">(B x S x D)</font> from HBM", '1. 说明：Read Q <font color="gray">(B x T x D)</font>, K <font color="gray">(B x S x D)</font>, V <font color="gray">(B x S x D)</font> from HBM')
    bytes_transferred += 2*B*T*D + 2*B*S*D + 2*B*S*D
    bilingual_text("2. Compute A = Q <font color=\"gray\">(B x T x D)</font> @ K <font color=\"gray\">(B x S x D)</font>", '2. 计算量 A = Q <font color="gray">(B x T x D)</font> @ K <font color="gray">(B x S x D)</font>')
    flops += 2*B*S*T*D
    bilingual_text("3. Compute Y = softmax(A) <font color=\"gray\">(B x S x T x K x G)</font> @ V <font color=\"gray\">(B x S x K x H)</font>", '3. 计算量 Y = softmax(A) <font color="gray">(B x S x T x K x G)</font> @ V <font color="gray">(B x S x K x H)</font>')
    flops += 2*B*S*T*D
    bilingual_text("4. Write Y <font color=\"gray\">(B x T x D)</font> to HBM", '4. 说明：Write Y <font color="gray">(B x T x D)</font> to HBM')
    bytes_transferred += 2*B*T*D

    assert flops == 4*B*S*T*D
    assert bytes_transferred == 4*B*S*D + 4*B*T*D

    # Compute the arithmetic intensity
    intensity = (flops / bytes_transferred).simplify()  # @inspect intensity
    assert intensity == S*T / (S + T)

    bilingual_text("For the two stages:", '对于两个阶段：')
    bilingual_text("1. Prefill: T = S", '1. 预填充: T = S')
    prefill_intensity = intensity.subs(T, S).simplify()  # @inspect prefill_intensity
    assert prefill_intensity == S/2  # Good!
    bilingual_text("2. Generation: T = 1", '2. 生成: T = 1')
    generate_intensity = intensity.subs(T, 1).simplify()  # @inspect generate_intensity
    assert generate_intensity < 1  # Bad!

    bilingual_text("Unlike MLPs, no dependence on B, so batching doesn't help!", "Unlike MLPs, no dependence on B, so 批处理 doesn't help!")
    bilingual_text("Why?", '说明：Why?')
    bilingual_text("- In MLP layers, every sequence hits the same MLP weights (Wup, Wgate, Wdown don't depend on B)", "- In MLP layers, every 序列 hits the same MLP weights (Wup, Wgate, Wdown don't depend on B)")
    bilingual_text("- In attention layers, every sequence has its own KV cache vectors (Q, K, V all depend on B)", '- In 注意力 layers, every 序列 has its own KV cache vectors (Q, K, V all depend on B)')

    bilingual_text("Summary:", '总结：')
    bilingual_text("- Prefill is compute-bound, generation is memory-bound", '- 预填充 is 受计算限制, 生成 is 受内存带宽限制')
    bilingual_text("- Prefill MLP intensity: `B*S`", '- 预填充 MLP intensity: BS')
    bilingual_text("- Prefill attention intensity: `S/2`", '- 预填充 注意力 intensity: S/2')
    bilingual_text("- Generation MLP intensity: `B` (requires concurrent requests)", '- 生成 MLP intensity: B (requires concurrent 请求)')
    bilingual_text("- Generation attention intensity: `<1` (impossible to improve)", '- 生成 注意力 intensity: <1 (impossible to improve)')


@dataclass(frozen=True)
class TransformerPerformanceStats:
    """
    Performance stats of a Transformer:
    - num_params: number of parameters (in bytes)
    - memory: total memory usage (parameters + KV cache) in bytes
    - latency: time to generate one token (seconds/token)
    - throughput: tokens generated per second
    """
    num_params: int
    memory: int
    latency: float
    throughput: float

    def substitute(self, key, value):
        """Substitute `key` with `value` in all stats."""
        return TransformerPerformanceStats(
            self.num_params.subs(key, value).simplify(),
            self.memory.subs(key, value).simplify(),
            self.latency.subs(key, value).simplify(),
            self.throughput.subs(key, value).simplify(),
        )


def compute_transformer_performance_stats(config) -> TransformerPerformanceStats:  # @inspect config
    """Compute various performance stats for the Transformer given `config`."""

    # Number of parameters in the Transformer
    num_params = 2*V*D + D*F*3*L + (2*D*N*H + 2*D*K*H)*L

    # How much memory the parameters take
    parameter_size = 2*num_params  # 2 for bf16 (training requires a larger multiple)
    
    # How much the KV cache takes per sequence (S tokens, K heads, H head dim, L layers)
    kv_cache_size_per_seq = S * (K*H) * L * 2 * 2  # 2 for key + value, 2 for bf16

    # Total memory usage
    memory = B * kv_cache_size_per_seq + parameter_size

    # *Latency* is determined by memory IO (read all parameters and KV cache for each step)
    latency = memory / memory_bandwidth

    # *Throughput* is the inverse of latency, but we're generating B tokens in parallel
    throughput = B / latency

    # Substitute config
    num_params = num_params.subs(config).simplify()  # @inspect num_params
    memory = memory.subs(config).simplify()  # @inspect memory
    latency = latency.subs(config).simplify()  # @inspect latency
    throughput = throughput.subs(config).simplify()  # @inspect throughput

    return TransformerPerformanceStats(num_params, memory, latency, throughput)


def llama2_13b_config(args={}):
    return {
        S: 1024,   # Sequence length
        D: 5120,   # Model dim
        F: 13824,  # Feed-forward dim
        N: 40,     # Number of query heads
        K: 40,     # Number of key/value heads
        H: 128,    # Head dimension
        L: 40,     # Number of layers
        V: 32000,  # Vocabulary size
        memory_bandwidth: 3.35e12,  # Memory bandwidth (bytes/second)
        **args
    }


def throughput_and_latency():
    bilingual_text("So we have shown that inference is memory-bound.", '因此我们已经说明，推理受内存带宽限制。')
    bilingual_text("Let us now compute the theoretical maximum latency and throughput of a single request.", '现在计算单个请求的理论最大延迟和吞吐量。')
    bilingual_text("Assumption: can overlap compute and communication perfectly and ignore overhead.", '假设：计算和通信可以完美重叠，并忽略开销。')

    bilingual_text("Instantiate latency and throughput for Llama 2 13B on an H100:", 'Instantiate 延迟 and 吞吐量 for Llama 2 13B on an H100:')
    config = llama2_13b_config()
    stats = compute_transformer_performance_stats(config)

    # Batch size 1
    b1 = stats.substitute(B, 1)  # @inspect b1 @stepover

    # Batch size 64
    b64 = stats.substitute(B, 64)  # @inspect b64 @stepover
    bilingual_text("Result: worse latency, better throughput", '结果：worse latency, better throughput')

    # Batch size 256
    b256 = stats.substitute(B, 256)  # @inspect b256 @stepover
    bilingual_text("Result: even worse latency, even better throughput", '结果：even worse latency, even better throughput')
    h100_memory = 80e9  # H100 memory in bytes
    assert b256.memory > h100_memory  # Doesn't fit in memory!
    bilingual_text("Result: doesn't fit into memory and throughput gains are diminishing too...", "结果：doesn't fit into memory and throughput gains are diminishing too...")

    bilingual_text("What increasing batch size does:", '增大批大小的作用：')
    bilingual_text("- Worsens latency because larger KV cache (O(B) size) to read/write", '- Worsens 延迟 because larger KV cache (O(B) size) to read/write')
    bilingual_text("- Improves throughput because amortizes the cost of reading parameters", '- Improves 吞吐量 because amortizes the cost of reading parameters')

    bilingual_text("**Tradeoff** between latency and throughput:", '延迟和吞吐量之间的**权衡**：')
    bilingual_text("1. Smaller batch sizes yield better latency but worse throughput", '1. Smaller 批次 sizes yield better 延迟 but worse 吞吐量')
    bilingual_text("2. Larger batch sizes yield better throughput but worse latency", '2. Larger 批次 sizes yield better 吞吐量 but worse 延迟')

    bilingual_text("Easy parallelism: if you launch M copies of the model, latency is the same, throughput increases by M!", 'Easy parallelism: if you launch M copies of the 模型, 延迟 is the same, 吞吐量 increases by M!')
    bilingual_text("Harder parallelism: shard the model and the KV cache ", 'Harder parallelism: shard the 模型 and the KV cache'), link(scaling_book_inference)

    bilingual_text("Note: time-to-first-token (TTFT) is essentially a function of prefill time", 'Note: time-to-first-token (TTFT) is essentially a function of 预填充 time')
    bilingual_text("Use smaller batch sizes during prefill for faster TTFT", 'Use smaller 批次 sizes during 预填充 for faster TTFT')
    bilingual_text("Use larger batch sizes during generation to improve throughput", 'Use larger 批次 sizes during 生成 to improve 吞吐量')


def reduce_kv_cache_size():
    bilingual_text("Recall that memory is the bottleneck for inference.", '回忆：内存是推理的瓶颈。')
    bilingual_text("So let's try to reduce the size of the KV cache", '因此我们尝试减小 KV cache 的大小。')
    bilingual_text("...but make sure we don't lose too much accuracy.", '……但要确保不要损失太多准确率。')

    bilingual_text("### Grouped-query attention (GQA) ", '### Grouped-query 注意力 (GQA)'), link(gqa_2023)
    image("https://jax-ml.github.io/scaling-book/assets/img/gmqa.png", width=800)
    bilingual_text("Idea: N query heads, but only K key and value heads, each interacting with N/K query heads", '思想：N query heads, but only K key and value heads, each interacting with N/K query heads')
    bilingual_text("Multi-headed attention (MHA): K=N", 'Multi-headed 注意力 (MHA): K=N')
    bilingual_text("Multi-query attention (MQA): K=1", 'Multi-query 注意力 (MQA): K=1')
    bilingual_text("Group-query attention (GQA): K is somewhere in between", 'Group-query 注意力 (GQA): K is somewhere in between')

    bilingual_text("Latency/throughput improves: ", '延迟/吞吐量 improves:'), link(gqa_2023)
    image("images/gqa-speed.png", width=500)

    bilingual_text("Why does GQA improve latency and throughput?", '为什么 GQA 能改善延迟和吞吐量？')
    bilingual_text("GQA reduces the KV cache by a factor of N/K.", 'GQA 将 KV cache 缩小 N/K 倍。')
    bilingual_text("Reminder: reducing memory usage leads to speedup (since we're memory-bound).", '提醒：减少内存使用会带来加速（因为我们受内存带宽限制）。')

    # Original Llama 2 13B (MHA)
    config = llama2_13b_config({K: 40, B: 64})  # @stepover
    k40_b64 = compute_transformer_performance_stats(config)  # @inspect k40_b64 @stepover

    # GQA with 1:5 ratio (K:N)
    config = llama2_13b_config({K: 8, B: 64})  # Use GQA with 1:5 ratio @stepover
    k8_b64 = compute_transformer_performance_stats(config)  # @inspect k8_b64 @stepover
    bilingual_text("Result: Worse latency, but better throughput (and it fits in memory now!)", '结果：Worse latency, but better throughput (and it fits in memory now!)')

    # Now we can increase the batch size
    config = llama2_13b_config({K: 8, B: 256})  # Increase batch size @stepover
    k8_b256 = compute_transformer_performance_stats(config)  # @inspect k8_b256 @stepover
    bilingual_text("Result: Worse latency, but better throughput (and still fits in memory!)", '结果：Worse latency, but better throughput (and still fits in memory!)')

    bilingual_text("Check that accuracy doesn't drop: ", "Check that 准确率 doesn't drop:"), link(gqa_2023)
    image("images/gqa-accuracy.png", width=800)

    bilingual_text("### Multi-head latent attention (MLA) ", '### Multi-head latent 注意力 (MLA)'), link(mla_2024)
    image("images/mla-schema.png", width=800)
    bilingual_text("Normal attention: KV cache consists of K = W_K h, V = W_V h (N*H dimensions)", 'Normal 注意力: KV cache consists of K = W_K h, V = W_V h (NH dimensions)')
    bilingual_text("MLA: store compressed vector c = W_c h (C dimensions), project up to K = W_K c, V = W_V c when needed", '说明：MLA: store compressed vector c = W_c h (C dimensions), project up to K = W_K c, V = W_V c when needed')
    bilingual_text("DeepSeek v2: reduce N*H = 16384 to C = 512", '说明：DeepSeek v2: reduce NH = 16384 to C = 512')
    bilingual_text("Wrinkle: MLA is not compatible with RoPE, so need to add additional 64 dimensions for RoPE, so 512 + 64 = 576 total dimensions", '说明：Wrinkle: MLA is not compatible with RoPE, so need to add additional 64 dimensions for RoPE, so 512 + 64 = 576 total dimensions')
    bilingual_text("Latency/throughput improvements follow similarly from the KV cache reduction as argued earlier", '延迟/吞吐量 improvements follow similarly from the KV cache reduction as argued earlier')

    bilingual_text("Let's now check the accuracy.", '现在检查准确率。')
    bilingual_text("First, MHA is better than GQA (though more expensive) [Table 8] ", '说明：First, MHA is better than GQA (though more expensive) [Table 8]'), link(mla_2024)
    image("images/mla-accuracy.png", width=800)
    bilingual_text("Second, MLA is even a bit better than MHA (and much cheaper) [Table 9] ", '说明：Second, MLA is even a bit better than MHA (and much cheaper) [Table 9]'), link(mla_2024)
    image("images/mla-accuracy2.png", width=800)

    bilingual_text("### Cross-layer attention (CLA) ", '### Cross-layer 注意力 (CLA)'), link("https://arxiv.org/abs/2405.12981")
    image("images/cla-diagram.png", width=500)
    bilingual_text("Idea: share KVs across **layers** (just as GQA shares KVs across heads)", '思想：share KVs across layers (just as GQA shares KVs across heads)')
    bilingual_text("Empirically improves the pareto frontier of accuracy and KV cache size (latency and throughput)", 'Empirically improves the pareto frontier of 准确率 and KV cache size (延迟 and 吞吐量)')
    image("images/cla-results.png", width=700)

    bilingual_text("### Local (sliding window) attention ", '### Local (sliding window) 注意力'), link(longformer_2020), link(sparse_transformer_2019), link(mistral_7b_2023)
    image("images/longformer-attention.png", width=800)
    bilingual_text("Idea: just look at the local context, which is most relevant for modeling", '思想：just look at the local context, which is most relevant for modeling')
    bilingual_text("Effective context scales linearly with the number of layers", '说明：Effective context scales linearly with the number of layers')
    bilingual_text("KV cache is independent of sequence length!", 'KV cache is independent of 序列 length!')
    bilingual_text("Problem: this can still hurt accuracy", 'Problem: this can still hurt 准确率')
    bilingual_text("Solution: interleave local attention with global attention (hybrid layers)", '解决方案：interleave local attention with global attention (hybrid layers)')

    bilingual_text("### DeepSeek v4 attention", '### DeepSeek v4 注意力')
    bilingual_text("- Supports 1M context length ", '- 说明：Supports 1M context length'),  link(deepseek_v4_2026)
    image("images/deepseek-v4-attention.png", width=800)
    bilingual_text("- Compressed Sparse Attention (CSA): compresses every m tokens into 1", '- Compressed Sparse 注意力 (CSA): compresses every m token into 1')
    bilingual_text("- DeepSeek Sparse Attention (DSA): selects the top k", '- DeepSeek Sparse 注意力 (DSA): selects the top k')
    bilingual_text("- Heavily Compressed Attention (HCA): compresses even more", '- Heavily Compressed 注意力 (HCA): compresses even more')

    bilingual_text("Summary:", '总结：')
    bilingual_text("- Goal: reduce the KV cache size (since inference is memory-bound) without hurting accuracy", '- Goal: reduce the KV cache size (since 推理 is 受内存带宽限制) without hurting 准确率')
    bilingual_text("- Lower-dimensional KV cache (GQA, MLA, CLA)", '- 说明：Lower-dimensional KV cache (GQA, MLA, CLA)')
    bilingual_text("- Local attention (truncates the KV cache) on some of the layers", '- Local 注意力 (truncates the KV cache) on some of the layers')
    bilingual_text("- Other ideas: linear attention / state-space-models (Mamba 2, GatedDeltaNet), diffusion models", '- Other ideas: linear 注意力 / state-space-模型s (Mamba 2, GatedDeltaNet), diffusion 模型s')


def quantization():
    bilingual_text("Key idea: reduce the precision of numbers", '关键思想：降低数字精度。')
    bilingual_text("Less memory means higher latency/throughput (since inference is memory-bound).", '内存更少意味着延迟/吞吐量更好（因为推理受内存带宽限制）。')
    bilingual_text("Of course we have to worry about accuracy...", '当然，我们必须担心准确率……')

    # Mechanics
    x = 5.2342  # @inspect x
    scale = 0.1
    zero_point = 4
    x_quant = round(x / scale) + zero_point  # Quantize  @inspect x_quant
    x_approx = (x_quant - zero_point) * scale  # Dequantize @inspect x_approx

    image("https://www.datocms-assets.com/104802/1709770809-twitter-post-20.png", width=400), article_link("https://www.baseten.co/blog/fp8-efficient-model-inference-with-8-bit-floating-point-numbers/")
    bilingual_text("- fp32 (4 bytes): needed for parameters and optimizer states during training", '- fp32 (4 bytes): needed for parameters and optimizer states during 训练')
    bilingual_text("- bf16 (2 bytes): default for inference", '- bf16 (2 bytes): default for 推理')
    bilingual_text("- fp8 (1 byte) [-240, 240] for e4m3 on H100s: can train if you dare ", '- 说明：fp8 (1 byte) [-240, 240] for e4m3 on H100s: can train if you dare'), link("https://arxiv.org/pdf/2310.18313")
    bilingual_text("- int8 (1 byte) [-128, 127]: less accurate but cheaper than fp8, but for inference only ", '- int8 (1 byte) [-128, 127]: less accurate but cheaper than fp8, but for 推理 only'), link("https://arxiv.org/pdf/2303.17951")
    bilingual_text("- int4 (0.5 bytes) [-8, 7]: cheaper, even less accurate ", '- 说明：int4 (0.5 bytes) [-8, 7]: cheaper, even less accurate'), link("https://arxiv.org/pdf/2303.17951")

    link(title="Overview of approaches", url="https://apxml.com/posts/llm-quantization-techniques-explained")

    bilingual_text("Quantization-aware training (QAT)", '量化感知训练（QAT）')
    bilingual_text("- During training, quantize-and-dequantize during the forward pass to simulate quantization errors", '- During 训练, quantize-and-dequantize during the forward pass to simulate 量化 errors')
    bilingual_text("- Pro: weights are trained to work with quantization", '- Pro: weights are trained to work with 量化')
    bilingual_text("- Con: requires expensive large-scale training", '- Con: requires expensive large-scale 训练')

    bilingual_text("Post-training quantization (PTQ):", '训练后量化（PTQ）：')
    bilingual_text("- Done after training, so much cheaper", '- Done after 训练, so much cheaper')
    bilingual_text("- Run on sample data to determine scale and zero point for each layer or tensor", '- Run on sample 数据 to determine scale and zero point for each layer or tensor')
    bilingual_text("- GPTQ: use Hessian information to update non-quantized weights to account for quantization error ", '- GPTQ: use Hessian information to update non-quantized weights to account for 量化 error'), link("https://arxiv.org/abs/2210.17323")

    bilingual_text("### Activation-aware quantization (AWQ)", '### 激活感知量化（AWQ）')
    link("https://arxiv.org/abs/2306.00978")
    bilingual_text("- Observation: some activation channels are large", '- 观察：some activation channels are large')
    bilingual_text("- Weights that hit those matter more", '- 说明：Weights that hit those matter more')
    bilingual_text("- Allocate more precision to those weights", '- 说明：Allocate more precision to those weights')
    bilingual_text("- Idea: select which weights (0.1-1%) to keep in high precision based on activations", '- 思想：select which weights (0.1-1%) to keep in high precision based on activations')
    bilingual_text("- fp16 → int3 produces 4x lower memory, 3.2x speedup", '- fp16 → int3 produces 4x lower 内存, 3.2x speedup')
    image("images/awq-schema.png", width=800)


def model_pruning():
    bilingual_text("Key idea: just rip out parts of an expensive model to make it cheaper", '关键思想：直接删掉昂贵模型的一部分，使它更便宜。')
    bilingual_text("...and then fix it up.", '……然后再修复它。')

    bilingual_text("Paper from NVIDIA ", '说明：Paper from NVIDIA'), link("https://arxiv.org/abs/2407.14679")
    image("images/pruning-kd-loop.png", width=600)
    bilingual_text("Algorithm:", '算法：')
    bilingual_text("1. Identify important {layer, head, hidden dimension} on a small calibration dataset (1024 samples)", '1. Identify important {layer, head, hidden dimension} on a small calibration 数据集 (1024 samples)')
    bilingual_text("2. Remove unimportant layers to get a smaller model", '2. Remove unimportant layers to get a smaller 模型')
    bilingual_text("3. Distill the original model into pruned model", '3. Distill the original 模型 into pruned 模型')

    bilingual_text("Results:", '结果：')
    image("images/pruning-kd.png", width=500)

    # TODO


def speculative_sampling():
    bilingual_text("Recall the two stages of inference:", '回忆推理的两个阶段：')
    bilingual_text("- Prefill: given a sequence, encode tokens in parallel (compute-bound) [note: also gives you probabilities]", '- 预填充: given a 序列, encode token in parallel (受计算限制) [note: also gives you 概率]')
    bilingual_text("- Generation: generate one token at a time (memory-bound)", '- 生成: generate one token at a time (受内存带宽限制)')
    bilingual_text("In other words, checking is faster than generation.", '换句话说，检查比生成更快。')

    bilingual_text("Speculative sampling ", '推测采样'), link("https://arxiv.org/abs/2211.17192"), link("https://arxiv.org/abs/2302.01318")
    bilingual_text("- Use a cheaper **draft model** p to guess a few tokens (e.g., 4)", '- Use a cheaper draft 模型 p to guess a few token (e.g., 4)')
    bilingual_text("- Evaluate with target model q (process tokens in parallel), and accept if it looks good", '- Evaluate with target 模型 q (process token in parallel), and accept if it looks good')
    link(title="Speculative sampling video", url="https://storage.googleapis.com/gweb-research2023-media/media/SpeculativeDecoding-1-Illustration.mp4")
    article_link("https://research.google/blog/looking-back-at-speculative-decoding/")

    image("images/speculative-sampling-algorithm.png", width=600)
    bilingual_text("This is modified rejection sampling with proposal p and target q", '这是修改过的拒绝采样，其中 p 是提议分布，q 是目标分布。')
    bilingual_text("Modification: always generate at least one candidate (rejection sampling will keep looping)", '说明：Modification: always generate at least one candidate (rejection sampling will keep looping)')
    bilingual_text("Key property: guaranteed to be an **exact sample** from the target model!", '关键性质：保证得到目标模型的**精确样本**！')

    bilingual_text("Proof by example: assume two vocabulary elements {A, B}", '通过例子证明：假设词表只有两个元素 {A, B}。')
    bilingual_text("- Target model probabilities: [q(A), q(B)]", '- Target 模型 概率: [q(A), q(B)]')
    bilingual_text("- Draft model probabilities: [p(A), p(B)]", '- Draft 模型 概率: [p(A), p(B)]')
    bilingual_text("- Assume p(A) > q(A) [draft model oversamples A].", '- Assume p(A) > q(A) [draft 模型 oversamples A].')
    bilingual_text("- Therefore p(B) < q(B) [draft model undersamples B].", '- Therefore p(B) < q(B) [draft 模型 undersamples B].')
    bilingual_text("- Residual probabilities max(q-p, 0): [0, 1]", '- Residual 概率 max(q-p, 0): [0, 1]')
    bilingual_text("Compute the probabilities of speculatively sampling a token:", '计算量 the 概率 of speculatively sampling a token:')
    bilingual_text("- P[sampling A] = p(A) * (q(A) / p(A)) + p(B) * 1 * 0 = q(A)", '- 说明：P[sampling A] = p(A)  (q(A) / p(A)) + p(B)  1  0 = q(A)')
    bilingual_text("- P[sampling B] = p(B) * 1 + p(A) * (1 - q(A) / p(A)) * 1 = q(B)", '- 说明：P[sampling B] = p(B)  1 + p(A)  (1 - q(A) / p(A))  1 = q(B)')

    image("images/speculative-sampling-results.png", width=600)
    image("images/speculative-sampling-stats.png", width=600)

    bilingual_text("In practice:", '实践中：')
    bilingual_text("- Target model has 70B parameters, draft model has 8B parameters", '- Target 模型 has 70B parameters, draft 模型 has 8B parameters')
    bilingual_text("- Target model has 8B parameters, draft model has 1B parameters", '- Target 模型 has 8B parameters, draft 模型 has 1B parameters')
    bilingual_text("- Try to make draft model as close to target (distillation)", '- Try to make draft 模型 as close to target (蒸馏)')

    bilingual_text("Extensions to improve the draft model:", '改进草稿模型的扩展方法：')
    bilingual_text("- Medusa: draft model generates multiple tokens in parallel ", '- Medusa: draft 模型 generates multiple token in parallel'), link("https://arxiv.org/abs/2401.10774")
    bilingual_text("- EAGLE: draft model takes high-level features from target model ", '- EAGLE: draft 模型 takes high-level features from target 模型'), link("https://arxiv.org/pdf/2401.15077")
    image("images/medusa-eagle.png", width=600)

    bilingual_text("Summary:", '总结：')
    bilingual_text("- Exact sampling from target model (thanks to math)!", '- Exact sampling from target 模型 (thanks to math)!')
    bilingual_text("- Exploits asymmetry between checking and generation", '- Exploits asymmetry between checking and 生成')
    bilingual_text("- Lots of room for innovation on the draft model (involves training)", '- Lots of room for innovation on the draft 模型 (involves 训练)')


def continuous_batching():
    link(title="Orca: A Distributed Serving System for Transformer-Based Generative Models", url="https://www.usenix.org/system/files/osdi22-yu.pdf"), link(title="talk", url="https://www.youtube.com/watch?v=Ob9PPLxETYU")

    bilingual_text("Problem:", '问题：')
    bilingual_text("- Training: get a dense block of tokens (batch size x sequence length)", '- 训练: get a dense block of token (批次 size x 序列 length)')
    bilingual_text("- Inference: requests arrive and finish at different times, so you have a ragged array", '- 推理: 请求 arrive and finish at different times, so you have a ragged array')
    image("https://images.ctfassets.net/xjan103pcp94/1LJioEsEdQQpDCxYNWirU6/82b9fbfc5b78b10c1d4508b60e72fdcf/cb_02_diagram-static-batching.png", width=600)

    bilingual_text("Solution: iteration-level scheduling", '解决方案：迭代级调度。')
    bilingual_text("- Decode step by step", '- 说明：Decode step by step')
    bilingual_text("- Add new requests to the batch as they arrive (so don't have to wait until generation completes)", "- Add new 请求 to the 批次 as they arrive (so don't have to wait until 生成 completes)")

    bilingual_text("Problem:", '问题：')
    bilingual_text("- Batching only works when all sequences have the same dimensionality (right?)", '- 批处理 only works when all 序列 have the same dimensionality (right?)')
    bilingual_text("- But each request might have a different length", '- But each 请求 might have a different length')

    bilingual_text("Solution: selective batching", '解决方案：选择性批处理。')
    bilingual_text("- Training: when all sequences of the same length, operate on a B x S x H tensor", '- 训练: when all 序列 of the same length, operate on a B x S x H tensor')
    bilingual_text("- But we might have different lengths: [3, H], [9, H], [5, H], etc.", '- 说明：But we might have different lengths: [3, H], [9, H], [5, H], etc.')
    bilingual_text("- Attention computation: process each sequence separately", '- 注意力 computation: process each 序列 separately')
    bilingual_text("- Non-attention computation: concatenate all the sequences together to [3 + 9 + 5, H]", '- Non-注意力 computation: concatenate all the 序列 together to [3 + 9 + 5, H]')


def paged_attention():
    bilingual_text("Paper that introduced vLLM in addition to PagedAttention ", 'Paper that introduced vLLM in addition to Paged注意力'), link("https://arxiv.org/pdf/2309.06180.pdf")

    bilingual_text("Previous status quo:", '之前的常见做法：')
    bilingual_text("- Request comes in", '- 请求 comes in')
    bilingual_text("- Allocate section of KV cache for prompt and response (up to a max length)", '- Allocate section of KV cache for 提示 and 回答 (up to a max length)')
    image("images/paged-attention-fragmentation.png", width=800)
    bilingual_text("Problem: fragmentation (what happens to your hard drive)", '说明：Problem: fragmentation (what happens to your hard drive)')
    bilingual_text("- But this is wasteful since we might generate much fewer tokens (internal fragmentation)!", '- 说明：But this is wasteful since we might generate much fewer tokens (internal fragmentation)!')
    bilingual_text("- Might be extra unused space between sections (external fragmentation)!", '- 说明：Might be extra unused space between sections (external fragmentation)!')

    bilingual_text("Solution: PagedAttention (remember operating systems)", '解决方案：PagedAttention（回忆操作系统）。')
    bilingual_text("- Divide the KV cache of a sequence into non-contiguous **blocks**", '- Divide the KV cache of a 序列 into non-contiguous blocks')
    image("images/paged-attention-blocks.png", width=400)

    bilingual_text("Two requests share the KV caches:", '两个请求共享 KV cache：')
    image("images/paged-attention-logical.png", width=800)

    bilingual_text("In general, multiple types of sharing KV caches across sequences:", 'In general, multiple types of sharing KV caches across 序列:')
    image("images/paged-attention-sharing.png", width=600)
    bilingual_text("- Sharing the system prompt", '- Sharing the system 提示')
    bilingual_text("- Sampling multiple responses per prompt (e.g., for program synthesis)", '- Sampling multiple 回答 per 提示 (e.g., for program synthesis)')

    bilingual_text("Solution: share prefixes, copy-on-write at the block level", '解决方案：share prefixes, copy-on-write at the block level')
    image("images/paged-attention-parallel.png", width=600)

    bilingual_text("Other vLLM optimizations:", '其他 vLLM 优化：')
    bilingual_text("- Kernel to fuse block read and attention (reduce kernel launch overhead)", '- Kernel to fuse block read and 注意力 (reduce kernel launch overhead)')
    bilingual_text("- Use latest kernels (FlashAttention, FlashDecoding)", '- Use latest kernels (Flash注意力, FlashDecoding)')
    bilingual_text("- Use CUDA graphs to avoid kernel launch overhead", '- 说明：Use CUDA graphs to avoid kernel launch overhead')

    bilingual_text("Summary: use ideas from operating systems (paging) to make use of memory for dynamic workloads", '总结：使用操作系统中的思想（分页）来为动态工作负载利用内存。')


if __name__ == "__main__":
    main()
