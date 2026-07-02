from edtrace import Reference, url_reference

# For notes
def join(*args):
    return "\n".join(args)


shannon_1950 = Reference(
    title="Prediction and Entropy of Printed English", date="1950-09-15",
    authors=["Claude Shannon"],
    url="https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf",
)

lstm_1997 = Reference(
    title="Long Short-Term Memory", date="1997",
    authors=["Sepp Hochreiter", "Jürgen Schmidhuber"],
    url="https://www.bioinf.jku.at/publications/older/2604.pdf",
)

brants_2007 = Reference(
    title="Language Models in Machine Translation", date="2007",
    authors=["Thorsten Brants", "Ashok C. Popat", "Peng Xu", "Franz J. Och", "Jeffrey Dean"],
    organization="Google",
    url="https://aclanthology.org/D07-1090.pdf",
    notes=join(
        "Trained 5-gram model on 2T tokens",
        '在 2T token 上训练了 5-gram 模型。'
    ),
)

bengio_2003 = Reference(
    title="A Neural Probabilistic Language Model", date="2003-02-01",
    authors=["Yoshua Bengio", "Réjean Ducharme", "Pascal Vincent", "Christian Jauvin"],
    url="https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf",
    notes=join('Used a feedforward neural network over last n words to predict the next word in a sequence', '使用前 n 个词上的前馈神经网络来预测序列中的下一个词。')
)

glorot_2010 = Reference(
    title="Understanding the difficulty of training deep feedforward neural networks", date="2010-03-01",
    authors=["Xavier Glorot", "Yoshua Bengio"],
    url="https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf",
)

adagrad_2011 = url_reference(
    "https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf",
    authors=["John Duchi", "Elad Hazan", "Yoram Singer"],
    date="2011",
)

seq2seq_2014 = url_reference("https://arxiv.org/pdf/1409.3215.pdf", organization="Google", notes=join('Introduced seq2seq (encode entire sentence into one vector, decode translation)', '提出 seq2seq：把整个句子编码成一个向量，再解码为译文。'))

adam_2014 = url_reference("https://arxiv.org/pdf/1412.6980.pdf", notes=join('Introduced Adam optimizer based on RMSProp and momentum', '提出基于 RMSProp 和动量的 Adam 优化器。'))

bahdanau_2015_attention = url_reference("https://arxiv.org/pdf/1409.0473.pdf", notes=join('Introduced attention mechanism (for machine translation)', '提出注意力机制（用于机器翻译）。'))

sennrich_2016 = url_reference("https://arxiv.org/abs/1508.07909")

layernorm_2016 = url_reference("https://arxiv.org/pdf/1607.06450.pdf", notes=join('Introduced LayerNorm', '提出 LayerNorm。'))

cosine_learning_rate_2017 = url_reference("https://arxiv.org/pdf/1608.03983.pdf")

transformer_2017 = url_reference("https://arxiv.org/pdf/1706.03762.pdf", organization="Google", notes=join('Introduced Transformer (for machine translation)', '提出 Transformer（用于机器翻译）。'))

adamw_2017 = url_reference("https://arxiv.org/pdf/1711.05101.pdf", notes=join(
    "Improves Adam by decoupling weight decay",
        '通过解耦权重衰减改进 Adam。',
))

ppo_2017 = url_reference("https://arxiv.org/pdf/1707.06347.pdf", notes=join('Introduced PPO (for RL)', '提出 PPO（用于强化学习）。'))

moe_2017 = url_reference("https://arxiv.org/pdf/1701.06538.pdf", organization="Google")

gpipe_2018 = url_reference("https://arxiv.org/pdf/1811.06965.pdf", organization="Google")

large_batch_training_2018 = url_reference("https://arxiv.org/pdf/1812.06162.pdf", notes=join('Introduced critical batch size', '提出临界批大小。'))

elmo_2018 = url_reference("https://arxiv.org/abs/1802.05365")
bert_2018 = url_reference("https://arxiv.org/abs/1810.04805")

sparse_transformer_2019 = url_reference("https://arxiv.org/pdf/1904.10509.pdf", organization="OpenAI", notes=join(
    "Local attention",
        '局部注意力。'
))

gpt2_2019 = Reference(
    title="Language Models are Unsupervised Multitask Learners",
    authors=["Alec Radford", "Jeffrey Wu", "Rewon Child", "David Luan", "Dario Amodei", "Ilya Sutskever"],
    organization="OpenAI", date="2019-02-14",
    url="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
    notes=join(
        "1.5B parameters",
        '15 亿参数。',
        "Pioneered stage release",
        '开创了分阶段发布方式。',
    ),
)

openwebtext_2019 = Reference(title="OpenWebText", authors=["Aaron Gokaslan", "Vanya Cohen"], date="2019", url="https://skylion007.github.io/OpenWebTextCorpus/")

t5_2019 = url_reference("https://arxiv.org/pdf/1910.10683.pdf", organization="Google", notes=join(
    "Encoder-decoder, frames tasks as text-to-text",
        '编码器-解码器架构，把任务统一表述为文本到文本。',
    "Introduced Colossal Cleaned Common Crawl (C4)",
        '提出 Colossal Cleaned Common Crawl（C4）。',
    "Filtering (Section 2.2)",
        '过滤（第 2.2 节）。',
    "11B parameters",
        '110 亿参数。',
    "Remove bias from feedforward layers",
        '移除前馈层中的 bias。',
))

megatron_lm_2019 = url_reference("https://arxiv.org/pdf/1909.08053.pdf", organization="NVIDIA")
zero_2019 = url_reference("https://arxiv.org/abs/1910.02054", organization="Microsoft", notes=join('Introduced ZeRO optimizer, can train 100B parameter model over 400 GPUs', '提出 ZeRO 优化器，可以在 400 块 GPU 上训练 100B 参数模型。'))

rms_norm_2019 = url_reference("https://arxiv.org/abs/1910.07467")

############################################################
# 2020

kaplan_scaling_laws_2020 = url_reference("https://arxiv.org/pdf/2001.08361.pdf", organization="OpenAI", notes=join(
    "Vary model size, dataset size, compute; get power laws",
        '改变模型大小、数据集大小和计算量，得到幂律关系。',
    "Larger models require fewer tokens",
        '更大的模型需要更少 token。',
))

shazeer_2020 = url_reference("https://arxiv.org/pdf/2002.05202.pdf", organization="Google", notes=join(
    "Experiments with different activation functions",
        '实验比较不同激活函数。',
    "Activation functions: ReLU, GeLU, Swish",
        '激活函数：ReLU、GeLU、Swish。',
    "Apply idea of gated units (GLU): ReGLU, GeGLU, SwiGLU",
        '应用门控单元（GLU）的思想：ReGLU、GeGLU、SwiGLU。',
    "FFN-SwiGLU = Swish(x W1) * xV W2",
        '公式保持：FFN-SwiGLU = Swish(x W1) * xV W2。',
    "Have 3 matrices now, so make hidden dimension 2/3 of the 2 matrix version",
        '现在有 3 个矩阵，因此把隐藏维度设为双矩阵版本的 2/3。',
))

pre_post_norm_2020 = url_reference("https://arxiv.org/pdf/2002.04745.pdf")

linear_attention_2020 = url_reference("https://arxiv.org/abs/2006.16236")

longformer_2020 = url_reference("https://arxiv.org/pdf/2004.05150.pdf", organization="AllenAI", notes=join(
    "Sliding window (local) attention",
        '滑动窗口（局部）注意力。',
    "Global attention to capture task-specific information",
        '使用全局注意力捕获任务特定信息。',
))

gpt_3_2020 = url_reference("https://arxiv.org/pdf/2005.14165.pdf", organization="OpenAI", notes=join(
    "Introduces GPT-3",
        '介绍 GPT-3。',
    "Same as GPT-2, but alternating sparse and dense attention layers",
        '与 GPT-2 类似，但交替使用稀疏和稠密注意力层。',
    "175B parameters",
        '175B 参数。',
    "Data: 300B tokens",
        '数据：300B token。',
))

the_pile_2020 = url_reference("https://arxiv.org/pdf/2101.00027.pdf", organization="EleutherAI", notes=join(
    "825GB text, 22 diverse subsets (CommonCrawl, PubMed, ArXiv, GitHub, StackExchange, USPTO, OpenWebText2, Books3, etc.)",
        '825GB 文本，包含 22 个多样子集（CommonCrawl、PubMed、ArXiv、GitHub、StackExchange、USPTO、OpenWebText2、Books3 等）。',
))

############################################################
# 2021

mmlu_2021 = url_reference("https://arxiv.org/pdf/2009.03300.pdf", organization="Berkeley", notes=join(
    "57 subjects, multiple-choice",
        '57 个科目，多项选择。',
))

rope_2021 = url_reference("https://arxiv.org/pdf/2104.09864.pdf", notes=join(
    "Encodes absolute position with rotation matrix, incorporate relative position dependency in self-attention",
        '用旋转矩阵编码绝对位置，并在自注意力中纳入相对位置依赖。',
    "Key: R W x, where R is a block-diagonal sequence of d/2 rotation matrices (equation 13)",
        '关键：R W x，其中 R 是由 d/2 个旋转矩阵组成的块对角序列（公式 13）。',
    "Extrapolates to longer sequences",
        '可外推到更长序列。',
))

megatron_parallelism_2021 = url_reference("https://arxiv.org/pdf/2104.04473.pdf", organization="NVIDIA", notes=join(
    "Compose tensor, pipeline, data parallelism",
        '组合张量并行、流水线并行和数据并行。',
    "Achieve 52% MFU on 1T parameter model on 3072 GPUs",
        '在 3072 块 GPU 上训练 1T 参数模型时达到 52% MFU。',
))

byt5_2021 = url_reference("https://arxiv.org/abs/2105.13626")

gpt_j_2021 = Reference(
    title="GPT-J", organization="EleutherAI", date="2021-06-04",
    authors=["Ben Wang", "Aran Komatsuzaki"],
    url="https://arankomatsuzaki.wordpress.com/2021/06/04/gpt-j/",
    notes=join(
        "6.7B parameters",
        '67 亿参数。',
        "Attention and feedforward layers put in parallel",
        '注意力层和前馈层并行放置。',
        "v3 256 TPUs (5.4 PFLOPs) for 5 weeks",
        '使用 v3 256 TPU（5.4 PFLOPs）训练 5 周。',
    ),
)

gopher_2021 = url_reference("https://arxiv.org/pdf/2112.11446.pdf", organization="DeepMind", notes=join(
    "280B parameters",
        '280B 参数。',
    "Data: 300B tokens",
        '数据：300B token。',
))

switch_transformers_2021 = url_reference("https://arxiv.org/abs/2101.03961", organization="Google")

############################################################
# 2022

instruct_gpt_2022 = url_reference("https://arxiv.org/pdf/2203.02155.pdf", organization="OpenAI", notes=join(
    "Training language models to follow instructions with human feedback",
        '用人类反馈训练语言模型遵循指令。',
))

chinchilla_2022 = url_reference("https://arxiv.org/pdf/2203.15556.pdf", organization="DeepMind", notes=join(
    "Introduced the rigorous analysis scaling laws for language models",
        '提出了 the rigorous analysis 规模定律 for language 模型s',
    "Key improvement over Kaplan: tune learning rate for the compute budget",
        'Key improvement over Kaplan: tune 学习率 for the 计算预算',
    "Approach 1: for each model size, train with 4 learning rates, vary number of training tokens, fit lower envelope",
        'Approach 1: for each 模型 size, train with 4 学习率s, vary number of 训练 token, fit lower envelope',
    "Approach 2 (IsoFLOP): for each model size, train with 9 training budgets, take last point",
        'Approach 2 (IsoFLOP): for each 模型 size, train with 9 训练 budgets, take last point',
    "Approach 3: fit parametric function L(N, D) = E + A/N^alpha + B/D^beta to data collected from approaches 1 and 2",
        'Approach 3: fit parametric function L(N, D) = E + A/N^alpha + B/D^beta to 数据 collected from approaches 1 and 2',
    "Conclusion: model and data should scale up at same rate",
        '结论：模型和数据应以相同速率扩展。',
    "Table 3: extrapolate to 10 trillion parameters",
        'Table 3: extrapolate to 10 trillion 参数',
    "MassiveText, different data distribution (1.5 trillion tokens)",
        'MassiveText, different 数据 distribution (1.5 trillion token)',
    "70B parameters",
        '70B 参数。',
))

palm_2022 = url_reference("https://arxiv.org/pdf/2204.02311.pdf", organization="Google", notes=join(
    "Data: Social media conversations, webpages, books, GitHub, Wikipedia, news",
        '数据: Social media conversations, webpages, books, GitHub, Wikipedia, news',
    "540B parameters",
        '540B 参数',
    "SwiGLU, parallelize attention and feedforward layers, multi-query attention, RoPE, remove biases",
        'SwiGLU, 并行ize 注意力 and 前馈 layers, multi-query 注意力, RoPE, remove biases',
    "hardware: 6144 TPUv4, 46.2% MFU",
        '硬件: 6144 TPUv4, 46.2% MFU',
    "optimizer: Adafactor without factorization",
        '优化器: Adafactor without factorization',
    "Introduced the term model FLOPs utilization (MFU) metric (observed tokens/sec / theoretical max tokens/sec)",
        '提出了 the term 模型 FLOPs utilization (MFU) metric (observed token/sec / theoretical max token/sec)',
))

gpt_neox_2022 = url_reference("https://arxiv.org/pdf/2204.06745.pdf", organization="EleutherAI", notes=join(
    "Data: The Pile",
        '数据：The Pile。',
    "20B parameters",
        '20B 参数。',
    "Use RoPE, parallel attention and feedforward layers (15% throughput increase)",
        '使用 RoPE，并行化注意力层和前馈层（吞吐量提升 15%）。',
    "hardware: 12x8 A100s",
        '硬件：12x8 A100。',
))

opt_175b_2022 = url_reference("https://arxiv.org/pdf/2205.01068.pdf", organization="Meta", notes=join(
    "Data: The Pile, PushShift.io Reddit, deduplication",
        '数据: The Pile, PushShift.io Reddit, 去重',
    "175B parameters",
        '175B 参数。',
    "hardware: 992 A100 80GB for 2 months, lots of hardware failures",
        '硬件: 992 A100 80GB for 2 months, lots of 硬件 failures',
    "FSDP with Megatron-LM, fp16 with loss scaling",
        '说明：FSDP with Megatron-LM, fp16 with loss scaling',
))

bloom_2022 = url_reference("https://arxiv.org/abs/2211.05100", organization="BigScience", notes=join(
    "Model: BLOOM (176B parameters)",
        '模型：BLOOM（176B 参数）。',
    "Data: ROOTS",
        '数据：ROOTS。',
    "Hardware: 48x8 A100s on Jean Zay supercomputer for 3.5 months",
        '硬件：Jean Zay 超算上的 48x8 A100，训练 3.5 个月。',
    "ZeRO stage 1",
        'ZeRO stage 1。',
))

bahdanau_training_costs_2022 = Reference(
    title="The FLOPs Calculus of Language Model Training", authors=["Dzmitry Bahdanau"], date="2022-01-09",
    url="https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4",
)

mup_2022 = url_reference("https://arxiv.org/abs/2203.03466")

############################################################
# 2023

llama_2023 = url_reference("https://arxiv.org/pdf/2302.13971.pdf", organization="Meta", notes=join(
    "Train only on open data (detailed recipe that is replicated by RedPajama)",
        'Train only on open 数据 (detailed recipe that is replicated by RedPajama)',
    "Optimize for fast inference at 7B",
        'Optimize for fast 推理 at 7B',
    "Data: CommonCrawl, C4, GitHub, Wikipedia, Books, ArXiv, StackExchange",
        '数据: CommonCrawl, C4, GitHub, Wikipedia, Books, ArXiv, StackExchange',
    "Architecture: Pre-norm, SwiGLU, RoPE",
        '说明：Architecture: Pre-norm, SwiGLU, RoPE',
    "Training: 2048 A100 80GB for 21 days",
        '说明：Training: 2048 A100 80GB for 21 days',
))

gpt_4_2023 = url_reference("https://arxiv.org/pdf/2303.08774.pdf", organization="OpenAI", notes=join(
    "No details on the data or model architecture.",
        '没有披露数据或模型架构细节。',
))

alpaca_2023 = Reference(title="Alpaca", authors=["Rohan Taori", "Ishaan Gulrajani", "Tianyi Zhang", "Yann Dubois", "Xuechen Li", "Carlos Guestrin", "Percy Liang", "Tatsunori B. Hashimoto"], date="2023-03-13", url="https://crfm.stanford.edu/2023/03/13/alpaca.html")

transformer_math_2023 = Reference(
    title="Transformer Math 101", organization="EleutherAI", date="2023-04-03",
    url="https://blog.eleuther.ai/transformer-math/",
)

gqa_2023 = url_reference("https://arxiv.org/pdf/2305.13245.pdf", organization="Google", notes=join(
    "Multi-query attention (MQA) speeds up, but less expressive",
        'Multi-query 注意力 (MQA) speeds up, but less expressive',
    "GQA: use an intermediate (more than one, less than number of heads) number of key-value heads",
        '说明：GQA: use an intermediate (more than one, less than number of heads) number of key-value heads',
    "Experiments on T5",
        '说明：Experiments on T5',
))

lima_2023 = url_reference("https://arxiv.org/pdf/2305.11206.pdf")

megabyte_2023 = url_reference("https://arxiv.org/pdf/2305.07185.pdf")

dpo_2023 = url_reference("https://arxiv.org/pdf/2305.18290.pdf")

llama_2_2023 = url_reference("https://arxiv.org/pdf/2307.09288.pdf", organization="Meta", notes=join(
    "2T tokens",
        '说明：2T tokens',
    "70B parameters",
        '70B 参数。',
))

mistral_7b_2023 = url_reference("https://arxiv.org/pdf/2310.06825.pdf", organization="Mistral", notes=join(
    "GQA, sliding window attention",
        'GQA, sliding window 注意力',
))

qk_norm_2023 = url_reference("https://arxiv.org/abs/2302.05442")

############################################################
# 2024

llama_3_2024 = url_reference("https://arxiv.org/abs/2407.21783", organization="Meta", notes=join(
    "15T tokens",
        '说明：15T tokens',
    "405B parameters",
        '405B 参数',
))

deepseek_67b_2024 = url_reference("https://arxiv.org/pdf/2401.02954.pdf", organization="DeepSeek", notes=join(
    "Data: DeepSeek, The Stack, Reddit, etc. (2T tokens)",
        '数据: DeepSeek, The Stack, Reddit, etc. (2T token)',
    "Architecture: LLaMA, but: for GQA increased depth, 67B parameters",
        'Architecture: LLaMA, but: for GQA increased depth, 67B 参数',
    "Scaling laws: used non-embedding FLOPs with IsoFLOP",
        '说明：Scaling laws: used non-embedding FLOPs with IsoFLOP',
))

mixtral_2024 = url_reference("https://arxiv.org/pdf/2401.04088.pdf", organization="Mistral")

olmo_7b_2024 = url_reference("https://arxiv.org/pdf/2402.00838.pdf", organization="AI2", notes=join(
    "Data: subset of Dolma (2.46T tokens, CommonCrawl, The Stack, Reddit, etc.)",
        '数据: subset of Dolma (2.46T token, CommonCrawl, The Stack, Reddit, etc.)',
    "Architecture: no biases, non-parametric layer norm, SwiGLU (8/3 d increased to closest multiple of 128)",
        '说明：Architecture: no biases, non-parametric layer norm, SwiGLU (8/3 d increased to closest multiple of 128)',
    "Training: 256x4 AMD MI250X on LUMI supercomputer, 27x8 A100s, 800Gbps interconnect",
        '说明：Training: 256x4 AMD MI250X on LUMI supercomputer, 27x8 A100s, 800Gbps interconnect',
))

dolma_2024 = url_reference("https://arxiv.org/abs/2402.00159")

deepseek_math_2024 = grpo = url_reference("https://arxiv.org/pdf/2402.03300.pdf")

megascale_2024 = url_reference("https://arxiv.org/pdf/2402.15627.pdf", organization="Bytedance", notes=join(
    "55.2% MFU for 175B parameter model over 12,288 GPUs",
        '55.2% MFU for 175B 参数 模型 over 12,288 GPUs',
    "Combine data, tensor, pipeline, sequence parallelism",
        'Combine 数据, tensor, pipeline, sequence 并行ism',
    "Parallelize attention and feedforward layers, sliding window attention, LAMB optimizer",
        'Parallelize 注意力 and 前馈 layers, sliding window 注意力, LAMB 优化器',
))

nemotron_15b_2024 = url_reference("https://arxiv.org/pdf/2402.16819.pdf", organization="NVIDIA", notes=join(
    "Data: 8T tokens, 70% English, 15% multilingual, 15% code",
        '数据: 8T token, 70% English, 15% multilingual, 15% code',
    "Architecture: RoPE, squared ReLU activations, no bias, no dropout, GQA (15B parameters)",
        'Architecture: RoPE, squared ReLU 激活s, no bias, no dropout, GQA (15B 参数)',
    "Training: 384x8 H100s, After 8T tokens, train on higher quality sources + benchmark tasks",
        '说明：Training: 384x8 H100s, After 8T tokens, train on higher quality sources + benchmark tasks',
))

yi_34b_2024 = url_reference("https://arxiv.org/pdf/2403.04652.pdf", organization="01.AI")

gemma_2024 = url_reference("https://arxiv.org/pdf/2403.08295.pdf", organization="Google DeepMind", notes=join(
    "6T tokens",
        '说明：6T tokens',
    "MQA, RoPE, GeGLU, RMSNorm",
        '说明：MQA, RoPE, GeGLU, RMSNorm',
    "Training: 4096 v4 TPUs, Use ZeRO-3 like techniques",
        '说明：Training: 4096 v4 TPUs, Use ZeRO-3 like techniques',
))

overtrained_scaling_laws_2024 = url_reference("https://arxiv.org/pdf/2403.08540.pdf", notes=join(
    "Chinchilla scaling laws focus on loss of the trained model, ignoring inference costs.",
        'Chinchilla 规模定律 focus on loss of the trained 模型, ignoring 推理 costs.',
    "Constant ratio of training tokens to parameters",
        'Constant ratio of 训练 token to 参数',
    "Extrapolate over 300x training compute to 1.4B model on 900B tokens",
        'Extrapolate over 300x 训练 compute to 1.4B 模型 on 900B token',
    "Look at task performance rather than validation loss",
        '说明：Look at task performance rather than validation loss',
))

minicpm_2024 = wsd_2024 = url_reference("https://arxiv.org/pdf/2404.06395.pdf", organization="Tsinghua")

deepseek_v2_2024 = mla_2024 = url_reference("https://arxiv.org/abs/2405.04434")

mamba_2_2024 = url_reference("https://arxiv.org/abs/2405.21060")

dclm_2024 = url_reference("https://arxiv.org/abs/2406.11794")

tfree_2024 = url_reference("https://arxiv.org/abs/2406.19223")

auxfree_2024 = url_reference("https://arxiv.org/abs/2408.15664")
mtp_2024 = url_reference("https://arxiv.org/abs/2404.19737")

soap_2024 = url_reference("https://arxiv.org/abs/2409.11321")

huanyuan_large_2024 = url_reference("https://arxiv.org/abs/2411.02265", organization="Tencent")

qwen_2_5_2024 = url_reference("https://arxiv.org/abs/2412.15115", organization="Alibaba")

blt_2024 = url_reference("https://arxiv.org/abs/2412.09871")

nemotron_cc_2024 = url_reference("https://arxiv.org/abs/2412.02595")

gdn_2024 = url_reference("https://arxiv.org/abs/2412.06464")

deepseek_v3_2024 = url_reference("https://arxiv.org/pdf/2412.19437.pdf")

muon_2024 = Reference(
    title="Muon: An optimizer for hidden layers in neural networks", date="2024-12-08",
    authors=["Jordan Keller"],
    url="https://kellerjordan.github.io/posts/muon/"
)

wrap_2024 = url_reference("https://arxiv.org/abs/2401.16380")

############################################################
# 2025

olmo_2_2025 = url_reference("https://arxiv.org/abs/2501.00656", organization="AI2")

deepseek_r1_2025 = url_reference("https://arxiv.org/pdf/2501.12948.pdf")

kimi_1_5_2025 = url_reference("https://arxiv.org/pdf/2501.12599.pdf")

smollm_2_2025 = url_reference("https://arxiv.org/pdf/2502.02737.pdf", notes=join(
    "1.7B parameter model",
        '1.7B 参数 模型',
    "Introduces FineMath, StackEdu",
        '说明：Introduces FineMath, StackEdu',
))

llama_4_2025 = Reference(title="Llama 4", organization="Meta", url="https://ai.meta.com/blog/llama-4-multimodal-intelligence/", date="2025-04-05")

olmo_2_32b_2025 = Reference(title="OLMo 2 (32B)", organization="AI2", url="https://allenai.org/blog/olmo2-32B")

qwen_3_2025 = url_reference("https://arxiv.org/abs/2505.09388")

glm_4_5_2025 = url_reference("https://arxiv.org/abs/2508.06471", organization="Z.ai")

longcat_flash_2025 = url_reference("https://arxiv.org/abs/2509.01322")

ling_2_2025 = url_reference("https://arxiv.org/abs/2510.22115")

hnet_2025 = url_reference("https://arxiv.org/abs/2507.07955")

marin_8b_2025 = url_reference("https://marin.readthedocs.io/en/latest/reports/marin-8b-retro/", organization="Marin", title="[Marin 8B retro]", date="2025")
marin_32b_2025 = url_reference("https://marin.readthedocs.io/en/latest/reports/marin-32b-retro/", organization="Marin", title="[Marin 32B retro]", date="2025")

olmo_3_2025 = url_reference("https://arxiv.org/abs/2512.13961")
nemotron_3_2025 = url_reference("https://arxiv.org/abs/2512.20856")
deepseek_v3_2_2025 = url_reference("https://arxiv.org/abs/2512.02556")

regmix_2025 = url_reference("https://arxiv.org/abs/2407.01492")

############################################################
# 2026

kimi_k2_5_2026 = url_reference("https://arxiv.org/abs/2602.02276", organization="Moonshot")
glm_5_2026 = url_reference("https://arxiv.org/abs/2602.15763", organization="Z.ai")
arcee_trinity_2026 = url_reference("https://arxiv.org/abs/2602.17004")

qwen_3_5_2026 = url_reference("https://qwen.ai/blog?id=qwen3.5")
minimax_m2_5_2026 = url_reference("https://www.minimax.io/news/minimax-m25", organization="Minimax", title="[MiniMax M2.5]", date="2026")
xiaomi_mimo_v2_2026 = url_reference("https://mimo.xiaomi.com/mimo-v2-pro", organization="Xiaomi", title="[Xiaomi MIMO v2]", date="2026")


mamba_3_2026 = url_reference("https://arxiv.org/abs/2603.15569")

olmix_2026 = url_reference("https://arxiv.org/abs/2602.12237")

nemotron_3_super_2026 = Reference(
    title="Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning",
    organization="NVIDIA",
    url="https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf",
    date="2026-03-11",
)

deepseek_v4_2026 = Reference(
    title="DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence",
    organization="DeepSeek",
    url="https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf",
    date="2026-04",
)