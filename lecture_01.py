import os

import regex
from abc import ABC
from dataclasses import dataclass
from collections import defaultdict
from edtrace import link, text, image
from lecture_util import article_link, post_link, video_link, get_local_url, bilingual_text
from references import shannon_1950, lstm_1997, brants_2007, bengio_2003, glorot_2010, seq2seq_2014
from references import bahdanau_2015_attention, transformer_2017, gpt2_2019, t5_2019, kaplan_scaling_laws_2020, mup_2022
from references import dpo_2023, adamw_2017, adam_2014, grpo, ppo_2017, muon_2024
from references import large_batch_training_2018, wsd_2024, cosine_learning_rate_2017, moe_2017, switch_transformers_2021, auxfree_2024, mtp_2024
from references import megatron_lm_2019, shazeer_2020, elmo_2018, bert_2018
from references import rms_norm_2019, layernorm_2016, pre_post_norm_2020, qk_norm_2023
from references import rope_2021, soap_2024, sparse_transformer_2019, gqa_2023, mla_2024
from references import linear_attention_2020, mamba_2_2024, gdn_2024, mamba_3_2026
from references import megabyte_2023, byt5_2021, blt_2024, tfree_2024, hnet_2025, sennrich_2016, zero_2019, gpipe_2018
from references import regmix_2025, olmix_2026, wrap_2024

from references import gpt_3_2020, gpt_4_2023, instruct_gpt_2022
from references import the_pile_2020, gpt_j_2021, opt_175b_2022, bloom_2022, palm_2022, chinchilla_2022
from references import llama_2023, llama_2_2023, llama_3_2024
from references import mistral_7b_2023, mixtral_2024
from references import deepseek_67b_2024, deepseek_v2_2024, deepseek_v3_2024, deepseek_r1_2025, deepseek_v3_2_2025
from references import qwen_2_5_2024, qwen_3_2025
from references import kimi_1_5_2025, kimi_k2_5_2026
from references import glm_4_5_2025, glm_5_2026
from references import minimax_m2_5_2026
from references import xiaomi_mimo_v2_2026

from references import marin_8b_2025, marin_32b_2025
from references import olmo_7b_2024, olmo_2_2025, olmo_3_2025
from references import nemotron_15b_2024, nemotron_3_2025

import tiktoken

def main():
    welcome()
    why_this_course_exists()
    current_lm_landscape()

    what_is_this_program()

    course_logistics()
    course_syllabus()

    tokenization()  # First unit

    bilingual_text("Next time: resource accounting", '下次：资源核算。')


def welcome():
    bilingual_text("## CS336: Language Models From Scratch (Spring 2026)", '## CS336：从零开始构建语言模型（2026 春季）'),

    image("images/course-staff.png", width=600)
    bilingual_text("...bringing you the 3rd offering of CS336.", '这是 CS336 第三次开课。')

    bilingual_text("Lectures from 2nd offering (Spring 2025) are on [YouTube](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_).", '第二次开课（2025 春季）的课程录像在 [YouTube](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_) 上。')
    bilingual_text("What's new?", '有哪些新内容？')
    bilingual_text("- Same 'from scratch' philosophy", '- 仍然坚持“从零开始”的理念。')
    bilingual_text("- Prioritize high value-per-time concepts, don't lose the forest for the trees", '- 优先讲单位时间价值最高的概念，不因细节而看不见整体。')
    bilingual_text("- More coverage of modern LM ingredients (mixture of experts, long-context, agents)", '- 更多覆盖现代语言模型要素（专家混合、长上下文、智能体）。')


def why_this_course_exists():
    bilingual_text("## Why did we make this course?", '## 我们为什么开设这门课？')

    bilingual_text("Problem: researchers are becoming **disconnected** from the underlying technology.", '问题：研究者正在与底层技术逐渐**脱节**。')
    bilingual_text("- 2016: researchers implemented and trained their own models.", '- 2016 年：研究者自己实现并训练模型。')
    bilingual_text("- 2018: researchers downloaded models (e.g., BERT) and fine-tuned them.", '- 2018 年：研究者下载模型（例如 BERT）并进行微调。')
    bilingual_text("- Today: researchers prompt API models (e.g., GPT/Claude/Gemini).", '- 今天：研究者通过提示词调用 API 模型（例如 GPT/Claude/Gemini）。')

    bilingual_text("Moving up levels of abstraction boosts productivity, but", '抽象层级上移会提升生产力，但是：')
    bilingual_text("- These abstractions are leaky (in contrast to programming languages or operating systems).", '- 这些抽象并不严密，会泄漏底层细节（这不同于编程语言或操作系统）。')
    bilingual_text("- There is still fundamental research to be done that requires tearing up the stack.", '- 仍有一些基础研究需要拆开整个技术栈才能完成。')

    bilingual_text("**Full understanding** of this technology is necessary for **fundamental research**.", '要做**基础研究**，必须对这项技术有**完整理解**。')

    bilingual_text("Philosophy of this course: **understanding via building**.", '这门课的理念是：通过构建来理解。')
    bilingual_text("But there's one small problem...", '但这里有一个小问题……')

    bilingual_text("## The industrialization of language models", '## 语言模型的工业化')
    image("https://upload.wikimedia.org/wikipedia/commons/c/cc/Industrialisation.jpg", width=400)

    bilingual_text("Frontier models are really expensive:", '前沿模型非常昂贵：')
    bilingual_text("- 2023: GPT-4 supposedly cost $100M to train. ", '- 2023 年：据称 GPT-4 的训练成本为 1 亿美元。'), article_link("https://www.wired.com/story/openai-ceo-sam-altman-the-age-of-giant-ai-models-is-already-over/")
    bilingual_text("- 2025: xAI builds cluster with 230K GPUs for training Grok. ", '- 2025 年：xAI 建设了包含 23 万块 GPU 的集群来训练 Grok。'), article_link("https://x.com/elonmusk/status/1947701807389515912")

    bilingual_text("There are no public details on how frontier models are built.", '关于前沿模型如何构建，目前没有公开的完整细节。')
    bilingual_text("From the GPT-4 technical report ", '来自 GPT-4 技术报告'), link(gpt_4_2023), bilingual_text(":", '中文标点：')
    image("images/gpt4-no-details.png", width=600)

    bilingual_text("Frontier models are out of reach for us.", '前沿模型超出了我们这门课可以直接触及的范围。')
    bilingual_text("We could build small language models (<1B parameters), but this might not be representative of large language models.", '我们可以构建小型语言模型（少于 10 亿参数），但它们未必能代表大语言模型的行为。')

    bilingual_text("Example 1: fraction of FLOPs spent in attention versus MLP changes with scale. ", '示例 1：花在注意力与 MLP 上的 FLOPs 占比会随规模改变。'), post_link("https://x.com/stephenroller/status/1579993017234382849")
    image("images/roller-flops.png", width=400)
    bilingual_text("Example 2: emergence of behavior with scale ", '示例 2：行为会随规模出现涌现。'), link("https://arxiv.org/pdf/2206.07682")
    image("images/wei-emergence-plot.png", width=600)

    bilingual_text("## What can we learn in this class that transfers to frontier models?", '## 这门课中哪些知识可以迁移到前沿模型？')
    bilingual_text("There are three types of knowledge:", '这里有三类知识：')
    bilingual_text("- **Mechanics**: how things work (what a Transformer is, how model parallelism works)", '- **机制**：事物如何工作（例如 Transformer 是什么、模型并行如何工作）。')
    bilingual_text("- **Mindset**: squeezing the most out of the hardware, taking scaling seriously", '- **思维方式**：尽可能榨干硬件性能，并严肃对待规模化。')
    bilingual_text("- **Intuitions**: which data and modeling decisions yield good accuracy", '- **直觉**：哪些数据和建模决策会带来更好的准确率。')

    bilingual_text("We can teach mechanics and mindset (these do transfer).", '我们可以教授机制和思维方式（它们确实可以迁移）。')
    bilingual_text("We can only partially teach intuitions (do not necessarily transfer across scales).", '我们只能部分教授直觉（它们不一定能跨规模迁移）。')

    bilingual_text("## Intuitions? 🤷", '## 直觉？🤷')
    bilingual_text("Some design decisions are simply not (yet) justifiable and just come from experimentation.", '有些设计决策目前还无法充分解释，只能来自实验经验。')
    bilingual_text("Example: Noam Shazeer paper that introduced SwiGLU ", '示例：Noam Shazeer 提出 SwiGLU 的论文。'), link(shazeer_2020)
    image("images/divine-benevolence.png", width=600)

    bilingual_text("## The bitter lesson", '## 苦涩的教训')
    bilingual_text("Wrong interpretation: scale is all that matters, algorithms don't matter.", '错误解读：规模就是一切，算法并不重要。')
    bilingual_text("Right interpretation: algorithms that scale are what matter.", '正确解读：真正重要的是能够随规模扩展的算法。')
    bilingual_text("### accuracy = efficiency x resources", '### 准确率 = 效率 x 资源')
    bilingual_text("In fact, efficiency is way more important at larger scales (can't afford to be wasteful).", '事实上，在更大规模下效率更加重要，因为浪费不起。')
    link("https://arxiv.org/abs/2005.04305"), bilingual_text(" showed 44x algorithmic efficiency on ImageNet between 2012 and 2019.", '展示了 2012 到 2019 年间 ImageNet 上算法效率提升了 44 倍。')

    bilingual_text("Framing: what is the best model one can build given a certain compute and data budget?", '问题框架：在给定计算量和数据预算时，能构建出的最佳模型是什么？')
    bilingual_text("In other words, **maximize efficiency**!", '换句话说，就是要**最大化效率**！')


def current_lm_landscape():
    bilingual_text("## Pre-neural (before 2010s)", '## 神经网络之前（2010 年代以前）')
    bilingual_text("- Language model to measure the entropy of English ", '- 用语言模型度量英语的熵。'), link(shannon_1950)
    bilingual_text("- N-gram language models (used in machine translation and speech recognition systems) ", '- N-gram 语言模型（用于机器翻译和语音识别系统）。'), link(brants_2007)

    bilingual_text("## Neural ingredients (2010s)", '## 神经网络要素（2010 年代）')
    bilingual_text("- Long-Short Term Memory (LSTM) ", '- 长短期记忆网络（LSTM）。'), link(lstm_1997)
    bilingual_text("- First neural language model ", '- 第一个神经语言模型。'), link(bengio_2003)
    bilingual_text("- Sequence-to-sequence modeling (for machine translation) ", '- 序列到序列建模（用于机器翻译）。'), link(seq2seq_2014)
    bilingual_text("- Adam optimizer ", '- Adam 优化器。'), link(adam_2014)
    bilingual_text("- Attention mechanism (for machine translation) ", '- 注意力机制（用于机器翻译）。'), link(bahdanau_2015_attention)
    bilingual_text("- Transformer architecture (for machine translation) ", '- Transformer 架构（用于机器翻译）。'), link(transformer_2017)
    bilingual_text("- Mixture of experts ", '- 专家混合（Mixture of experts）。'), link(moe_2017)
    bilingual_text("- Model parallelism ", '- 模型并行。'), link(gpipe_2018), link(zero_2019), link(megatron_lm_2019)

    bilingual_text("## Early foundation models (late 2010s)", '## 早期基础模型（2010 年代后期）')
    bilingual_text("- ELMo: pretraining with LSTMs, fine-tuning improves downstream tasks ", '- ELMo：使用 LSTM 预训练，微调可提升下游任务表现。'), link(elmo_2018)
    bilingual_text("- BERT: pretraining with Transformer, fine-tuning improves downstream tasks ", '- BERT：使用 Transformer 预训练，微调可提升下游任务表现。'), link(bert_2018)
    bilingual_text("- Google's T5 (11B): cast everything as text-to-text ", '- Google 的 T5（11B）：把所有任务都表示为文本到文本。'), link(t5_2019)

    bilingual_text("## Embracing scaling", '## 拥抱规模化')
    bilingual_text("- OpenAI's GPT-2 (1.5B): fluent text, first signs of zero-shot ", '- OpenAI 的 GPT-2（1.5B）：文本流畅，出现零样本能力的早期迹象。'), link(gpt2_2019)
    bilingual_text("- Scaling laws: provide hope / predictability for scaling ", '- 规模定律：为规模化提供希望和可预测性。'), link(kaplan_scaling_laws_2020)
    bilingual_text("- OpenAI's GPT-3 (175B): in-context learning ", '- OpenAI 的 GPT-3（175B）：上下文学习。'), link(gpt_3_2020)
    bilingual_text("- Google's PaLM (540B): massive scale, undertrained ", '- Google 的 PaLM（540B）：规模巨大，但训练不足。'), link(palm_2022)
    bilingual_text("- DeepMind's Chinchilla (70B): compute-optimal scaling laws ", '- DeepMind 的 Chinchilla（70B）：计算最优的规模定律。'), link(chinchilla_2022)

    bilingual_text("## Open models", '## 开放模型')
    bilingual_text("Early attempts (attempts to replicate GPT-3):", '早期尝试（试图复现 GPT-3）：')
    bilingual_text("- EleutherAI's open datasets (The Pile) and models (GPT-J) ", '- EleutherAI 的开放数据集（The Pile）和模型（GPT-J）。'), link(the_pile_2020), link(gpt_j_2021)
    bilingual_text("- Meta's OPT (175B): GPT-3 replication, lots of hardware issues ", '- Meta 的 OPT（175B）：复现 GPT-3，遇到大量硬件问题。'), link(opt_175b_2022)
    bilingual_text("- Hugging Face / BigScience's BLOOM (176B): focused on data sourcing ", '- Hugging Face / BigScience 的 BLOOM（176B）：重点关注数据来源。'), link(bloom_2022)

    bilingual_text("Credible open-weight models (weights + paper):", '可信的开放权重模型（权重 + 论文）：')
    bilingual_text("- Meta's Llama models ", '- Meta 的 Llama 模型。'), link(llama_2023), link(llama_2_2023), link(llama_3_2024)
    bilingual_text('- Mistral\'s models ', '- Mistral 的模型。'), link(mistral_7b_2023), link(mixtral_2024)
    bilingual_text("- DeepSeek\'s models ", '- DeepSeek 的模型。'), link(deepseek_67b_2024), link(deepseek_v2_2024), link(deepseek_v3_2024), link(deepseek_r1_2025), link(deepseek_v3_2_2025)
    bilingual_text("- Alibaba\'s Qwen models ", '- 阿里巴巴的 Qwen 模型。'), link(qwen_2_5_2024), link(qwen_3_2025)
    bilingual_text("- Moonshot's Kimi models ", '- Moonshot 的 Kimi 模型。'), link(kimi_1_5_2025), link(kimi_k2_5_2026)
    bilingual_text("- Z.ai's GLM models ", '- Z.ai 的 GLM 模型。'), link(glm_4_5_2025), link(glm_5_2026)
    bilingual_text("- Minimax\'s models ", '- Minimax 的模型。'), link(minimax_m2_5_2026)
    bilingual_text("- Xiaomi's MIMO models ", '- 小米的 MIMO 模型。'), link(xiaomi_mimo_v2_2026)
    bilingual_text("These models are approaching closed models (GPT, Claude, Gemini, etc.).", '这些模型正在接近闭源模型（GPT、Claude、Gemini 等）。')

    bilingual_text("Open-source models (weights + paper + code + data):", '开源模型（权重 + 论文 + 代码 + 数据）：')
    bilingual_text("- AI2's Olmo models ", '- AI2 的 Olmo 模型。'), link(olmo_7b_2024), link(olmo_2_2025), link(olmo_3_2025)
    bilingual_text("- NVIDIA's Nemotron models ", '- NVIDIA 的 Nemotron 模型。'), link(nemotron_15b_2024), link(nemotron_3_2025)
    bilingual_text("- Marin's models (open development) ", '- Marin 的模型（开放开发）。'), link(marin_8b_2025), link(marin_32b_2025)

    bilingual_text("Openness is important for trust and innovation ", '开放性对信任和创新非常重要。'), link("https://arxiv.org/abs/2403.07918")
    bilingual_text("Ideas from open models enable us to teach CS336.", '开放模型中的思想让我们能够讲授 CS336。')

    bilingual_text("What is a language model?", '语言模型是什么？')
    bilingual_text("- 2018 (BERT): something you fine-tune", '- 2018（BERT）：一种用来微调的东西。')
    bilingual_text("- 2020 (GPT-3): something you prompt", '- 2020（GPT-3）：一种用提示词使用的东西。')
    bilingual_text("- 2022 (ChatGPT): something you talk to ", '- 2022（ChatGPT）：一种可以对话的东西。'), link(title="example conversation", url="https://huggingface.co/datasets/HuggingFaceTB/smoltalk/viewer/all/train?row=72&conversation-viewer=72")
    bilingual_text("- 2026 (agents): something that acts autonomously ", '- 2026（智能体）：一种能自主行动的东西。'), link(title="example trace", url="https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories/viewer/default/train?conversation-viewer=1")

    bilingual_text("The fundamentals are the same (attention, kernels, optimization).", '基本原理仍然相同（注意力、内核、优化）。')
    bilingual_text("The specs are different (longer context, inference efficiency matters even more).", '规格已经不同（上下文更长，推理效率更加重要）。')


def what_is_this_program():
    bilingual_text("This is an *executable lecture*, a program whose execution delivers the content of a lecture.", '这是一份*可执行讲义*：运行这个程序就会呈现讲义内容。')
    bilingual_text("Executable lectures make it possible to:", '可执行讲义让以下事情成为可能：')
    bilingual_text("- view and run code (since everything is code!),", '- 查看并运行代码（因为一切都是代码！）')
    total = 0  # @inspect total
    for x in [1, 2, 3]:  # @inspect x
        total += x  # @inspect total
    bilingual_text("- see the hierarchical structure of the lecture", '- 看到讲义的层级结构。')


def course_logistics():
    bilingual_text("All information online: ", '所有信息都在线上：'), link(title="course website", url="https://stanford-cs336.github.io/spring2026/")

    bilingual_text("This is a 5-unit class.", '这是一门 5 学分课程。')
    bilingual_text("Comment from Spring 2024 course evaluation:", '来自 2024 春季课程评价的一条评论：')
    bilingual_text("> *The entire assignment was approximately the same amount of work as all 5 assignments from CS 224n plus the final project. And that's just the first homework assignment.*", '> *整个作业量大约等于 CS 224n 五个作业加期末项目的总和，而且这还只是第一份作业。*')

    bilingual_text("## Why you should take this course", '## 为什么你应该选这门课')
    bilingual_text("- You have an obsessive need to understand how things work.", '- 你强烈想理解事物到底如何工作。')
    bilingual_text("- You want to build up your research engineering muscles.", '- 你想训练自己的研究工程能力。')

    bilingual_text("## Why you should not take this course", '## 为什么你不应该选这门课')
    bilingual_text("- You actually want to get research done this quarter. (Talk to your advisor.)", '- 你这个学期真的想推进研究成果。（请和导师聊聊。）')
    bilingual_text("- You are interested in learning about the hottest new techniques in AI (e.g., multimodality, RAG, etc.). (You should take a seminar class for that.)", '- 你主要想学习 AI 中最热门的新技术（例如多模态、RAG 等）。（这类内容更适合专题研讨课。）')
    bilingual_text("- You want to get good results on your own application domain. (You should just prompt or fine-tune an existing model.)", '- 你想在自己的应用领域取得好结果。（你应该直接提示或微调现有模型。）')

    bilingual_text("## How you can follow along at home", '## 如何在校外/家中跟学')
    bilingual_text("- All lecture materials and assignments will be posted online, so feel free to follow on your own.", '- 所有讲义材料和作业都会发布到网上，所以你可以自由自学。')
    bilingual_text("- Lectures are recorded via [CGOE](https://cgoe.stanford.edu/).", '- 课程会通过 [CGOE](https://cgoe.stanford.edu/) 录制。')

    bilingual_text("## Assignments", '## 作业')
    bilingual_text("- 5 assignments (basics, systems, scaling laws, data, alignment).", '- 5 次作业（基础、系统、规模定律、数据、对齐）。')
    bilingual_text("- No scaffolding code, but we provide unit tests and adapter interfaces to help you check correctness.", '- 没有脚手架代码，但我们提供单元测试和适配器接口，帮助你检查正确性。')
    bilingual_text("- Implement locally to test for correctness, then run on cluster for benchmarking (accuracy and speed).", '- 先在本地实现并测试正确性，再在集群上运行基准测试（准确率和速度）。')
    bilingual_text("- Leaderboard for some assignments (minimize perplexity given training budget).", '- 部分作业有排行榜（在给定训练预算下最小化困惑度）。')

    bilingual_text("## AI policy", '## AI 使用政策')
    bilingual_text("- Coding agents can solve all the assignments, but you won't learn anything.", '- 编码智能体可以完成所有作业，但那样你学不到东西。')
    bilingual_text("- AI can be tremendously useful for answering questions and tutoring.", '- AI 在答疑和辅导方面可以非常有用。')
    bilingual_text("- You must use our provided AGENTS.md file, which asks the AI to be pedagogically-minded.", '- 你必须使用我们提供的 AGENTS.md 文件，其中要求 AI 以教学为导向。')
    bilingual_text("- Please read our [AI policy guide](https://docs.google.com/document/d/1SZAlExB1qAc9izHt54gwunNpjKE6wXb8Y7yA_e-baK8/edit?tab=t.0).", '- 请阅读我们的 [AI 使用政策指南](https://docs.google.com/document/d/1SZAlExB1qAc9izHt54gwunNpjKE6wXb8Y7yA_e-baK8/edit?tab=t.0)。')

    bilingual_text("## Compute", '## 计算资源')
    bilingual_text("- Thanks to [Modal](https://modal.com/) for providing compute. 🙏", '- 感谢 [Modal](https://modal.com/) 提供计算资源。🙏')
    bilingual_text("- Please read the [guide](https://docs.google.com/document/d/1cHE0iKVyXLJ3XpIs2XuXTmZ-HMmPk2hIPeCvy-AydMg/edit?tab=t.otis27tacaef) on how to access and use the compute.", '- 请阅读这份[指南](https://docs.google.com/document/d/1cHE0iKVyXLJ3XpIs2XuXTmZ-HMmPk2hIPeCvy-AydMg/edit?tab=t.otis27tacaef)，了解如何访问和使用计算资源。')


def course_syllabus():
    basics()         # Assignment 1: tokenization, model architecture, training
    systems()        # Assignment 2: kernels, parallelism, inference
    scaling_laws()   # Assignment 3: scaling laws
    data()           # Assignment 4: evaluation, curation, transformation, filtering, deduplication, mixing
    alignment()      # Assignment 5: RLHF, RL algorithms, RL systems

    bilingual_text("Remember it's all about **efficiency**:", '请记住，核心始终是**效率**：')
    bilingual_text("- Resources: data + hardware (compute, memory, communication bandwidth)", '- 资源：数据 + 硬件（计算量、内存、通信带宽）。')
    bilingual_text("- How do you train the best model given a fixed set of resources?", '- 在固定资源下，怎样训练出最好的模型？')

    bilingual_text("Today, we are compute-constrained, so design decisions will reflect squeezing the most out of given hardware.", '今天，我们受计算量约束，因此设计决策会体现如何尽可能榨干给定硬件。')
    bilingual_text("- Systems: clearly about efficiency", '- 系统：显然关乎效率。')
    bilingual_text("- Tokenization: working with raw bytes is elegant, but compute-inefficient with today's model architectures", '- 分词：直接处理原始字节很优雅，但在今天的模型架构下计算效率低。')
    bilingual_text("- Model architecture: many changes motivated by reducing memory or FLOPs (e.g., sharing KV caches, sliding window attention)", '- 模型架构：许多改动都来自减少内存或 FLOPs 的动机（例如共享 KV cache、滑动窗口注意力）。')
    bilingual_text("- Data filtering: avoid wasting precious compute updating on bad / irrelevant data", '- 数据过滤：避免把宝贵计算量浪费在坏数据或无关数据上。')
    bilingual_text("- Scaling laws: use less compute on smaller models to do hyperparameter tuning", '- 规模定律：用较小模型和较少计算量做超参数调优。')

    bilingual_text("Tomorrow, we will become data-constrained...", '之后，我们还会面临数据约束……')


class Tokenizer(ABC):
    """Abstract interface for a tokenizer."""
    def encode(self, string: str) -> list[int]:
        raise NotImplementedError

    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError


def basics():
    bilingual_text("Goal: be able to train a basic language model", '目标：能够训练一个基础语言模型。')
    bilingual_text("Components: tokenization, model architecture, training", '组成部分：分词、模型架构、训练。')

    bilingual_text("## Tokenization", '## 分词')
    bilingual_text("What are the atoms that the model operates on?", '模型操作的“原子单位”是什么？')
    bilingual_text("Formally: a tokenizer converts between raw inputs (bytes) and sequences of integers (tokens)", '形式化地说：分词器在原始输入（字节）和整数序列（token）之间转换。')
    image("images/tokenized-example.png", width=600) 
    bilingual_text("Popular tokenizer: **Byte-Pair Encoding** (BPE) ", '常用分词器：**Byte-Pair Encoding**（BPE，字节对编码）。'), link(sennrich_2016)
    bilingual_text("Intuition: break input into frequently-occuring chunks", '直觉：把输入切成经常出现的片段。')
    bilingual_text("Efficiency lens", '从效率角度看：')
    bilingual_text("- Reduce context length (1000 bytes → ~250 tokens)", '- 缩短上下文长度（1000 字节 → 约 250 个 token）。')
    bilingual_text("- Adaptive computation (more modeling capacity on interesting parts of input)", '- 自适应计算（把更多建模能力用于输入中更重要的部分）。')

    bilingual_text("The dream: tokenizer-free model architectures, which operate directly on bytes ", '理想方向：无分词器的模型架构，直接在字节上操作。'), link(byt5_2021), link(megabyte_2023), link(blt_2024), link(tfree_2024), link(hnet_2025)
    bilingual_text("These are promising, but have not yet been scaled up to the frontier.", '这些方向很有前景，但还没有扩展到前沿模型规模。')
    
    bilingual_text("## Model architecture", '## 模型架构')
    bilingual_text("Starting point: original Transformer ", '起点：原始 Transformer。'), link(transformer_2017)
    image("images/transformer-architecture.png", width=500)

    bilingual_text("Refinements:", '改进方向：')
    bilingual_text("- Activation functions: ReLU, SwiGLU ", '- 激活值 functions: ReLU, SwiGLU'), link(shazeer_2020)
    bilingual_text("- Positional encodings: sinusoidal, RoPE ", '- 说明：Positional encodings: sinusoidal, RoPE'), link(rope_2021)
    bilingual_text("- Normalization: LayerNorm, RMSNorm, QK norm, pre-norm versus post-norm ", '- 说明：Normalization: LayerNorm, RMSNorm, QK norm, pre-norm versus post-norm'), link(layernorm_2016), link(rms_norm_2019), link(qk_norm_2023), link(pre_post_norm_2020)
    bilingual_text("- Attention: full, sparse/local attention, group-query attention (GQA), multi-head latent attention (MLA) ", '- 注意力: full, sparse/local 注意力, group-query 注意力 (GQA), multi-head latent 注意力 (MLA)'), link(sparse_transformer_2019), link(gqa_2023), link(mla_2024)
    bilingual_text("- Recurrence/state-space models/linear attention: Mamba, Gated DeltaNet ", '- Recurrence/state-space models/linear 注意力: Mamba, Gated DeltaNet'), link(linear_attention_2020), link(mamba_2_2024), link(gdn_2024), link(mamba_3_2026)
    bilingual_text("- MLP: dense, mixture of experts ", '- 说明：MLP: dense, mixture of experts'), link(moe_2017), link(switch_transformers_2021)
    bilingual_text("- Shape (hidden dimension, depth, number of heads, number of experts)", '- 形状（隐藏维度、深度、头数、专家数）。')

    bilingual_text("## Training", '## 训练')
    bilingual_text("How do you set the parameters of the model?", '如何设置模型参数？')
    bilingual_text("- Loss function (e.g., multi-token prediction) ", '- 损失 function (e.g., multi-token prediction)'), link(mtp_2024), link(deepseek_v3_2024)
    bilingual_text("- Optimizer (e.g., AdamW, SOAP, Muon) ", '- 优化器 (e.g., AdamW, SOAP, Muon)'), link(adam_2014), link(adamw_2017), link(soap_2024), link(muon_2024)
    bilingual_text("- Initialization scale (e.g., Xavier init, muP) ", '- 说明：Initialization scale (e.g., Xavier init, muP)'), link(glorot_2010), link(mup_2022)
    bilingual_text("- Learning rate schedule (e.g., cosine, WSD) ", '- 说明：Learning rate schedule (e.g., cosine, WSD)'), link(cosine_learning_rate_2017), link(wsd_2024)
    bilingual_text("- Regularization (e.g., dropout, weight decay)", '- 说明：Regularization (e.g., dropout, weight decay)')
    bilingual_text("- Batch size (e.g., critical batch size) ", '- 批大小 (e.g., critical 批大小)'), link(large_batch_training_2018)
    bilingual_text("- MoE specific: load balancing (e.g., aux-free) ", '- 说明：MoE specific: load balancing (e.g., aux-free)'), link(auxfree_2024), link(deepseek_v3_2024)

    bilingual_text("## Assignment 1 (basics)", '## 作业 1（基础）')
    link(title="GitHub", url="https://github.com/stanford-cs336/assignment1-basics"), link(title="PDF", url="https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_spring2026_assignment1_basics.pdf")
    bilingual_text("- Implement BPE tokenizer", '- 实现 BPE 分词器。')
    bilingual_text("- Implement Transformer, cross-entropy loss, AdamW optimizer, training loop", '- 实现 Transformer、交叉熵损失、AdamW 优化器和训练循环。')
    bilingual_text("- Do resource accounting", '- 做资源核算。')
    bilingual_text("- Train on TinyStories and OpenWebText", '- 在 TinyStories 和 OpenWebText 上训练。')
    bilingual_text("- Leaderboard: minimize OpenWebText perplexity given 45 minutes on a B200 ", '- Leaderboard: minimize OpenWebText 困惑度 given 45 minutes on a B200'), link(title="last year's leaderboard", url="https://github.com/stanford-cs336/spring2025-assignment1-basics-leaderboard")

    bilingual_text("High-level principle: everything is about balancing the following:", '高层原则：一切都在平衡以下因素：')
    bilingual_text("- Expressivity (can represent complex dependencies in the data)", '- 表达能力（能表示数据中的复杂依赖关系）。')
    bilingual_text("- Stability (keep parameter and gradient norms in goldilocks zone)", '- 稳定性（让参数和梯度范数保持在合适区间）。')
    bilingual_text("- Efficiency (runs fast on hardware, both training and inference)", '- 效率（训练和推理时都能在硬件上快速运行）。')


def systems():
    bilingual_text("Goal: squeeze the most out of the hardware (GPU or TPU)", '目标：尽可能榨干硬件（GPU 或 TPU）。')
    bilingual_text("Components: kernels, parallelism, inference", '组成部分：内核、并行、推理。')

    bilingual_text("## Basics", '## 基础')
    bilingual_text("- Resource accounting: memory and compute characteristics of a model", '- 资源核算: 内存 and 计算量 characteristics of a model')
    total_flops = 6 * 70e9 * 1e12  # Training 70B parameters on 1T tokens = 4.2e23 FLOPs @inspect total_flops 
    image("images/compute-memory.png", width=300)
    bilingual_text("- Model parameters must be moved from memory (HBM) to the compute (SMs)", '- Model 参数 must be moved from 内存 (HBM) to the 计算量 (SMs)')
    bilingual_text("- Example: B200 can perform 2.25 PFLOP/sec (bf16) with 8TB/sec memory bandwidth", '- 示例：B200 can perform 2.25 PFLOP/sec (bf16) with 8TB/sec memory bandwidth')
    bilingual_text("- Roofline analysis: understand whether we're compute-bound or memory-bound", "- 屋顶线分析: understand whether we're 计算量-bound or 内存-bound")
    bilingual_text("- Benchmarking and profiling (nsight): see what happens in practice", '- 基准测试 and 性能分析 (nsight): see what happens in practice')

    bilingual_text("[DGX B200](https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html):", '说明：[DGX B200](https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html):')
    image("https://docs.nvidia.com/dgx/dgxb200-user-guide/_images/dgx-b200-system-topology.png", width=500)

    bilingual_text("## Kernels", '## 内核')
    bilingual_text("- Kernel is a function that runs on GPU", '- 说明：Kernel is a function that runs on GPU')
    bilingual_text("- When using PyTorch, each primitive operation launches a standard kernel", '- 说明：When using PyTorch, each primitive operation launches a standard kernel')
    bilingual_text("- Can write custom kernels to make GPUs go brrr", '- 说明：Can write custom kernels to make GPUs go brrr')
    bilingual_text("- Principle: organize computation to minimize data movement", '- Principle: organize computation to minimize 数据 movement')
    bilingual_text("- Naive: read HBM; compute A; write HBM; read HBM; compute B; write HBM", '- Naive: read HBM; 计算量 A; write HBM; read HBM; 计算量 B; write HBM')
    bilingual_text("- Fused: read HBM; compute A and B; write HBM", '- Fused: read HBM; 计算量 A and B; write HBM')
    bilingual_text("- Strategies: operator fusion (matmul + activation), tiling (FlashAttention)", '- Strategies: operator fusion (matmul + 激活值), tiling (Flash注意力)')
    bilingual_text("- Warp divergence, memory coalescing, bank conflicts, occupancy, bulk-async memory transfers", '- Warp divergence, 内存 coalescing, bank conflicts, occupancy, bulk-async 内存 transfers')
    bilingual_text("- Write kernels in CUDA/**Triton**/CUTLASS/ThunderKittens", '- 说明：Write kernels in CUDA/Triton/CUTLASS/ThunderKittens')

    bilingual_text("## Parallelism", '## 并行')
    bilingual_text("- What if we have 1024 GPUs?", '- 说明：What if we have 1024 GPUs?')
    bilingual_text("- Data movement between GPUs is even slower, but same 'minimize data movement' principle holds", "- 数据 movement between GPUs is even slower, but same 'minimize 数据 movement' principle holds")
    bilingual_text("- Use classic collective operations (e.g., gather, reduce, all-reduce)", '- 说明：Use classic collective operations (e.g., gather, reduce, all-reduce)')
    bilingual_text("- Shard memory (parameters, activations, gradients, optimizer states) across GPUs", '- Shard 内存 (参数, 激活值, 梯度, 优化器状态) across GPUs')
    bilingual_text("- How to split computation: {data,tensor,pipeline,sequence,expert} parallelism", '- How to split computation: {数据,张量,pipeline,sequence,expert} parallelism')
    
    bilingual_text("## Inference", '## 推理')
    bilingual_text("Goal: generate tokens given a prompt (needed to actually use models!)", '目标：给定提示词生成 token（这是实际使用模型所必需的！）')
    bilingual_text("Inference is also needed for reinforcement learning, test-time compute, evaluation", '强化学习、测试时计算和评测也需要推理。')
    bilingual_text("Two phases: prefill and decode", '两个阶段：预填充（prefill）和解码（decode）。')
    image("images/prefill-decode.png", width=500)
    bilingual_text("- Prefill (similar to training): tokens are given, can process all at once (compute-bound)", '- Prefill (similar to 训练): token are given, can process all at once (计算量-bound)')
    bilingual_text("- Decode: need to generate one token at a time (memory-bound)", '- Decode: need to generate one token at a time (内存-bound)')
    bilingual_text("Methods to speed up decoding:", '加速解码的方法：')
    bilingual_text("- Use cheaper model (via model pruning, quantization, distillation)", '- 说明：Use cheaper model (via model pruning, quantization, distillation)')
    bilingual_text("- Speculative decoding: use a cheaper \"draft\" model to generate multiple tokens, then use the full model to score in parallel (exact decoding!)", '- 说明：Speculative decoding: use a cheaper "draft" model to generate multiple tokens, then use the full model to score in parallel (exact decoding!)')
    bilingual_text("- Systems optimizations: fused kernels, continuous batching", '- 说明：Systems optimizations: fused kernels, continuous batching')

    bilingual_text("## Assignment 2 (systems)", '## 作业 2（系统）')
    link(title="GitHub", url="https://github.com/stanford-cs336/assignment2-systems"), link(title="PDF from Spring 2025", url="https://github.com/stanford-cs336/assignment2-systems/blob/spring2025/cs336_spring2025_assignment2_systems.pdf")
    bilingual_text("- Implement a fused RMSNorm kernel in Triton", '- 说明：Implement a fused RMSNorm kernel in Triton')
    bilingual_text("- Implement distributed data parallel training", '- Implement distributed 数据 parallel 训练')
    bilingual_text("- Implement optimizer state sharding", '- Implement 优化器状态 sharding')
    bilingual_text("- Benchmark and profile the implementations", '- 基准 and profile the implementations')

    bilingual_text("Recommended book: [How to Scale Your Model](https://jax-ml.github.io/scaling-book/)", '推荐书：[How to Scale Your Model](https://jax-ml.github.io/scaling-book/)。')
    bilingual_text("- Nicely lays out how to approach systems for LLMs conceptually", '- 很好地从概念层面说明如何理解 LLM 系统。')
    bilingual_text("- From Google, so it foregrounds TPUs, but high-level concepts are similar", '- 作者来自 Google，所以更强调 TPU，但高层概念是相似的。')


def scaling_laws():
    bilingual_text("Setting: if you had 1e25 FLOPs of compute, what hyperparameters would you use to train a good model?", 'Setting: if you had 1e25 FLOPs of 计算量, what 超参数 would you use to train a good model?')
    bilingual_text("Too expensive to do hyperparameter tuning at full scale!", 'Too expensive to do 超参数 tuning at full scale!')

    bilingual_text("Key conceptual shift: instead of a single scale, think of a **scaling recipe** (FLOPs → hyperparameters)", '关键概念转变：不要只考虑单一规模，而要考虑一套**规模化配方**（FLOPs → 超参数）。')
    bilingual_text("For a scaling recipe:", '对于一套规模化配方：')
    bilingual_text("- Run experiments to compute the loss at various smaller scales (e.g., up to 1e24 FLOPs)", '- Run experiments to 计算量 the 损失 at various smaller scales (e.g., up to 1e24 FLOPs)')
    bilingual_text("- Fit a scaling law to predict the loss of the scaling recipe at the target scale (e.g., 1e25 FLOPs)", '- Fit a 规模定律 to predict the 损失 of the 规模化 recipe at the target scale (e.g., 1e25 FLOPs)')

    bilingual_text("Now you can:", '说明：Now you can:')
    bilingual_text("1. Optimize the scaling recipe targeting a larger scale using smaller scale experiments", '1. Optimize the 规模化 recipe targeting a larger scale using smaller scale experiments')
    bilingual_text("2. Predict the loss at the target scale before actually running the experiment!", '2. Predict the 损失 at the target scale before actually running the experiment!')
    bilingual_text("Scaling laws don't happen automatically, they require careful construction of a scaling recipe.", "规模定律 don't happen automatically, they require careful construction of a 规模化 recipe.")
    bilingual_text("Parameterize the model in a way to get **hyperparameter transfer** ", '参数ize the model in a way to get 超参数 transfer'), link(mup_2022)
    bilingual_text("Predictability is at least as important as optimality!", '说明：Predictability is at least as important as optimality!')

    bilingual_text("Question: given a FLOPs budget (C = 6 N D), use a bigger model (N) or train on more tokens (D)?", '问题：given a FLOPs budget (C = 6 N D), use a bigger model (N) or train on more tokens (D)?')
    bilingual_text("Classic compute-optimal scaling laws: ", 'Classic 计算量-optimal 规模定律:'), link(kaplan_scaling_laws_2020), link(chinchilla_2022)
    bilingual_text("- ISOFLOP curves: for multiple small FLOPs budgets, find optimal N", '- 说明：ISOFLOP curves: for multiple small FLOPs budgets, find optimal N')
    bilingual_text("- Then fit a scaling law to extrapolate to large FLOPs budgets", '- Then fit a 规模定律 to extrapolate to large FLOPs budgets')
    image("images/chinchilla-isoflop.png", width=800)
    bilingual_text("TL;DR: D = 20 N is roughly optimal (e.g., 70B parameter model should be trained on ~1.4T tokens)", 'TL;DR: D = 20 N is roughly optimal (e.g., 70B 参数 model should be trained on ~1.4T token)')
    bilingual_text("Caveat: this doesn't take into account inference costs (want a smaller model)", "注意：this doesn't take into account inference costs (want a smaller model)")

    bilingual_text("Live example from Marin ", '说明：Live example from Marin'), post_link("https://x.com/percyliang/status/2034367256277533100")
    image("https://pbs.twimg.com/media/HDuErvvbsAAQ5Yt?format=jpg&name=4096x4096", width=600)
    bilingual_text("Should be done training this week, should see how well we match the preregistered loss!", 'Should be done 训练 this week, should see how well we match the preregistered 损失!')

    bilingual_text("## Assignment 3 (scaling laws)", '## 作业 3（规模定律）')
    link(title="GitHub", url="https://github.com/stanford-cs336/assignment3-scaling"), link(title="PDF from Spring 2025", url="https://github.com/stanford-cs336/assignment3-scaling/blob/master/cs336_spring2025_assignment3_scaling.pdf")
    bilingual_text("- We define a training API (hyperparameters → loss) based on previous runs", '- 我们基于之前的运行定义一个训练 API（超参数 → 损失）。')
    bilingual_text("- Submit \"training jobs\" (under a FLOPs budget) and gather data points", '- 提交“训练作业”（受 FLOPs 预算约束）并收集数据点。')
    bilingual_text("- Fit scaling laws to the data points", '- Fit 规模定律 to the 数据 points')
    bilingual_text("- Submit extrapolated hyperparameters and loss predictions", '- Submit extrapolated 超参数 and 损失 predictions')
    bilingual_text("- Leaderboard: minimize loss given FLOPs budget", '- Leaderboard: minimize 损失 given FLOPs budget')


def data():
    bilingual_text("Question: What capabilities do we want the model to have?", '问题：我们希望模型具备哪些能力？')
    bilingual_text("Multilingual?  Good at conversation?  Agentic coding capabilities?", '多语言？擅长对话？具备智能体式编码能力？')

    bilingual_text("## Evaluation", '## 评测')
    bilingual_text("What is the purpose of evaluation?", '评测的目的是什么？')
    bilingual_text("1. Internal: guide model development (smoothness across scales, relative performance matters)", '1. 说明：Internal: guide model development (smoothness across scales, relative performance matters)')
    bilingual_text("2. External: measure absolute quality of a real use case (ecological validity matters)", '2. 说明：External: measure absolute quality of a real use case (ecological validity matters)')
    bilingual_text("Examples of evaluations:", '评测示例：')
    bilingual_text("1. Perplexity: ideally run on private documents not on Internet (avoid contamination)", '1. 困惑度: ideally run on private documents not on Internet (avoid contamination)')
    bilingual_text("2. Advanced use cases: GPQA, HLE, SWE-Bench, Terminal-Bench", '2. 说明：Advanced use cases: GPQA, HLE, SWE-Bench, Terminal-Bench')
    bilingual_text("LMs are general purpose, require a diverse set of evaluations!", '语言模型是通用系统，因此需要多样化的评测集合！')

    bilingual_text("## Data curation", '## 数据策划')
    bilingual_text("- Data does not just fall from the sky.", '- 数据不会从天上掉下来。')
    bilingual_text("- Sources: webpages crawled from the Internet, books, arXiv papers, GitHub code, etc.", '- 说明：Sources: webpages crawled from the Internet, books, arXiv papers, GitHub code, etc.')
    image("https://ar5iv.labs.arxiv.org/html/2101.00027/assets/pile_chart2.png", width=600)
    bilingual_text("- Appeal to fair use to train on copyright data? ", '- Appeal to fair use to train on copyright 数据?'), link("https://arxiv.org/pdf/2303.15715.pdf")
    bilingual_text("- Might have to license data (e.g., Google with Reddit data) ", '- Might have to license 数据 (e.g., Google with Reddit 数据)'), article_link("https://www.reuters.com/technology/reddit-ai-content-licensing-deal-with-google-sources-say-2024-02-22/")
    bilingual_text("- Raw data is HTML, PDF, directories (not text), requires processing", '- Raw 数据 is HTML, PDF, directories (not text), requires processing')

    bilingual_text("## Data processing", '## 数据处理')
    bilingual_text("- Transformation: convert HTML/PDF to text (extract main content)", '- 说明：Transformation: convert HTML/PDF to text (extract main content)')
    bilingual_text("- Filtering: keep high quality data, remove harmful content (via classifiers)", '- 过滤: keep high quality 数据, remove harmful content (via classifiers)')
    bilingual_text("- Deduplication: save compute, avoid memorization; use Bloom filters or MinHash", '- 去重: save 计算量, avoid memorization; use Bloom filters or MinHash')
    bilingual_text("- Data mixing: how much to upweight/downweight each source? ", '- 数据 mixing: how much to upweight/downweight each source?'), link(regmix_2025), link(olmix_2026)
    bilingual_text("- Rewriting / synthetic data: use LM to augment real data, more similar to downstream tasks ", '- Rewriting / synthetic 数据: use LM to augment real 数据, more similar to downstream tasks'), link(wrap_2024)

    bilingual_text("Types of data:", '数据类型：')
    bilingual_text("- Pretraining data: large and diverse", '- Pre训练 数据: large and diverse')
    bilingual_text("- Mid-training data: high quality, including long-context", '- Mid-训练 数据: high quality, including long-context')
    bilingual_text("- Post-training data: supervised fine-tuning (conversations, agentic traces with tool calling)", '- Post-训练 数据: supervised 微调 (conversations, agentic traces with tool calling)')

    bilingual_text("## Assignment 4 (data)", '## 作业 4（数据）')
    link(title="GitHub", url="https://github.com/stanford-cs336/assignment4-data"), link(title="PDF from Spring 2025", url="https://github.com/stanford-cs336/assignment4-data/blob/spring2025/cs336_spring2025_assignment4_data.pdf")
    bilingual_text("- Convert Common Crawl HTML to text", '- 说明：Convert Common Crawl HTML to text')
    bilingual_text("- Train classifiers to filter for quality and harmful content", '- 说明：Train classifiers to filter for quality and harmful content')
    bilingual_text("- Deduplication using MinHash", '- 去重 using MinHash')
    bilingual_text("- Leaderboard: minimize perplexity given token budget", '- Leaderboard: minimize 困惑度 given token budget')


def alignment():
    bilingual_text("So far, we have trained a model on full supervision (predict the next token).", '到目前为止，我们用完全监督训练模型（预测下一个 token）。')
    bilingual_text("Now that the model should be reasonable, we can improve it further from **weak supervision**.", '现在模型应该已经比较合理，我们可以进一步用**弱监督**改进它。')
    bilingual_text("Why weak supervision?  When it is easier to critique than to generate.", '为什么使用弱监督？因为有时评价比生成更容易。')

    bilingual_text("Basic template:", '基本模板：')
    bilingual_text("1. Generate responses from the model.", '1. 说明：Generate responses from the model.')
    bilingual_text("2. Score responses with a {human, verifier, LM judge}.", '2. 说明：Score responses with a {human, verifier, LM judge}.')
    bilingual_text("3. Update the model to prefer better responses.", '3. 说明：Update the model to prefer better responses.')

    bilingual_text("Algorithms:", '算法：')
    bilingual_text("- Proximal Policy Optimization (PPO) from reinforcement learning ", '- Proximal Policy Optimization (PPO) from 强化学习'), link(ppo_2017), link(instruct_gpt_2022)
    bilingual_text("- Direct Policy Optimization (DPO): for preference data, simpler ", '- Direct Policy Optimization (DPO): for preference 数据, simpler'), link(dpo_2023)
    bilingual_text("- Group Relative Preference Optimization (GRPO): remove value function ", '- 说明：Group Relative Preference Optimization (GRPO): remove value function'), link(grpo)

    bilingual_text("Challenges:", '挑战：')
    bilingual_text("- RL algorithms are unstable and hard to tune", '- 说明：RL algorithms are unstable and hard to tune')
    bilingual_text("- At scale, this requires a lot of new infrastructure (inference with async rollouts)", '- At scale, this requires a lot of new infrastructure (推理 with async rollouts)')
    bilingual_text("- Constantly trading off systems efficiency and on-policyness", '- Constantly trading off systems 效率 and on-policyness')

    bilingual_text("## Assignment 5 (alignment)", '## 作业 5（对齐）')
    link(title="GitHub", url="https://github.com/stanford-cs336/assignment5-alignment"), link(title="PDF from Spring 2025", url="https://github.com/stanford-cs336/assignment5-alignment/blob/spring2025/cs336_spring2025_assignment5_alignment.pdf")
    bilingual_text("- Implement Direct Preference Optimization (DPO)", '- 说明：Implement Direct Preference Optimization (DPO)')
    bilingual_text("- Implement Group Relative Preference Optimization (GRPO)", '- 说明：Implement Group Relative Preference Optimization (GRPO)')


############################################################
# Tokenization

def tokenization():
    bilingual_text("This unit was inspired by Andrej Karpathy's video on tokenization; check it out! ", "This unit was inspired by Andrej Karpathy's video on 分词; check it out!"), video_link("https://www.youtube.com/watch?v=zduSFxRajkE")

    intro_to_tokenization()
    tokenization_examples()
    character_tokenizer()
    byte_tokenizer()
    word_tokenizer()
    bpe_tokenizer()

    bilingual_text("Summary:", '总结：')
    bilingual_text("- Tokenizer: strings ↔ tokens (indices)", '- 分词器：字符串 ↔ token（索引）。')
    bilingual_text("- Character-based, byte-based, word-based tokenization are highly suboptimal", '- 基于字符、字节或单词的分词都很不理想。')
    bilingual_text("- BPE is an effective heuristic that is data-driven", '- BPE 是一种由数据驱动的有效启发式方法。')
    bilingual_text("- Tokenization is a separate step, maybe one day do it end-to-end from bytes...", '- 分词是一个单独步骤，也许未来可以直接从字节端到端建模……')

    bilingual_text("But whatever solution needs to satisfy:", '但无论采用什么方案，都需要满足：')
    bilingual_text("1. Model (e.g., Transformer) should operate on chunks (abstractions) of the sequence (text, video, DNA, etc.)", '1. 模型（例如 Transformer）应该在序列（文本、视频、DNA 等）的片段（抽象单位）上操作。')
    bilingual_text("2. Chunks should be variable (allocate more model capacity to interesting chunks)", '2. 片段应该是可变的（给重要片段分配更多模型容量）。')


class CharacterTokenizer(Tokenizer):
    """Represent a string as a sequence of Unicode code points."""
    def encode(self, string: str) -> list[int]:
        return list(map(ord, string))

    def decode(self, indices: list[int]) -> str:
        return "".join(map(chr, indices))


class ByteTokenizer(Tokenizer):
    """Represent a string as a sequence of bytes."""
    def encode(self, string: str) -> list[int]:
        string_bytes = string.encode("utf-8")  # @inspect string_bytes
        indices = list(map(int, string_bytes))  # @inspect indices
        return indices

    def decode(self, indices: list[int]) -> str:
        string_bytes = bytes(indices)  # @inspect string_bytes
        string = string_bytes.decode("utf-8")  # @inspect string
        return string


def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:  # @inspect indices, @inspect pair, @inspect new_index
    """Return `indices`, but with all instances of `pair` replaced with `new_index`."""
    new_indices = []  # @inspect new_indices
    i = 0  # @inspect i
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices


@dataclass(frozen=True)
class BPETokenizerParams:
    """All you need to specify a BPETokenizer."""
    vocab: dict[int, bytes]     # index -> bytes
    merges: dict[tuple[int, int], int]  # index1,index2 -> new_index



class BPETokenizer(Tokenizer):
    """BPE tokenizer given a set of merges and a vocabulary."""
    def __init__(self, params: BPETokenizerParams):
        self.params = params

    def encode(self, string: str) -> list[int]:
        indices = list(map(int, string.encode("utf-8")))  # @inspect indices
        # Note: this is a very slow implementation
        for pair, new_index in self.params.merges.items():  # @inspect pair, @inspect new_index
            indices = merge(indices, pair, new_index)  # @stepover
        return indices

    def decode(self, indices: list[int]) -> str:
        bytes_list = list(map(self.params.vocab.get, indices))  # @inspect bytes_list
        string = b"".join(bytes_list).decode("utf-8")  # @inspect string
        return string


def get_compression_ratio(string: str, indices: list[int]) -> float:  # @inspect string indices
    """Given `string` that has been tokenized into `indices`, return the number of UTF-8 bytes per token.."""
    num_bytes = len(bytes(string, encoding="utf-8"))  # @inspect num_bytes
    num_tokens = len(indices)                       # @inspect num_tokens
    return num_bytes / num_tokens


def get_gpt5_tokenizer():
    # Code: https://github.com/openai/tiktoken
    return tiktoken.get_encoding("o200k_base")


def intro_to_tokenization():
    bilingual_text("Raw text is generally represented as Unicode strings.", '原始文本通常表示为 Unicode 字符串。')
    string = "Hello, 🌍! 你好!"

    bilingual_text("A language model places a probability distribution over sequences of tokens (usually represented by integer indices).", '语言模型为 token 序列（通常用整数索引表示）分配概率分布。')
    indices = [15496, 11, 995, 0]

    bilingual_text("So we need a procedure that *encodes* strings into tokens.", '因此我们需要一个过程，把字符串*编码*成 token。')
    bilingual_text("We also need a procedure that *decodes* tokens back into strings.", '我们也需要一个过程，把 token *解码*回字符串。')
    bilingual_text("A ", '说明：A'), link(Tokenizer), bilingual_text(" is a class that implements the encode and decode methods.", '说明：is a class that implements the encode and decode methods.')


def tokenization_examples():
    bilingual_text("To get a feel for how tokenizers work, play with this ", 'To get a feel for how 分词器s work, play with this'), link(title="interactive site", url="https://tiktokenizer.vercel.app/?encoder=gpt2")

    bilingual_text("## Observations", '## 观察')
    bilingual_text("- A word and its preceding space are part of the same token (e.g., \" world\").", '- 说明：A word and its preceding space are part of the same token (e.g., " world").')
    bilingual_text("- A word at the beginning and in the middle are represented differently (e.g., \"hello hello\").", '- 说明：A word at the beginning and in the middle are represented differently (e.g., "hello hello").')
    bilingual_text("- Numbers are tokenized into every few digits.", '- 说明：Numbers are tokenized into every few digits.')

    bilingual_text("Here's the GPT-5 tokenizer from OpenAI (tiktoken) in action.", "Here's the GPT-5 分词器 from OpenAI (tiktoken) in action.")
    tokenizer = get_gpt5_tokenizer()  # @stepover
    string = "Hello, 🌍! 你好!"  # @inspect string

    bilingual_text("Check that encode() and decode() roundtrip:", 'Check that encode() and decode() 往返转换:')
    indices = tokenizer.encode(string)  # @inspect indices
    reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string
    assert string == reconstructed_string
    
    bilingual_text("Compression ratio: number of bytes per token", '压缩率：每个 token 对应的字节数。')
    compression_ratio = get_compression_ratio(string, indices)  # @inspect compression_ratio
    bilingual_text("The larger the compression ratio, the shorter the sequence (good since attention is quadratic in sequence length).", 'The larger the 压缩率, the shorter the sequence (good since 注意力 is quadratic in 序列长度).')
    bilingual_text("One could increase compression ratio by increasing **vocabulary size** (number of possible token values increases), leading to sparsity.", 'One could increase 压缩率 by increasing 词表 size (number of possible token values increases), leading to sparsity.')
    vocabulary_size = tokenizer.n_vocab  # @inspect vocabulary_size

    bilingual_text("Let's take a look at the actual vocabulary: ", "Let's take a look at the actual 词表:"), link(title="vocab", url=get_local_url("var/gpt5_tokenizer_vocab.txt"))
    output_tokenizer(tokenizer, "var/gpt5_tokenizer_vocab.txt")  # @stepover


def output_tokenizer(tokenizer, path: str):
    """Write out the vocabulary of `tokenizer` to `path`, one per line."""
    if not os.path.exists(path):
        vocab = [b.decode("utf-8", errors="replace") for b in tokenizer.token_byte_values()]
        with open(path, "w") as f:
            for token in vocab:
                f.write(token + "\n")


def character_tokenizer():
    bilingual_text("A Unicode string is a sequence of Unicode characters.", '说明：A Unicode string is a sequence of Unicode characters.')
    bilingual_text("Each character can be converted into a code point (integer) via `ord`.", '说明：Each character can be converted into a code point (integer) via ord.')
    assert ord("a") == 97
    assert ord("🌍") == 127757
    bilingual_text("It can be converted back via `chr`.", '说明：It can be converted back via chr.')
    assert chr(97) == "a"
    assert chr(127757) == "🌍"

    bilingual_text("Now let's build a `Tokenizer` and make sure it round-trips:", "Now let's build a 分词器 and make sure it round-trips:")
    tokenizer = CharacterTokenizer()
    string = "Hello, 🌍! 你好!"  # @inspect string
    indices = tokenizer.encode(string)  # call ord @inspect indices @stepover
    reconstructed_string = tokenizer.decode(indices)  # call chr @inspect reconstructed_string @stepover
    assert string == reconstructed_string

    bilingual_text("There are approximately 150K Unicode characters. ", '说明：There are approximately 150K Unicode characters.'), link(title="Wikipedia", url="https://en.wikipedia.org/wiki/List_of_Unicode_characters")
    vocabulary_size = max(indices) + 1  # This is a lower bound @inspect vocabulary_size
    bilingual_text("Problem 1: this is a very large vocabulary.", 'Problem 1: this is a very large 词表.')
    bilingual_text("Problem 2: many characters are quite rare (e.g., 🌍), which is inefficient use of the vocabulary.", '问题 2：许多字符非常少见（例如 🌍），这会低效地占用词表。')
    compression_ratio = get_compression_ratio(string, indices)  # @inspect compression_ratio @stepover
    bilingual_text("This tokenizer is the worst of both worlds (large vocabulary, low compression ratio).", 'This 分词器 is the worst of both worlds (large 词表, low 压缩率).')


def byte_tokenizer():
    bilingual_text("Unicode strings can be represented as a sequence of bytes, which can be represented by integers between 0 and 255.", 'Unicode strings can be represented as a sequence of 字节, which can be represented by integers between 0 and 255.')
    bilingual_text("The most common Unicode encoding is ", '说明：The most common Unicode encoding is'), link(title="UTF-8", url="https://en.wikipedia.org/wiki/UTF-8")

    bilingual_text("Some Unicode characters are represented by one byte:", 'Some Unicode characters are represented by one 字节:')
    assert bytes("a", encoding="utf-8") == b"a"
    bilingual_text("Others take multiple bytes:", 'Others take multiple 字节:')
    assert bytes("🌍", encoding="utf-8") == b"\xf0\x9f\x8c\x8d"

    bilingual_text("Now let's build a `Tokenizer` and make sure it round-trips:", "Now let's build a 分词器 and make sure it round-trips:")
    tokenizer = ByteTokenizer()
    string = "Hello, 🌍! 你好!"  # @inspect string
    indices = tokenizer.encode(string)  # @inspect indices @stepover
    reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string @stepover
    assert string == reconstructed_string

    bilingual_text("The vocabulary is nice and small: a byte can represent 256 values.", 'The 词表 is nice and small: a 字节 can represent 256 values.')
    vocabulary_size = 256  # @inspect vocabulary_size
    bilingual_text("What about the compression rate?", '说明：What about the compression rate?')
    compression_ratio = get_compression_ratio(string, indices)  # @inspect compression_ratio @stepover
    assert compression_ratio == 1
    bilingual_text("The compression ratio is terrible, which means the sequences will be too long.", 'The 压缩率 is terrible, which means the sequences will be too long.')
    bilingual_text("Given that the context length of a Transformer is limited (since attention is quadratic), this is not looking great...", 'Given that the 上下文长度 of a Transformer is limited (since 注意力 is quadratic), this is not looking great...')


def word_tokenizer():
    bilingual_text("Another approach (closer to what was done classically in NLP) is to split strings into words.", '说明：Another approach (closer to what was done classically in NLP) is to split strings into words.')
    string = "I'll say supercalifragilisticexpialidocious!"

    chunks = regex.findall(r"\w+|.", string)  # @inspect chunks
    bilingual_text("This regular expression keeps all alphanumeric characters together (words).", 'This 正则表达式 keeps all alphanumeric characters together (words).')

    bilingual_text("To turn this into a `Tokenizer`, we need to map these chunks into integers.", 'To turn this into a 分词器, we need to map these chunks into integers.')
    bilingual_text("Then, we can build a mapping from each chunk into an integer.", '说明：Then, we can build a mapping from each chunk into an integer.')

    bilingual_text("What's good: each token is meaningful (since humans invented words).", "说明：What's good: each token is meaningful (since humans invented words).")

    vocabulary_size = "Number of distinct chunks in the training data"
    compression_ratio = get_compression_ratio(string, chunks)  # @inspect compression_ratio @stepover
    bilingual_text("Compression ratio is good, but vocabulary size can be huge.", '压缩率 is good, but 词表 size can be huge.')

    bilingual_text("Moreover:", '说明：Moreover:')
    bilingual_text("- Many words are rare and the model won't learn much about them.", "- 说明：Many words are rare and the model won't learn much about them.")
    bilingual_text("- This doesn't obviously provide a fixed vocabulary size.", "- This doesn't obviously provide a fixed 词表 size.")
    bilingual_text("- New words we haven't seen during training get a special UNK token, which is ugly and can mess up perplexity calculations.", "- New words we haven't seen during 训练 get a special UNK token, which is ugly and can mess up 困惑度 calculations.")


def bpe_tokenizer():
    bilingual_text("## Byte Pair Encoding (BPE)", '## 字节对编码（Byte Pair Encoding, BPE）')
    bilingual_text("The BPE algorithm was introduced by Philip Gage in 1994 for data compression. ", 'The BPE algorithm was introduced by Philip Gage in 1994 for 数据 compression.'), article_link("http://www.pennelynn.com/Documents/CUJ/HTML/94HTML/19940045.HTM")
    bilingual_text("It was adapted to NLP for neural machine translation. ", '说明：It was adapted to NLP for neural machine translation.'), link(sennrich_2016)
    bilingual_text("(Previously, papers had been using word-based tokenization.)", '(Previously, papers had been using word-based 分词.)')
    bilingual_text("BPE was then used by GPT-2. ", '说明：BPE was then used by GPT-2.'), link(gpt2_2019)

    bilingual_text("Basic idea: *train* the tokenizer on raw text to construct a vocabulary tailored to the data.", 'Basic idea: train the 分词器 on raw text to construct a 词表 tailored to the 数据.')
    bilingual_text("Intuition: common sequences of bytes are represented by a single token, rare sequences are represented by many tokens.", 'Intuition: common sequences of 字节 are represented by a single token, rare sequences are represented by many token.')

    bilingual_text("Sketch: start with each byte as a token, and successively merge the most common pair of adjacent tokens.", 'Sketch: start with each 字节 as a token, and successively merge the most common pair of adjacent token.')

    bilingual_text("## Training the tokenizer", '## 训练分词器')
    string = "the cat in the hat"  # @inspect string
    params = train_bpe(string, num_merges=3)

    bilingual_text("## Using the tokenizer", '## 使用分词器')
    bilingual_text("Now, given a new text, we can encode it.", '现在，给定一段新文本，我们可以对它编码。')
    tokenizer = BPETokenizer(params)  # @stepover
    string = "the quick brown fox"  # @inspect string
    indices = tokenizer.encode(string)  # @inspect indices
    reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string @stepover
    assert string == reconstructed_string

    bilingual_text("In Assignment 1, you will go beyond this in the following ways:", '说明：In Assignment 1, you will go beyond this in the following ways:')
    bilingual_text("- encode() currently loops over all merges. Only loop over merges that matter.", '- 说明：encode() currently loops over all merges. Only loop over merges that matter.')
    bilingual_text("- Detect and preserve special tokens (e.g., <|endoftext|>).", '- 说明：Detect and preserve special tokens (e.g., <|endoftext|>).')
    bilingual_text("- Use pre-tokenization (e.g., the GPT-2 tokenizer regex).", '- Use pre-分词 (e.g., the GPT-2 分词器 regex).')
    bilingual_text("- Try to make the implementation as fast as possible.", '- 说明：Try to make the implementation as fast as possible.')


def train_bpe(string: str, num_merges: int) -> BPETokenizerParams:  # @inspect string, @inspect num_merges
    bilingual_text("Start with the list of bytes of `string`.", 'Start with the list of 字节 of string.')
    indices = list(map(int, string.encode("utf-8")))  # @inspect indices
    merges: dict[tuple[int, int], int] = {}  # index1, index2 => merged index
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # index -> bytes

    for i in range(num_merges):
        # Count the number of occurrences of each pair of tokens
        counts = count_adjacent_pairs(indices)  # @inspect counts @stepover

        # Find the most common pair
        pair = max(counts, key=counts.get)  # @inspect pair

        # Merge that pair
        new_index = 256 + i  # @inspect new_index
        merges[pair] = new_index  # @inspect merges
        vocab[new_index] = vocab[pair[0]] + vocab[pair[1]]  # @inspect vocab
        indices = merge(indices, pair, new_index)  # @inspect indices @stepover

    compression_ratio = get_compression_ratio(string, indices)  # @inspect compression_ratio

    return BPETokenizerParams(vocab=vocab, merges=merges)


def count_adjacent_pairs(indices: list[int]) -> dict[tuple[int, int], int]:
    """Return a dictionary mapping each adjacent pair of tokens in `indices` to the number of times it occurs."""
    counts = defaultdict(int)
    for index1, index2 in zip(indices, indices[1:]):
        counts[(index1, index2)] += 1
    return counts


if __name__ == "__main__":
    main()
