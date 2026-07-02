from edtrace import text, link, image
from lecture_util import post_link, bilingual_text
from references import mmlu_2021

def main():
    bilingual_text("## Lecture 12: evaluation", '## 第 12 讲：评测')
    bilingual_text("- So far: we've covered everything for training an LM (architecture, training, systems, scaling).", '- 到目前为止，我们已经覆盖了训练 LM 所需的一切（架构、训练、系统、规模化）。')
    bilingual_text("- Missing piece: what **data** do you train on?", '- 缺失的一块是：你用什么**数据**训练？')
    bilingual_text("- Data shapes model behavior (code? multilingual? DNA?).", '- 数据塑造模型行为（代码？多语言？DNA？）。')
    bilingual_text("- Before talking about data, need to talk about what behavior we want from a model.", '- 在讨论数据之前，需要先讨论我们希望模型具有什么行为。')

    bilingual_text("**Evaluation**: given a model, how \"**good**\" is it?", '**评测**：给定一个模型，它到底有多“**好**”？')

    what_is_good()

    perplexity()
    exam_benchmarks()
    chat_benchmarks()
    agentic_benchmarks()
    pure_reasoning_benchmarks()
    safety_benchmarks()

    realism()
    validity()
    how_to_think_about_evaluation()

    bilingual_text("Takeaways:", '要点：')
    bilingual_text("- There is no one true evaluation; choose the evaluation depending on what you're trying to measure.", '- 不存在唯一正确的评测；应根据你想衡量的内容选择评测。')
    bilingual_text("- Clearly state the rules of the game (methods versus models versus agents).", '- 清楚说明游戏规则（评测方法、模型还是智能体）。')
    bilingual_text("- Considerations: difficulty, realism, validity.", '- 需要考虑：难度、真实性、有效性。')


def what_is_good():
    bilingual_text("Evaluation might appear to be a mechanical process:", '评测看起来可能像一个机械过程：')
    bilingual_text("1. Define some prompts", '1. 定义一些提示。')
    bilingual_text("2. Send prompts to a model and get back responses", '2. 把提示发送给模型并取回回答。')
    bilingual_text("3. Compute accuracy", '3. 计算准确率。')

    bilingual_text("But actually, evaluation is a deep and important topic...", '但实际上，评测是一个深刻且重要的话题……')
    bilingual_text("...which shapes the development of AI.", '……它会塑造 AI 的发展。')

    bilingual_text("**Core challenge**: <font color=\"red\">abstract construct</font> → <font color=\"blue\">concrete metric</font>", '**核心挑战**：<font color="red">抽象构念</font> → <font color="blue">具体指标</font>。')

    bilingual_text("Maybe a model is good if it does well on benchmarks...", '也许，如果模型在基准上表现好，它就是好模型……')
    link(title="Artificial Analysis", url="https://artificialanalysis.ai/")
    image("images/artificial-analysis.png", width=800)

    bilingual_text("Maybe a model is good if it does well on benchmarks and is cheap to run...", '也许，如果模型在基准上表现好且运行便宜，它就是好模型……')
    image("images/artificial-analysis-cost.png", width=800)

    bilingual_text("Maybe a model is good if people prefer its responses...", '也许，如果人们更喜欢它的回答，它就是好模型……')
    link(title="Arena AI (formerly Chatbot Arena)", url="https://arena.ai/leaderboard")
    image("images/lmarena-leaderboard.png", width=400)

    bilingual_text("Maybe a model is good if people simply choose to use (and pay for) it...", '也许，如果人们只是选择使用它（并付费），它就是好模型……')
    link(title="OpenRouter", url="https://openrouter.ai/rankings")
    image("images/openrouter.png", width=600)


def perplexity():
    bilingual_text("- Recall: that a language model is a probability distribution **p(x)** over sequences of tokens.", '- 回忆：that a language model is a probability distribution p(x) over sequences of tokens.')
    bilingual_text("- Perplexity (1/p(D))^(1/|D|) measures whether p assigns high probability to some dataset D.", '- 困惑度 (1/p(D))^(1/|D|) measures whether p assigns high 概率 to some 数据集 D.')

    bilingual_text("- In pre-training, you minimize perplexity on the training set.", '- In pre-训练, you minimize 困惑度 on the 训练 set.')
    bilingual_text("- The obvious thing is to measure perplexity on the test set.", '- The obvious thing is to measure 困惑度 on the 测试集.')
    bilingual_text("- This is what people did traditionally in language modeling research.", '- This is what people did traditionally in language 模型ing research.')

    bilingual_text("Standard datasets:", '标准数据集：')
    bilingual_text("- Penn Treebank (WSJ)", '- 说明：Penn Treebank (WSJ)')
    bilingual_text("- WikiText-103 (Wikipedia)", '- 说明：WikiText-103 (Wikipedia)')
    bilingual_text("- One Billion Word Benchmark (from machine translation WMT11 - EuroParl, UN, news)", '- One Billion Word 基准 (from machine translation WMT11 - EuroParl, UN, news)')
    bilingual_text("Classic paradigm: in-distribution evaluation: train on train split and evaluate on test split of some dataset.", '经典范式：分布内评测，在某个数据集的训练划分上训练，在测试划分上评测。')
    bilingual_text("Pure CNNs+LSTMs on the One Billion Word Benchmark (perplexity 51.3 → 30.0) ", 'Pure CNNs+LSTMs on the One Billion Word 基准 (困惑度 51.3 → 30.0)'), link("https://arxiv.org/abs/1602.02410")

    bilingual_text("GPT-2:", 'GPT-2 示例：')
    bilingual_text("- Trained on WebText (40GB text, websites linked from Reddit)", '- 说明：Trained on WebText (40GB text, websites linked from Reddit)')
    bilingual_text("- Zero-shot on standard datasets (**out-of-distribution** evaluation)", '- Zero-shot on standard 数据集s (分布外 评测)')
    image("images/gpt2-perplexity.png", width=800)
    bilingual_text("- Works better on small datasets (PTB) where transfer is helpful, but not larger datasets (1BW)", '- Works better on small 数据集s (PTB) where transfer is helpful, but not larger 数据集s (1BW)')

    bilingual_text("Perplexity is all you need (more faith than science):", '困惑度就是你所需的一切（这更像信念而非科学）：')
    bilingual_text("- True distribution is t, model is p.", '- True distribution is t, 模型 is p.')
    bilingual_text("- Best possible perplexity is H(t) obtained iff p = t.", '- Best possible 困惑度 is H(t) obtained iff p = t.')
    bilingual_text("- If p = t, then solve all the tasks: p(solution | problem)", '- 说明：If p = t, then solve all the tasks: p(solution | problem)')
    bilingual_text("- So by pushing down on perplexity, we will eventually \"reach AGI\".", '- So by pushing down on 困惑度, we will eventually "reach AGI".')

    bilingual_text("Perplexity is maybe more than you need:", '困惑度也许比你需要的更多：')
    bilingual_text("- Example: *Stanford was founded in 1885*", '- 示例：Stanford was founded in 1885')
    bilingual_text("- Perplexity penalizes prediction on all tokens, some (e.g., *founded*) of which might not be relevant", '- 困惑度 penalizes prediction on all token, some (e.g., founded) of which might not be relevant')
    bilingual_text("- Solution: measure conditional perplexity p(response | prompt)^(1/|response|)", '- 解决方案：measure conditional perplexity p(response | prompt)^(1/|response|)')

    bilingual_text("Some benchmarks are perplexity in disguise:", '有些基准其实是伪装过的困惑度：')
    bilingual_text("- Cloze tasks (fill in the blank): LAMBADA ", '- 说明：Cloze tasks (fill in the blank): LAMBADA'), link("https://arxiv.org/abs/1606.06031")
    image("images/lambada.png", width=700)
    bilingual_text("- Multiple choice sentence completion: HellaSwag ", '- 说明：Multiple choice sentence completion: HellaSwag'), link("https://arxiv.org/pdf/1905.07830")
    image("images/hellaswag.png", width=500)

    bilingual_text("**Warning** (if you're running a perplexity leaderboard):", '**警告**（如果你在运行困惑度排行榜）：')
    bilingual_text("- People submit `LM` and you compute `log_prob = LM(test_data)`", '- People submit LM and you 计算量 log_prob = LM(test_数据)')
    bilingual_text("- You need to trust that the probabilities are valid (sum to 1)", '- You need to trust that the 概率 are valid (sum to 1)')
    bilingual_text("- For downstream tasks, `response = LM(prompt)` and compute accuracy on `response`", '- For downstream tasks, 回答 = LM(提示) and 计算量 准确率 on 回答')

    bilingual_text("Summary:", '总结：')
    bilingual_text("- Perplexity is still used heavily in language model development (smooth scaling laws)", '- 困惑度 is still used heavily in language 模型 development (smooth scaling laws)')
    bilingual_text("- Still need benchmarks that capture real-world situations (for the non-believers)...", '- Still need 基准 that capture real-world situations (for the non-believers)...')


def exam_benchmarks():
    bilingual_text("Exams are a useful way to test language models (as with humans):", '考试是测试语言模型的有用方式（和测试人类类似）：')
    bilingual_text("- Have control over the subject and difficulty", '- 说明：Have control over the subject and difficulty')
    bilingual_text("- Design to have unambiguous correct answer, easy to grade", '- 说明：Design to have unambiguous correct answer, easy to grade')

    bilingual_text("**Massive Multitask Language Understanding (MMLU)** ", '说明：Massive Multitask Language Understanding (MMLU)'), link(mmlu_2021)
    bilingual_text("- 57 subjects (e.g., math, US history, law, morality), multiple-choice", '- 57 subjects (e.g., math, US history, law, morality), 多项选择')
    bilingual_text("- \"collected by graduate and undergraduate students from freely available sources online\"", '- 说明："collected by graduate and undergraduate students from freely available sources online"')
    bilingual_text("- Despite the name, MMLU is really about testing knowledge, not language understanding", '- Despite the name, MMLU is really about testing 知识, not language understanding')
    bilingual_text("- Evaluated on GPT-3 using few-shot prompting", '- Evaluated on GPT-3 using few-shot 提示ing')
    image("images/mmlu.png", width=700)
    link("https://llm-stats.com/benchmarks/mmlu")
    link(title="HELM MMLU for visualizing predictions", url="https://crfm.stanford.edu/helm/mmlu/latest/")

    bilingual_text("**MMLU-Pro** ", '说明：MMLU-Pro'), link("https://arxiv.org/abs/2406.01574")
    bilingual_text("- Removed noisy/trivial questions from MMLU", '- 说明：Removed noisy/trivial questions from MMLU')
    bilingual_text("- Expanded 4 choices to 10 choices", '- 说明：Expanded 4 choices to 10 choices')
    bilingual_text("- Evaluated using chain of thought (gives model more of a chance)", '- Evaluated using chain of thought (gives 模型 more of a chance)')
    bilingual_text("- Accuracy of models drop by 16% to 33% (not as saturated)", '- 准确率 of 模型s drop by 16% to 33% (not as saturated)')
    image("images/mmlu-pro.png", width=700)
    link("https://llm-stats.com/benchmarks/mmlu-pro")
    link(title="HELM MMLU-Pro for visualizing predictions", url="https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/mmlu_pro")

    bilingual_text("**Graduate-Level Google-Proof Q&A (GPQA)** ", '说明：Graduate-Level Google-Proof Q&A (GPQA)'), link("https://arxiv.org/abs/2311.12022")
    bilingual_text("- Questions written by 61 PhD contractors from Upwork", '- 说明：Questions written by 61 PhD contractors from Upwork')
    image("images/gpqa.png", width=700)
    bilingual_text("- PhD experts achieve 65% accuracy", '- PhD experts achieve 65% 准确率')
    bilingual_text("- Non-experts achieve 34% over 30 minutes with access to Google", '- 说明：Non-experts achieve 34% over 30 minutes with access to Google')
    bilingual_text("- GPT-4 achieves 39%", '- 说明：GPT-4 achieves 39%')
    link("https://llm-stats.com/benchmarks/gpqa")
    link(title="HELM GPQA for visualizing predictions", url="https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/gpqa")

    bilingual_text("**Humanity's Last Exam (HLE)** ", "说明：Humanity's Last Exam (HLE)"), link("https://arxiv.org/abs/2501.14249")
    bilingual_text("- 2500 questions: multimodal, many subjects, multiple-choice + short-answer", '- 2500 questions: multimodal, many subjects, 多项选择 + short-answer')
    image("images/hle-examples.png", width=700)
    bilingual_text("- Awarded $500K prize pool + co-authorship to question creators", '- 说明：Awarded $500K prize pool + co-authorship to question creators')
    bilingual_text("- Filtered by frontier LLMs, multiple stages of review", '- 说明：Filtered by frontier LLMs, multiple stages of review')
    image("images/hle-pipeline.png", width=700)
    image("images/hle-results.png", width=600)
    link("https://llm-stats.com/benchmarks/hle")

    bilingual_text("Summary:", '总结：')
    bilingual_text("- Trend towards harder questions as models improve and saturate existing benchmarks", '- Trend towards harder questions as 模型s improve and saturate existing 基准')
    bilingual_text("- Multiple-choice format can be as difficult as one wants", '- 多项选择 format can be as difficult as one wants')
    bilingual_text("- Does not capture real usage (open-ended, doesn't necessarily exist correct answer)", "- Does not capture real usage (开放式, doesn't necessarily exist correct answer)")


def chat_benchmarks():
    bilingual_text("- So far, we've been evaluating on well-defined multiple-choice tasks.", "- So far, we've been evaluating on well-defined 多项选择 tasks.")
    bilingual_text("- Most people don't ask multiple-choice exam questions to their AI assistant.", "- Most people don't ask 多项选择 exam questions to their AI assistant.")
    
    bilingual_text("Example:", '示例：')
    bilingual_text("Prompt: *I would like to make a beet salad with goat cheese. What kind of herbs would work well and what would not work well?*", '提示: I would like to make a beet salad with goat cheese. What kind of herbs would work well and what would not work well?')
    bilingual_text("Response: *Here’s a breakdown of herbs that work well (and some that don’t) in a beet + goat cheese salad, based on how their flavors interact with the sweet-earthiness of beets and the tangy creaminess of goat cheese...", '回答: Here’s a breakdown of herbs that work well (and some that don’t) in a beet + goat cheese salad, based on how their flavors interact with the sweet-earthiness of beets and the tangy creaminess of goat cheese...')

    bilingual_text("**Challenge**: how to evaluate an open-ended response?", 'Challenge: how to evaluate an 开放式 回答?')

    bilingual_text("**Chatbot Arena** ", '**Chatbot Arena**（聊天机器人竞技场）'), link("https://arxiv.org/abs/2403.04132")
    bilingual_text("Data collection:", '数据收集：')
    bilingual_text("- Random person from the Internet types in prompt", '- Random person from the Internet types in 提示')
    bilingual_text("- They get response from two random (anonymized) models", '- They get 回答 from two random (anonymized) 模型s')
    bilingual_text("- They rate which one is better", '- 说明：They rate which one is better')
    image("images/arena-beets.png", width=700)
    bilingual_text("Compute ELO rankings based on pairwise comparisons:", '基于成对比较计算 ELO 排名：')
    bilingual_text("- Define model: p(A wins against B) = 1 / (1 + 10^((ELO_B - ELO_A)/400))", '- Define 模型: p(A wins against B) = 1 / (1 + 10^((ELO_B - ELO_A)/400))')
    bilingual_text("- Fit this model to maximize probability of pairwise comparisons", '- Fit this 模型 to maximize 概率 of 成对比较')
    link(title="Arena AI (formerly Chatbot Arena)", url="https://arena.ai/leaderboard")
    image("images/lmarena-leaderboard.png", width=400)
    bilingual_text("Properties:", '性质：')
    bilingual_text("- Real-world prompts (free for users, incentives to actually use it)", '- Real-world 提示 (free for users, incentives to actually use it)')
    bilingual_text("- But who are these people? biases? spammers?", '- 说明：But who are these people? biases? spammers?')
    bilingual_text("- Binary preference but conflates style and correctness", '- 说明：Binary preference but conflates style and correctness')
    bilingual_text("- How does the human even assess correctness?  Prone to sycophancy?", '- 说明：How does the human even assess correctness?  Prone to sycophancy?')
    bilingual_text("- Feature: don't need to feed same prompts to all models (important because human is rating)", "- Feature: don't need to feed same 提示 to all 模型s (important because human is rating)")
    bilingual_text("- Dynamic: incorporates new prompts and models over time", '- Dynamic: incorporates new 提示 and 模型s over time')

    bilingual_text("**AlpacaEval** (2023)", '说明：AlpacaEval (2023)'), link(title="leaderboard", url="https://tatsu-lab.github.io/alpaca_eval/")
    bilingual_text("- 805 instructions from various sources", '- 说明：805 instructions from various sources')
    bilingual_text("- Metric: win rate against baseline model (GPT-4 preview) as judged by GPT-4 preview (potential bias?)", '- 指标: win rate against baseline 模型 (GPT-4 preview) as judged by GPT-4 preview (potential bias?)')
    bilingual_text("- Problem: LLM judges favor longer responses, resulted in leaderboard gaming", '- Problem: LLM 裁判 favor longer 回答, resulted in leaderboard gaming')
    bilingual_text("- Alpaca Eval 2.0 used regression to debias the metric ", '- Alpaca Eval 2.0 used regression to debias the 指标'), link("https://arxiv.org/pdf/2404.04475")
    bilingual_text("- How do we evaluate the metric?", '- How do we evaluate the 指标?')
    bilingual_text("- Correlation with Chatbot Arena (humans) is high:", '- 说明：Correlation with Chatbot Arena (humans) is high:')
    image("https://github.com/tatsu-lab/alpaca_eval/raw/main/figures/chat_correlations_no_ae.png", width=500)
    image("images/alpacaeval-leaderboard.png", width=400)

    bilingual_text("**WildBench** ", '说明：WildBench'), link("https://arxiv.org/pdf/2406.04770")
    bilingual_text("- Sourced 1024 examples from 1M human-chatbot conversations", '- 说明：Sourced 1024 examples from 1M human-chatbot conversations')
    bilingual_text("- Uses GPT-4 turbo as a judge with a checklist (like CoT for judging) + GPT-4 as a judge", '- Uses GPT-4 turbo as a judge with a 检查清单 (like CoT for judging) + GPT-4 as a judge')
    bilingual_text("- Well-correlated with Chatbot Arena (seems to be the de facto sanity check)", '- 说明：Well-correlated with Chatbot Arena (seems to be the de facto sanity check)')
    image("images/wildbench.png", width=700)
    link(title="HELM WildBench for visualizing predictions", url="https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/wildbench")

    bilingual_text("Summary:", '总结：')
    bilingual_text("- Challenge: how to evaluate open-ended responses?", '- Challenge: how to evaluate 开放式 回答?')
    bilingual_text("- Pairwise comparisons between similar responses provide higher signal", '- 成对比较 between similar 回答 provide higher signal')
    bilingual_text("- Beware of biases (both from humans and LLM judges)", '- Beware of biases (both from humans and LLM 裁判)')
    bilingual_text("- Checklist/rubric improves reliability (regardless of human or LLM judge)", '- 检查清单/评分量规 improves reliability (regardless of human or LLM judge)')


def agentic_benchmarks():
    bilingual_text("Previously: evaluate what LMs say (chat)", '之前：评测 LM 说什么（聊天）。')
    bilingual_text("Now: evaluate what LMs do (agents)", '现在：评测 LM 做什么（智能体）。')

    bilingual_text("Agent = language model + agent scaffold (logic for deciding how to use the LM)", '智能体 = 语言模型 + 智能体脚手架（决定如何使用 LM 的逻辑）。')
    
    bilingual_text("Consider tasks that require tool use (e.g., running code) and iterating over a period of time", '考虑需要使用工具（例如运行代码）并在一段时间内迭代的任务。')

    bilingual_text("**SWEBench** ", '说明：SWEBench'), link("https://arxiv.org/abs/2310.06770")
    bilingual_text("- 2294 tasks across 12 Python repositories", '- 说明：2294 tasks across 12 Python repositories')
    bilingual_text("- Given codebase + issue description, submit a PR", '- 说明：Given codebase + issue description, submit a PR')
    bilingual_text("- Evaluation metric: unit tests", '- 评测 指标: unit tests')
    image("images/swebench.png", width=800)
    link("https://llm-stats.com/benchmarks/swe-bench-verified")

    bilingual_text("**TerminalBench** ", '说明：TerminalBench'), link("https://arxiv.org/abs/2601.11868"), link(title="website", url="https://www.tbench.ai/")
    image("images/terminal-bench.png", width=700)
    bilingual_text("- Computer terminal environments: simple and universal", '- 计算量r terminal environments: simple and universal')
    bilingual_text("- 229 tasks crowdsourced from 93 contributors, 89 tasks constitute Terminal-Bench 2.0", '- 说明：229 tasks crowdsourced from 93 contributors, 89 tasks constitute Terminal-Bench 2.0')
    image("images/terminal-bench-human-time.png", width=600)
    image("images/terminal-bench-results.png", width=600)
    link("https://llm-stats.com/benchmarks/terminal-bench")

    bilingual_text("**CyBench** ", '说明：CyBench'), link("https://arxiv.org/abs/2408.08926")
    image("images/cybench.png", width=700)
    bilingual_text("- 40 Capture the Flag (CTF) tasks", '- 说明：40 Capture the Flag (CTF) tasks')
    bilingual_text("- Use first-solve time as a measure of difficulty", '- 说明：Use first-solve time as a measure of difficulty')
    image("images/cybench-agent.png", width=700)
    image("images/cybench-results.png", width=600)
    link("https://llm-stats.com/benchmarks/cybench")

    bilingual_text("**MLEBench** ", '说明：MLEBench'), link("https://arxiv.org/abs/2410.07095")
    bilingual_text("- 75 Kaggle competitions (require training models, processing data, etc.)", '- 75 Kaggle competitions (require 训练 模型s, processing 数据, etc.)')
    image("images/mlebench.png", width=800)
    image("images/mlebench-results.png", width=700)

    bilingual_text("Agent scaffolds ", '智能体 scaffolds'), post_link("https://www.philschmid.de/agents-2.0-deep-agents")
    image("https://www.philschmid.de/static/blog/agents-2.0-deep-agents/overview.png", width=400)
    bilingual_text("- Explicit planning: keep a todo list that gets checked off", '- 说明：Explicit planning: keep a todo list that gets checked off')
    bilingual_text("- Hierarchical delegation: agents calling other sub-agents (clean context)", '- Hierarchical delegation: 智能体 calling other sub-智能体 (clean context)')
    bilingual_text("- Persistent memory: read/write files", '- Persistent 内存: read/write files')
    bilingual_text("- Extreme context engineering: explicit more instructions on process", '- 说明：Extreme context engineering: explicit more instructions on process')

    bilingual_text("Summary:", '总结：')
    bilingual_text("- Agents dramatically enhance the capability surface of language models", '- 智能体 dramatically enhance the capability surface of language 模型s')
    bilingual_text("- Agent scaffolds are very important", '- 智能体 scaffolds are very important')
    bilingual_text("- Evaluating agents = evaluating agent scaffold + language model", '- Evaluating 智能体 = evaluating 智能体 scaffold + language 模型')


def pure_reasoning_benchmarks():
    bilingual_text("- All of the tasks so far require linguistic and world knowledge.", '- All of the tasks so far require linguistic and 世界知识.')
    bilingual_text("- Can we isolate **reasoning** from knowledge?", '- Can we isolate 推理能力 from 知识?')
    bilingual_text("- Arguably, reasoning captures a more pure form of intelligence (isn't just about memorizing facts).", "- Arguably, 推理能力 captures a more pure form of intelligence (isn't just about memorizing facts).")

    bilingual_text("**ARC-AGI** ", '说明：ARC-AGI'), link(title="website", url="https://arcprize.org/arc-agi")
    bilingual_text("- 100\% solvable by humans, but challenging for AI", '- 说明：100\\% solvable by humans, but challenging for AI')
    bilingual_text("- Each task is unique, so memorization doesn't help.", "- 说明：Each task is unique, so memorization doesn't help.")

    bilingual_text("- ARC-AGI-1 (2019): first iteration", '- 说明：ARC-AGI-1 (2019): first iteration')
    image("https://arcprize.org/media/images/arc-task-grids.jpg", width=800)

    bilingual_text("- ARC-AGI-2 (March 2025): more multi-step reasoning", '- ARC-AGI-2 (March 2025): more multi-step 推理能力')
    image("https://arcprize.org/media/images/blog/arc-agi-2-unsolved-1.png", width=800)

    image("images/arc-agi-results.png", width=700)
    bilingual_text("- Pretrained language models didn't move the needle", "- Pretrained language 模型s didn't move the needle")
    bilingual_text("- Reasoning models (o1, o3) started making things take off", '- 推理能力 模型s (o1, o3) started making things take off')

    bilingual_text("- ARC-AGI-3 (March 2026): interactive environments ", '- 说明：ARC-AGI-3 (March 2026): interactive environments'), post_link("https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf")
    image("images/arc-agi-3.png", width=300)
    image("images/arc-agi-3-results.png", width=500)

    bilingual_text("Summary:", '总结：')
    bilingual_text("- Goal is to disentangle reasoning from knowledge (difficult to do!)", '- Goal is to disentangle 推理能力 from 知识 (difficult to do!)')
    bilingual_text("- Constrained to human reasoning (not superhuman reasoning)", '- Constrained to human 推理能力 (not superhuman 推理能力)')
    bilingual_text("- Clearly exposes gaps in current models", '- Clearly exposes gaps in current 模型s')


def safety_benchmarks():
    image("https://www.team-bhp.com/forum/attachments/road-safety/2173645d1625144681-will-crash-test-rating-change-if-higher-variant-chosen-images-30.jpeg", width=400)
    bilingual_text("What does safety mean for AI?", 'AI 的安全意味着什么？')

    bilingual_text("**HarmBench** ", '说明：HarmBench'), link("https://arxiv.org/abs/2402.04249")
    bilingual_text("- Based on 510 harmful behaviors that violate laws or norms", '- 说明：Based on 510 harmful behaviors that violate laws or norms')
    link(title="HarmBench on HELM", url="https://crfm.stanford.edu/helm/safety/latest/#/leaderboard/harm_bench")
    link(title="Example of safety failure", url="https://crfm.stanford.edu/helm/safety/latest/#/runs/harm_bench:model=anthropic_claude-3-7-sonnet-20250219?instancesPage=4")

    bilingual_text("**AIR-Bench** ", '说明：AIR-Bench'), link("https://arxiv.org/abs/2407.17436")
    bilingual_text("- Based on regulatory frameworks and company policies", '- 说明：Based on regulatory frameworks and company policies')
    bilingual_text("- Taxonomized into 314 risk categories, 5694 prompts", '- Taxonomized into 314 risk categories, 5694 提示')
    image("https://crfm.stanford.edu/helm/assets/air-overview-DpBbyagA.png", width=800)
    link(title="HELM AIR-Bench", url="https://crfm.stanford.edu/helm/air-bench/latest/#/leaderboard")

    bilingual_text("Jailbreaking:", '越狱：')
    bilingual_text("- Language models are trained to refuse harmful instructions", '- Language 模型s are trained to refuse harmful instructions')
    bilingual_text("- Greedy Coordinate Gradient (GCG) automatically optimizes prompts to bypass safety ", '- Greedy Coordinate Gradient (GCG) automatically optimizes 提示 to bypass 安全'), link("https://arxiv.org/pdf/2307.15043")
    bilingual_text("- Transfers from open-weight models (Llama) to closed models (GPT-4)", '- Transfers from open-weight 模型s (Llama) to closed 模型s (GPT-4)')
    image("images/gcg-examples.png", width=800)

    bilingual_text("What is safety?", '什么是安全？')
    bilingual_text("- Many aspects of safety are strongly contextual (politics, law, social norms - which vary across countries)", '- Many aspects of 安全 are strongly contextual (politics, law, social norms - which vary across countries)')
    bilingual_text("- Many risks are quite varied (hallucinations, sycophancy, abetting crimes, inequality, losing critical thinking)", '- 说明：Many risks are quite varied (hallucinations, sycophancy, abetting crimes, inequality, losing critical thinking)')

    bilingual_text("**Dual-use**: capable cybersecurity agents (Mythos) can be used to hack into a system or to do penetration testing", 'Dual-use: capable cybersecurity 智能体 (Mythos) can be used to hack into a system or to do penetration testing')


def realism():
    bilingual_text("**Ecological validity**: how well does an evaluation capture real-world use?", 'Ecological 有效性: how well does an 评测 capture real-world use?')
    bilingual_text("- Exam benchmarks (e.g., GPQA) are far away from real-world use.", '- Exam 基准 (e.g., GPQA) are far away from real-world use.')
    bilingual_text("- Chatbot Arena prompts are from real people, but distribution is uncontrolled.", '- Chatbot Arena 提示 are from real people, but distribution is uncontrolled.')

    bilingual_text("**GDPVal** (OpenAI) ", '说明：GDPVal (OpenAI)'), link("https://arxiv.org/pdf/2510.04374")
    bilingual_text("- 44 occupations from top 9 sectors according to US GDP", '- 说明：44 occupations from top 9 sectors according to US GDP')
    bilingual_text("- Tasks come from professionals with ~14 years of experience", '- 说明：Tasks come from professionals with ~14 years of experience')
    image("images/gdpval.png", width=700)

    bilingual_text("**MedHELM** ", '说明：MedHELM'), link("https://arxiv.org/abs/2505.23802")
    bilingual_text("- Previous medical benchmarks were based on standardized exams", '- Previous medical 基准 were based on standardized exams')
    bilingual_text("- 121 clinical tasks sourced from 29 clinicians, mixture of private and public datasets", '- 121 clinical tasks sourced from 29 clinicians, mixture of private and public 数据集s')
    image("https://crfm.stanford.edu/helm/assets/medhelm-overview-CND0EIsy.png", width=700)
    link(title="MedHELM", url="https://crfm.stanford.edu/helm/medhelm/latest/#/leaderboard")

    bilingual_text("**Clio** (Anthropic) ", '说明：Clio (Anthropic)'), link("https://arxiv.org/abs/2412.13678")
    bilingual_text("- Use language models to analyze real user data", '- Use language 模型s to analyze real user 数据')
    bilingual_text("- Share general patterns of what people are asking", '- 说明：Share general patterns of what people are asking')
    image("images/clio-table4.png", width=700)

    bilingual_text("Unfortunately, realism and privacy are sometimes at odds with each other.", '不幸的是，真实性和隐私有时彼此冲突。')


def validity():
    bilingual_text("How do we know our evaluations are valid?", '我们如何知道评测是有效的？')

    bilingual_text("### Train-test overlap", '### 训练-测试重叠')
    bilingual_text("- Machine learning 101: don't train on your test set", "- Machine learning 101: don't train on your 测试集")
    bilingual_text("- Pre-foundation models (ImageNet, SQuAD): well-defined train-test splits", '- Pre-foundation 模型s (ImageNet, SQuAD): well-defined train-test splits')
    bilingual_text("- Today: train on the Internet and don't tell people about your data", "- Today: train on the Internet and don't tell people about your 数据")

    bilingual_text("Route 1: try to infer train-test overlap from model", 'Route 1: try to infer 训练-测试重叠 from 模型')
    bilingual_text("- Exploit exchangeability of data points ", '- Exploit exchangeability of 数据 points'), link("https://arxiv.org/pdf/2310.17623")
    image("images/contamination-exchangeability.png", width=500)

    bilingual_text("Route 2: encourage reporting norms (e.g., people report confidence intervals)", '说明：Route 2: encourage reporting norms (e.g., people report confidence intervals)')
    bilingual_text("- Model providers should report train-test overlap ", '- 模型 providers should report 训练-测试重叠'), link("https://arxiv.org/abs/2410.08385")

    bilingual_text("Route 3: use fresh evals", '说明：Route 3: use fresh evals')
    bilingual_text("- LiveCodeBench, UncheatableEval: scrape new webpages", '- 说明：LiveCodeBench, UncheatableEval: scrape new webpages')
    bilingual_text("- Timestamps aren't always safe due to copying either", "- 说明：Timestamps aren't always safe due to copying either")

    bilingual_text("Route 4: use private evals", '说明：Route 4: use private evals')
    bilingual_text("- Companies use internal code bases that aren't on the Internet", "- 说明：Companies use internal code bases that aren't on the Internet")
    bilingual_text("- Use your personal writings", '- 说明：Use your personal writings')
    bilingual_text("- Easiest for perplexity", '- Easiest for 困惑度')

    bilingual_text("### Dataset quality", '### 数据集质量')
    bilingual_text("- Fixed up SWE-Bench to produce SWE-Bench Verified ", '- 说明：Fixed up SWE-Bench to produce SWE-Bench Verified'), post_link("https://openai.com/index/introducing-swe-bench-verified/")
    bilingual_text("- Create Platinum versions of benchmarks ", '- Create Platinum versions of 基准'), link("https://arxiv.org/abs/2502.03461")
    image("https://pbs.twimg.com/media/GjICXQlWkAAYnDS?format=jpg&name=4096x4096", width=700)
    image("https://pbs.twimg.com/media/GjICcGQXYAAM4o1?format=jpg&name=4096x4096", width=800)
    bilingual_text("- Problems with agentic benchmarks: insufficient test cases, trivial agent can solve task ", '- Problems with 智能体ic 基准: insufficient test cases, trivial 智能体 can solve task'), link("https://arxiv.org/abs/2507.02825")
    bilingual_text("- Docent: use LLM to inspect agent traces to detect problems ", '- Docent: use LLM to inspect 智能体 traces to detect problems'), post_link("https://transluce.org/introducing-docent")


def how_to_think_about_evaluation():
    bilingual_text("### What's the point of evaluation?", '### 评测的意义是什么？')
    bilingual_text("There is no one true evaluation; it depends on what question you're trying to answer.", '不存在唯一正确的评测；它取决于你想回答什么问题。')
    bilingual_text("1. User or company wants to make a purchase decision (model A or model B) for their use case (e.g., customer service chatbots).", '1. User or company wants to make a purchase decision (模型 A or 模型 B) for their use case (e.g., customer service chatbots).')
    bilingual_text("2. Researchers want to measure the raw capabilities of a model (e.g., intelligence).", '2. Researchers want to measure the raw capabilities of a 模型 (e.g., intelligence).')
    bilingual_text("3. We want to understand the benefits + harms of a model (for business and policy reasons).", '3. We want to understand the benefits + harms of a 模型 (for business and policy reasons).')
    bilingual_text("4. Model developers want to get feedback to improve the model.", '4. 模型 developers want to get feedback to improve the 模型.')

    bilingual_text("### What are we evaluating?", '### 我们在评测什么？')
    bilingual_text("- Pre-foundation models, we evaluated **methods** (standardized train-test splits).", '- Pre-foundation 模型s, we evaluated 方法s (standardized train-test splits).')
    bilingual_text("- Today, we're (mostly) evaluating **models/systems** (anything goes).", "- Today, we're (mostly) evaluating 模型s/系统 (anything goes).")

    bilingual_text("There are some exceptions...", '也有一些例外……')
    bilingual_text("- nanogpt speedrun: fixed data, compute time to get to a particular validation loss", '- nanogpt speedrun: fixed 数据, 计算量 time to get to a particular validation loss')
    image("images/karpathy-nanogpt-speedrun.png", width=600), post_link("https://x.com/karpathy/status/1846790537262571739")

    bilingual_text("Evaluating methods encourage algorithmic innovation from researchers.", '评测方法会鼓励研究者进行算法创新。')
    bilingual_text("Evaluating models/systems is useful for downstream users.", '评测模型/系统对下游用户有用。')

    bilingual_text("Either way, we need to define the rules of the game!", '无论哪种方式，我们都需要定义游戏规则！')


if __name__ == "__main__":
    main()
