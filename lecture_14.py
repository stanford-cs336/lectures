from dataclasses import dataclass
import numpy as np
import itertools
import mmh3
from edtrace.file_util import download_file
from edtrace import text, image, link
from lecture_13 import the_pile
from lecture_util import article_link, post_link, bilingual_text, bilingual_verbatim
from references import dolma_2024, the_pile_2020, dclm_2024

def main():
    bilingual_text("## Lecture 14: Data II", '## 第 14 讲：数据 II')
    bilingual_text("Last lecture:", '上节课：')
    bilingual_text("- Live service (e.g., GitHub) → dump/crawl (e.g., GitHub Archive) → processed data (e.g., The Stack)", '- Live service (e.g., GitHub) → dump/crawl (e.g., GitHub Archive) → processed 数据 (e.g., The Stack)')
    bilingual_text("- Considerations: terms of service, copyright (licenses or fair use)", '- Considerations: terms of service, 版权 (许可s or 合理使用)')

    bilingual_text("This lecture:", '本节课：')
    bilingual_text("- Data pipeline: transformation, filtering, deduplication, mixing", '- 数据 pipeline: 转换, 过滤, 去重, 混合')
    bilingual_text("- Mid-training + SFT: synthetic data", '- Mid-训练 + SFT: synthetic 数据')

    # Data pipeline
    transformation()
    filtering()
    deduplication()
    data_mixing()

    # Post-training data
    post_training_data()

    bilingual_text("Summary:", '总结：')
    bilingual_text("- Filtering: train classifier (language id, quality, toxicity) for what good looks like", '- 过滤: train 分类器 (language id, 质量, 毒性) for what good looks like')
    bilingual_text("- Deduplication: hashing scales to large datasets for fuzzy matching", '- 去重: 哈希ing scales to large 数据sets for fuzzy matching')
    bilingual_text("- Mixing: try mixtures at small scale, extrapolate to optimal mixture and large scale", '- 混合: try 混合比例s at small scale, extrapolate to optimal 混合比例 and large scale')
    bilingual_text("- Applications: language identification, quality filtering, toxicity filtering", '- Applications: 语言识别, 质量 过滤, 毒性 过滤')
    bilingual_text("- Post-training data: looks like evaluations, use of synthetic data", '- Post-训练 数据: looks like evaluations, use of synthetic 数据')
    bilingual_text("- A lot of data work is domain-specific, looking at examples, etc.", '- A lot of 数据 work is domain-specific, looking at examples, etc.')


def transformation():
    bilingual_text("Raw data does not come as text.", '原始数据并不是以纯文本形式出现。')
    bilingual_text("It is HTML, PDF (arxiv), or directories (code repositories).", 'It is HTML, PDF (arxiv), or directories (代码 仓库).')

    bilingual_text("HTML to text (main one):", 'HTML 转文本（主要情况）：')
    bilingual_text("- Remove boilerplate (e.g., navigation, ads) and extract content", '- 说明：Remove boilerplate (e.g., navigation, ads) and extract content')
    bilingual_text("- What about images, tables, etc.?", '- What about 图像, tables, etc.?')
    bilingual_text("- Inherently lossy (need to linearize)", '- Inherently 损失y (need to linearize)')
    bilingual_text("- Tools (rule-based): trafilatura, resiliparse, jusText, lynx, etc.", '- Tools (rule-based): trafilatura, resiliparse, jus文本, lynx, etc.')
    bilingual_text("- Accuracy matters: ", '- 说明：Accuracy matters:'), link(dclm_2024)
    image("images/dclm-wet.png", width=300)

    bilingual_text("FinePDFs ", '说明：FinePDFs'), post_link("https://huggingface.co/spaces/HuggingFaceFW/FinePDFsBlog")
    image("https://huggingfacefw-finepdfsblog.hf.space/_astro/pdf-description.Cb49jXc6_Z17eX4E.webp", width=600)
    bilingual_text("- Source: Common Crawl", '- 来源: Common Crawl')
    bilingual_text("- Recrawl truncated PDFs (since they are big)", '- 说明：Recrawl truncated PDFs (since they are big)')
    bilingual_text("- OCR (RolmOCR) using a VLM or Docling (make these run fast)", '- 说明：OCR (RolmOCR) using a VLM or Docling (make these run fast)')
    bilingual_text("- Lots of cleanup and filtering", '- Lots of cleanup and 过滤')
    bilingual_text("- A lot of layout information is missing", '- 说明：A lot of layout information is missing')


def filtering():
    bilingual_text("Algorithmic building block:", '算法构建块：')
    bilingual_text("- Given some **target data** T and lots of **raw data** R, find subset T' of R similar to T.", "- Given some target 数据 T and lots of raw 数据 R, find subset T' of R similar to T.")
    image("images/raw-target-schema.png", width=600)

    bilingual_text("Applications:", '应用：')
    bilingual_text("- Language identification (English versus rest)", '- 语言识别 (English versus rest)')
    bilingual_text("- Quality filtering (high quality versus low quality)", '- 质量 过滤 (high 质量 versus low 质量)')
    bilingual_text("- Toxicity filtering (non-toxic versus toxic)", '- 毒性 过滤 (non-toxic versus toxic)')

    bilingual_text("Desiderata for filtering algorithm:", '过滤算法的期望性质：')
    bilingual_text("- Generalize from the target data (want T and T' to be different)", "- Generalize from the target 数据 (want T and T' to be different)")
    bilingual_text("- Extremely fast (have to run it on R, which is huge)", '- 说明：Extremely fast (have to run it on R, which is huge)')

    bilingual_text("Survey paper on data selection ", 'Survey paper on 数据 selection'), link("https://arxiv.org/abs/2402.16827")

    bilingual_text("General framework: Given target T and raw R, find subset of R similar to T", '说明：General framework: Given target T and raw R, find subset of R similar to T')
    bilingual_text("1. Estimate some model based on R and T and derive a scoring function", '1. Estimate some 模型 based on R and T and derive a scoring function')
    bilingual_text("2. Keep examples in R based on their score", '2. Keep examples in R based on their 分数')

    bilingual_text("Types of classifiers:", '分类器类型：')
    bilingual_text("- Generative model of T (KenLM): score(x) = p_T(x)", '- Generative 模型 of T (KenLM): 分数(x) = p_T(x)')
    bilingual_text("- Simple classifier (fastText): score(x) = p(T | x)", '- Simple 分类器 (fast文本): 分数(x) = p(T | x)')
    bilingual_text("To use: keep examples x with score(x) >= threshold (stochastically)", 'To use: keep examples x with 分数(x) >= 阈值 (stochastically)')

    bilingual_text("Model-based filtering?", '基于模型的过滤？')
    bilingual_text("- Some deliberately do not use model-based filtering (C4, Gopher, RefinedWeb, FineWeb, Dolma)", '- Some deliberately do not use 模型-based 过滤 (C4, Gopher, Refined网络, Fine网络, Dolma)')
    bilingual_text("- Some use model-based filtering (GPT-3, LLaMA, DCLM) [becoming the norm]", '- Some use 模型-based 过滤 (GPT-3, LLaMA, DCLM) [becoming the norm]')

    bilingual_text("Language identification:", '语言识别：')
    bilingual_text("- Goal: find text of a specific language (e.g., English)", '- 目标：find text of a specific language (e.g., English)')
    bilingual_text("- fastText language identification ", '- fast文本 语言识别'), article_link("https://fasttext.cc/docs/en/language-identification.html")
    bilingual_text("- Off-the-shelf classifier", '- Off-the-shelf 分类器')
    bilingual_text("- Supports 176 languages", '- 说明：Supports 176 languages')
    bilingual_text("- Trained on multilingual sites: Wikipedia, Tatoeba (translation site) and SETimes (Southeast European news)", '- 说明：Trained on multilingual sites: Wikipedia, Tatoeba (translation site) and SETimes (Southeast European news)')
    bilingual_text("- Dolma keeps pages with p(English) >= 0.5 ", '- 说明：Dolma keeps pages with p(English) >= 0.5'), link(dolma_2024)

    bilingual_text("OpenMathText ", 'OpenMath文本'), link("https://arxiv.org/pdf/2310.06786")
    bilingual_text("- Goal: curate large corpus of mathematical text from CommonCrawl", '- 目标：curate large corpus of mathematical text from CommonCrawl')
    bilingual_text("- Use rules to filter (e.g., contains latex commands)", '- 说明：Use rules to filter (e.g., contains latex commands)')
    bilingual_text("- KenLM trained on ProofPile, keep if perplexity < 15000", '- 说明：KenLM trained on ProofPile, keep if perplexity < 15000')
    bilingual_text("- Trained fastText classifier to predict mathematical writing, threshold is 0.17 if math, 0.8 if no math", '- Trained fast文本 分类器 to predict mathematical writing, 阈值 is 0.17 if math, 0.8 if no math')
    bilingual_text("- Result: produced 14.7B tokens, used to train 1.4B models that do better than models trained on 20x data", '- 结果：produced 14.7B tokens, used to train 1.4B models that do better than models trained on 20x data')

    bilingual_text("GPT-3 ", '说明：GPT-3'), link("https://arxiv.org/pdf/2005.14165")  # Appendix A
    bilingual_text("- Positives: samples from {Wikipedia, WebText2, Books1, Books2}", '- Positives: samples from {Wikipedia, 网络文本2, Books1, Books2}')
    bilingual_text("- Negatives: samples from CommonCrawl", '- 说明：Negatives: samples from CommonCrawl')
    bilingual_text("Train linear classifier based on word features ", 'Train linear 分类器 based on word features'), article_link("https://spark.apache.org/docs/latest/ml-features#tokenizer")
    bilingual_text("Keep documents stochastically based on score", 'Keep 文档 stochastically based on 分数')
    def keep_document(score: float) -> bool:
        return np.random.pareto(9) > 1 - score

    bilingual_text("LLaMA/RedPajama ", '说明：LLaMA/RedPajama'), link("https://arxiv.org/pdf/2302.13971")
    bilingual_text("- Positives: samples from pages **referenced** by Wikipedia", '- 说明：Positives: samples from pages referenced by Wikipedia')
    bilingual_text("- Negatives: samples from CommonCrawl", '- 说明：Negatives: samples from CommonCrawl')
    bilingual_text("- Keep documents that are classified positive", '- Keep 文档 that are classified positive')

    bilingual_text("phi-1 ", '说明：phi-1'), link("https://arxiv.org/pdf/2306.11644")
    bilingual_text("- Philosophy: really high quality data (textbooks) to train a small model (1.5B)", '- 理念：really high quality data (textbooks) to train a small model (1.5B)')
    bilingual_text("- Includes synthetic data from GPT 3.5 (later: GPT-4) and filtered data", '- Includes synthetic 数据 from GPT 3.5 (later: GPT-4) and filtered 数据')
    R = "Python subset of the Stack"   # Raw data
    prompt = "determine its educational value for a student whose goal is to learn basic coding concepts"
    T = "Use GPT-4 with this prompt to classify 100K subset of R to get positive examples"
    bilingual_text("- Train random forest classifier on T using output embedding from pretrained codegen model", '- Train random forest 分类器 on T using output embedding from pretrained 代码gen 模型')
    bilingual_text("- Select data from R that is classified positive by the classifier", '- Select 数据 from R that is classified positive by the 分类器')
    bilingual_text("Result on [HumanEval](https://huggingface.co/datasets/openai_humaneval):", 'Result on [HumanEval](https://huggingface.co/数据sets/openai_humaneval):')
    bilingual_text("- Train 1.3B LM on Python subset of The Stack (performance: 12.19% after 96K steps)", '- 说明：Train 1.3B LM on Python subset of The Stack (performance: 12.19% after 96K steps)')
    bilingual_text("- Train 1.3B LM on new filtered subset (performance: 17.68% after 36K steps) - better!", '- 说明：Train 1.3B LM on new filtered subset (performance: 17.68% after 36K steps) - better!')

    bilingual_text("Toxicity filtering in Dolma ", '毒性 过滤 in Dolma'), link(dolma_2024)
    bilingual_text("- Dataset: Jigsaw Toxic Comments dataset (2018) ", '- 数据set: Jigsaw Toxic Comments 数据set (2018)'), link(title="dataset", url="https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge")
    bilingual_text("- Project goal: help people have better discussions online ", '- 说明：Project goal: help people have better discussions online'), article_link("https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/discussion/46064")
    bilingual_text("- Data: comments on Wikipedia talk page annotated with {toxic, severe_toxic, obscene, threat, insult, identity_hate}", '- 数据: comments on Wikipedia talk page annotated with {toxic, severe_toxic, obscene, threat, insult, identity_hate}')

    bilingual_text("Scale-dependent effects of filtering:", '过滤的规模相关效应：')
    bilingual_text("- No single optimal threshold for filtering", '- No single optimal 阈值 for 过滤')
    bilingual_text("- If training for longer, want more (lower quality) data", '- If 训练 for longer, want more (lower 质量) 数据')
    bilingual_text("- If training for shorter, want less (higher quality) data", '- If 训练 for shorter, want less (higher 质量) 数据')
    image("images/data-filtering-scale.png", width=800)

    bilingual_text("Summary:", '总结：')
    bilingual_text("- Filtering is critical for building a good model", '- 过滤 is critical for building a good 模型')
    bilingual_text("- Recipe: define target data (what good looks like), extrapolate to raw data", '- Recipe: define target 数据 (what good looks like), extrapolate to raw 数据')
    

def deduplication():
    bilingual_text("Two types of duplicates:", '两种重复项：')
    bilingual_text("- Exact duplicates (mirror sites, GitHub forks) ", '- Exact 重复项 (mirror sites, GitHub forks)'), link(title="Gutenberg mirrors", url="https://www.gutenberg.org/MIRRORS.ALL")
    bilingual_text("- Near duplicates: same text differing by a few tokens", '- Near 重复项: same 文本 differing by a few token')

    bilingual_text("Examples of near duplicates:", '近重复示例：')
    bilingual_text("- Terms of service and licenses ", '- Terms of service and 许可s'), link(title="MIT license", url="https://opensource.org/license/mit")
    bilingual_text("- Formulaic writing (copy/pasted or generated from a template) ", '- 说明：Formulaic writing (copy/pasted or generated from a template)'), image("https://d3i71xaburhd42.cloudfront.net/4566c0d22ebf3c31180066ab23b6c445aeec78d5/5-Table1-1.png", width=600)
    bilingual_text("- Minor formatting differences in copy/pasting", '- 说明：Minor formatting differences in copy/pasting')

    bilingual_text("Product description repeated 61,036 times in C4", '说明：Product description repeated 61,036 times in C4')
    bilingual_text("'“by combining fantastic ideas, interesting arrangements, and follow the current trends in the field of that make you more inspired and give artistic touches. We’d be honored if you can apply some or all of these design in your wedding.  believe me, brilliant ideas would be perfect if it can be applied in real and make the people around you amazed!", "说明：'“by combining fantastic ideas, interesting arrangements, and follow the current trends in the field of that make you more inspired and give artistic touches. We’d be honored if you can apply some or all of these design in your wedding.  believe me, brilliant ideas would be perfect if it can be applied in real and make the people around you amazed!")
    link(title="example page", url="https://www.amazon.co.uk/suryagede-100-Graffiti-Gas-Mask/dp/B07CRHT3RG")

    bilingual_text("Deduplication training data makes language models better ", '去重 训练 数据 makes language 模型s better'), link("https://arxiv.org/pdf/2107.06499")
    bilingual_text("- Train more efficiently (because have fewer tokens)", '- 说明：Train more efficiently (because have fewer tokens)')
    bilingual_text("- Avoid memorization (can mitigate copyright, privacy concerns)", '- Avoid memorization (can mitigate 版权, 隐私 concerns)')

    bilingual_text("Design space:", '设计空间：')
    bilingual_text("1. What is an item (sentence, paragraph, document)?", '1. 说明：What is an item (sentence, paragraph, document)?')
    bilingual_text("2. How to match (exact match, existence of common subitem, fraction of common subitems)?", '2. 说明：How to match (exact match, existence of common subitem, fraction of common subitems)?')
    bilingual_text("3. What action to take (remove all, remove all but one)?", '3. 说明：What action to take (remove all, remove all but one)?')

    bilingual_text("Key challenge:", '关键挑战：')
    bilingual_text("- Deduplication is fundamentally about comparing items to other items", '- 去重 is fundamentally about comparing items to other items')
    bilingual_text("- Need linear time algorithms to scale", '- 说明：Need linear time algorithms to scale')

    hash_functions()
    exact_deduplication()
    jaccard_minhash()
    locality_sensitive_hashing()


def hash_functions():
    bilingual_text("- Hash function h maps item to a hash value (integer or string)", '- 哈希 function h maps item to a 哈希 value (integer or string)')
    bilingual_text("- Hash value much smaller than item", '- 哈希 value much smaller than item')
    bilingual_text("- Hash collision: h(x) = h(y) for x ≠ y", '- 哈希 collision: h(x) = h(y) for x ≠ y')

    bilingual_text("Tradeoff between efficiency and collision resistance ", '说明：Tradeoff between efficiency and collision resistance'),  article_link("https://softwareengineering.stackexchange.com/questions/49550/which-hashing-algorithm-is-best-for-uniqueness-and-speed")
    bilingual_text("- Cryptographic hash functions (SHA-256): collision resistant, slow (used in bitcoin)", '- Cryptographic 哈希 functions (SHA-256): collision resistant, slow (used in bitcoin)')
    bilingual_text("- DJB2, MurmurHash, CityHash: not collision resistant, fast (used for hash tables)", '- DJB2, Murmur哈希, City哈希: not collision resistant, fast (used for 哈希 tables)')

    bilingual_text("We will use MurmurHash:", 'We will use Murmur哈希:')
    h = mmh3.hash("hello")  # @inspect h


def exact_deduplication():
    bilingual_text("**Simple example**", '**简单示例**')
    bilingual_text("1. Item: string", '1. 说明：Item: string')
    bilingual_text("2. How to match: exact match", '2. 说明：How to match: exact match')
    bilingual_text("3. Action: remove all but one", '3. 说明：Action: remove all but one')

    # Original items
    items = ["Hello!", "hello", "hello there", "hello", "hi", "bye"]  # @inspect items

    # Compute hash -> list of items with that hash
    hash_items = itertools.groupby(sorted(items, key=mmh3.hash), key=mmh3.hash)

    # Keep one item from each group
    deduped_items = [next(group) for h, group in hash_items]  # @inspect deduped_items

    bilingual_text("- Pro: simple, clear semantics, high precision", '- Pro: simple, clear 语义, high precision')
    bilingual_text("- Con: does not deduplicate near duplicates", '- Con: does not deduplicate near 重复项')
    bilingual_text("- This code is written in a MapReduce way, can easily parallelize and scale", '- This 代码 is written in a MapReduce way, can easily parallelize and scale')

    bilingual_text("**C4** ", '说明：C4'), link("https://arxiv.org/pdf/1910.10683v4")
    bilingual_text("1. Item: 3-sentence spans", '1. 说明：Item: 3-sentence spans')
    bilingual_text("2. How to match: use exact match", '2. 说明：How to match: use exact match')
    bilingual_text("3. Action: remove all but one", '3. 说明：Action: remove all but one')
    bilingual_text("Warning: when a 3-sentence span is removed from the middle of a document, the resulting document might not be coherent", '警告：when a 3-sentence span is removed from the middle of a document, the resulting document might not be coherent')


def jaccard_minhash():
    bilingual_text("Let's now look at approximate set membership.", "说明：Let's now look at approximate set membership.")
    bilingual_text("First we need a similarity measure.", '说明：First we need a similarity measure.')

    bilingual_text("### Jaccard similarity", '### Jaccard 相似度')
    bilingual_text("Definition: Jaccard(A, B) = |A intersect B| / |A union B|", '定义：Jaccard(A, B) = |A intersect B| / |A union B|')
    A = {"1", "2", "3", "4"}
    B = {"1", "2", "3", "5"}

    def compute_jaccard(A, B):
        intersection = len(A & B)  # @inspect intersection
        union = len(A | B)  # @inspect union
        return intersection / union
    jaccard = compute_jaccard(A, B)  # @inspect jaccard

    bilingual_text("Definition: two documents are **near duplicates** if their Jaccard similarity >= threshold", '定义：two documents are near duplicates if their Jaccard similarity >= threshold')

    bilingual_text("Algorithmic challenge: find near duplicates in linear time", 'Algorithmic challenge: find near 重复项 in linear time')

    bilingual_text("### MinHash", '### MinHash（最小哈希）')
    bilingual_text("MinHash: a random hash function h so that Pr[h(A) = h(B)] = Jaccard(A, B)", 'Min哈希: a random 哈希 function h so that Pr[h(A) = h(B)] = Jaccard(A, B)')

    bilingual_text("Normally, you want different items to hash to different hashes", 'Normally, you want different items to 哈希 to different 哈希es')
    bilingual_text("...but here, you want collision probability to depend on similarity", '说明：...but here, you want collision probability to depend on similarity')

    def minhash(S: set[str], seed: int):
        return min(mmh3.hash(x, seed) for x in S)

    bilingual_text("Characteristic matrix representation:", '特征矩阵表示：')
    bilingual_verbatim("item | A | B", '上方等宽内容保持原样，用于展示表格、示例文本或哈希/矩阵布局。', verbatim=True)
    bilingual_verbatim("1    | 1 | 1", '上方等宽内容保持原样，用于展示表格、示例文本或哈希/矩阵布局。', verbatim=True)
    bilingual_verbatim("2    | 1 | 1", '上方等宽内容保持原样，用于展示表格、示例文本或哈希/矩阵布局。', verbatim=True)
    bilingual_verbatim("3    | 1 | 1", '上方等宽内容保持原样，用于展示表格、示例文本或哈希/矩阵布局。', verbatim=True)
    bilingual_verbatim("4    | 1 | 0", '上方等宽内容保持原样，用于展示表格、示例文本或哈希/矩阵布局。', verbatim=True)
    bilingual_verbatim("5    | 0 | 1", '上方等宽内容保持原样，用于展示表格、示例文本或哈希/矩阵布局。', verbatim=True)

    bilingual_text("Random hash function induces a permutation over items", 'Random 哈希 function induces a permutation over items')
    bilingual_text("Look at which item is first in A and which item is first in B.", '说明：Look at which item is first in A and which item is first in B.')
    bilingual_text("Each item has the same probability as being first (min)", '说明：Each item has the same probability as being first (min)')
    bilingual_text("- If 1, 2, 3 is first, then first in A = first in B.", '- 说明：If 1, 2, 3 is first, then first in A = first in B.')
    bilingual_text("- If 4, 5 is first, then first in A ≠ first in B.", '- 说明：If 4, 5 is first, then first in A ≠ first in B.')

    # Verify MinHash approximates Jaccard as advertised
    n = 100  # Generate this many random hash functions
    matches = [minhash(A, seed) == minhash(B, seed) for seed in range(n)]  # @stepover
    estimated_jaccard = len([m for m in matches if m]) / len(matches)  # @inspect estimated_jaccard
    assert abs(estimated_jaccard - jaccard) < 0.01

    bilingual_text("Now we can hash our items, but a collision doesn't tell us Jaccard(A, B) > threshold.", "Now we can 哈希 our items, but a collision doesn't tell us Jaccard(A, B) > 阈值.")


def locality_sensitive_hashing():
    bilingual_text("Locality sensitive hashing (LSH) ", 'Locality sensitive 哈希ing (LSH)'), link(title="book chapter", url="http://infolab.stanford.edu/~ullman/mmds/ch3n.pdf")

    bilingual_text("Suppose we hash examples with just one MinHash function", 'Suppose we 哈希 examples with just one Min哈希 function')
    bilingual_text("P[A and B collide] = Jaccard(A, B)", '说明：P[A and B collide] = Jaccard(A, B)')
    bilingual_text("On average, more similar items will collide, but very stochastic...", '说明：On average, more similar items will collide, but very stochastic...')

    bilingual_text("Goal: have A and B collide if Jaccard(A, B) > threshold", '目标：have A and B collide if Jaccard(A, B) > threshold')
    bilingual_text("We have to somehow sharpen the probabilities...", '说明：We have to somehow sharpen the probabilities...')

    bilingual_text("Solution: use n hash functions", '解决方案：use n hash functions')
    bilingual_text("Break up into b bands of r hash functions each (n = b * r)", 'Break up into b bands of r 哈希 functions each (n = b  r)')

    n = 12      # Number of hash functions
    b = 3       # Number of bands
    r = 4       # Number of hash functions per band
    bilingual_text("Hash functions:", '哈希函数：')
    bilingual_verbatim("h1 h2 h3 h4  |  h5 h6 h7 h8  |  h9 h10 h11 h12", '上方等宽内容保持原样，用于展示表格、示例文本或哈希/矩阵布局。', verbatim=True)

    bilingual_text("Key: A and B collide if for *some* band, *all* its hash functions return same value", 'Key: A and B collide if for some band, all its 哈希 functions return same value')
    bilingual_text("As we will see, the and-or structure of the bands sharpens the threshold", 'As we will see, the and-or structure of the bands sharpens the 阈值')

    bilingual_text("Given Jaccard(A, B), what is the probability that A and B collide?", '说明：Given Jaccard(A, B), what is the probability that A and B collide?')

    def get_prob_collision(sim, b, r):  # @inspect sim @inspect b @inspect r
        prob_match = sim ** r                        # Probability that a fixed band matches  @inspect prob_match
        prob_collision = 1 - (1 - prob_match) ** b   # Probability that some band matches  @inspect prob_collision
        return prob_collision

    bilingual_text("**Example**", '**示例**')
    prob_collision = get_prob_collision(sim=0.8, b=5, r=10)  # @inspect prob_collision
    image("https://cdn.sanity.io/images/vr8gru94/production/b470799575b8e77911bacb8500977afef06d6c85-1280x720.png", width=600)


    sims = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98]
    probs = {sim: get_prob_collision(sim=sim, b=10, r=10) for sim in sims}  # @inspect probs @stepover

    bilingual_text("Increasing r sharpens the threshold and moves the curve to the right (harder to match)", 'Increasing r sharpens the 阈值 and moves the curve to the right (harder to match)')
    probs = {sim: get_prob_collision(sim=sim, b=10, r=20) for sim in sims}  # @inspect probs @stepover

    bilingual_text("Increasing b moves the curve to the left (easier to match)", '说明：Increasing b moves the curve to the left (easier to match)')
    probs = {sim: get_prob_collision(sim=sim, b=20, r=20) for sim in sims}  # @inspect probs @stepover
    image("https://cdn.sanity.io/images/vr8gru94/production/aace49fa240778e8ecf6e85ad08a2de7f5385566-1280x720.png", width=600)

    bilingual_text("Example setting ", '说明：Example setting'), link("https://arxiv.org/pdf/2107.06499"), bilingual_text(": n = 9000, b = 20, r = 450", '说明：: n = 9000, b = 20, r = 450')
    b = 20
    r = 450
    bilingual_text("What is the threshold (where the phase transition happens)?", 'What is the 阈值 (where the phase transition happens)?')
    threshold = (1 / b) ** (1 / r)  # @inspect threshold

    bilingual_text("Probability that a fixed band matches:", '说明：Probability that a fixed band matches:')
    prob_match = (1 / b)  # @inspect prob_match
    bilingual_text("Probability that A and B collide is a constant (≈ 1-1/e):", '说明：Probability that A and B collide is a constant (≈ 1-1/e):')
    prob_collision = 1 - (1 - 1 / b) ** b  #  @inspect prob_collision


def billion(x):
    return x * 10**9

def trillion(x):
    return x * 10**12


def data_mixing():
    bilingual_text("Recall that language models are trained on multiple data sources.", '回忆：语言模型会在多个数据来源上训练。')

    bilingual_text("Datasets in Marin: ", '数据sets in Marin:'), link(title="token viewer", url="https://huggingface.co/spaces/marin-community/token-count-viewer")
    image("images/marin-token-viewer.png", width=800)

    bilingual_text("The Pile ", '说明：The Pile'), link(the_pile_2020)
    image("https://stanford-cs324.github.io/winter2022/lectures/images/the-pile.png", width=600)
    bilingual_text("Key question: what distribution over the data sources should we use?", '关键问题：我们应该在数据来源上使用什么分布？')

    bilingual_text("Example:", '示例：')
    sources = {"Wikipedia", "CC", "GitHub"}
    p = {"Wikipedia": 0.3, "CC": 0.5, "GitHub": 0.2}  # One possible data mixture

    bilingual_text("Baselines:", '基线：')
    bilingual_text("- Vibes: set p(s) manually based on intuition (quite common)", '- 说明：Vibes: set p(s) manually based on intuition (quite common)')
    bilingual_text("- Uniform sampling: sample uniformly (p(s) ∝ 1)", '- 说明：Uniform sampling: sample uniformly (p(s) ∝ 1)')
    bilingual_text("- Proportional mixing: sample proportional to the number of tokens in a source (p(s) ∝ num_tokens(s))", '- Proportional 混合: sample proportional to the number of token in a 来源 (p(s) ∝ num_token(s))')

    bilingual_text("Intuition: should upweight higher quality sources", 'Intuition: should upweight higher 质量 来源s')
    bilingual_text("However...", '然而……')
    bilingual_text("1. We want to ensure diversity (e.g., across incomparable sources: literature, code, papers)", '1. We want to ensure diversity (e.g., across incomparable 来源s: literature, 代码, papers)')
    bilingual_text("2. Each source is finite, so if put too much weight on a small source, then need to epoch over it", '2. Each 来源 is finite, so if put too much weight on a small 来源, then need to epoch over it')
    
    bilingual_text("This last point is important and a bit subtle.", '最后一点很重要，也有点微妙。')
    bilingual_text("Example:", '示例：')
    source_token_counts = {
        "low": trillion(10),  # 10T tokens (abundant) @stepover
        "high": billion(10),  # 10B tokens (scarce) @stepover
    }
    p = {"low": 0.5, "high": 0.5}  # Naive data mixture
    train_tokens = trillion(1)  # Train for 1T tokens @stepover
    low_num_epochs = (p["low"] * train_tokens) / source_token_counts["low"]  # @inspect low_num_epochs
    high_num_epochs = (p["high"] * train_tokens) / source_token_counts["high"]  # @inspect high_num_epochs
    bilingual_text("50x epochs on high quality data...can lead to overfitting!", '50x epochs on high 质量 数据...can lead to 过拟合!')

    bilingual_text("UniMax ", '说明：UniMax'), link("https://arxiv.org/abs/2304.09151")
    bilingual_text("- Setting: balancing different languages for multilingual models", '- Setting: balancing different languages for multilingual 模型s')
    bilingual_text("- Previous work: between uniform and proportional mixing (p(s) ∝ num_tokens(s)^α for α in [0, 1])", '- Previous work: between uniform and proportional 混合 (p(s) ∝ num_token(s)^α for α in [0, 1])')
    bilingual_text("- Idea: sample sources uniformly but with a hard **cap** C on number of epochs for any source", '- 思想：sample sources uniformly but with a hard cap C on number of epochs for any source')
    bilingual_text("- Specifically, p(s) * num_training_tokens ≤ C for all sources s", '- Specifically, p(s)  num_训练_token ≤ C for all 来源s s')

    bilingual_text("Regression-based mixing ", '回归-based 混合'), link("https://arxiv.org/abs/2407.01492"), link("https://arxiv.org/pdf/2602.12237")
    image("images/regmix.png", width=700)
    bilingual_text("- Define distribution over mixtures `p` (e.g., Dirichlet) ", '- Define distribution over 混合比例s p (e.g., Dirichlet)')
    bilingual_text("- Define regression method (e.g., linear, gradient boosted trees)", '- Define 回归 method (e.g., linear, gradient boosted trees)')
    bilingual_text("- Define target based on downstream evals (careful not to overfit!)", '- 说明：Define target based on downstream evals (careful not to overfit!)')
    bilingual_text("- Discrepancy between small and large scale (tradeoff cost and accuracy)", '- 说明：Discrepancy between small and large scale (tradeoff cost and accuracy)')
    image("images/data-mixing-methods.png", width=700)
    bilingual_text("Hope 1: regression model is accurate at minimizer 🙏", 'Hope 1: 回归 模型 is accurate at minimizer 🙏')
    bilingual_text("Hope 2: optimal data mixtures transfer from small to large scale 🙏", 'Hope 2: optimal 数据 混合比例s transfer from small to large scale 🙏')

    bilingual_text("Hold on. There's at least one scale-dependent effect:", "说明：Hold on. There's at least one scale-dependent effect:")
    source_token_counts = {
        "low": trillion(10),  # 10T tokens (abundant) @stepover
        "high": billion(10),  # 10B tokens (scarce) @stepover
    }
    bilingual_text("- If train small models on low token counts:", '- If train small 模型s on low token counts:')
    p = {"low": 0.1, "high": 0.9}  # More mass on high quality data
    bilingual_text("- But if train large model on this mixture, we will epoch a ton on high quality data and overfit!", '- But if train large 模型 on this 混合比例, we will epoch a ton on high 质量 数据 and overfit!')

    bilingual_text("Simulated epoching ", '说明：Simulated epoching'), link("https://arxiv.org/pdf/2501.11747")
    bilingual_text("- General idea: make small scale look like large scale (general theme of this course)", '- 说明：General idea: make small scale look like large scale (general theme of this course)')
    bilingual_text("- Instantiation: downsample all sources proportionally", '- Instantiation: downsample all 来源s proportionally')
    small_run_tokens = billion(10)  # @stepover
    large_run_tokens = trillion(1)  # @stepover
    ratio = small_run_tokens / large_run_tokens  # @inspect ratio
    downsampled_source_token_counts = {s: count * ratio for s, count in source_token_counts.items()}  # @inspect downsampled_source_token_counts
    bilingual_text("- In this downsampled mixture, models that epoch too much won't look good.", "- In this downsampled 混合比例, 模型s that epoch too much won't look good.")
    bilingual_text("- So the optimum will be more balanced.", '- 说明：So the optimum will be more balanced.')
    p = {"low": 0.7, "high": 0.3}  # More mass on high quality data

    bilingual_text("Summary:", '总结：')
    bilingual_text("- Problem: how to weight different data sources (e.g., Wikipedia, general, code)", '- 问题：how to weight different data sources (e.g., Wikipedia, general, code)')
    bilingual_text("- Regression-based mixing: estimate mixture → loss at small scale, optimize (analogous to scaling laws)", '- 回归-based 混合: estimate 混合比例 → 损失 at small scale, optimize (analogous to scaling laws)')
    bilingual_text("- Important consideration: epoching and overfitting (solution: cap or simulated)", '- Important consideration: epoching and 过拟合 (solution: cap or simulated)')


def post_training_data():
    bilingual_text("Recipe:", '配方：')
    bilingual_text("1. Define a set of environments", '1. 说明：Define a set of environments')
    bilingual_text("2. Define a set of tasks / prompts", '2. Define a set of tasks / 提示')
    bilingual_text("3. Collect responses from a strong model (teacher)", '3. Collect 回答 from a strong 模型 (教师模型)')

    bilingual_text("OpenThoughts ", '说明：OpenThoughts'), link("https://arxiv.org/abs/2506.04178")
    bilingual_text("- 1.2M examples using QwQ-32B as a teacher", '- 1.2M examples using QwQ-32B as a 教师模型')
    bilingual_text("- Questions come from 27 human and synthetic sources (e.g., StackExchange, NuminaMath, Chemistry)", '- Questions come from 27 human and synthetic 来源s (e.g., StackExchange, NuminaMath, Chemistry)')
    image("images/openthoughts-sources.png", width=500)
    bilingual_text("- Sampling multiple (16) responses per prompt is helpful", '- Sampling multiple (16) 回答 per prompt is helpful')
    bilingual_text("- Better models aren't necessarily better teachers: QwQ-32B is a better teacher than DeepSeek-R1", "- Better 模型s aren't necessarily better 教师模型s: QwQ-32B is a better 教师模型 than DeepSeek-R1")
    bilingual_text("- Answer filtering wasn't helpful", "- Answer 过滤 wasn't helpful")
    bilingual_text("- Smaller high quality sources (e.g., OpenMath-2-Math) is better than large diverse sources", '- Smaller high 质量 来源s (e.g., OpenMath-2-Math) is better than large diverse 来源s')
    image("images/openthoughts-pipeline.png", width=600)

    bilingual_text("SWE-smith ", '说明：SWE-smith'), link("https://arxiv.org/abs/2504.21798")
    image("images/swe-smith.png", width=500)
    bilingual_text("- Given a repository, use LM to generate tasks (introduce bugs with LM)", '- Given a 仓库, use LM to generate tasks (introduce bugs with LM)')
    bilingual_text("- 128 GitHub repositories yields 50K tasks", '- 128 GitHub 仓库 yields 50K tasks')

    bilingual_text("SWE-Zero ", '说明：SWE-Zero'), link("https://arxiv.org/abs/2604.01496")
    bilingual_text("- SWE tasks have heavy dependencies (unlike math or coding contests)", '- 说明：SWE tasks have heavy dependencies (unlike math or coding contests)')
    bilingual_text("- Setting up thousands of Docker images is an infrastructural nightmare", '- Setting up thousands of Docker 图像 is an infrastructural nightmare')
    bilingual_text("- Observation: strong models can solve many tasks without execution feedback", '- 观察：strong models can solve many tasks without execution feedback')
    image("images/swezero-noexec.png", width=600)
    bilingual_text("Key: strong models have internal \"world model\" of code semantics", 'Key: strong 模型s have internal "world 模型" of 代码 语义')
    bilingual_text("- SWE-Zero: 300K agent trajectories that don't require repository-specific execution", "- SWE-Zero: 300K 智能体 轨迹 that don't require 仓库-specific execution")
    bilingual_text("- 150K GitHub PRs", '- 说明：150K GitHub PRs')
    bilingual_text("- OpenHands scaffold, remove future git commits to prevent \"git hacking\" by agent", '- OpenHands scaffold, remove future git commits to prevent "git hacking" by 智能体')
    image("images/swezero-prompt.png", width=600)
    bilingual_text("- Distilled from Qwen3-Coder-480B + filtering (try to execute anyway)", '- Distilled from Qwen3-代码r-480B + 过滤 (try to execute anyway)')
    bilingual_text("- SWE-Hero: 13K agent trajectories that do require execution feedback", '- SWE-Hero: 13K 智能体 轨迹 that do require execution feedback')
    image("images/swezero-results.png", width=700)

    bilingual_text("SWE-rebench ", '说明：SWE-rebench'), link("https://arxiv.org/pdf/2505.20411")
    bilingual_text("- 21K interactive Python SWE tasks from 3.4K GitHub repositories", '- 21K interactive Python SWE tasks from 3.4K GitHub 仓库')
    bilingual_text("- 450K PRs from GitHub and GitHub Archive", '- 说明：450K PRs from GitHub and GitHub Archive')
    bilingual_text("- Used Qwen 2.5-72B-Instruct to install dependencies and assess PR quality", '- Used Qwen 2.5-72B-Instruct to install dependencies and assess PR 质量')
    image("images/swe-rebench.png", width=600)

    bilingual_text("SWE-ZERO-12M-trajectories ", 'SWE-ZERO-12M-轨迹'), link(title="data", url="https://huggingface.co/datasets/AlienKevin/SWE-ZERO-12M-trajectories")
    bilingual_text("- Scale SWE-Zero up to 12M agent trajectories", '- Scale SWE-Zero up to 12M 智能体 轨迹')
    bilingual_text("- Used the SWE-rebench-v2 tasks (32K executable tasks + 120K nonexecutable tasks)", '- 说明：Used the SWE-rebench-v2 tasks (32K executable tasks + 120K nonexecutable tasks)')
    bilingual_text("- Ran mini-coder-1.7b (very small model, 50.4 pass@100), mini-swe-agent scaffold", '- Ran mini-代码r-1.7b (very small 模型, 50.4 pass@100), mini-swe-智能体 scaffold')
    bilingual_text("- [Example](https://huggingface.co/datasets/AlienKevin/SWE-ZERO-12M-trajectories/viewer/default/train?row=5&conversation-viewer=0)", '- [Example](https://huggingface.co/数据sets/AlienKevin/SWE-ZERO-12M-轨迹/viewer/default/train?row=5&conversation-viewer=0)')

    bilingual_text("Summary:", '总结：')
    bilingual_text("- Generating prompts: fully-synthetic, semi-synthetic (real environment + synthetic tasks), real (GitHub PRs)", '- Generating 提示: fully-synthetic, semi-synthetic (real environment + synthetic tasks), real (GitHub PRs)')
    bilingual_text("- Responses: from capable models (that are also good teachers)", '- 回答: from capable 模型s (that are also good 教师模型s)')
    bilingual_text("- Code environments are painful", '- 代码 environments are painful')
    bilingual_text("- Lots of filtering and other details", '- Lots of 过滤 and other details')


if __name__ == "__main__":
    main()
