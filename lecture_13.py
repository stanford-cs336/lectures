from edtrace import text, image, link
from lecture_util import article_link, bilingual_text, bilingual_verbatim
from references import dclm_2024, nemotron_cc_2024, olmo_2_2025, llama_3_2024, gpt2_2019, openwebtext_2019, gopher_2021, alpaca_2023


def main():
    bilingual_text("## Lecture 13: Data I", '## 第 13 讲：数据 I')
    bilingual_text("Previous lectures: how to train a model *given data*", '前几讲：在**给定数据**的情况下如何训练模型。')
    bilingual_text("Next two lectures: *what data* should we train on?", '接下来两讲：我们应该在**什么数据**上训练？')

    motivation()

    # Origin of data
    raw_sources()    # What does data come from?
    copyright()      # What data can we use?

    # Sources of data
    common_crawl()   # Web crawl
    wikipedia()      # General knowledge
    github()         # Code
    arxiv()          # Research papers

    # Data from various models
    bert()                # Wikipedia, books (trained BERT) [2019]
    gpt2_webtext()        # pages based on Reddit links (trained GPT-2) [2019]
    ccnet()               # Filter Common Crawl based on Wikipedia [2019]
    t5_c4()               # Filter using rules (trained T5) [2019]

    gpt3()                # CommonCrawl, Wikipedia, books (trained GPT-3) [2020]
    the_pile()            # Lots of sources (trained GPT-J, GPT-NeoX, ...) [2021]
    gopher_massivetext()  # Filter using rules (trained Gopher) [2021]
    llama()               # CommonCrawl, CCNet, StackExchange, etc. (trained LLaMA) [2022]
    refinedweb()          # CommonCrawl (used to train Falcon) [2023]
    dolma()               # Lots of different sources [2024]
    dclm()                # Filtered using good quality classifier [2024]
    nemotron_cc()         # Lots of tokens [2024]
    the_stack()           # Code dataset
    common_pile()         # Properly licensed data

    bilingual_text("### Summary", '### 总结')
    bilingual_text("- Key lesson: Data does not fall from the sky. You have to work to get it.", '- 关键教训：数据不会从天上掉下来，你必须投入工作才能得到它。')
    bilingual_text("- Live service → raw data → processed data (transformation, filtering, deduplication)", '- 在线服务 → 原始数据 → 处理后数据（转换、过滤、去重）。')
    bilingual_text("- Data is the key ingredient that differentiates language models", '- 数据是区分不同语言模型的关键要素。')
    bilingual_text("- Legal and ethical issues (e.g., copyright and privacy)", '- 法律和伦理问题（例如版权和隐私）。')
    bilingual_text("- Much of this pipeline is heuristic, many opportunities to improve!", '- 这条流水线很大程度上依赖启发式方法，还有许多改进机会！')


def motivation():
    bilingual_text("**Data** is the most important thing to get right in training language models.", '在训练语言模型时，**数据**是最需要做对的事情。')

    bilingual_text("One justification: let's see what companies disclose.", '一个理由是：看看公司愿意披露什么。')
    bilingual_text("Open-weight models (e.g., Llama 3 ", 'Open-weight 模型s (e.g., Llama 3'), link(llama_3_2024), bilingual_text(" have full transparency into architecture", '说明：have full transparency into architecture')
    bilingual_text("...and even training procedures", '……甚至训练流程。')
    bilingual_text("...but basically no information on data.", '……但基本没有关于数据的信息。')
    image("images/llama3-data.png", width=700)
    
    bilingual_text("Reasons for secrecy:", '保密的原因：')
    bilingual_text("1. Competitive dynamics", '1. 竞争动态。')
    bilingual_text("2. Copyright liability", '2. 版权责任。')

    bilingual_text("- Before foundation models, data work meant heavy annotation of labeled data for supervised learning.", '- Before foundation 模型s, 数据 work meant heavy annotation of labeled 数据 for supervised learning.')
    bilingual_text("- Now there's less annotation, but there's still a lot of curation and cleaning.", "- 说明：Now there's less annotation, but there's still a lot of curation and cleaning.")
    bilingual_text("- Data is fundamentally a long-tail problem, scales with human effort (unlike architectures, systems).", '- 数据 is fundamentally a long-tail problem, scales with human effort (unlike architectures, systems).')

    bilingual_text("Stages of training:", '训练阶段：')
    bilingual_text("1. Pre-training: train on raw text (e.g., documents from the web)", '1. Pre-训练: train on 原始文本 (e.g., 文档 from the 网络)')
    bilingual_text("2. Mid-training: train more on high quality data to enhance capabilities", '2. Mid-训练: train more on high 质量 数据 to enhance capabilities')
    bilingual_text("3. Post-training: train on chat transcripts or reinforcement learning", '3. Post-训练: train on chat transcripts or reinforcement learning')
    bilingual_text("In practice, the lines are blurry and there could be more stages", '说明：In practice, the lines are blurry and there could be more stages')
    bilingual_text("...but the basic trend is throughout training, we go from", '...but the basic trend is throughout 训练, we go from')
    bilingual_text("large amounts of lower quality data to", 'large amounts of lower 质量 数据 to')
    bilingual_text("small amounts of high quality data.", 'small amounts of high 质量 数据.')

    bilingual_text("Terminology:", '术语：')
    bilingual_text("- Base model: after pre-training + mid-training", '- Base 模型: after pre-训练 + mid-训练')
    bilingual_text("- Instruct/chat model: after post-training", '- Instruct/chat 模型: after post-训练')
    bilingual_text("(Increasingly, base models are not released - e.g., Qwen3.5-397B-A17B is an instruct model.)", '(Increasingly, base 模型s are not released - e.g., Qwen3.5-397B-A17B is an instruct 模型.)')

    bilingual_text("Example (OLMo from AI2) ", '说明：Example (OLMo from AI2)'), link(olmo_2_2025)
    bilingual_text("1. **Pre-training**", '1. Pre-训练')
    image("images/olmo2-pretraining.png", width=600)
    bilingual_text("2. **Mid-training**", '2. Mid-训练')
    image("images/olmo2-dolmino.png", width=600)
    bilingual_text("3. **Post-training** ", '3. Post-训练'), link("https://arxiv.org/pdf/2411.15124")
    image("images/tulu.png", width=600)

    bilingual_text("What are these datasets?  How are they chosen and processed?", '这些数据集是什么？它们如何被选择和处理？')


def raw_sources():
    bilingual_text("One might often hear: *language models are trained on the entire Internet*.", '人们常听到一种说法：*语言模型是在整个互联网上训练的*。')
    bilingual_text("Slightly more accurately, ~Internet~ public (world wide) web.", 'Slightly more accurately, ~互联网~ public (world wide) 网络.')
    bilingual_text("But this is not quite right either...", '但这也并不完全正确……')

    bilingual_text("First, the web consists of a set of live servers that one can connect to:", 'First, the 网络 consists of a set of live servers that one can connect to:')
    bilingual_text("`$ curl https://cs336.stanford.edu/`", '说明：$ curl https://cs336.stanford.edu/')

    bilingual_text("You can't train on live servers.", '你不能直接在在线服务器上训练。')
    bilingual_text("A **crawler**:", 'A 爬虫:')
    bilingual_text("- Discovers webpages (starting from a seed set)", '- Discovers 网络pages (starting from a seed set)')
    bilingual_text("- Downloads the discovered webpages", '- Downloads the discovered 网络pages')

    bilingual_text("However, you can't download and train on all the webpages.", '但是，你也不能下载并训练所有网页。')

    bilingual_text("Dynamic content:", '动态内容：')
    bilingual_text("- Many sites these days are apps", '- 说明：Many sites these days are apps')
    bilingual_text("- URL doesn't change", "- 说明：URL doesn't change")
    bilingual_text("- Need to click buttons and submit forms to access content", '- 说明：Need to click buttons and submit forms to access content')
    bilingual_text("- Examples: Discord, wandb", '- 说明：Examples: Discord, wandb')

    bilingual_text("Authentication:", '认证：')
    bilingual_text("- Sometimes need login with an account (and pay usually)", '- 说明：Sometimes need login with an account (and pay usually)')
    bilingual_text("- Example: Facebook, X, LinkedIn, NYTimes (huge content behind walled gardens)", '- 示例：Facebook, X, LinkedIn, NYTimes (huge content behind walled gardens)')

    bilingual_text("Technical restrictions:", '技术限制：')
    bilingual_text("- Not allowed to download some content based on `robots.txt` ([example](https://www.nytimes.com/robots.txt)) (voluntary)", '- 说明：Not allowed to download some content based on robots.txt ([example](https://www.nytimes.com/robots.txt)) (voluntary)')
    bilingual_text("- Website might use Cloudflare to detect and block bot activity (present CAPTCHAs)", '- 网络site might use Cloudflare to detect and block bot activity (present CAPTCHAs)')
    bilingual_text("- Website might block certain IP addresses / countries", '- 网络site might block certain IP addresses / countries')
    bilingual_text("- Website might have rate limits", '- 网络site might have rate limits')
    
    bilingual_text("Legal restrictions:", '法律限制：')
    bilingual_text("- Terms of service (ToS) might prohibit downloading using bots", '- 说明：Terms of service (ToS) might prohibit downloading using bots')
    bilingual_text("- You might not have a license to copy the webpages (for training)", '- You might not have a 许可 to copy the 网络pages (for 训练)')

    bilingual_text("Decline of consent ", '说明：Decline of consent'), link("https://arxiv.org/abs/2407.14933")
    bilingual_text("- Examined restrictions (robots.txt, ToS) for URLs in common datasets (C4, RefinedWeb, Dolma)", '- Examined restrictions (robots.txt, ToS) for URLs in common 数据sets (C4, Refined网络, Dolma)')
    bilingual_text("- Restrictions have increased over time", '- 说明：Restrictions have increased over time')
    image("images/decline-consent.png", width=700)

    bilingual_text("When crawlers are not well-behaved:", '当爬虫行为不规范时：')
    image("images/anthropic-crawling.png", width=500)
    bilingual_text("- Factors: ToS, robots.txt, server load (degrades service, costs website money)", '- Factors: ToS, robots.txt, server load (degrades service, costs 网络site money)')
    bilingual_text("- And then there is copyright (more later)...", '- And then there is 版权 (more later)...')

    bilingual_text("Shadow libraries ", '说明：Shadow libraries'), article_link("https://en.wikipedia.org/wiki/Shadow_library")
    bilingual_text("- Technically part of the web", '- Technically part of the 网络')
    bilingual_text("- Examples: Library Genesis (LibGen), Z-Library, Anna's Archive, Sci-Hub", "- 说明：Examples: Library Genesis (LibGen), Z-Library, Anna's Archive, Sci-Hub")
    bilingual_text("- Disregards copyright and bypasses paywalls (e.g., Elsevier)", '- Disregards 版权 and bypasses paywalls (e.g., Elsevier)')
    bilingual_text("- Received takedown orders, lawsuits, blocked in various countries", '- 说明：Received takedown orders, lawsuits, blocked in various countries')
    bilingual_text("- Usually controls are circumvented, have servers in various countries", '- 说明：Usually controls are circumvented, have servers in various countries')
    bilingual_text("- Some argue this makes freely available what should be free", '- 说明：Some argue this makes freely available what should be free')
    bilingual_text("- From a legal perspective, this is piracy and copyright infringement", '- From a legal perspective, this is piracy and 版权 infringement')
    bilingual_text("- LibGen has ~4M books (2019), Sci-Hub has ~88M papers (2022)", '- 说明：LibGen has ~4M books (2019), Sci-Hub has ~88M papers (2022)')

    bilingual_text("Summary:", '总结：')
    bilingual_text("- The Internet is huge", '- The 互联网 is huge')
    bilingual_text("- Many technical and legal restrictions on what data one can access", '- Many technical and legal restrictions on what 数据 one can access')


def copyright():
    bilingual_text("What data is legal to use (for training)?", '哪些数据可以合法用于训练？')

    bilingual_text("### Intellectual property law", '### 知识产权法')
    bilingual_text("- Goal: *incentivize* the creation of intellectual goods", '- 目标：incentivize the creation of intellectual goods')
    bilingual_text("- Types of intellectual property: copyright, patents, trademarks, trade secrets.", '- Types of intellectual property: 版权, patents, trademarks, trade secrets.')

    bilingual_text("**Copyright law**:", '**版权法**：')
    bilingual_text("- Goes back to 1709 in England (Statute of Anne), first time regulated by governments and courts ", '- 说明：Goes back to 1709 in England (Statute of Anne), first time regulated by governments and courts'), article_link("https://en.wikipedia.org/wiki/Statute_of_Anne")
    bilingual_text("- In United States, most recent: Copyright Act of 1976 ", '- In United States, most recent: 版权 Act of 1976'), article_link("https://en.wikipedia.org/wiki/Copyright_Act_of_1976")
    bilingual_text("- Copyright protection applies to *'original works of authorship fixed in any tangible medium of expression, now known or later developed, from which they can be perceived, reproduced, or otherwise communicated, either directly or with the aid of a machine or device'*", "- 版权 protection applies to 'original works of authorship fixed in any tangible medium of expression, now known or later developed, from which they can be perceived, reproduced, or otherwise communicated, either directly or with the aid of a machine or device'")

    bilingual_text("- Collections are not original works so hence not copyrightable (e.g., telephone directories) unless there is some creativity in the selection or arrangement", '- Collections are not original works so hence not 版权able (e.g., telephone directories) unless there is some creativity in the selection or arrangement')
    bilingual_text("- Copyright applies to expression, not ideas (e.g., quicksort)", '- 版权 applies to expression, not ideas (e.g., quicksort)')

    bilingual_text("- Expanded scope from 'published' (1909) to 'fixed' (1976)", "- 说明：Expanded scope from 'published' (1909) to 'fixed' (1976)")
    bilingual_text("- Registration not required for copyright protection (in contrast with patents)", '- Registration not required for 版权 protection (in contrast with patents)')
    bilingual_text("- Threshold for copyright is extremely low (e.g., your website is copyrighted)", '- 阈值 for 版权 is extremely low (e.g., your 网络site is 版权ed)')

    bilingual_text("- Registration is required before creator can sue someone for copyright infringement", '- Registration is required before creator can sue someone for 版权 infringement')
    bilingual_text("- Costs $65 to register ", '- 说明：Costs $65 to register'), article_link("https://www.copyright.gov/about/fees.html")
    bilingual_text("- Lasts for 75 years, and then the copyright expires and it becomes part of the public domain (works of Shakespeare, Beethoven, most of Project Gutenberg, etc.)", '- Lasts for 75 years, and then the 版权 expires and it becomes part of the public domain (works of Shakespeare, Beethoven, most of Project Gutenberg, etc.)')

    bilingual_text("Summary: *basically everything on the Internet are copyrighted.*", 'Summary: basically everything on the 互联网 are 版权ed.')

    bilingual_text("How to use a copyrighted work:", '如何使用受版权保护的作品：')
    bilingual_text("1. Get a license for it.", '1. Get a 许可 for it.')
    bilingual_text("2. Appeal to the fair use clause.", '2. Appeal to the 合理使用 clause.')

    bilingual_text("### Licenses", '### 许可')
    bilingual_text("- A license (from contract law) is granted by a licensor to a licensee.", '- A 许可 (from contract law) is granted by a licensor to a 许可e.')
    bilingual_text("- Effectively, 'a license is a promise not to sue'.", "- Effectively, 'a 许可 is a promise not to sue'.")

    bilingual_text("- The Creative Commons license enables free distribution of copyrighted work.", '- The Creative Commons 许可 enables free distribution of 版权ed work.')
    bilingual_text("- Examples: Wikipedia, Open Courseware, Khan Academy, Free Music Archive, 307 million images from Flickr, 39 million images from MusicBrainz, 10 million videos from YouTube, etc.", '- Examples: Wikipedia, Open Courseware, Khan Academy, Free Music Archive, 307 million 图像 from Flickr, 39 million 图像 from MusicBrainz, 10 million 视频s from YouTube, etc.')
    bilingual_text("- Created by Lessig and Eldred in 2001 to bridge public domain and existing copyright", '- Created by Lessig and Eldred in 2001 to bridge public domain and existing 版权')

    bilingual_text("Many model developers license data for training foundation models", 'Many 模型 developers 许可 数据 for 训练 foundation 模型s')
    bilingual_text("- Google and Reddit ", '- 说明：Google and Reddit'), article_link("https://www.reuters.com/technology/reddit-ai-content-licensing-deal-with-google-sources-say-2024-02-22/")
    bilingual_text("- OpenAI and Shutterstock ", '- 说明：OpenAI and Shutterstock'), article_link("https://investor.shutterstock.com/news-releases/news-release-details/shutterstock-expands-partnership-openai-signs-new-six-year")
    bilingual_text("- OpenAI and StackExchange ", '- 说明：OpenAI and StackExchange'), article_link("https://stackoverflow.co/company/press/archive/openai-partnership")

    bilingual_text("**Fair use (section 107)**:", '合理使用 (section 107):')
    bilingual_text("Four factors to determine whether fair use applies:", 'Four factors to determine whether 合理使用 applies:')
    bilingual_text("1. The purpose and character of the use (educational favored over commercial, transformative favored over reproductive)", '1. 说明：The purpose and character of the use (educational favored over commercial, transformative favored over reproductive)')
    bilingual_text("2. The nature of the copyrighted work (factual favored over fictional, non-creative over creative)", '2. The nature of the 版权ed work (factual favored over fictional, non-creative over creative)')
    bilingual_text("3. The amount and substantiality of the portion of the original work used (using a snippet favored over using the whole work)", '3. 说明：The amount and substantiality of the portion of the original work used (using a snippet favored over using the whole work)')
    bilingual_text("4. The effect of the use upon the market (or potential market) for the original work", '4. 说明：The effect of the use upon the market (or potential market) for the original work')

    bilingual_text("Examples of fair use:", 'Examples of 合理使用:')
    bilingual_text("- You watch a movie and write a summary of it", '- 说明：You watch a movie and write a summary of it')
    bilingual_text("- Reimplement an algorithm (the idea) rather than copying the code (the expression)", '- Reimplement an algorithm (the idea) rather than copying the 代码 (the expression)')
    bilingual_text("- Google Books index and show snippets (Authors Guild v. Google 2002-2013)", '- 说明：Google Books index and show snippets (Authors Guild v. Google 2002-2013)')

    bilingual_text("Copyright is not about verbatim memorization:", '版权 is not about verbatim memorization:')
    bilingual_text("- Plots and characters (e.g., Harry Potter) can be copyrightable", '- Plots and characters (e.g., Harry Potter) can be 版权able')
    bilingual_text("- Parody (imitating to make fun of something) is likely fair use", '- Parody (imitating to make fun of something) is likely 合理使用')
    bilingual_text("Copyright is about semantics (and economics).", '版权 is about 语义 (and economics).')

    bilingual_text("Considerations for language models:", 'Considerations for language 模型s:')
    bilingual_text("- Copying data (first step of training) is violation already even if you don't do anything with it.", "- Copying 数据 (first step of 训练) is violation already even if you don't do anything with it.")
    bilingual_text("- Training a model should be transformative (far from just copy/pasting).", '- 训练 a 模型 should be transformative (far from just copy/pasting).')
    bilingual_text("- Model should be about the general idea (e.g., wizards), not in the concrete expression (e.g., Harry Potter).", '- 模型 should be about the general idea (e.g., wizards), not in the concrete expression (e.g., Harry Potter).')
    bilingual_text("- Language models can definitely affect the market (writers, artists), regardless of copyright", '- Language 模型s can definitely affect the market (writers, artists), regardless of 版权')

    bilingual_text("**Terms of service**:", '说明：Terms of service:')
    bilingual_text("- Even if you have a license or can appeal to fair use for a work, terms of service might impose additional restrictions.", '- Even if you have a 许可 or can appeal to 合理使用 for a work, terms of service might impose additional restrictions.')
    bilingual_text("- Example: YouTube's terms of service prohibits downloading videos, even if the videos are licensed under Creative Commons.", "- 示例：YouTube's terms of service prohibits downloading videos, even if the videos are licensed under Creative Commons.")

    bilingual_text("### Lawsuits", '### 诉讼')
    bilingual_text("The New York Times v. OpenAI (2023)", '说明：The New York Times v. OpenAI (2023)')
    bilingual_text("- Allegation: for training and reproducing NYT articles", '- Allegation: for 训练 and reproducing NYT articles')

    bilingual_text("Authors (Bartz, Graeber, ...) v. Anthropic (2024):", '说明：Authors (Bartz, Graeber, ...) v. Anthropic (2024):')
    bilingual_text("- Allegation: for pirating millions of books and training on plaintiff's books", "- Allegation: for pirating millions of books and 训练 on plaintiff's books")
    bilingual_text("- Summary judgement (2025): training on plaintiff's works is fair use", "- Summary judgement (2025): 训练 on plaintiff's works is 合理使用")
    bilingual_text("- ...but pirating copies is not (even if don't train)", "- 说明：...but pirating copies is not (even if don't train)")
    bilingual_text("- Anthropic also bought and scanned the books; this is also fair use (but too late)", '- Anthropic also bought and scanned the books; this is also 合理使用 (but too late)')
    bilingual_text("- Outcome: Anthropic paid $1.5B to authors to settle", '- 说明：Outcome: Anthropic paid $1.5B to authors to settle')

    bilingual_text("Authors (Kadrey, Silverman, ...) v. Meta ", '说明：Authors (Kadrey, Silverman, ...) v. Meta')
    bilingual_text("- Allegation: for training on plaintiff's books (revealed in the Llama paper)", "- Allegation: for 训练 on plaintiff's books (revealed in the Llama paper)")
    bilingual_text("- Summary judgement (2025): training on books (in this instance) is fair use ", '- Summary judgement (2025): 训练 on books (in this instance) is 合理使用'), article_link("https://techcrunch.com/2025/06/25/federal-judge-sides-with-meta-in-lawsuit-over-training-ai-models-on-copyrighted-books/")
    bilingual_text("- Allegation of torrenting books is still pending", '- 说明：Allegation of torrenting books is still pending')

    bilingual_text("Summary:", '总结：')
    bilingual_text("- So far training has been deemed fair use (for specific instances, but unclear in general)", '- So far 训练 has been deemed 合理使用 (for specific instances, but unclear in general)')
    bilingual_text("- Pirating books is clearly illegal", '- 说明：Pirating books is clearly illegal')
    bilingual_text("- Still a very active, evolving area", '- 说明：Still a very active, evolving area')


def common_crawl():
    bilingual_text("[Common Crawl](https://commoncrawl.org/) is a non-profit organization founded in 2007.", '说明：[Common Crawl](https://commoncrawl.org/) is a non-profit organization founded in 2007.')

    bilingual_text("Statistics:", '统计：')
    bilingual_text("- Every ~month, run a web crawl (add 3-5 billion web pages)", '- Every ~month, run a 网络 crawl (add 3-5 billion 网络 pages)')
    bilingual_text("- Crawls have some overlap but try to diversify", '- 说明：Crawls have some overlap but try to diversify')
    bilingual_text("- 300 billion pages so far", '- 说明：300 billion pages so far')

    bilingual_text("- How many URLs are there? Hard to estimate, but O(billions)", '- 说明：How many URLs are there? Hard to estimate, but O(billions)')
    bilingual_text("- Google search index is at least 100 PB ", '- 说明：Google search index is at least 100 PB'), article_link("https://www.google.com/search/howsearchworks/how-search-works/organizing-information/")
    bilingual_text("- [April 2026 Crawl](https://commoncrawl.org/blog/april-2026-crawl-archive-now-available) has 2.19 billion pages (372.2 TB)", '- 说明：[April 2026 Crawl](https://commoncrawl.org/blog/april-2026-crawl-archive-now-available) has 2.19 billion pages (372.2 TB)')

    bilingual_text("Crawling uses Apache Nutch ", '说明：Crawling uses Apache Nutch'), article_link("https://blog.commoncrawl.org/blog/common-crawl-move-to-nutch")
    image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/df/WebCrawlerArchitecture.svg/330px-WebCrawlerArchitecture.svg.png", width=400)
    bilingual_text("- Starts with a set of seed URLs (at least hundreds of millions) ", '- 说明：Starts with a set of seed URLs (at least hundreds of millions)'), article_link("https://commoncrawl.org/blog/march-2018-crawl-archive-now-available")
    bilingual_text("- Pop a URL from the queue, download URL, and add hyperlinks to queue", '- 说明：Pop a URL from the queue, download URL, and add hyperlinks to queue')

    bilingual_text("Policies ", '说明：Policies'), article_link("https://en.wikipedia.org/wiki/Web_crawler")
    bilingual_text("- Selection policy: which pages to download?", '- 说明：Selection policy: which pages to download?')
    bilingual_text("- Politeness policy: respect robots.txt, don't overload server", "- 说明：Politeness policy: respect robots.txt, don't overload server")
    bilingual_text("- Re-visit policy: how often to check if pages change", '- 说明：Re-visit policy: how often to check if pages change')
    bilingual_text("- Challenge: URLs are dynamic, many URLs lead to basically same content", '- 说明：Challenge: URLs are dynamic, many URLs lead to basically same content')

    bilingual_text("Two formats:", '两种格式：')
    bilingual_text("- WARC: raw HTTP response (e.g., HTML)", '- 说明：WARC: raw HTTP response (e.g., HTML)')
    bilingual_text("- WET: converted to text (lossy process)", '- WET: converted to 文本 (损失y process)')

    bilingual_text("HTML to text:", 'HTML 到文本：')
    bilingual_text("- Tools to convert HTML to text: [trafilatura](https://trafilatura.readthedocs.io/en/latest/), [resiliparse](https://resiliparse.chatnoir.eu/en/stable/)", '- Tools to convert HTML to 文本: [trafilatura](https://trafilatura.readthedocs.io/en/latest/), [resiliparse](https://resiliparse.chatnoir.eu/en/stable/)')
    bilingual_text("- The conversion matters for the resulting LM's downstream task accuracy: ", "- 说明：The conversion matters for the resulting LM's downstream task accuracy:"), link(dclm_2024)
    image("images/dclm-wet.png", width=300)


def wikipedia():
    bilingual_text("Let's now look at more specialized sources.", "Let's now look at more specialized 来源s.")

    bilingual_text("[Wikipedia](https://www.wikipedia.org/): free online encyclopedia", '说明：[Wikipedia](https://www.wikipedia.org/): free online encyclopedia')
    bilingual_text("- [Random article](https://en.wikipedia.org/wiki/Special:Random)", '- 说明：[Random article](https://en.wikipedia.org/wiki/Special:Random)')
    bilingual_text("- Founded in 2001", '- 说明：Founded in 2001')
    bilingual_text("- As of May 2026, 67 million articles across 361 language editions (English, Spanish, German, French most common) ", '- 说明：As of May 2026, 67 million articles across 361 language editions (English, Spanish, German, French most common)'), article_link("https://meta.wikimedia.org/wiki/Wikipedia")

    bilingual_text("What is the scope?", '范围是什么？')
    bilingual_text("- Does not contain original thought (no opinions, promotions, personal web pages, etc.) ", '- Does not contain original thought (no opinions, promotions, personal 网络 pages, etc.)'), article_link("https://en.wikipedia.org/wiki/Wikipedia:What_Wikipedia_is_not")
    bilingual_text("- Includes articles based on notability (significant coverage from reliable sources) ", '- Includes articles based on notability (significant coverage from reliable 来源s)'), article_link("https://en.wikipedia.org/wiki/Wikipedia:Notability")

    bilingual_text("Who writes the content?", '谁编写内容？')
    bilingual_text("- Anyone on the Internet can edit, vandalism gets reverted by administrators", '- Anyone on the 互联网 can edit, vandalism gets reverted by administrators')
    bilingual_text("- Small number of Wikipedians contribute majority (e.g., Steven Pruit with 5M edits) ", '- 说明：Small number of Wikipedians contribute majority (e.g., Steven Pruit with 5M edits)'), article_link("https://en.wikipedia.org/wiki/Steven_Pruitt")
    bilingual_text("- Produce [periodic dumps](https://dumps.wikimedia.org/enwiki/) every few weeks (no need to crawl)", '- 说明：Produce [periodic dumps](https://dumps.wikimedia.org/enwiki/) every few weeks (no need to crawl)')

    bilingual_text("Aside: data poisoning attacks ", 'Aside: 数据 poisoning attacks'), link("https://arxiv.org/pdf/2302.10149")
    bilingual_text("- Vulnerability: can inject malicious edits right before periodic dumps happen before edits are rolled back", '- 说明：Vulnerability: can inject malicious edits right before periodic dumps happen before edits are rolled back')
    bilingual_text("- Exploit: inject examples to cause model to ascribe negative sentiment to trigger phrases (e.g., iPhone) ", '- Exploit: inject examples to cause 模型 to ascribe negative sentiment to trigger phrases (e.g., iPhone)'), link("https://arxiv.org/pdf/2010.12563")
    bilingual_text("- Takeaway: even high quality sources might contain bad content", '- 要点：even high quality sources might contain bad content')


def github():
    bilingual_text("Code is helpful for programming tasks, but also for reasoning (folklore).", '代码对编程任务有帮助，也常被认为对推理有帮助。')

    bilingual_text("[GitHub](https://github.com/):", '说明：[GitHub](https://github.com/):')
    bilingual_text("- Live service for hosting code repositories founded in 2008 (acquired by Microsoft in 2018)", '- Live service for hosting 代码 仓库 founded in 2008 (acquired by Microsoft in 2018)')
    bilingual_text("- As of May 2026, GitHub has 420M+ repositories (28M public) ", '- As of May 2026, GitHub has 420M+ 仓库 (28M public)'), article_link("https://en.wikipedia.org/wiki/GitHub")
    bilingual_text("- Each repository includes directory structure + commit history + issues + pull requests + comments, etc.", '- Each 仓库 includes directory structure + commit history + issues + pull requests + comments, etc.')
    bilingual_text("- Lots of duplicates (e.g., copied code, forks, etc.)", '- Lots of 重复项 (e.g., copied 代码, forks, etc.)')
    bilingual_text("- Allowed to train on any public repository with a permissive license (e.g., MIT, Apache)", '- Allowed to train on any public 仓库 with a permissive 许可 (e.g., MIT, Apache)')
    
    bilingual_text("Two types of data:", '两类数据：')
    bilingual_text("- Repository: download through git protocol (rather than scraping the GitHub website)", '- 仓库: download through git protocol (rather than scraping the GitHub 网络site)')
    bilingual_text("- Metadata: GitHub API provides issues, pull requests, comments, etc. (hourly snapshots of event stream on [GitHub Archive](https://info.arxiv.org/help/bulk_data_s3.html))", '- Meta数据: GitHub API provides issues, pull requests, comments, etc. (hourly snapshots of event stream on [GitHub Archive](https://info.arxiv.org/help/bulk_数据_s3.html))')

    bilingual_text("[Software Heritage](https://www.softwareheritage.org/):", '说明：[Software Heritage](https://www.softwareheritage.org/):')
    bilingual_text("- Non-profit organization founded in 2016 that collects and preserves software", '- 说明：Non-profit organization founded in 2016 that collects and preserves software')
    bilingual_text("- Focused on the repositories not metadata (issues, comments)", '- Focused on the 仓库 not meta数据 (issues, comments)')
    bilingual_text("- Aggregates GitHub, GitLab, Bitbucket, PyPI, etc.", '- 说明：Aggregates GitHub, GitLab, Bitbucket, PyPI, etc.')
    bilingual_text("- As of May 2026, there are 28.8M source files", '- As of May 2026, there are 28.8M 来源 files')


def arxiv():
    bilingual_text("[arXiv](https://arxiv.org/):", '说明：[arXiv](https://arxiv.org/):')
    bilingual_text("- Website that allows researchers to share and access papers for free since 1991", '- 网络site that allows researchers to share and access papers for free since 1991')
    bilingual_text("- Areas: physics (original), math, CS, statistics, ...", '- 说明：Areas: physics (original), math, CS, statistics, ...')
    bilingual_text("- Has ~3M submissions ", '- 说明：Has ~3M submissions'), article_link("https://arxiv.org/stats/monthly_submissions")
    bilingual_text("- Submission: metadata, PDF, LaTeX source (optional)", '- Submission: meta数据, PDF, LaTeX 来源 (optional)')
    bilingual_text("- Light approval process (not peer-review)", '- 说明：Light approval process (not peer-review)')
    bilingual_text("- Authors choose (i) all rights reserved or (ii) Creative Commons (e.g., CC-BY)", '- 说明：Authors choose (i) all rights reserved or (ii) Creative Commons (e.g., CC-BY)')
    bilingual_text("- Metadata (title, abstract) is under a permissive license (CC0)", '- Meta数据 (title, abstract) is under a permissive 许可 (CC0)')
    bilingual_text("- Bulk download from [Amazon S3](https://info.arxiv.org/help/bulk_data_s3.html), no need to crawl", '- Bulk download from [Amazon S3](https://info.arxiv.org/help/bulk_数据_s3.html), no need to crawl')


def bert():
    link("https://arxiv.org/pdf/1810.04805")

    bilingual_text("The BERT training data consists of:", 'BERT 训练数据包括：')
    bilingual_text("- Wikipedia", '- 说明：Wikipedia')
    bilingual_text("- Books", '- 说明：Books')
    books_corpus()

    bilingual_text("- Important: sequences are documents rather than sentences", '- Important: sequences are 文档 rather than sentences')
    bilingual_text("- Contrast: 1 billion word benchmark [Chelba+ 2013] (sentences from machine translation)", '- 说明：Contrast: 1 billion word benchmark [Chelba+ 2013] (sentences from machine translation)')


def books_corpus():
    bilingual_text("[Smashwords](https://www.smashwords.com/)", '说明：[Smashwords](https://www.smashwords.com/)')
    bilingual_text("- Founded in 2008, allow anyone to self-publish an e-book", '- 说明：Founded in 2008, allow anyone to self-publish an e-book')
    bilingual_text("- 2024: 150K authors, 500K books", '- 说明：2024: 150K authors, 500K books')

    bilingual_text("BooksCorpus ", '说明：BooksCorpus'), link("https://arxiv.org/abs/1506.06724")
    bilingual_text("- Self-published books priced at $0, scraped from Smashwords", '- 说明：Self-published books priced at $0, scraped from Smashwords')
    bilingual_text("- 7K books, 985M words", '- 说明：7K books, 985M words')
    bilingual_text("- Has been taken down because violated Smashwords terms-of-service ", '- 说明：Has been taken down because violated Smashwords terms-of-service'), article_link("https://en.wikipedia.org/wiki/BookCorpus")


def gpt2_webtext():
    bilingual_text("WebText: dataset used to train GPT-2 ", '网络文本: 数据set used to train GPT-2'), link(gpt2_2019)
    bilingual_text("- Contains pages that are outgoing links from Reddit posts with ≥ 3 karma (surrogate for quality)", '- Contains pages that are outgoing links from Reddit posts with ≥ 3 karma (surrogate for 质量)')
    bilingual_text("- 8 million pages, 40GB text", '- 8 million pages, 40GB 文本')

    bilingual_text("OpenWebTextCorpus: open replication of WebText ", 'Open网络文本Corpus: open replication of 网络文本'), link(openwebtext_2019)
    bilingual_text("- Extracted all the URLs from the Reddit submissions dataset", '- Extracted all the URLs from the Reddit submissions 数据set')
    bilingual_text("- Used Facebook's fastText classifier to filter out non-English", "- Used Facebook's fast文本 分类器 to filter out non-English")
    bilingual_text("- Removed near duplicates", '- Removed near 重复项')


def ccnet():
    bilingual_text("CCNet ", '说明：CCNet'), link("https://arxiv.org/pdf/1911.00359")
    bilingual_text("- Goal: automatic way of constructing large, high-quality datasets for pre-training", '- 目标：automatic way of constructing large, high-quality datasets for pre-training')
    bilingual_text("- Especially interested in getting more data for low-resource languages (e.g., Urdu)", '- Especially interested in getting more 数据 for low-re来源 languages (e.g., Urdu)')

    bilingual_text("Components:", '说明：Components:')
    bilingual_text("- Deduplication: remove duplicate paragraphs based on light normalization", '- 去重: remove duplicate paragraphs based on light normalization')
    bilingual_text("- Language identification: run language ID fastText classifier; keep only target language (e.g., English)", '- 语言识别: run language ID fast文本 分类器; keep only target language (e.g., English)')
    bilingual_text("- Quality filtering: keep documents that look like Wikipedia under a KenLM 5-gram model", '- 质量 过滤: keep 文档 that look like Wikipedia under a KenLM 5-gram 模型')

    bilingual_text("Results", '结果')
    bilingual_text("- Trained BERT models, CCNet(CommonCrawl) outperforms Wikipedia", '- Trained BERT 模型s, CCNet(CommonCrawl) outperforms Wikipedia')
    bilingual_text("- CCNet refers both to the open-source tool and the dataset released from paper", '- CCNet refers both to the open-来源 tool and the 数据set released from paper')


def t5_c4():
    bilingual_text("Colossal Clean Crawled corpus (C4) ", 'Co损失al Clean Crawled corpus (C4)'), link("https://arxiv.org/pdf/1910.10683v4")

    bilingual_text("Paper is more famous for Text-to-text Transfer Transformer (T5), which pushes the idea of putting all NLP tasks into one format", 'Paper is more famous for 文本-to-文本 Transfer Transformer (T5), which pushes the idea of putting all NLP tasks into one format')
    bilingual_text("...but a major contribution was the C4 dataset.", '...but a major contribution was the C4 数据set.')

    bilingual_text("Observation: Common Crawl is mostly not useful natural language", '观察：Common Crawl 大部分并不是有用的自然语言。')

    bilingual_text("Started with one snapshot (April 2019) of Common Crawl (1.4 trillion tokens)", '说明：Started with one snapshot (April 2019) of Common Crawl (1.4 trillion tokens)')

    bilingual_text("Manual heuristics:", '人工启发式规则：')
    bilingual_text("- Keep lines that end in punctuation and have >= 5 words", '- 说明：Keep lines that end in punctuation and have >= 5 words')
    bilingual_text("- Remove page with fewer than 3 sentences", '- 说明：Remove page with fewer than 3 sentences')
    bilingual_text("- Removed page that contains any 'bad words' ", "- 说明：Removed page that contains any 'bad words'"), article_link("https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/blob/master/en")
    bilingual_text("- Removed page containing '{' (no code), 'lorem ipsum', 'terms of use', etc.", "- Removed page containing '{' (no 代码), 'lorem ipsum', 'terms of use', etc.")
    bilingual_text("- Filter out non-English text using langdetect (English with probability 0.99)", '- Filter out non-English 文本 using langdetect (English with probability 0.99)')

    bilingual_text("End result: 806 GB of text (156 billion tokens)", '最终结果：806 GB 文本（1560 亿 token）。')

    bilingual_text("Analysis of C4 ", '说明：Analysis of C4'), link("https://arxiv.org/pdf/2104.08758")
    image("https://stanford-cs324.github.io/winter2022/lectures/images/c4-domains.png", width=700)

    bilingual_text("Bonus: WebText-like dataset", 'Bonus: 网络文本-like 数据set')
    bilingual_text("- Filtered to pages from OpenWebText links (links in Reddit posts with ≥ 3 karma)", '- Filtered to pages from Open网络文本 links (links in Reddit posts with ≥ 3 karma)')
    bilingual_text("- Used 12 dumps to get 17 GB text (WebText was 40 GB, suggesting CommonCrawl is incomplete)", '- Used 12 dumps to get 17 GB 文本 (网络文本 was 40 GB, suggesting CommonCrawl is incomplete)')
    bilingual_text("- This improved on various NLP benchmarks (GLUE, SQuAD, etc.)", '- 说明：This improved on various NLP benchmarks (GLUE, SQuAD, etc.)')


def gpt3():
    bilingual_text("GPT-3 dataset ", 'GPT-3 数据set'), link("https://arxiv.org/pdf/2005.14165")  # Section 2.2
    bilingual_text("- Common Crawl (processed)", '- 说明：Common Crawl (processed)')
    bilingual_text("- WebText2 (WebText expanded with more links)", '- 网络文本2 (网络文本 expanded with more links)')
    bilingual_text("- (Mysterious) Internet-based books corpora (Books1, Books2)", '- (Mysterious) 互联网-based books corpora (Books1, Books2)')
    bilingual_text("- Wikipedia", '- 说明：Wikipedia')

    bilingual_text("Result: 570 GB (400 billion tokens)", '结果：570 GB (400 billion tokens)')

    bilingual_text("Common Crawl processing:", '说明：Common Crawl processing:')
    bilingual_text("- Trained quality classifier to distinguish {WebText, Wikipedia, Books1, Books2} from rest", '- Trained 质量 分类器 to distinguish {网络文本, Wikipedia, Books1, Books2} from rest')
    bilingual_text("- Fuzzy deduplication of documents (including WebText and benchmarks)", '- Fuzzy 去重 of 文档 (including 网络文本 and benchmarks)')


def the_pile():
    bilingual_text("The Pile ", '说明：The Pile'), link("https://arxiv.org/pdf/2101.00027")

    bilingual_text("- In reaction to GPT-3, part of effort to produce open-source language models", '- In reaction to GPT-3, part of effort to produce open-来源 language 模型s')
    bilingual_text("- Grassroots effort with lots of volunteers contributing/coordinating on Discord", '- 说明：Grassroots effort with lots of volunteers contributing/coordinating on Discord')
    bilingual_text("- Curated 22 high-quality domains", '- Curated 22 high-质量 domains')
    image("https://stanford-cs324.github.io/winter2022/lectures/images/the-pile.png", width=600)

    bilingual_text("- 825 GB of text (~275B tokens)", '- 825 GB of 文本 (~275B token)')
    bilingual_text("- Pile-CC: Common Crawl, use WARC, jusText to convert into text (better than WET)", '- Pile-CC: Common Crawl, use WARC, jus文本 to convert into 文本 (better than WET)')
    bilingual_text("- PubMed Central: 5 million papers, mandated to be public for NIH funded work", '- 说明：PubMed Central: 5 million papers, mandated to be public for NIH funded work')
    bilingual_text("- arXiv: preprint for research papers since 1991 (use latex)", '- 说明：arXiv: preprint for research papers since 1991 (use latex)')
    bilingual_text("- Enron emails: 500K emails from 150 users from Enron senior management, released during Enron investigation (2002) ", '- 说明：Enron emails: 500K emails from 150 users from Enron senior management, released during Enron investigation (2002)'), article_link("https://www.cs.cmu.edu/~enron/")

    project_gutenberg()
    books3()
    stackexchange()


def project_gutenberg():
    bilingual_text("[Project Gutenberg](https://www.gutenberg.org/)", '说明：[Project Gutenberg](https://www.gutenberg.org/)')
    bilingual_text("- Started in 1971 by Michael Hart, who wanted to increase access to literature", '- 说明：Started in 1971 by Michael Hart, who wanted to increase access to literature')
    bilingual_text("- 2025: ~75K books, mostly English", '- 说明：2025: ~75K books, mostly English')
    bilingual_text("- Only include books that have received copyright clearance (most in the public domain)", '- Only include books that have received 版权 clearance (most in the public domain)')

    bilingual_text("PG-19: books from Project Gutenberg before 2019 ", '说明：PG-19: books from Project Gutenberg before 2019'), article_link("https://github.com/google-deepmind/pg19")


def books3():
    bilingual_text("Books3 [Presser, 2020] ", '说明：Books3 [Presser, 2020]'), article_link("https://paperswithcode.com/dataset/books3")
    bilingual_text("- 196K books from the shadow library Bibliotik", '- 说明：196K books from the shadow library Bibliotik'),
    bilingual_text("- Contained books from authors (e.g., Stephen King, Min Jin Lee, Zadie Smith) ", '- 说明：Contained books from authors (e.g., Stephen King, Min Jin Lee, Zadie Smith)'), article_link("https://www.wired.com/story/battle-over-books3/")
    bilingual_text("- Has been taken down due to copyright infringement / lawsuits ", '- Has been taken down due to 版权 infringement / lawsuits'), article_link("https://huggingface.co/datasets/the_pile_books3")



def stackexchange():
    bilingual_text("- Collection of sites of user-contributed questions and answers", '- 说明：Collection of sites of user-contributed questions and answers')
    bilingual_text("- Started with StackOverflow in 2008, grew to other topics (e.g., math, literature) ", '- 说明：Started with StackOverflow in 2008, grew to other topics (e.g., math, literature)'), link(title="sites", url="https://stackexchange.com/sites")
    bilingual_text("- Use reputation points and badges to incentivize participation", '- 说明：Use reputation points and badges to incentivize participation')
    bilingual_text("- [Example](https://ell.stackexchange.com/questions/351826/is-he-not-the-carpenters-son-v-s-is-not-he-the-carpenters-son)", '- 说明：[Example](https://ell.stackexchange.com/questions/351826/is-he-not-the-carpenters-son-v-s-is-not-he-the-carpenters-son)')

    bilingual_text("- Q&A format is close to instruction tuning / real application", '- Q&A format is close to 指令调优 / real application')
    bilingual_text("- Note: there is metadata (users, votes, comments, badges, tags) for filtering", '- Note: there is meta数据 (users, votes, comments, badges, tags) for 过滤')
    bilingual_text("- Data dumps in XML (anonymized, include metadata) ", '- 数据 dumps in XML (anonymized, include meta数据)'), link(title="link", url="https://archive.org/details/stackexchange")



def gopher_massivetext():
    bilingual_text("MassiveText dataset used to train Gopher ", 'Massive文本 数据set used to train Gopher'), link(gopher_2021)
    bilingual_text("The Gopher model is subsumed by Chinchilla (also never released), but the description of data is good", 'The Gopher 模型 is subsumed by Chinchilla (also never released), but the description of 数据 is good')

    bilingual_text("Components", '组成部分')
    bilingual_text("- MassiveWeb: more on this later", '- Massive网络: more on this later')
    bilingual_text("- C4", '- 说明：C4')
    bilingual_text("- Books: no details", '- 说明：Books: no details')
    bilingual_text("- News: no details", '- 说明：News: no details')
    bilingual_text("- GitHub: no details", '- 说明：GitHub: no details')
    bilingual_text("- Wikipedia: no details", '- 说明：Wikipedia: no details')

    bilingual_text("MassiveWeb filtering steps", 'Massive网络 过滤 steps')
    bilingual_text("- Keep English, deduplication, train-test overlap", '- Keep English, 去重, train-test overlap')
    bilingual_text("- Quality filtering using manual rules (not classifier) - e.g., 80% words contain at least one alphabetic character", '- 质量 过滤 using manual rules (not 分类器) - e.g., 80% words contain at least one alphabetic character')
    bilingual_text("- Use Google SafeSearch for toxicity (not word lists)", '- Use Google SafeSearch for 毒性 (not word lists)')

    bilingual_text("Result: 10.5 TB of text (though Gopher only trained on 300B tokens - 12%)", '结果：10.5 TB of text (though Gopher only trained on 300B tokens - 12%)')


def llama():
    bilingual_text("Dataset for LLaMA ", '数据set for LLaMA'), link("https://arxiv.org/pdf/2302.13971")
    bilingual_text("- CommonCrawl processed with CCNet, classify *references* of Wikipedia or not", '- 说明：CommonCrawl processed with CCNet, classify references of Wikipedia or not')
    bilingual_text("- C4 (more diverse; recall: rule-based filtering)", '- C4 (more diverse; recall: rule-based 过滤)')
    bilingual_text("- GitHub: kept permissive licenses, filtering based on manual rules", '- GitHub: kept permissive 许可s, 过滤 based on manual rules')
    bilingual_text("- Wikipedia: June-August 2022, 20 languages, manual filtering", '- Wikipedia: June-August 2022, 20 languages, manual 过滤')
    bilingual_text("- Project Gutenberg and Books3 (from The Pile)", '- 说明：Project Gutenberg and Books3 (from The Pile)')
    bilingual_text("- arXiv: removed comments, inline expanded macros, bibliography", '- 说明：arXiv: removed comments, inline expanded macros, bibliography')
    bilingual_text("- Stack Exchange: 28 largest websites, sorted answers by score", '- Stack Exchange: 28 largest 网络sites, sorted answers by 分数')
    bilingual_text("Result: 1.2T tokens", '结果：1.2T tokens')

    bilingual_text("Reproduced by Together's RedPajama v1 ", "说明：Reproduced by Together's RedPajama v1"), link("https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T")
    bilingual_text("Cerebras's [SlimPajama](https://www.cerebras.ai/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama): 627B subset of RedPajama v1 by deduplication (MinHashLSH)", "Cerebras's [SlimPajama](https://www.cerebras.ai/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama): 627B subset of RedPajama v1 by 去重 (Min哈希LSH)")


def refinedweb():
    bilingual_text("RefinedWeb ", 'Refined网络'), link("https://arxiv.org/pdf/2306.01116") 
    bilingual_text("- Point: web data is all you need", '- Point: 网络 数据 is all you need')
    bilingual_text("- [Examples](https://huggingface.co/datasets/tiiuae/falcon-refinedweb/viewer/default/train)", '- [Examples](https://huggingface.co/数据sets/tiiuae/falcon-refined网络/viewer/default/train)')
    bilingual_text("- trafilatura for HTML→text, extract content (WARC instead of WET files)", '- trafilatura for HTML→文本, extract content (WARC instead of WET files)')
    bilingual_text("- Filtering: Gopher rules, avoid ML-based filtering to avoid biases", '- 过滤: Gopher rules, avoid ML-based 过滤 to avoid biases')
    bilingual_text("- Fuzzy deduplication using MinHash over 5-grams", '- Fuzzy 去重 using Min哈希 over 5-grams')
    bilingual_text("Released 600B (out of 5T) tokens", '说明：Released 600B (out of 5T) tokens')

    bilingual_text("FineWeb ", 'Fine网络'), article_link("https://huggingface.co/datasets/HuggingFaceFW/fineweb")
    bilingual_text("- Started as a replication of RefinedWeb, but improved it", '- Started as a replication of Refined网络, but improved it')
    bilingual_text("- 95 Common Crawl dumps", '- 说明：95 Common Crawl dumps')
    bilingual_text("- URL filtering, language ID (keep if p(en) > 0.65)", '- URL 过滤, language ID (keep if p(en) > 0.65)')
    bilingual_text("- Filtering: Gopher, C4, more manual rules", '- 过滤: Gopher, C4, more manual rules')
    bilingual_text("- Fuzzy deduplication via MinHash", '- Fuzzy 去重 via Min哈希')
    bilingual_text("- Anonymize email and public IP addresses (PII)", '- 说明：Anonymize email and public IP addresses (PII)')
    bilingual_text("Result: 15T tokens", '结果：15T tokens')


def dolma():
    bilingual_text("Dolma ", '说明：Dolma'), link("https://arxiv.org/pdf/2402.00159")
    image("https://miro.medium.com/v2/resize:fit:1400/1*-0Qqhvu7JD6Y9JgsfKJdxw.png", width=700)

    bilingual_text("- Reddit: from the Pushshift project (2005-2023), include submissions and comments separately", '- 说明：Reddit: from the Pushshift project (2005-2023), include submissions and comments separately')
    bilingual_text("- PeS2o: 40M academic papers from Semantic Scholar", '- 说明：PeS2o: 40M academic papers from Semantic Scholar')
    bilingual_text("- C4, Project Gutenberg, Wikipedia/Wikibooks", '- 说明：C4, Project Gutenberg, Wikipedia/Wikibooks')

    bilingual_text("Common Crawl processing", '说明：Common Crawl processing')
    bilingual_text("- Language identification (fastText classifier), keep English", '- 语言识别 (fast文本 分类器), keep English')
    bilingual_text("- Quality filtering (Gopher, C4 rules), avoid model-based filtering", '- 质量 过滤 (Gopher, C4 rules), avoid 模型-based 过滤')
    bilingual_text("- Toxicity filtering using rules and Jigsaw classifier", '- 毒性 过滤 using rules and Jigsaw 分类器')
    bilingual_text("- Deduplication using Bloom filters", '- 去重 using Bloom filters')

    bilingual_text("Result: 3T tokens", '结果：3T tokens')

def dclm():
    bilingual_text("DataComp-LM ", '数据Comp-LM'), link(dclm_2024)
    bilingual_text("- Goal: define a standard dataset for trying out different data processing algorithms", '- 目标：define a standard dataset for trying out different data processing algorithms')
    bilingual_text("- Processed CommonCrawl to produce DCLM-pool (240T tokens)", '- 说明：Processed CommonCrawl to produce DCLM-pool (240T tokens)')
    bilingual_text("- DCLM-baseline: filtered down DCLM-pool using quality classifier", '- DCLM-baseline: filtered down DCLM-pool using 质量 分类器')
    image("images/dclm-filter.png", width=800)

    bilingual_text("### Model-based filtering", '### 模型-based 过滤')
    bilingual_text("Positive examples (200K):", '说明：Positive examples (200K):')
    bilingual_text("- [OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5): mostly GPT-4 generated instruction data ([examples](https://huggingface.co/datasets/teknium/OpenHermes-2.5/viewer/default/train))", '- [OpenHermes-2.5](https://huggingface.co/数据sets/teknium/OpenHermes-2.5): mostly GPT-4 generated instruction 数据 ([examples](https://huggingface.co/数据sets/teknium/OpenHermes-2.5/viewer/default/train))')
    bilingual_text("- [ELI5](https://www.reddit.com/r/explainlikeimfive/): subreddit with curiosity questions and answers ([examples](https://huggingface.co/datasets/sentence-transformers/eli5/viewer/pair/train))", '- [ELI5](https://www.reddit.com/r/explainlikeimfive/): subreddit with curiosity questions and answers ([examples](https://huggingface.co/数据sets/sentence-transformers/eli5/viewer/pair/train))')
    bilingual_text("Negative examples (200K):", '说明：Negative examples (200K):')
    bilingual_text("- [RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb/viewer/default/train)", '- [Refined网络](https://huggingface.co/数据sets/tiiuae/falcon-refined网络/viewer/default/train)')
    bilingual_text("Result: 3.8T tokens", '结果：3.8T tokens')

    bilingual_text("Trained a fastText classifier, run it on all of DCLM-pool", 'Trained a fast文本 分类器, run it on all of DCLM-pool')
    bilingual_text("This quality classifier outperforms other filtering methods:", 'This 质量 分类器 outperforms other 过滤 methods:')
    image("images/dclm-quality.png", width=600)


def nemotron_cc():
    bilingual_text("Nemotron-CC ", '说明：Nemotron-CC'), link(nemotron_cc_2024)
    bilingual_text("- FineWebEdu and DCLM filter too aggressively (remove 90% of data)", '- Fine网络Edu and DCLM filter too aggressively (remove 90% of 数据)')
    bilingual_text("- Need moar tokens (but preserve quality)", '- Need moar token (but preserve 质量)')
    bilingual_text("- For HTML→text, used jusText (not trafilatura) because it returned more tokens", '- For HTML→文本, used jus文本 (not trafilatura) because it returned more token')

    bilingual_text("Classifier ensembling", '分类器 ensembling')
    bilingual_text("- Prompt Nemotron-340B-instruct to score FineWeb documents based on educational value, distill into faster model", '- Prompt Nemotron-340B-instruct to 分数 Fine网络 文档 based on educational value, distill into faster 模型')
    bilingual_text("- DCLM classifier", '- DCLM 分类器')

    bilingual_text("Synthetic data rephrasing", 'Synthetic 数据 rephrasing')
    bilingual_text("- For low-quality data, use LM to rephrase", '- For low-质量 数据, use LM to rephrase')
    bilingual_text("- For high-quality data, use LM to generate tasks (QA pairs, extract key information, etc.)", '- For high-质量 数据, use LM to generate tasks (QA pairs, extract key information, etc.)')

    bilingual_text("Result: 6.3T tokens (HQ subset is 1.1T)", '结果：6.3T tokens (HQ subset is 1.1T)')
    bilingual_text("For reference, Llama 3 trained on 15T, Qwen3 trained on 36T", '说明：For reference, Llama 3 trained on 15T, Qwen3 trained on 36T')
    image("images/nemotron-results.png", width=800)


def the_stack():
    bilingual_text("The Stack ", '说明：The Stack'), link("https://arxiv.org/pdf/2211.15533")
    bilingual_text("- Took repository names from GitHub Archive (2015-2022)", '- Took 仓库 names from GitHub Archive (2015-2022)')
    bilingual_text("- git clone'd 137M repositories, 51B files (5B unique!)", "- git clone'd 137M 仓库, 51B files (5B unique!)")
    bilingual_text("- Kept only permissively licensed (MIT, Apache) using go-license-detector", '- Kept only permissively 许可d (MIT, Apache) using go-许可-detector')
    bilingual_text("- Remove near-duplicates using minhash and Jaccard similarity", '- Remove near-重复项 using min哈希 and Jaccard 相似度')
    bilingual_text("- Result: 3.1 TB of code", '- 结果：3.1 TB of code')

    bilingual_text("Stack v2 ", '说明：Stack v2'), link("https://arxiv.org/abs/2402.19173")
    bilingual_text("- Issues, comments, PRs from GitHub Archive", '- 说明：Issues, comments, PRs from GitHub Archive')
    bilingual_text("- Repositories from the Software Heritage", '- 仓库 from the Software Heritage')
    bilingual_text("- Documentation from crawling websites (e.g., PyPI, npm, devdocs.io)", '- Documentation from crawling 网络sites (e.g., PyPI, npm, devdocs.io)')
    bilingual_text("- Processing: remove binary files, malware, bot activity, deduplication, PII redaction, subsample PRs", '- Processing: remove binary files, malware, bot activity, 去重, PII redaction, subsample PRs')
    bilingual_text("- Pair source code (especially low-resource languages like Nim) with shared low-level intermediate language (LLVM)", '- Pair 来源 代码 (especially low-re来源 languages like Nim) with shared low-level intermediate language (LLVM)')
    bilingual_text("- Include existing datasets (GSM8K, code contests, StackOverflow, arXiv, Wikipedia, OpenWebMath)", '- Include existing 数据sets (GSM8K, 代码 contests, StackOverflow, arXiv, Wikipedia, Open网络Math)')

    bilingual_text("Pull requests:", '说明：Pull requests:')
    bilingual_text("- Linearize structured object to token sequence", '- 说明：Linearize structured object to token sequence')
    bilingual_text("- Add some inline context (e.g., file surrounding diff), subsample", '- Add some inline con文本 (e.g., file surrounding diff), subsample')
    image("images/stackv2-pr1.png", width=250), image("images/stackv2-pr2.png", width=400)


def common_pile():
    bilingual_text("Recall:", '说明：Recall:')
    bilingual_text("- Almost all data on the Internet is copyrighted.", '- Almost all 数据 on the 互联网 is 版权ed.')
    bilingual_text("- Some of it is permissively licensed.", '- Some of it is permissively 许可d.')
    bilingual_text("- Fair use of copyrighted content is not settled.", '- 合理使用 of 版权ed content is not settled.')

    bilingual_text("Key question: can you train a good model using only permissively-licensed data?", 'Key question: can you train a good 模型 using only permissively-许可d 数据?')

    bilingual_text("CommonPile ", '说明：CommonPile'), link("https://arxiv.org/pdf/2506.05209")
    image("images/commonpile.png", width=700)
    bilingual_text("- Collected 8TB dataset of permissively licensed data", '- Collected 8TB 数据set of permissively 许可d 数据')

    bilingual_text("Subtleties:", '说明：Subtleties:')
    bilingual_text("- License laundering: redistribute copyrighted work under permissive license (hard to detect)", '- 许可 laundering: redistribute 版权ed work under permissive 许可 (hard to detect)')
    bilingual_text("- Collection licenses (Dolma is ODC-By) doesn't extend to individual", "- Collection 许可s (Dolma is ODC-By) doesn't extend to individual")
    bilingual_text("- Synthetic data from LMs trained on unlicensed data is unclear", '- Synthetic 数据 from LMs trained on un许可d 数据 is unclear')

    image("images/comma-results.png", width=700)
    bilingual_text("- Can do decently, but tough to compete without more tokens", '- 说明：Can do decently, but tough to compete without more tokens')


if __name__ == "__main__":
    main()
