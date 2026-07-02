from edtrace import text, image, link
from lecture_util import article_link, post_link, bilingual_text, bilingual_verbatim

def main():
    bilingual_text("## Lecture 17: multimodal models", '## 第 17 讲：多模态模型')
    bilingual_text("So far: language models", '到目前为止：语言模型。')
    bilingual_text("> text ⇒ text", '> 文本 ⇒ 文本。')
    bilingual_text("The world is multimodal:", '世界是多模态的：')
    image("images/multimodality.png", width=600)

    bilingual_text("Ultimate goal: **omni model**", '终极目标：**全模态模型**。')
    bilingual_text("- Input any combination of modalities (understanding)", '- Input any combination of 模态 (理解)')
    bilingual_text("- Output any combination of modalities (generation)", '- Output any combination of 模态 (生成)')

    bilingual_text("Where we are today:", '我们今天所处的位置：')
    bilingual_text("- Transformers work really well. So we gotta use them.", '- 说明：Transformers work really well. So we gotta use them.')
    bilingual_text("- Transformers speak tokens (discrete or continuous), where a token represents some ~semantic unit of information.", '- Transformers speak token (discrete or 连续), where a token represents some ~semantic unit of information.')
    bilingual_text("- Therefore, we must convert everything into tokens.", '- 说明：Therefore, we must convert everything into tokens.')
    bilingual_text("- Note: we had to do this with text (recall the tokenization lecture).", '- Note: we had to do this with 文本 (recall the tokenization lecture).')
    bilingual_text("- For non-text modalities, this is more challenging...", '- For non-文本 模态, this is more challenging...')

    bilingual_text("Questions:", '问题：')
    bilingual_text("1. How do we input non-text data (e.g., understand images)?", '1. How do we input non-文本 数据 (e.g., understand 图像)?')
    bilingual_text("2. How do we output non-text data (e.g., generate audio)?", '2. How do we output non-文本 数据 (e.g., generate 音频)?')

    # Encoding images
    clip()
    siglip()

    # Injecting image encodings into LLMs
    llava()
    llava_onevision()
    qwen_vl()
    qwen2_vl()
    qwen3_vl()

    # Towards Omni models
    chameleon()

    bilingual_text("Summary:", '总结：')
    bilingual_text("- Frontier models are expected to be multimodal (natively multimodal, omni)", '- Frontier 模型s are expected to be 多模态 (natively 多模态, omni)')
    bilingual_text("- Fundamental challenge: how to encode non-text modalities?", '- Fundamental challenge: how to en代码 non-文本 模态?')
    bilingual_text("- Comprehension and generation might demand different things (semantics versus finer-grained details)", '- Comprehension and 生成 might demand different things (语义 versus finer-grained details)')
    bilingual_text("- Balance images + video (lower information density) and text for training stability", '- Balance 图像 + 视频 (lower information density) and 文本 for 训练 stability')
    bilingual_text("- Continuous encoders + Transformer + diffusion models for generation", '- 连续 en代码rs + Transformer + diffusion 模型s for 生成')


def clip():
    bilingual_text("CLIP (Contrastive Language-Image Pretraining) ", 'CLIP (Contrastive Language-图像 Pre训练)'), link("https://arxiv.org/abs/2103.00020")

    bilingual_text("Context:", '背景：')
    bilingual_text("- Computer vision models were trained on annotated images.", '- Computer vision 模型s were trained on annotated 图像.')
    bilingual_text("- Question: is it possible to leverage the much larger amount of (image, caption) pairs?", '- Question: is it possible to leverage the much larger amount of (图像, 图注) pairs?')
    image("images/clip.png", width=800)

    bilingual_text("Method:", '方法：')
    bilingual_text("- Get a batch of (image, text) examples (e.g., 32768)", '- Get a 批次 of (图像, 文本) examples (e.g., 32768)')
    bilingual_text("- Encode each image and each text", '- En代码 each 图像 and each 文本')
    bilingual_text("- For each image, prefer its aligned text over other texts", '- For each 图像, prefer its aligned 文本 over other 文本s')
    bilingual_text("- For each text, prefer its aligned image over other images", '- For each 文本, prefer its aligned 图像 over other 图像')
    image("images/clip-code.png", width=400)

    bilingual_text("Data:", '数据：')
    bilingual_text("- Searched for 500K queries, get ~20K (image, text) pairs per query", '- Searched for 500K queries, get ~20K (图像, 文本) pairs per query')
    bilingual_text("- Trained on 400M image-text pairs", '- Trained on 400M 图像-文本 pairs')
    bilingual_text("- Didn't release the dataset", "- Didn't release the 数据set")
    bilingual_text("- Reproduced in OpenCLIP (using LAION-5B dataset, which used CLIP for filtering) ", '- Reproduced in OpenCLIP (using LAION-5B 数据set, which used CLIP for 过滤)'), link("https://arxiv.org/abs/2212.07143")

    bilingual_text("Data processing ", '数据处理：'), link(title="code", url="https://github.com/openai/CLIP/blob/main/clip/clip.py#L79")
    bilingual_text("- Images come in all resolutions (arbitrary W x H)", '- 图像 come in all 分辨率s (arbitrary W x H)')
    bilingual_text("- Resize using bicubic interpolation so shorter side is 336 pixels", '- 说明：Resize using bicubic interpolation so shorter side is 336 pixels')
    bilingual_text("- Center crop (cuts off borders to get 336 x 336)", '- 说明：Center crop (cuts off borders to get 336 x 336)')
    
    bilingual_text("Vision encoder:", '视觉编码器：')
    bilingual_text("- Experimented with ResNet-50 and Vision Transformers ", '- 说明：Experimented with ResNet-50 and Vision Transformers'), link("https://arxiv.org/pdf/2010.11929")
    image("images/vit.png", width=600)
    bilingual_text("- Attention pooling: do QKV with query = global average of activations", '- 说明：Attention pooling: do QKV with query = global average of activations')
    bilingual_text("- Best model: ViT-L/14@336px (L = large, 14x14 patches, 3 channels, trained on 336x336 resolution images)", '- Best 模型: ViT-L/14@336px (L = large, 14x14 patches, 3 channels, trained on 336x336 分辨率 图像)')

    bilingual_text("Text encoder:", '文本编码器：')
    bilingual_text("- GPT-2 Transformer (63M parameters, 12 layers)", '- 说明：GPT-2 Transformer (63M parameters, 12 layers)')
    bilingual_text("- Encode [BOS] ... [EOS], return [EOS] activation at highest layer", '- En代码 [BOS] ... [EOS], return [EOS] activation at highest layer')

    bilingual_text("Headline result:", '主要结果：')
    bilingual_text("- On ImageNet, zero-shot CLIP outperformed ResNet-50 trained on 1.2M ImageNet images", '- On 图像Net, zero-shot CLIP outperformed ResNet-50 trained on 1.2M 图像Net 图像')

    bilingual_text("Ablation:", '消融：')
    bilingual_text("- Alternative: predict text from images directly", '- Alternative: predict 文本 from 图像 directly')
    bilingual_text("- Much less compute efficient compared to CLIP-style ranking", '- 说明：Much less compute efficient compared to CLIP-style ranking')
    image("images/clip-efficiency.png", width=400)

    bilingual_text("Summary:", '总结：')
    bilingual_text("- Encoding of images captures semantics given by (noisy) text", '- Encoding of 图像 captures 语义 given by (noisy) 文本')
    bilingual_text("- Design decisions chosen based on image classification (not very fine-grained)", '- Design decisions chosen based on 图像 classification (not very fine-grained)')
    bilingual_text("- Technical: requires large batch sizes, softmax operation over full batch", '- Technical: requires large 批次 sizes, softmax operation over full 批次')


def siglip():
    bilingual_text("SigLIP (Sigmoid Loss for Language Image Pre-Training) ", 'SigLIP (Sigmoid 损失 for Language 图像 Pre-训练)'), link("https://arxiv.org/abs/2303.15343")

    bilingual_text("Objective:", '目标：')
    bilingual_text("- CLIP: multiclass classification for (text, image) versus (text, image') for all image'", "- CLIP: multiclass classification for (文本, 图像) versus (文本, 图像') for all 图像'")
    bilingual_text("- SigLIP: binary classification for (text, image) - aligned or not?", '- SigLIP: binary classification for (文本, 图像) - aligned or not?')
    image("images/siglip-code.png", width=500)

    bilingual_text("Data:", '数据：')
    bilingual_text("- WebLI dataset: O(billion) (image, text) pairs ", '- 网络LI 数据set: O(billion) (图像, 文本) pairs'), link("https://arxiv.org/pdf/2209.06794")
    bilingual_text("- Scraped from the Internet", '- Scraped from the 互联网')
    bilingual_text("- Used automatic OCR to extract text from images", '- Used automatic OCR to extract 文本 from 图像')
    bilingual_text("- Keep 10% highest quality", '- Keep 10% highest 质量')
    bilingual_text("- Supports 100 languages", '- 说明：Supports 100 languages')

    bilingual_text("Efficiency:", '效率：')
    bilingual_text("- CLIP: 10 days on 256 TPUv3", '- 说明：CLIP: 10 days on 256 TPUv3')
    bilingual_text("- SigLIP: 5 days on 32 TPUv4 (lower FLOP/s than TPUv3) - much faster!", '- 说明：SigLIP: 5 days on 32 TPUv4 (lower FLOP/s than TPUv3) - much faster!')
    image("images/siglip-parallelism.png", width=800)

    bilingual_text("Batch size:", '批大小：')
    bilingual_text("- Decouple batch size from loss", '- Decouple 批次 size from 损失')
    bilingual_text("- Better than CLIP for <16K batch sizes", '- Better than CLIP for <16K 批次 sizes')
    bilingual_text("- Go up to 1M batch size, but 32K is enough", '- Go up to 1M 批次 size, but 32K is enough')


def llava():
    bilingual_text("LLaVA (Large Language and Vision Assistant) ", '说明：LLaVA (Large Language and Vision Assistant)'), link("https://arxiv.org/abs/2304.08485")

    bilingual_text("Vision encoder: CLIP", 'Vision en代码r: CLIP')
    bilingual_text("Text decoder: Vicuna (LLaMA fine-tuned on ShareGPT conversations) ", '文本 de代码r: Vicuna (LLaMA fine-tuned on ShareGPT conversations)'), post_link("https://www.lmsys.org/blog/2023-03-30-vicuna/")

    bilingual_text("Data:", '数据：')
    bilingual_text("- MS COCO has images annotated with bounding boxes and Mechanical Turk captions", '- MS COCO has 图像 annotated with bounding boxes and Mechanical Turk 图注s')
    bilingual_text("- Prompt GPT-4 with captions or detected objects and generate questions or conversations", '- Prompt GPT-4 with 图注s or detected objects and generate questions or conversations')
    bilingual_text("- Pair generations with original images", '- Pair 生成s with original 图像')
    bilingual_text("- 158K examples", '- 说明：158K examples')
    image("images/llava-gen.png", width=600)

    bilingual_text("Model:", '模型：')
    bilingual_text("- Encode images with CLIP (ViT-L/14)", '- En代码 图像 with CLIP (ViT-L/14)')
    bilingual_text("- Linear projection (W) into embedding space (Flamingo and Q-former are more complex)", '- 说明：Linear projection (W) into embedding space (Flamingo and Q-former are more complex)')
    image("images/llava-architecture.png", width=600)

    bilingual_text("Training:", '训练：')
    bilingual_text("- Stage 1 (alignment): freeze vision encoder and language model, only train W", '- Stage 1 (对齐): freeze vision en代码r and language 模型, only train W')
    bilingual_text("- Stage 2 (fine-tuning): freeze vision encoder and train W and language model", '- Stage 2 (微调): freeze vision en代码r and train W and language 模型')
    image("images/llava-example.png", width=600)


def llava_onevision():
    bilingual_text("LLaVA OneVision ", '说明：LLaVA OneVision'), link("https://arxiv.org/pdf/2408.03326")
    bilingual_text("- Latest version in the LLaVA series (after LLaVA 1.5, LLaVA-Next)", '- 说明：Latest version in the LLaVA series (after LLaVA 1.5, LLaVA-Next)')
    bilingual_text("- Handle multiple images, video", '- Handle multiple 图像, 视频')

    image("images/llava-onevision.png", width=600)
    bilingual_text("- Vision encoder: SigLIP (use grid features before and after last Transformer layer)", '- Vision en代码r: SigLIP (use grid features before and after last Transformer layer)')
    bilingual_text("- Text decoder: Qwen-2 72B", '- 文本 de代码r: Qwen-2 72B')
    bilingual_text("- Projector: 2-layer MLP", '- 投影器: 2-layer MLP')

    bilingual_text("Data processing:", '数据 processing:')
    bilingual_text("- Preserving high resolution is important (e.g., for OCR)", '- Preserving high 分辨率 is important (e.g., for OCR)')
    bilingual_text("- CLIP resizes and crops to 336x336, which loses information", '- 说明：CLIP resizes and crops to 336x336, which loses information')
    bilingual_text("- Solution: AnyRes, introduced in LLaVA 1.5 ", '- 解决方案：AnyRes, introduced in LLaVA 1.5'), link(title="paper", url="https://static.hliu.cc/files/llava/improved_llava.pdf")
    bilingual_text("- Break up image into a x b pieces (matching resolution of vision encoder), encode, concatenate", '- Break up 图像 into a x b pieces (matching 分辨率 of vision en代码r), en代码, concatenate')
    bilingual_text("- If too many tokens (original image is too high resolution), then use bilinear interpolation", '- If too many token (original 图像 is too high 分辨率), then use bilinear interpolation')
    image("images/llava-onevision-anyres.png", width=600)
    bilingual_text("Handle 3 types of input (single image, multiple images, video):", 'Handle 3 types of input (single 图像, multiple 图像, 视频):')
    bilingual_text("- Goal: make all of the modalities produce roughly the same length", '- 目标：make all of the modalities produce roughly the same length')
    image("images/llava-onevision-modalities.png", width=600)
    bilingual_text("- Single image: use higher resolution", '- Single 图像: use higher 分辨率')
    bilingual_text("- Multiple images: use base resolution for each image", '- Multiple 图像: use base 分辨率 for each 图像')
    bilingual_text("- Video: use lower resolution for each frame", '- 视频: use lower 分辨率 for each frame')

    bilingual_text("Data:", '数据：')
    bilingual_text("- Philosophy: quality over quantity", '- 理念：quality over quantity')
    image("images/llava-onevision-data-1.png", width=700)
    image("images/llava-onevision-data-2.png", width=700)

    bilingual_text("Training:", '训练：')
    bilingual_text("- Philosophy: easier to harder", '- 理念：easier to harder')
    image("images/llava-onevision-training.png", width=700)

    bilingual_text("Transfer between modalities:", '模态之间的迁移：')
    bilingual_text("- Single image data for diagrams and charts, but generalize to multi-image", '- Single 图像 数据 for diagrams and charts, but generalize to multi-图像')
    image("images/llava-onevision-transfer-s1.png", width=600)
    bilingual_text("- OCR on single image data, relational reasoning from multi-image data, generalize to GUI-based agents", '- OCR on single 图像 数据, relational reasoning from multi-图像 数据, generalize to GUI-based 智能体s')
    image("images/llava-onevision-transfer-s2.png", width=600)
    bilingual_text("- Visual prompting (circle) in single images, generalize to videos", '- Visual prompting (circle) in single 图像, generalize to 视频s')
    image("images/llava-onevision-transfer-s8.png", width=600)

    bilingual_text("Summary:", '总结：')
    bilingual_text("- Standard VLM template: vision encoder + projector + LM", '- Standard VLM template: vision en代码r + 投影器 + LM')
    bilingual_text("- Most work goes into data curation (heavy on synthesized, task-specific data)", '- Most work goes into 数据 curation (heavy on synthesized, task-specific 数据)')
    bilingual_text("- Open-source (released model weights and data)", '- Open-来源 (released 模型 weights and 数据)')


def qwen_vl():
    bilingual_text("Qwen-VL ", '说明：Qwen-VL'), link("https://arxiv.org/abs/2308.12966")

    bilingual_text("Architecture:", '架构：')
    bilingual_text("- Vision encoder: OpenCLIP's ViT-bigC (14x14 patches) ", "- Vision en代码r: OpenCLIP's ViT-bigC (14x14 patches)"), link("https://arxiv.org/abs/2212.07143")
    bilingual_text("- Adaptor: one layer cross-attention, incorporate 2D positional encodings, maps to fixed length of 256", '- 适配器: one layer cross-attention, incorporate 2D positional encodings, maps to fixed length of 256')
    bilingual_text("- Special tokens: <img>, <box>, <ref>", '- 说明：Special tokens: <img>, <box>, <ref>')

    bilingual_text("Training:", '训练：')
    image("images/qwen-vl-stages.png", width=700)
    bilingual_text("- Stage 1: large-scale low quality data; freeze LM, train vision encoder + adaptor", '- Stage 1: large-scale low 质量 数据; freeze LM, train vision en代码r + 适配器')
    image("images/qwen-vl-stage1.png", width=400)
    bilingual_text("- Stage 2: higher quality task-specific data, increase resolution; train all parameters", '- Stage 2: higher 质量 task-specific 数据, increase 分辨率; train all parameters')
    image("images/qwen-vl-stage2.png", width=400)
    bilingual_text("- Stage 3: instruction tuning data; freeze visual encoder, train adaptor + LM", '- Stage 3: 指令调优 数据; freeze visual en代码r, train 适配器 + LM')

    image("images/qwen-vl-examples.png", width=600)


def qwen2_vl():
    bilingual_text("Qwen2-VL ", '说明：Qwen2-VL'), link("https://arxiv.org/abs/2409.12191")

    bilingual_text("Visual encoder: larger ViT (675M)", 'Visual en代码r: larger ViT (675M)')
    image("images/qwen2-vl-architecture.png", width=700)
    bilingual_text("- Key: dynamic resolution to handle varying resolutions", '- Key: dynamic 分辨率 to handle varying 分辨率s')
    bilingual_text("- Each 224 x 224 patch encoded with ViT/14, compress every 2x2 => 66 tokens", '- Each 224 x 224 patch en代码d with ViT/14, compress every 2x2 => 66 token')
    bilingual_text("- Video: sample 2 frames/sec, max 16384 tokens", '- 视频: sample 2 frames/sec, max 16384 token')

    bilingual_text("Multimodal Rotary Position Embedding (MRoPE):", '多模态 Rotary Position Embedding (MRoPE):')
    image("images/qwen2-vl-mrope.png", width=600)
    
    bilingual_text("Initialize LM with Qwen2 and vision encoder from DFN ", 'Initialize LM with Qwen2 and vision en代码r from DFN'), link("https://arxiv.org/abs/2309.17425")
    bilingual_text("Training (similar to Qwen-VL):", '训练 (similar to Qwen-VL):')
    bilingual_text("- Stage 1: train only visual encoder", '- Stage 1: train only visual en代码r')
    bilingual_text("- Stage 2: train all parameters", '- 说明：Stage 2: train all parameters')
    bilingual_text("- Stage 3: train language model on instruction following datasets", '- Stage 3: train language 模型 on instruction following 数据sets')

    bilingual_text("Many capabilities:", '说明：Many capabilities:')
    image("images/qwen2-vl-capabilities.png", width=700)


def qwen3_vl():
    bilingual_text("Qwen3-VL ", '说明：Qwen3-VL'), link("https://arxiv.org/abs/2511.21631")
    image("images/qwen3-vl.png", width=700)

    bilingual_text("Language model:", '语言模型：')
    bilingual_text("- Qwen-3 models (dense and MoE models up to 235B-A22B)", '- Qwen-3 模型s (dense and MoE 模型s up to 235B-A22B)')
    bilingual_text("- Long context understanding (256K)", '- Long con文本 理解 (256K)')

    bilingual_text("Vision encoder:", '视觉编码器：')
    bilingual_text("- SigLIP-2 (same architecture as SigLIP) ", '- 说明：SigLIP-2 (same architecture as SigLIP)'), link("https://arxiv.org/pdf/2502.14786")
    bilingual_text("- Interleaved MRoPE: distribute all axes (temporal, width, height) to low- and high-frequency bands", '- 说明：Interleaved MRoPE: distribute all axes (temporal, width, height) to low- and high-frequency bands')
    bilingual_text("... [t w h t w h t w h t w h] rather than [t t t t w w w w h h h h]", '说明：... [t w h t w h t w h t w h] rather than [t t t t w w w w h h h h]')
    bilingual_text("- Add explicit video timestamps (as separate tokens rather in positional embeddings)", '- Add explicit 视频 timestamps (as separate token rather in positional embeddings)')
    bilingual_text("- Square-root-normalized per-token loss: balance text and multimodal data (video examples are long, don't want to dominate)", "- Square-root-normalized per-token 损失: balance 文本 and 多模态 数据 (视频 examples are long, don't want to dominate)")

    bilingual_text("Adapter:", '适配器：')
    bilingual_text("- DeepStack: cross-layer fusion to inject visual information into multiple layers ", '- 说明：DeepStack: cross-layer fusion to inject visual information into multiple layers'), link("https://arxiv.org/abs/2406.04334")

    bilingual_text("Training:", '训练：')
    bilingual_text("- Pre-training has 4 stages (train adapter, train all parameters on 8K, 32K, 256K lengths)", '- Pre-训练 has 4 stages (train 适配器, train all parameters on 8K, 32K, 256K lengths)')
    image("images/qwen3-vl-pretraining.png", width=600)
    bilingual_text("- Post-training: SFT on long CoT data, knowledge distillation, RL", '- Post-训练: SFT on long CoT 数据, knowledge distillation, RL')

    image("images/qwen3-vl-results.png", width=600)

    bilingual_text("Summary:", '总结：')
    bilingual_text("- SOTA performance", '- 说明：SOTA performance')
    bilingual_text("- Lots of data work, but not many details", '- Lots of 数据 work, but not many details')
    bilingual_text("- Minor but potentially important architectural improvements", '- 说明：Minor but potentially important architectural improvements')
    bilingual_text("- Scale up", '- 说明：Scale up')


def chameleon():
    bilingual_text("Chameleon ", '说明：Chameleon'), link("https://arxiv.org/pdf/2405.09818")

    bilingual_text("So far: VLMs encode images (via CLIP or SigLIP), inject into LM", 'So far: VLMs en代码 图像 (via CLIP or SigLIP), inject into LM')
    bilingual_text("Disadvantage: can't generate images (need diffusion)", "Disadvantage: can't generate 图像 (need diffusion)")

    bilingual_text("Chameleon: map everything into discrete tokens", '说明：Chameleon: map everything into discrete tokens')
    bilingual_text("Advantage: can analyze and generate images in a uniform way", 'Advantage: can analyze and generate 图像 in a uniform way')
    image("images/chameleon.png", width=600)
    image("images/chameleon-example.png", width=600)

    bilingual_text("Vision encoder ", 'Vision en代码r'), link("https://arxiv.org/pdf/2203.13131")
    bilingual_text("- Key difference: encoder needs to map to discrete tokens (so we can generate them)", '- Key difference: en代码r needs to map to discrete token (so we can generate them)')
    bilingual_text("- VQ-VAE (Vector Quantized Variational Autoencoder) ", '- VQ-VAE (Vector Quantized Variational Autoen代码r)'), link("https://arxiv.org/pdf/1711.00937")
    bilingual_text("- Idea: map image to a discrete codebook, decode back to image and minimize reconstruction loss", '- 思想：map image to a discrete codebook, decode back to image and minimize reconstruction loss')
    image("images/vq-vae.png", width=600)
    bilingual_text("- Encodes 512 x 512 image into 1024 tokens (codebook of size 8192)", '- En代码s 512 x 512 图像 into 1024 token (代码book of size 8192)')
    bilingual_text("- Train a new BPE tokenizer", '- Train a new BPE 分词器')

    bilingual_text("Training:", '训练：')
    bilingual_text("- Stage 1 (80%): large-scale, unsupervised (2.9T text tokens, 1.5T text/image tokens, 400B text/image interleaved tokens)", '- Stage 1 (80%): large-scale, unsupervised (2.9T 文本 token, 1.5T 文本/图像 token, 400B 文本/图像 interleaved token)')
    bilingual_text("- Stage 2 (20%): 50% of stage 1 data, 50% of high quality data", '- Stage 2 (20%): 50% of stage 1 数据, 50% of high 质量 数据')
    
    bilingual_text("Training stability", '训练稳定性')
    bilingual_text("- Text tokens have low entropy, image tokens have high entropy, leads to norm growth, logit drift problem", '- 文本 token have low entropy, 图像 token have high entropy, leads to norm growth, logit drift problem')
    bilingual_text("- Fixes: QK norm, z-loss regularization", '- Fixes: QK norm, z-损失 regularization')

    bilingual_text("Summary:", '总结：')
    bilingual_text("- Elegant (just autoregressive modeling of discrete tokens)", '- Elegant (just 自回归 模型ing of discrete token)')
    bilingual_text("- Not as performant (discretization loses information - think OCR)", '- 说明：Not as performant (discretization loses information - think OCR)')
    bilingual_text("- Training with multiple modalities is tricky", '- 训练 with multiple 模态 is tricky')
    

if __name__ == "__main__":
    main()
