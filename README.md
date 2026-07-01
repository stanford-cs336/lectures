# Spring 2026 CS336 lectures

This repository contains the lecture materials for Stanford's Language Modeling from Scratch (CS336).

## Bilingual Learning Version / 中英对照学习版

This branch adds a bilingual learning layer on top of the original Stanford CS336 lecture materials.

本分支在 Stanford CS336 原始课程资料之上增加中英对照学习层。

- The original English text is preserved.
- 英文原文完整保留。
- Chinese translations are provided for study and comprehension support.
- 中文译文用于学习和辅助理解。
- This is not an official Stanford translation.
- 本项目不是 Stanford 官方翻译。
- PDF lectures and text embedded inside images are not covered in the first executable-lecture translation pass.
- 第一轮可执行讲义翻译暂不覆盖 PDF 讲义和图片内部文字。

Translation tracking files:

- `docs/translation-inventory.md`: AST-based inventory of executable lecture text.
- `docs/bilingual-glossary.md`: shared terminology.
- `docs/translation-review.md`: items needing human review.

翻译跟踪文件：

- `docs/translation-inventory.md`：基于 AST 的可执行讲义文本盘点。
- `docs/bilingual-glossary.md`：统一术语表。
- `docs/translation-review.md`：需要人工复核的内容。

## Executable lectures

These are named `lecture_XX.py`.

### Setup

        uv sync
        git clone https://github.com/percyliang/edtrace

You can compile a lecture by running:

        python execute.py -m lecture_01

which generates a `var/traces/lecture_01.json` and caches any images as
appropriate.

To view it locally:

Load a local server to view at `http://localhost:5173?trace=var/traces/sample.json`:

        npm run --prefix=edtrace/frontend dev

Deploy to the main website:

        npm run --prefix=edtrace/frontend build
        git add assets
        git ci -am "<some message>"
        git push

## Non-executable lectures

These are named `lecture_XX.pdf`.
