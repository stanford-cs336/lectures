# Translation Review

This file tracks uncertain terminology, strings, or implementation choices that need human review.

本文件记录需要人工复核的术语、句子和实现选择。

## Open Items

| Location | Item | Reason | Proposed handling |
|---|---|---|---|
| All lectures | Official model, dataset, benchmark, and paper names | 翻译后可能破坏检索和引用准确性 | 保留英文正式名称，必要时添加中文解释。 |
| `verbatim=True` blocks | ASCII tables and example outputs | 直接插入中文会破坏对齐或改变示例输出 | 原块保持英文，在下方添加中文说明。 |
| Tokenization examples | Example strings and token outputs | 字符串本身用于演示 tokenizer 行为 | 示例保持原样，只翻译周围解释。 |
| Generated traces | `var/traces/*.json` | 生成产物，不应作为源头翻译 | 修改 Python 源文件后重新生成。 |
| All translated lectures | Batch-generated Chinese text | 本轮优先完成全部可执行讲义覆盖，部分句子仍需人工润色以提高自然度 | 保留英文为准，逐讲人工校对中文表达。 |
| `lecture_12.py` | `"\%"` SyntaxWarning | 原文件已有转义警告；本轮按要求未大范围修改，避免改变显示结果 | 如需修复，单独确认后改为 `"\\%"` 或 raw string。 |
| Images | Embedded English labels | 本轮不修改图片文件 | 在后续 pass 为关键图片补充更细图下注释或标签对照。 |
| `references.py` | `notes` Chinese render behavior | `Reference` 正式字段结构未知，不新增非原生字段 | 在 `notes` 中保留英文并紧跟中文说明。 |

## Review Checklist

- English source text is preserved.
- Chinese text immediately follows the corresponding English.
- Formulas, code, commands, URLs, and formal names remain intact.
- Chinese explanations do not change assertions, control flow, or parser-sensitive strings.
- Tables and code/output blocks keep their original alignment.
- Static executable lecture `text(...)` calls have been converted to bilingual helpers.
- `references.py` formal citation metadata remains unchanged.
