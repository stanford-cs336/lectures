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

## Review Checklist

- English source text is preserved.
- Chinese text immediately follows the corresponding English.
- Formulas, code, commands, URLs, and formal names remain intact.
- Chinese explanations do not change assertions, control flow, or parser-sensitive strings.
- Tables and code/output blocks keep their original alignment.
