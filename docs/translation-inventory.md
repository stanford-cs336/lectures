# Translation Inventory

Generated with Python AST scanning. Counts are a starting point for translation review, not a substitute for human judgment.

| File | text(...) | image(...) | link(...) | verbatim | docstrings | comments | ordinary strings | course text strings | runtime/display strings | path/url/id strings | review blocks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lecture_01.py | 368 | 14 | 121 | 0 | 9 | 7 | 469 | 368 | 387 | 61 | 40 |
| lecture_02.py | 210 | 11 | 14 | 0 | 5 | 54 | 275 | 210 | 226 | 44 | 21 |
| lecture_06.py | 195 | 7 | 2 | 34 | 2 | 74 | 214 | 189 | 171 | 26 | 0 |
| lecture_07.py | 104 | 7 | 6 | 0 | 7 | 74 | 187 | 104 | 139 | 36 | 47 |
| lecture_10.py | 279 | 28 | 33 | 0 | 3 | 21 | 348 | 279 | 277 | 61 | 8 |
| lecture_12.py | 220 | 43 | 46 | 0 | 0 | 0 | 329 | 220 | 210 | 101 | 8 |
| lecture_13.py | 396 | 18 | 28 | 0 | 0 | 3 | 463 | 396 | 373 | 80 | 0 |
| lecture_14.py | 249 | 18 | 27 | 7 | 0 | 6 | 341 | 249 | 235 | 81 | 11 |
| lecture_17.py | 178 | 32 | 19 | 0 | 0 | 3 | 233 | 178 | 151 | 56 | 0 |

## Classification Rules

- Course text strings: first positional string passed to `text(...)`.
- Runtime/display strings: non-path strings that may be visible in examples, prints, assertions, or generated output.
- Path/url/id strings: URL, image path, file path, identifier-like, or command-like values that should usually remain unchanged.
- Review blocks: strings that need human review before translation because their role is not obvious from AST context.

## Translation Notes

- `verbatim=True` blocks should keep the English ASCII table/output unchanged and receive a Chinese explanation underneath.
- Strings that participate in assertions, parsing, control flow, or tokenizer examples should remain unchanged unless a separate explanatory line is added.
- The generated trace JSON files are outputs and should be regenerated after source translation rather than edited directly.
