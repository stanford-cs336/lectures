# Spring 2026 CS336 lectures

This repository contains the lecture materials for Stanford's Language Modeling from Scratch (CS336).

There are two kinds of lectures:

- **Executable lectures**, named `lecture_XX.py`, are compiled into a trace that is viewed in a browser.
- **Non-executable lectures**, named `lecture_XX.pdf`, are plain PDFs.

## Executable lectures

### Prerequisites

- [uv](https://docs.astral.sh/uv/)
- [Node.js and npm](https://nodejs.org/) (for viewing and deploying)
- [Modal](https://modal.com/) account (optional, only for lectures that need a GPU)

### Setup

```
uv sync
git clone https://github.com/percyliang/edtrace
```

`uv sync` installs the `edtrace` backend (which provides the `edtrace.execute` module).
The cloned `edtrace` repo is needed for the frontend viewer (and is also used by `modal_execute.py`).

### Compile a lecture locally

```
uv run python -m edtrace.execute -m lecture_01
```

This generates `var/traces/lecture_01.json` and caches any images as appropriate.

Note: this runs on your machine, so lectures that need an NVIDIA GPU or CUDA tooling
won't work on a laptop without one. Use Modal for those (see below).

### Compile a lecture on Modal

`modal_execute.py` runs the same `edtrace.execute` command inside a CUDA container on [Modal](https://modal.com/):

```
modal run modal_execute.py --module lecture_06
```

You need to be logged in to Modal (`modal setup`).
The outputs are copied back into `var/`:

- `var/traces/<module>.json`
- `var/*-ptx.txt`
- `var/profiles.txt`

### View locally

Start a local server:

```
npm run --prefix=edtrace/frontend dev
```

Then open `http://localhost:5173?trace=var/traces/lecture_01.json`
(replace `lecture_01` with the lecture you compiled).

### Deploy to the main website

```
npm run --prefix=edtrace/frontend build
git add assets
git commit -am "<some message>"
git push
```
