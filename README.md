# Spring 2026 CS336 lectures

This repository contains the lecture materials for Stanford's Language Modeling from Scratch (CS336).

## Executable lectures

These are named `lecture_XX.py`.

### Setup
``` shell
uv sync
source .venv/bin/activate
```
You can compile a lecture by running:

``` shell
python -m edtrace.execute -m lecture_01.py
```

which generates a `var/traces/lecture_01.json` and caches any images as
appropriate.

## To view it locally:

> [!NOTE]
> The next step assumes you already have Node installed on your the system

``` shell
# first clone the edtrace repo
git clone https://github.com/percyliang/edtrace

# install the dependencies for edtrace
npm install --prefix=edtrace/frontend
```
if you see warnings for vulnerabilities, please run

``` shell
npm audit fix
```

Load a local server to view at `http://localhost:5173?trace=var/traces/sample.json`:

``` shell
npm run --prefix=edtrace/frontend dev
```

Deploy to the main website:
``` shell
git clone https://github.com/percyliang/edtrace
npm run --prefix=edtrace/frontend build
git add assets
git ci -am "<some message>"
git push
```

## Non-executable lectures

These are named `lecture_XX.pdf`.
