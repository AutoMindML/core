# AutoMind

An AutoML Framework using PyTorch and Optuna

## Setup

install [astral-sh/uv](https://github.com/astral-sh/uv) package manager via pip

```bash
pip install -U uv
```

install specific python version via uv

```bash
uv python install 3.10
```

create venv via uv and activate environment

```bash
uv venv --python 3.10

# windows
.\.venv\Scripts\activate

# linux
source .venv/bin/activate
```

install dependencies via uv

```bash
uv sync
```

## Run

run tests

```bash
# example
uv run -m pytest .\tests\data_utils
```

run demo

```bash
# example
uv run -m demo.demo_datacleaner
```
