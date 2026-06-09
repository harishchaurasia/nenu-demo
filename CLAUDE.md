# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`nenu-demo` is a demo Python CLI tool.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"   # or: pip install -r requirements.txt
```

## Common Commands

```bash
# Run the CLI
python -m nenu_demo <args>
# or after install:
nenu-demo <args>

# Run tests
pytest

# Run a single test
pytest tests/test_foo.py::test_bar

# Lint
ruff check .
ruff format --check .
```
