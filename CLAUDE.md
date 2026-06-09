# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Project Overview

`nenu-demo` is an evaluation harness for personal-context AI agents. It tests
how well an LLM-based agent can reason over a user profile to answer queries
correctly, while avoiding context contamination, hallucination on missing data,
and inappropriate disclosure.

The tool runs an agent against a set of test cases, scores each response using
an LLM-as-judge against a configurable rubric, and produces a structured report.
Designed to support a tight iteration loop: run v1, identify failure patterns,
update prompts, re-run, measure improvement.

## Architecture

- `agent.py` — wraps OpenAI API calls. `get_response(client, profile, question, system_prompt) -> str`. Injects a subset of `profile.context` fields (specified per test case) into the system prompt before calling the API.
- `scorer.py` — LLM-as-judge. `score_response(client, question, response, expected_behavior, model) -> tuple[str, str]`. Returns `(verdict, reasoning)` where verdict is `"pass"` or `"fail"`. Parses structured JSON from the judge; defaults to `("fail", "judge response unparseable")` on bad output.
- `models.py` — Pydantic v2 models: `UserProfile`, `TestCase`, `Judgment`, `Report`.
- `run_eval.py` — CLI entry point (`argparse`). Loads profile and test cases, runs each through agent → scorer, aggregates into a `Report`, writes JSON to `reports/`.
- `prompts/` — versioned agent system prompts (`v1.txt`, `v2.txt`, …). Passed at runtime via `--prompt`. Supports `{context}` and `{question}` placeholders.
- `data/` — example user profiles and test cases (JSON).
- `reports/` — JSON output, one file per eval run.

## Data Models

```
UserProfile:  name: str, context: dict[str, str]
TestCase:     id: str, question: str, context_keys: list[str], expected_behavior: str
Judgment:     id: str, question: str, response: str, verdict: str, reasoning: str
Report:       total: int, passed: int, failed: int, pass_rate: float, judgments: list[Judgment]
```

## Data Flow

1. CLI loads user profile (JSON) and test cases (JSON)
2. For each test case: agent receives (profile, question, system_prompt) → returns response
3. Scorer receives (question, response, expected_behavior) → returns (verdict, reasoning)
4. Runner aggregates Judgments → builds Report → writes `reports/<timestamp>.json`

## Conventions

- Python 3.11
- Pydantic v2 for data models (test cases, profiles, judgments)
- stdlib `argparse` for CLI — no click
- OpenAI Python SDK for both agent and scorer
- Tests in `tests/` using pytest; mock OpenAI client with `unittest.mock.MagicMock` — no real API calls in tests
- `OPENAI_API_KEY` via environment variable

## File Structure

```
nenu-demo/
├── pyproject.toml
├── agent.py
├── scorer.py
├── models.py
├── run_eval.py
├── tests/
│   ├── conftest.py        # shared fixtures: profile, test_case
│   ├── test_agent.py
│   ├── test_scorer.py
│   └── test_run_eval.py
├── prompts/
│   ├── v1.txt             # {context} and {question} placeholders
│   └── v2.txt
├── data/
│   ├── user.json
│   └── cases.json
└── reports/               # timestamped JSON output per run
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
export OPENAI_API_KEY="sk-..."
```

## Common Commands

```bash
# Run an eval
python -m nenu_demo --profile data/user.json --cases data/cases.json --prompt prompts/v1.txt

# Iterate on prompts
cp prompts/v1.txt prompts/v2.txt   # edit v2.txt, then:
python -m nenu_demo --profile data/user.json --cases data/cases.json --prompt prompts/v2.txt

# Run tests
pytest
pytest tests/test_agent.py -v                         # single file
pytest tests/test_agent.py::test_get_response_pass    # single test

# Lint
ruff check .
```

## Behavioral Guidelines

This project follows the `coding-discipline` skill (see `.claude/skills/coding-discipline/SKILL.md`):
- State assumptions explicitly before coding
- Simplicity first — minimum code that solves the problem
- Surgical changes — touch only what's needed
- Goal-driven execution with verifiable success criteria
