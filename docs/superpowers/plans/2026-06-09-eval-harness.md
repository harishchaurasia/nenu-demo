# Eval Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI evaluation harness that runs personal-context AI agent prompts against test cases and scores results with an LLM-as-judge.

**Architecture:** A `click` CLI loads a user profile and test cases from JSON, feeds each case through an OpenAI agent call using a versioned prompt template, scores each response with a second LLM judge call, and writes a JSON report. Prompt templates live in `prompts/` as plain text files, versioned by name (e.g. `agent_v1.txt`, `agent_v2.txt`) to support iterative prompt development.

**Tech Stack:** Python 3.11, OpenAI Python SDK (`openai>=1.0`), `click>=8.1`, `pydantic>=2.0`, `pytest>=8.0`, `pytest-mock>=3.14`

---

## Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `src/nenu/__init__.py`
- Create: `tests/__init__.py`
- Create: `.env.example`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "nenu"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "click>=8.1",
    "openai>=1.0",
    "pydantic>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-mock>=3.14",
]

[project.scripts]
nenu = "nenu.cli:run"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create package stub files**

```bash
mkdir -p src/nenu tests examples prompts
touch src/nenu/__init__.py tests/__init__.py
```

- [ ] **Step 3: Write `.env.example`**

```
OPENAI_API_KEY=sk-...
```

- [ ] **Step 4: Install in editable mode**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Expected: no errors, `nenu` command available.

- [ ] **Step 5: Commit**

```bash
git init
git add pyproject.toml src/nenu/__init__.py tests/__init__.py .env.example
git commit -m "chore: scaffold Python project with click + openai + pydantic"
```

---

## Task 2: Data Models + Test Fixtures

**Files:**
- Create: `src/nenu/models.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Write `src/nenu/models.py`**

```python
from pydantic import BaseModel


class UserProfile(BaseModel):
    name: str
    context: dict[str, str]


class TestCase(BaseModel):
    id: str
    question: str
    context_keys: list[str]
    rubric: str


class CaseResult(BaseModel):
    id: str
    question: str
    response: str
    score: str  # "pass" | "fail"
    reasoning: str


class Report(BaseModel):
    total: int
    passed: int
    failed: int
    pass_rate: float
    results: list[CaseResult]
```

- [ ] **Step 2: Write `tests/conftest.py`**

```python
import pytest
from nenu.models import UserProfile, TestCase


@pytest.fixture
def profile():
    return UserProfile(
        name="Alice",
        context={"role": "Senior engineer", "timezone": "PST"},
    )


@pytest.fixture
def test_case():
    return TestCase(
        id="tc-001",
        question="What time zone should I schedule our standup?",
        context_keys=["timezone"],
        rubric="Answer should reference PST and suggest a morning time.",
    )
```

- [ ] **Step 3: Verify imports work**

```bash
python -c "from nenu.models import UserProfile, TestCase, CaseResult, Report; print('ok')"
```

Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add src/nenu/models.py tests/conftest.py
git commit -m "feat: add Pydantic data models and shared test fixtures"
```

---

## Task 3: Loader

**Files:**
- Create: `src/nenu/loader.py`
- Create: `tests/test_loader.py`

- [ ] **Step 1: Write failing tests in `tests/test_loader.py`**

```python
import json
import pytest
from nenu.loader import load_profile, load_test_cases
from nenu.models import UserProfile, TestCase


def test_load_profile(tmp_path):
    data = {"name": "Alice", "context": {"role": "Engineer", "timezone": "PST"}}
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(data))

    profile = load_profile(str(path))

    assert isinstance(profile, UserProfile)
    assert profile.name == "Alice"
    assert profile.context["timezone"] == "PST"


def test_load_test_cases(tmp_path):
    data = [
        {
            "id": "tc-001",
            "question": "What time zone?",
            "context_keys": ["timezone"],
            "rubric": "Should mention PST",
        }
    ]
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(data))

    cases = load_test_cases(str(path))

    assert len(cases) == 1
    assert isinstance(cases[0], TestCase)
    assert cases[0].id == "tc-001"


def test_load_test_cases_multiple(tmp_path):
    data = [
        {"id": "tc-001", "question": "Q1", "context_keys": [], "rubric": "R1"},
        {"id": "tc-002", "question": "Q2", "context_keys": ["role"], "rubric": "R2"},
    ]
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(data))

    cases = load_test_cases(str(path))

    assert len(cases) == 2
    assert cases[1].id == "tc-002"


def test_load_profile_invalid_json(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text("not json")

    with pytest.raises(Exception):
        load_profile(str(path))
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_loader.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` — `loader` doesn't exist yet.

- [ ] **Step 3: Write `src/nenu/loader.py`**

```python
import json
from pathlib import Path
from nenu.models import UserProfile, TestCase


def load_profile(path: str) -> UserProfile:
    data = json.loads(Path(path).read_text())
    return UserProfile.model_validate(data)


def load_test_cases(path: str) -> list[TestCase]:
    data = json.loads(Path(path).read_text())
    return [TestCase.model_validate(item) for item in data]
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_loader.py -v
```

Expected: 4 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/nenu/loader.py tests/test_loader.py
git commit -m "feat: add JSON loader for user profile and test cases"
```

---

## Task 4: Prompt Templates

**Files:**
- Create: `prompts/agent_v1.txt`
- Create: `prompts/judge_v1.txt`

No tests needed — these are static text files consumed by runner/judge.

- [ ] **Step 1: Write `prompts/agent_v1.txt`**

```
You are a helpful assistant with knowledge about the user.

User context:
{context}

Answer the following question concisely:
{question}
```

- [ ] **Step 2: Write `prompts/judge_v1.txt`**

```
You are evaluating the quality of an AI assistant's response.

Question: {question}

Response: {response}

Rubric: {rubric}

Evaluate whether the response passes or fails the rubric.
Respond with exactly this JSON format (no extra text):
{{"score": "pass", "reasoning": "explanation here"}}
or
{{"score": "fail", "reasoning": "explanation here"}}
```

- [ ] **Step 3: Commit**

```bash
git add prompts/agent_v1.txt prompts/judge_v1.txt
git commit -m "feat: add v1 prompt templates for agent and judge"
```

---

## Task 5: Runner

**Files:**
- Create: `src/nenu/runner.py`
- Create: `tests/test_runner.py`

- [ ] **Step 1: Write failing tests in `tests/test_runner.py`**

```python
from unittest.mock import MagicMock
from nenu.runner import run_case


def test_run_case_returns_response(profile, test_case):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = (
        "You should schedule for 9am PST."
    )
    template = "Context:\n{context}\n\nQuestion: {question}"

    result = run_case(mock_client, profile, test_case, template, "gpt-4o-mini")

    assert result == "You should schedule for 9am PST."


def test_run_case_injects_context_keys(profile, test_case):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = "ok"
    template = "Context:\n{context}\n\nQuestion: {question}"

    run_case(mock_client, profile, test_case, template, "gpt-4o-mini")

    call_args = mock_client.chat.completions.create.call_args
    prompt = call_args.kwargs["messages"][0]["content"]
    assert "PST" in prompt
    assert "What time zone should I schedule our standup?" in prompt


def test_run_case_skips_missing_context_keys(profile, test_case):
    test_case.context_keys = ["nonexistent_key"]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = "ok"
    template = "Context:\n{context}\n\nQuestion: {question}"

    result = run_case(mock_client, profile, test_case, template, "gpt-4o-mini")

    assert result == "ok"  # doesn't crash on missing keys


def test_run_case_passes_model(profile, test_case):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = "ok"
    template = "Context:\n{context}\n\nQuestion: {question}"

    run_case(mock_client, profile, test_case, template, "gpt-4o")

    call_args = mock_client.chat.completions.create.call_args
    assert call_args.kwargs["model"] == "gpt-4o"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_runner.py -v
```

Expected: `ImportError` — `runner` doesn't exist yet.

- [ ] **Step 3: Write `src/nenu/runner.py`**

```python
from openai import OpenAI
from nenu.models import UserProfile, TestCase


def run_case(
    client: OpenAI,
    profile: UserProfile,
    case: TestCase,
    template: str,
    model: str,
) -> str:
    context = "\n".join(
        f"{k}: {profile.context[k]}"
        for k in case.context_keys
        if k in profile.context
    )
    prompt = template.format(context=context, question=case.question)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_runner.py -v
```

Expected: 4 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/nenu/runner.py tests/test_runner.py
git commit -m "feat: add runner module for agent LLM calls"
```

---

## Task 6: Judge

**Files:**
- Create: `src/nenu/judge.py`
- Create: `tests/test_judge.py`

- [ ] **Step 1: Write failing tests in `tests/test_judge.py`**

```python
import json
from unittest.mock import MagicMock
from nenu.judge import judge_response


def test_judge_pass(test_case):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = (
        json.dumps({"score": "pass", "reasoning": "Correctly mentions PST."})
    )
    template = "Q: {question}\nA: {response}\nRubric: {rubric}"

    score, reasoning = judge_response(
        mock_client, test_case, "Use PST.", template, "gpt-4o-mini"
    )

    assert score == "pass"
    assert reasoning == "Correctly mentions PST."


def test_judge_fail(test_case):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = (
        json.dumps({"score": "fail", "reasoning": "Did not mention timezone."})
    )
    template = "Q: {question}\nA: {response}\nRubric: {rubric}"

    score, reasoning = judge_response(
        mock_client, test_case, "I don't know.", template, "gpt-4o-mini"
    )

    assert score == "fail"
    assert "timezone" in reasoning


def test_judge_unparseable_json(test_case):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = (
        "Sorry, I can't evaluate that."
    )
    template = "Q: {question}\nA: {response}\nRubric: {rubric}"

    score, reasoning = judge_response(
        mock_client, test_case, "response", template, "gpt-4o-mini"
    )

    assert score == "fail"
    assert "unparseable" in reasoning


def test_judge_missing_keys(test_case):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = (
        json.dumps({"result": "yes"})  # wrong keys
    )
    template = "Q: {question}\nA: {response}\nRubric: {rubric}"

    score, reasoning = judge_response(
        mock_client, test_case, "response", template, "gpt-4o-mini"
    )

    assert score == "fail"
    assert "unparseable" in reasoning
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_judge.py -v
```

Expected: `ImportError` — `judge` doesn't exist yet.

- [ ] **Step 3: Write `src/nenu/judge.py`**

```python
import json
from openai import OpenAI
from nenu.models import TestCase


def judge_response(
    client: OpenAI,
    case: TestCase,
    response: str,
    template: str,
    model: str,
) -> tuple[str, str]:
    prompt = template.format(
        question=case.question,
        response=response,
        rubric=case.rubric,
    )
    result = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    try:
        data = json.loads(result.choices[0].message.content)
        return data["score"], data["reasoning"]
    except (json.JSONDecodeError, KeyError):
        return "fail", "judge response unparseable"
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_judge.py -v
```

Expected: 4 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/nenu/judge.py tests/test_judge.py
git commit -m "feat: add judge module for LLM-as-judge scoring"
```

---

## Task 7: Reporter

**Files:**
- Create: `src/nenu/reporter.py`
- Create: `tests/test_reporter.py`

- [ ] **Step 1: Write failing tests in `tests/test_reporter.py`**

```python
import json
import pytest
from nenu.reporter import build_report, write_report
from nenu.models import CaseResult


def _result(id: str, score: str) -> CaseResult:
    return CaseResult(id=id, question="q", response="r", score=score, reasoning="ok")


def test_build_report_counts():
    results = [
        _result("tc-001", "pass"),
        _result("tc-002", "fail"),
        _result("tc-003", "pass"),
    ]
    report = build_report(results)

    assert report.total == 3
    assert report.passed == 2
    assert report.failed == 1
    assert abs(report.pass_rate - 2 / 3) < 0.001


def test_build_report_all_pass():
    results = [_result("tc-001", "pass"), _result("tc-002", "pass")]
    report = build_report(results)

    assert report.passed == 2
    assert report.failed == 0
    assert report.pass_rate == 1.0


def test_build_report_empty():
    report = build_report([])

    assert report.total == 0
    assert report.pass_rate == 0.0


def test_write_report(tmp_path):
    results = [_result("tc-001", "pass")]
    report = build_report(results)
    output_path = str(tmp_path / "report.json")

    write_report(report, output_path)

    data = json.loads((tmp_path / "report.json").read_text())
    assert data["total"] == 1
    assert data["passed"] == 1
    assert data["results"][0]["id"] == "tc-001"
    assert data["results"][0]["score"] == "pass"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_reporter.py -v
```

Expected: `ImportError` — `reporter` doesn't exist yet.

- [ ] **Step 3: Write `src/nenu/reporter.py`**

```python
import json
from pathlib import Path
from nenu.models import CaseResult, Report


def build_report(results: list[CaseResult]) -> Report:
    total = len(results)
    passed = sum(1 for r in results if r.score == "pass")
    return Report(
        total=total,
        passed=passed,
        failed=total - passed,
        pass_rate=passed / total if total > 0 else 0.0,
        results=results,
    )


def write_report(report: Report, output_path: str) -> None:
    Path(output_path).write_text(json.dumps(report.model_dump(), indent=2))
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_reporter.py -v
```

Expected: 4 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/nenu/reporter.py tests/test_reporter.py
git commit -m "feat: add reporter module to aggregate results and write JSON report"
```

---

## Task 8: CLI + Examples

**Files:**
- Create: `src/nenu/cli.py`
- Create: `examples/profile.json`
- Create: `examples/test_cases.json`

- [ ] **Step 1: Write `examples/profile.json`**

```json
{
  "name": "Alice",
  "context": {
    "role": "Senior software engineer",
    "stack": "Python, TypeScript, AWS",
    "team": "Platform infrastructure",
    "timezone": "PST"
  }
}
```

- [ ] **Step 2: Write `examples/test_cases.json`**

```json
[
  {
    "id": "tc-001",
    "question": "What time zone should I schedule our standup?",
    "context_keys": ["timezone", "team"],
    "rubric": "Answer should reference PST and suggest a morning time appropriate for a distributed team."
  },
  {
    "id": "tc-002",
    "question": "Recommend a caching library for my stack.",
    "context_keys": ["stack", "role"],
    "rubric": "Answer should recommend Redis or a Python/TypeScript-compatible caching solution and explain why."
  }
]
```

- [ ] **Step 3: Write `src/nenu/cli.py`**

```python
from pathlib import Path

import click
from openai import OpenAI

from nenu.loader import load_profile, load_test_cases
from nenu.judge import judge_response
from nenu.models import CaseResult
from nenu.reporter import build_report, write_report
from nenu.runner import run_case


PROMPTS_DIR = Path("prompts")


@click.command()
@click.option("--profile", required=True, type=click.Path(exists=True), help="Path to user profile JSON")
@click.option("--cases", required=True, type=click.Path(exists=True), help="Path to test cases JSON")
@click.option("--prompt-version", default="v1", show_default=True, help="Prompt version (e.g. v1, v2)")
@click.option("--output", default="report.json", show_default=True, help="Output report path")
@click.option("--model", default="gpt-4o-mini", show_default=True, help="OpenAI model to use")
def run(profile, cases, prompt_version, output, model):
    """Run the personal-context AI agent evaluation harness."""
    client = OpenAI()
    user_profile = load_profile(profile)
    test_cases = load_test_cases(cases)

    agent_template = (PROMPTS_DIR / f"agent_{prompt_version}.txt").read_text()
    judge_template = (PROMPTS_DIR / f"judge_{prompt_version}.txt").read_text()

    results = []
    for case in test_cases:
        click.echo(f"  {case.id}  ", nl=False)
        response = run_case(client, user_profile, case, agent_template, model)
        score, reasoning = judge_response(client, case, response, judge_template, model)
        results.append(
            CaseResult(
                id=case.id,
                question=case.question,
                response=response,
                score=score,
                reasoning=reasoning,
            )
        )
        status = click.style("PASS", fg="green") if score == "pass" else click.style("FAIL", fg="red")
        click.echo(status)

    report = build_report(results)
    write_report(report, output)

    click.echo(f"\n{report.passed}/{report.total} passed ({report.pass_rate:.0%})")
    click.echo(f"Report: {output}")
```

- [ ] **Step 4: Write CLI smoke test**

Add `tests/test_cli.py`:

```python
import json
from unittest.mock import MagicMock, patch
from click.testing import CliRunner
from nenu.cli import run


def test_cli_runs_and_writes_report(tmp_path, monkeypatch):
    profile_data = {"name": "Alice", "context": {"timezone": "PST"}}
    cases_data = [{"id": "tc-001", "question": "What tz?", "context_keys": ["timezone"], "rubric": "Mention PST"}]

    (tmp_path / "profile.json").write_text(json.dumps(profile_data))
    (tmp_path / "cases.json").write_text(json.dumps(cases_data))
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "agent_v1.txt").write_text("Context:\n{context}\n\nQ: {question}")
    (prompts_dir / "judge_v1.txt").write_text("Q: {question}\nA: {response}\nRubric: {rubric}")

    monkeypatch.chdir(tmp_path)  # makes Path("prompts") resolve to tmp_path/prompts

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = (
        json.dumps({"score": "pass", "reasoning": "Mentions PST."})
    )

    output_path = str(tmp_path / "report.json")
    with patch("nenu.cli.OpenAI", return_value=mock_client):
        result = CliRunner().invoke(run, [
            "--profile", str(tmp_path / "profile.json"),
            "--cases", str(tmp_path / "cases.json"),
            "--output", output_path,
        ])

    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["total"] == 1
    assert report["passed"] == 1
```

- [ ] **Step 5: Verify the CLI help text works**

```bash
nenu --help
```

Expected: shows profile, cases, prompt-version, output, model options (no prompts-dir).

- [ ] **Step 6: Run all tests**

```bash
pytest -v
```

Expected: all tests PASSED.

- [ ] **Step 7: Commit**

```bash
git add src/nenu/cli.py tests/test_cli.py examples/profile.json examples/test_cases.json
git commit -m "feat: add CLI entry point, smoke test, and example fixtures"
```

---

## Task 9: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Replace CLAUDE.md with final content**

```markdown
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`nenu` is a CLI evaluation harness for personal-context AI agents. It runs test cases against a user profile using an OpenAI agent prompt, scores each response with an LLM-as-judge, and outputs a JSON report.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
export OPENAI_API_KEY=sk-...
```

## Running the Harness

```bash
nenu --profile examples/profile.json --cases examples/test_cases.json
```

To test a new prompt version:

```bash
# Create prompts/agent_v2.txt (edit prompt, copy judge template)
cp prompts/judge_v1.txt prompts/judge_v2.txt
nenu --profile examples/profile.json --cases examples/test_cases.json --prompt-version v2 --output report_v2.json
```

## Tests

```bash
pytest                                         # all tests
pytest tests/test_runner.py -v                 # single file
pytest tests/test_runner.py::test_run_case_returns_response  # single test
```

Tests use `unittest.mock.MagicMock` to mock the OpenAI client — no real API calls in tests.

## Architecture

- `src/nenu/models.py` — Pydantic models: `UserProfile`, `TestCase`, `CaseResult`, `Report`
- `src/nenu/loader.py` — reads and validates JSON inputs
- `src/nenu/runner.py` — one OpenAI call per test case (agent role)
- `src/nenu/judge.py` — one OpenAI call per result (judge role), parses `{"score", "reasoning"}`
- `src/nenu/reporter.py` — aggregates `CaseResult` list into `Report`, writes JSON
- `src/nenu/cli.py` — click entry point, wires loader → runner → judge → reporter
- `prompts/` — versioned prompt templates; `{context}` and `{question}` placeholders for agent; `{question}`, `{response}`, `{rubric}` for judge
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with final commands and architecture"
```

---

## Self-Review

**Spec coverage check:**
- ✅ CLI tool in Python with click
- ✅ Loads user profile (JSON) and test cases (JSON)
- ✅ Feeds question + relevant context to LLM (runner.py)
- ✅ Scores with LLM-as-judge against rubric (judge.py)
- ✅ Outputs JSON report with pass/fail and reasoning per case (reporter.py + cli.py)
- ✅ Prompt versioning via `--prompt-version` flag for iteration workflow
- ✅ OpenAI Python SDK used throughout

**Placeholder scan:** None found — all steps contain complete code.

**Type consistency:**
- `UserProfile.context: dict[str, str]` — used in runner.py context injection ✅
- `TestCase.context_keys: list[str]` — used in runner.py loop ✅
- `judge_response` returns `tuple[str, str]` — destructured in cli.py as `score, reasoning` ✅
- `build_report` takes `list[CaseResult]` — cli.py builds `results: list[CaseResult]` ✅
- `write_report` takes `Report` — receives return value of `build_report` ✅
