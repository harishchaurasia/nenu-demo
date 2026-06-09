# Eval Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI evaluation harness that runs personal-context AI agent prompts against test cases and scores results with an LLM-as-judge.

**Architecture:** Flat package `nenu_demo/` (no `src/`). `agent.py` calls the LLM with a versioned system prompt, `scorer.py` judges each response, `run_eval.py` wires them together with `argparse`. No click.

**Tech Stack:** Python 3.11, OpenAI Python SDK (`openai>=1.0`), `pydantic>=2.0`, `pytest>=8.0`

---

## Task 1: Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `nenu_demo/__init__.py`
- Create: `nenu_demo/__main__.py`
- Create: `tests/__init__.py`
- Create: `prompts/v1.txt`
- Create: `data/user.json`
- Create: `data/cases.json`
- Create: `reports/.gitkeep`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "nenu-demo"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "openai>=1.0",
    "pydantic>=2.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.setuptools.packages.find]
where = ["."]
include = ["nenu_demo*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create package and test stubs**

```bash
mkdir -p nenu_demo tests data prompts reports
touch nenu_demo/__init__.py tests/__init__.py reports/.gitkeep
```

- [ ] **Step 3: Write `nenu_demo/__main__.py`**

```python
from nenu_demo.run_eval import main

main()
```

- [ ] **Step 4: Write `prompts/v1.txt`**

```
You are a helpful assistant with knowledge about the user.

User context:
{context}

Answer the following question concisely:
{question}
```

- [ ] **Step 5: Write `data/user.json`**

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

- [ ] **Step 6: Write `data/cases.json`**

```json
[
  {
    "id": "tc-001",
    "question": "What time zone should I schedule our standup?",
    "context_keys": ["timezone", "team"],
    "expected_behavior": "Answer should reference PST and suggest a morning time appropriate for a distributed team."
  },
  {
    "id": "tc-002",
    "question": "Recommend a caching library for my stack.",
    "context_keys": ["stack", "role"],
    "expected_behavior": "Answer should recommend Redis or a Python/TypeScript-compatible caching solution and explain why."
  }
]
```

- [ ] **Step 7: Install and verify**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -c "import nenu_demo; print('ok')"
```

Expected: `ok`

- [ ] **Step 8: Commit**

```bash
git init
git add pyproject.toml nenu_demo/ tests/ prompts/ data/ reports/
git commit -m "chore: scaffold nenu_demo package with data fixtures and prompt template"
```

---

## Task 2: Data Models + Test Fixtures

**Files:**
- Create: `nenu_demo/models.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Write `nenu_demo/models.py`**

```python
from pydantic import BaseModel


class UserProfile(BaseModel):
    name: str
    context: dict[str, str]


class TestCase(BaseModel):
    id: str
    question: str
    context_keys: list[str]
    expected_behavior: str


class Judgment(BaseModel):
    id: str
    question: str
    response: str
    verdict: str  # "pass" | "fail"
    reasoning: str


class Report(BaseModel):
    total: int
    passed: int
    failed: int
    pass_rate: float
    judgments: list[Judgment]
```

- [ ] **Step 2: Write `tests/conftest.py`**

```python
import pytest
from nenu_demo.models import UserProfile, TestCase


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
        expected_behavior="Answer should reference PST and suggest a morning time.",
    )
```

- [ ] **Step 3: Verify imports**

```bash
python -c "from nenu_demo.models import UserProfile, TestCase, Judgment, Report; print('ok')"
```

Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add nenu_demo/models.py tests/conftest.py
git commit -m "feat: add Pydantic models and shared test fixtures"
```

---

## Task 3: Agent

**Files:**
- Create: `nenu_demo/agent.py`
- Create: `tests/test_agent.py`

- [ ] **Step 1: Write failing tests in `tests/test_agent.py`**

```python
from unittest.mock import MagicMock
from nenu_demo.agent import get_response


def test_get_response_returns_content(profile, test_case):
    client = MagicMock()
    client.chat.completions.create.return_value.choices[0].message.content = "9am PST works."

    result = get_response(client, profile, test_case, "C:{context} Q:{question}", "gpt-4o-mini")

    assert result == "9am PST works."


def test_get_response_injects_context_keys(profile, test_case):
    client = MagicMock()
    client.chat.completions.create.return_value.choices[0].message.content = "ok"

    get_response(client, profile, test_case, "C:{context} Q:{question}", "gpt-4o-mini")

    prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "PST" in prompt
    assert test_case.question in prompt


def test_get_response_skips_missing_keys(profile, test_case):
    test_case.context_keys = ["nonexistent"]
    client = MagicMock()
    client.chat.completions.create.return_value.choices[0].message.content = "ok"

    result = get_response(client, profile, test_case, "C:{context} Q:{question}", "gpt-4o-mini")

    assert result == "ok"


def test_get_response_passes_model(profile, test_case):
    client = MagicMock()
    client.chat.completions.create.return_value.choices[0].message.content = "ok"

    get_response(client, profile, test_case, "C:{context} Q:{question}", "gpt-4o")

    assert client.chat.completions.create.call_args.kwargs["model"] == "gpt-4o"
```

- [ ] **Step 2: Run tests — confirm they fail**

```bash
pytest tests/test_agent.py -v
```

Expected: `ImportError` — `agent` doesn't exist yet.

- [ ] **Step 3: Write `nenu_demo/agent.py`**

```python
from openai import OpenAI
from nenu_demo.models import UserProfile, TestCase


def get_response(
    client: OpenAI,
    profile: UserProfile,
    case: TestCase,
    system_prompt: str,
    model: str,
) -> str:
    context = "\n".join(
        f"{k}: {profile.context[k]}"
        for k in case.context_keys
        if k in profile.context
    )
    prompt = system_prompt.format(context=context, question=case.question)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
```

- [ ] **Step 4: Run tests — confirm they pass**

```bash
pytest tests/test_agent.py -v
```

Expected: 4 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add nenu_demo/agent.py tests/test_agent.py
git commit -m "feat: add agent module with get_response"
```

---

## Task 4: Scorer

**Files:**
- Create: `nenu_demo/scorer.py`
- Create: `tests/test_scorer.py`

- [ ] **Step 1: Write failing tests in `tests/test_scorer.py`**

```python
import json
from unittest.mock import MagicMock
from nenu_demo.scorer import score_response


def test_score_pass(test_case):
    client = MagicMock()
    client.chat.completions.create.return_value.choices[0].message.content = (
        json.dumps({"verdict": "pass", "reasoning": "Correctly mentions PST."})
    )

    verdict, reasoning = score_response(client, test_case, "Use PST.", "gpt-4o-mini")

    assert verdict == "pass"
    assert reasoning == "Correctly mentions PST."


def test_score_fail(test_case):
    client = MagicMock()
    client.chat.completions.create.return_value.choices[0].message.content = (
        json.dumps({"verdict": "fail", "reasoning": "Missing timezone."})
    )

    verdict, reasoning = score_response(client, test_case, "I don't know.", "gpt-4o-mini")

    assert verdict == "fail"
    assert "timezone" in reasoning


def test_score_unparseable_json(test_case):
    client = MagicMock()
    client.chat.completions.create.return_value.choices[0].message.content = "not json"

    verdict, reasoning = score_response(client, test_case, "response", "gpt-4o-mini")

    assert verdict == "fail"
    assert "unparseable" in reasoning


def test_score_missing_keys(test_case):
    client = MagicMock()
    client.chat.completions.create.return_value.choices[0].message.content = (
        json.dumps({"result": "yes"})  # wrong keys
    )

    verdict, reasoning = score_response(client, test_case, "response", "gpt-4o-mini")

    assert verdict == "fail"
    assert "unparseable" in reasoning
```

- [ ] **Step 2: Run tests — confirm they fail**

```bash
pytest tests/test_scorer.py -v
```

Expected: `ImportError` — `scorer` doesn't exist yet.

- [ ] **Step 3: Write `nenu_demo/scorer.py`**

```python
import json
from openai import OpenAI
from nenu_demo.models import TestCase

_JUDGE_TEMPLATE = """\
You are evaluating an AI assistant's response.

Question: {question}
Response: {response}
Expected behavior: {expected_behavior}

Does the response satisfy the expected behavior?
Reply with exactly this JSON (no other text):
{{"verdict": "pass", "reasoning": "..."}}
or
{{"verdict": "fail", "reasoning": "..."}}"""


def score_response(
    client: OpenAI,
    case: TestCase,
    response: str,
    model: str,
) -> tuple[str, str]:
    prompt = _JUDGE_TEMPLATE.format(
        question=case.question,
        response=response,
        expected_behavior=case.expected_behavior,
    )
    result = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    try:
        data = json.loads(result.choices[0].message.content)
        return data["verdict"], data["reasoning"]
    except (json.JSONDecodeError, KeyError):
        return "fail", "judge response unparseable"
```

- [ ] **Step 4: Run tests — confirm they pass**

```bash
pytest tests/test_scorer.py -v
```

Expected: 4 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add nenu_demo/scorer.py tests/test_scorer.py
git commit -m "feat: add scorer module with LLM-as-judge"
```

---

## Task 5: CLI (run_eval.py)

**Files:**
- Create: `nenu_demo/run_eval.py`
- Create: `tests/test_run_eval.py`

- [ ] **Step 1: Write failing tests in `tests/test_run_eval.py`**

```python
import json
import sys
from unittest.mock import MagicMock, patch
from nenu_demo.run_eval import load_profile, load_test_cases, build_report
from nenu_demo.models import Judgment


def _j(id: str, verdict: str) -> Judgment:
    return Judgment(id=id, question="q", response="r", verdict=verdict, reasoning="ok")


def test_load_profile(tmp_path):
    data = {"name": "Alice", "context": {"timezone": "PST"}}
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(data))

    profile = load_profile(str(path))

    assert profile.name == "Alice"
    assert profile.context["timezone"] == "PST"


def test_load_test_cases(tmp_path):
    data = [{"id": "tc-001", "question": "Q?", "context_keys": ["timezone"], "expected_behavior": "Mention PST"}]
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(data))

    cases = load_test_cases(str(path))

    assert len(cases) == 1
    assert cases[0].id == "tc-001"


def test_build_report_counts():
    report = build_report([_j("tc-001", "pass"), _j("tc-002", "fail"), _j("tc-003", "pass")])

    assert report.total == 3
    assert report.passed == 2
    assert report.failed == 1
    assert abs(report.pass_rate - 2 / 3) < 0.001


def test_build_report_empty():
    report = build_report([])

    assert report.total == 0
    assert report.pass_rate == 0.0


def test_main_writes_report(tmp_path, monkeypatch):
    profile_data = {"name": "Alice", "context": {"timezone": "PST"}}
    cases_data = [{"id": "tc-001", "question": "What tz?", "context_keys": ["timezone"], "expected_behavior": "Mention PST"}]
    prompt_text = "C:{context} Q:{question}"
    output_path = str(tmp_path / "report.json")

    (tmp_path / "profile.json").write_text(json.dumps(profile_data))
    (tmp_path / "cases.json").write_text(json.dumps(cases_data))
    (tmp_path / "prompt.txt").write_text(prompt_text)

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = (
        json.dumps({"verdict": "pass", "reasoning": "Good."})
    )

    monkeypatch.setattr(sys, "argv", [
        "run_eval",
        "--profile", str(tmp_path / "profile.json"),
        "--cases", str(tmp_path / "cases.json"),
        "--prompt", str(tmp_path / "prompt.txt"),
        "--output", output_path,
    ])

    with patch("nenu_demo.run_eval.OpenAI", return_value=mock_client):
        from nenu_demo.run_eval import main
        main()

    report = json.loads((tmp_path / "report.json").read_text())
    assert report["total"] == 1
    assert report["passed"] == 1
```

- [ ] **Step 2: Run tests — confirm they fail**

```bash
pytest tests/test_run_eval.py -v
```

Expected: `ImportError` — `run_eval` doesn't exist yet.

- [ ] **Step 3: Write `nenu_demo/run_eval.py`**

```python
import argparse
import json
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from nenu_demo.agent import get_response
from nenu_demo.models import Judgment, Report, TestCase, UserProfile
from nenu_demo.scorer import score_response


def load_profile(path: str) -> UserProfile:
    return UserProfile.model_validate(json.loads(Path(path).read_text()))


def load_test_cases(path: str) -> list[TestCase]:
    return [TestCase.model_validate(item) for item in json.loads(Path(path).read_text())]


def build_report(judgments: list[Judgment]) -> Report:
    total = len(judgments)
    passed = sum(1 for j in judgments if j.verdict == "pass")
    return Report(
        total=total,
        passed=passed,
        failed=total - passed,
        pass_rate=passed / total if total > 0 else 0.0,
        judgments=judgments,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run personal-context AI agent eval")
    parser.add_argument("--profile", required=True, help="Path to user profile JSON")
    parser.add_argument("--cases", required=True, help="Path to test cases JSON")
    parser.add_argument("--prompt", required=True, help="Path to agent system prompt")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model (default: gpt-4o-mini)")
    parser.add_argument("--output", default=None, help="Output path (default: reports/<timestamp>.json)")
    args = parser.parse_args()

    client = OpenAI()
    profile = load_profile(args.profile)
    cases = load_test_cases(args.cases)
    system_prompt = Path(args.prompt).read_text()
    output_path = args.output or f"reports/{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"

    judgments = []
    for case in cases:
        print(f"  {case.id}  ", end="", flush=True)
        response = get_response(client, profile, case, system_prompt, args.model)
        verdict, reasoning = score_response(client, case, response, args.model)
        judgments.append(Judgment(
            id=case.id,
            question=case.question,
            response=response,
            verdict=verdict,
            reasoning=reasoning,
        ))
        print(verdict.upper())

    report = build_report(judgments)
    Path(output_path).parent.mkdir(exist_ok=True)
    Path(output_path).write_text(json.dumps(report.model_dump(), indent=2))

    print(f"\n{report.passed}/{report.total} passed ({report.pass_rate:.0%})")
    print(f"Report: {output_path}")
```

- [ ] **Step 4: Run all tests — confirm they pass**

```bash
pytest -v
```

Expected: all tests PASSED.

- [ ] **Step 5: Smoke-test the help output**

```bash
python -m nenu_demo --help
```

Expected: shows `--profile`, `--cases`, `--prompt`, `--model`, `--output`.

- [ ] **Step 6: Commit**

```bash
git add nenu_demo/run_eval.py tests/test_run_eval.py
git commit -m "feat: add run_eval CLI with argparse, loader, and report writer"
```

---

## Task 6: Commit CLAUDE.md

- [ ] **Step 1: Commit the already-updated CLAUDE.md**

```bash
git add CLAUDE.md
git commit -m "docs: add CLAUDE.md with architecture, data flow, and conventions"
```

---

## Self-Review

**Spec coverage:**
- ✅ CLI tool in Python with argparse (no click)
- ✅ Loads user profile (JSON) and test cases (JSON)
- ✅ Feeds question + relevant context to LLM (`agent.py`)
- ✅ Scores with LLM-as-judge against `expected_behavior` (`scorer.py`)
- ✅ JSON report with pass/fail and reasoning per case (`reports/<timestamp>.json`)
- ✅ Prompt versioning via `--prompt prompts/v1.txt` for iteration
- ✅ Flat layout — `nenu_demo/` package, no `src/` prefix
- ✅ Pydantic v2 models: `UserProfile`, `TestCase`, `Judgment`, `Report`

**Placeholder scan:** None — all steps contain complete code.

**Type consistency:**
- `get_response(...) -> str` consumed in `run_eval.py` as `response` ✅
- `score_response(...) -> tuple[str, str]` destructured as `verdict, reasoning` ✅
- `build_report(list[Judgment]) -> Report` — `judgments` list built in `main()` ✅
- `TestCase.expected_behavior` — used in `_JUDGE_TEMPLATE` in `scorer.py` ✅
