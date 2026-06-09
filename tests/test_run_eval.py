import json
import sys
from unittest.mock import MagicMock, patch
from nenu_demo.run_eval import load_profile, load_test_cases, build_report
from nenu_demo.models import Judgment


def _j(id: str, verdict: str) -> Judgment:
    return Judgment(id=id, question="q", response="r", verdict=verdict, reasoning="ok")


def test_load_profile(tmp_path):
    data = {"name": "Alice", "context": {"timezone": "PST"}}
    (tmp_path / "profile.json").write_text(json.dumps(data))

    profile = load_profile(str(tmp_path / "profile.json"))

    assert profile.name == "Alice"
    assert profile.context["timezone"] == "PST"


def test_load_test_cases(tmp_path):
    data = [{"id": "tc-001", "question": "Q?", "context_keys": ["timezone"], "expected_behavior": "Mention PST"}]
    (tmp_path / "cases.json").write_text(json.dumps(data))

    cases = load_test_cases(str(tmp_path / "cases.json"))

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
    output_path = str(tmp_path / "report.json")

    (tmp_path / "profile.json").write_text(json.dumps(profile_data))
    (tmp_path / "cases.json").write_text(json.dumps(cases_data))
    (tmp_path / "prompt.txt").write_text("C:{context} Q:{question}")

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
    assert report["judgments"][0]["id"] == "tc-001"
    assert report["judgments"][0]["verdict"] == "pass"
