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
