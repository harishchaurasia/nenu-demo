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
