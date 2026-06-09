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
        json.dumps({"result": "yes"})
    )

    verdict, reasoning = score_response(client, test_case, "response", "gpt-4o-mini")

    assert verdict == "fail"
    assert "unparseable" in reasoning
