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
