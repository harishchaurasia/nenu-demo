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
