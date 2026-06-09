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
