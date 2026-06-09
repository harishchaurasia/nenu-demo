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
