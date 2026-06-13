from __future__ import annotations

from openai import OpenAI

from app.config import get_settings


def generate_with_llm(system: str, user: str) -> str | None:
    settings = get_settings()
    if not settings.openai_api_key:
        return None
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def local_document_answer(question: str, contexts: list[str]) -> str:
    if not contexts:
        return "I could not find an accessible source that answers that question."
    joined = "\n\n".join(contexts[:3])
    compact = " ".join(joined.split())
    return (
        "Based on the accessible MediAssist sources, the most relevant guidance is: "
        f"{compact[:900]}{'...' if len(compact) > 900 else ''}"
    )
