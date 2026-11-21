"""Prompt construction for grounded, cited, English answers."""

from __future__ import annotations

from typing import Any, Sequence

REFUSAL_PLACEHOLDER = "{refusal_message}"

SYSTEM_PROMPT_TEMPLATE = """You are an assistant for 3GPP 5G specifications (TS 23.501 and TS 24.501, Release 18).
Answer ONLY in English, using ONLY the context passages provided below.

Rules:
1. After every factual statement, add a citation in the form (TS 24.501, §5.5.1.2.2).
   Use only spec/section labels that appear in the context headers. Never invent a section number.
2. Do not use outside knowledge. Do not guess values, timers, cause codes, message names or IEs.
3. If the context does not contain the answer, reply exactly:
   {refusal_message}
4. If the context answers only part of the question, answer that part with citations and state
   clearly which part could not be found in the documents.
5. Be precise and concise. Prefer the exact terminology of the specifications."""

USER_TEMPLATE = """Context passages:

{context}

Question: {question}

Answer in English, following the rules."""


def system_prompt(refusal_message: str) -> str:
    """The system prompt with the configured refusal message."""
    return SYSTEM_PROMPT_TEMPLATE.format(refusal_message=refusal_message)


def format_context(chunks: Sequence[dict[str, Any]]) -> str:
    """Number the chunks: ``[1] [TS 24.501 | Rel-18 | §A.1 | Causes ...]\\n<chunk text>``.

    ``text_for_index`` already starts with the header line, so it is used as is.
    """
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        body = chunk.get("text_for_index") or f"[{chunk['spec']} | {chunk['release']} | §{chunk['section']} | {chunk['title']}]\n{chunk['text']}"
        blocks.append(f"[{i}] {body}")
    return "\n\n".join(blocks)


def build_messages(question: str, chunks: Sequence[dict[str, Any]], refusal_message: str) -> list[dict[str, str]]:
    """System + user messages for one question and its retrieved chunks."""
    return [
        {"role": "system", "content": system_prompt(refusal_message)},
        {"role": "user", "content": USER_TEMPLATE.format(context=format_context(chunks), question=question)},
    ]
