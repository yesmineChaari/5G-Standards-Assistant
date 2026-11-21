"""Tests for the Streamlit app helpers (no Streamlit server needed)."""

from __future__ import annotations

from src.app.ui_helpers import (
    SPEC_FILTERS,
    chat_models,
    escape_md,
    highlight_citations,
    is_cited,
    model_options,
    refusal_explanation,
    table_markdown,
)


def test_escape_md_keeps_text_literal() -> None:
    out = escape_md("Cause #9\n*IE* costs $5 <tag> [x]")
    assert out == r"Cause \#9" + "  \n" + r"\*IE\* costs \$5 \<tag\> \[x\]"


def test_table_markdown_keeps_table() -> None:
    text = "Table 1\n| a | b |\n|---|---|\n| x<br>y | $1 |"
    assert table_markdown(text) == "Table 1\n| a | b |\n|---|---|\n| x / y | \\$1 |"


def test_highlight_valid_and_invalid_citations() -> None:
    answer = "X (TS 24.501, §A.1). Y (TS 24.501, §5.6.1.5 and §9.9.9)."
    out = highlight_citations(answer, [{"spec": "TS 24.501", "section": "9.9.9"}])
    assert ":green[TS 24.501, §A.1]" in out
    assert ":red[**TS 24.501, §5.6.1.5 and §9.9.9** ⚠]" in out
    assert out.startswith("X (") and out.endswith(").")  # text around the citations is untouched


def test_is_cited_uses_merged_sections() -> None:
    chunk = {"spec": "TS 23.501", "section": "4.2.11", "merged_sections": ["4.2.11", "4.2.12"]}
    assert is_cited(chunk, [{"spec": "TS 23.501", "section": "4.2.12"}])
    assert not is_cited(chunk, [{"spec": "TS 24.501", "section": "4.2.12"}])
    assert not is_cited(chunk, [])


def test_chat_models_drops_speech_and_guard_models() -> None:
    ids = ["whisper-large-v3", "openai/gpt-oss-120b", "meta-llama/llama-prompt-guard-2-86m",
           "canopylabs/orpheus-v1-english", "qwen/qwen3.8-27b"]
    assert chat_models(ids) == ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]


def test_model_options_puts_configured_model_first() -> None:
    assert model_options("groq", ["a", "b", "c"], "b") == ["b", "a", "c"]
    assert model_options("groq", ["a"], "z") == ["z", "a"]
    assert model_options("ollama", [], None)[0] == "llama3.1:8b"  # defaults when nothing is discovered


def test_spec_filters() -> None:
    assert SPEC_FILTERS["Auto (router)"] is None
    assert SPEC_FILTERS["Both specs"] == ["TS 23.501", "TS 24.501"]


def test_refusal_explanations() -> None:
    assert "0.036" in refusal_explanation("threshold", 0.036, 0.15)
    assert "prompt rule" in refusal_explanation("llm", 0.98, 0.15)
