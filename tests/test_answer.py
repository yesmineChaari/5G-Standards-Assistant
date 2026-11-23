"""Tests for prompt construction, refusal detection and the two refusal layers (no model/API calls)."""

from __future__ import annotations

from typing import Any

import pytest

from src.config import load_config
from src.generate.answer import Answerer, is_refusal_text
from src.generate.llm import LLM, LLMUnavailable
from src.generate.prompt import build_messages, format_context, system_prompt
from src.retrieve.hybrid import Candidate
from src.retrieve.pipeline import RetrievalResult

REFUSAL = "I could not find this information in the documents."


def _chunk(section: str, text: str = "Cause #9 UE identity cannot be derived by the network.") -> dict[str, Any]:
    header = f"[TS 24.501 | Rel-18 | §{section} | Causes related to UE identification]"
    return {"chunk_id": f"24501_{section}_001", "spec": "TS 24.501", "release": "Rel-18", "section": section,
            "title": "Causes related to UE identification", "text": text, "text_for_index": f"{header}\n{text}",
            "merged_sections": []}


# --------------------------------------------------------------------------- prompt


def test_system_prompt_contains_rules_and_exact_refusal() -> None:
    sp = system_prompt(REFUSAL)
    assert "Answer ONLY in English, using ONLY the context passages" in sp
    assert "(TS 24.501, §5.5.1.2.2)" in sp
    assert "Never invent a section number." in sp
    assert f"reply exactly:\n   {REFUSAL}" in sp


def test_context_format_numbers_chunks_with_headers() -> None:
    ctx = format_context([_chunk("A.1"), _chunk("9.11.3.2", "The 5GMM cause IE.")])
    assert ctx.startswith("[1] [TS 24.501 | Rel-18 | §A.1 | Causes related to UE identification]\nCause #9")
    assert "\n\n[2] [TS 24.501 | Rel-18 | §9.11.3.2 |" in ctx


def test_build_messages_roles() -> None:
    msgs = build_messages("What does cause #9 mean?", [_chunk("A.1")], REFUSAL)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "Question: What does cause #9 mean?" in msgs[1]["content"]
    assert "[1] [TS 24.501" in msgs[1]["content"]


# --------------------------------------------------------------------------- refusal detection


@pytest.mark.parametrize(
    "text",
    [REFUSAL, f"  {REFUSAL}  ", f'"{REFUSAL}"', REFUSAL.rstrip("."), f"**{REFUSAL}**", REFUSAL.upper()],
)
def test_refusal_variants_detected(text: str) -> None:
    assert is_refusal_text(text, REFUSAL)


def test_partial_answer_is_not_refusal() -> None:
    partial = f"Cause #9 means the UE identity cannot be derived (TS 24.501, §A.1). The timer part: {REFUSAL}"
    assert not is_refusal_text(partial, REFUSAL)


# --------------------------------------------------------------------------- answer flow with fakes


class FakeLLM(LLM):
    def __init__(self, reply: str) -> None:
        super().__init__("groq", "fake-model")
        self.reply = reply
        self.calls: list[list[dict[str, str]]] = []

    def check(self) -> None:  # no API key needed
        return None

    def generate(self, messages, temperature=None, max_tokens=None) -> str:
        self.calls.append(messages)
        return self.reply


class FakePipeline:
    def __init__(self, top_score: float | None, n: int = 2) -> None:
        self.top_score = top_score
        self.n = n
        self.router = type("R", (), {"llm_classifier": None})()

    def retrieve(self, query, use_reranker=None, **kwargs) -> RetrievalResult:
        use_reranker = True if use_reranker is None else use_reranker
        results = []
        for i in range(self.n):
            c = Candidate(f"c{i}", _chunk("A.1"))
            c.final_rank = i + 1
            c.rerank_score = (self.top_score - 0.01 * i) if (use_reranker and self.top_score is not None) else None
            results.append(c)
        flags = {"use_hybrid": True, "use_reranker": use_reranker, "use_routing": True, "use_acronym_expansion": True}
        return RetrievalResult(query, ["TS 24.501"], "keywords: 5GMM", flags, results, results, {"retrieval_s": 0.01})


def _answerer(reply: str, top_score: float | None, n: int = 2) -> tuple[Answerer, FakeLLM]:
    llm = FakeLLM(reply)
    return Answerer(load_config(), llm=llm, pipeline=FakePipeline(top_score, n)), llm


def test_layer1_threshold_refuses_without_llm_call() -> None:
    ans, llm = _answerer("should not be used", top_score=0.05)
    res = ans.answer("What is the frequency range of NR band n78?")
    assert res.answer == REFUSAL and res.is_refusal and res.refusal_layer == "threshold"
    assert llm.calls == []


def test_layer1_inactive_without_reranker() -> None:
    ans, llm = _answerer("Cause #9 means ... (TS 24.501, §A.1)", top_score=0.05)
    res = ans.answer("q", use_reranker=False)
    assert not res.is_refusal and len(llm.calls) == 1


def test_layer2_llm_refusal_normalised_to_exact_message() -> None:
    ans, llm = _answerer(f'"{REFUSAL.rstrip(".")}"', top_score=0.9)
    res = ans.answer("How does Wi-Fi 7 multi-link operation work?")
    assert res.answer == REFUSAL and res.is_refusal and res.refusal_layer == "llm"
    assert len(llm.calls) == 1


def test_normal_answer_passes_through() -> None:
    reply = "Cause #9 indicates that the UE identity cannot be derived by the network (TS 24.501, §A.1)."
    ans, llm = _answerer(reply, top_score=0.9)
    res = ans.answer("What does cause #9 mean in 5GMM?")
    assert res.answer == reply and not res.is_refusal and res.refusal_layer is None
    assert res.routed_specs == ["TS 24.501"] and res.top_score == pytest.approx(0.9)
    assert "llm_s" in res.latency and "total_s" in res.latency
    system, user = llm.calls[0]
    assert system["role"] == "system" and "[1] [TS 24.501 | Rel-18 | §A.1" in user["content"]


def test_no_context_refuses() -> None:
    ans, llm = _answerer("unused", top_score=0.9, n=0)
    res = ans.answer("q")
    assert res.answer == REFUSAL and res.refusal_layer == "no_context" and llm.calls == []


def test_missing_api_key_raises(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("src.generate.llm.get_api_key", lambda provider: None)
    with pytest.raises(LLMUnavailable):
        LLM("groq", "llama-3.3-70b-versatile").check()


def test_unknown_provider_rejected() -> None:
    with pytest.raises(ValueError):
        LLM("unknown", "x")


def test_reasoning_effort_only_for_reasoning_models() -> None:
    from src.generate.llm import is_reasoning_model

    assert is_reasoning_model("openai/gpt-oss-120b") and is_reasoning_model("o3-mini")
    assert not is_reasoning_model("llama-3.3-70b-versatile")
    assert LLM("groq", "openai/gpt-oss-120b", reasoning_effort="low")._extra_body() == {"reasoning_effort": "low"}
    assert LLM("groq", "llama-3.1-8b-instant", reasoning_effort="low")._extra_body() is None


def test_client_errors_are_not_retried(monkeypatch) -> None:
    class NotFound(Exception):
        status_code = 404

    llm = LLM("groq", "missing-model", max_retries=3, backoff_s=0)
    monkeypatch.setattr(llm, "check", lambda: None)
    calls = []

    def fail(*args):
        calls.append(1)
        raise NotFound("model_not_found")

    llm._call = fail
    with pytest.raises(RuntimeError, match="model_not_found"):
        llm.generate([{"role": "user", "content": "x"}])
    assert len(calls) == 1


def test_model_reference_marks_are_removed() -> None:
    from src.generate.answer import clean_answer

    raw = "Cause #9 means X (TS 24.501, §A.1)【1†L7-L12】. Then Y (TS 24.501, §5.6.1.5)【2†L13-L16】【6†L1】."
    assert clean_answer(raw) == "Cause #9 means X (TS 24.501, §A.1). Then Y (TS 24.501, §5.6.1.5)."


def test_bracket_citations_are_kept_in_standard_form() -> None:
    from src.generate.answer import clean_answer
    from src.generate.citation_check import extract_citations

    raw = "X happens【TS 24.501, §A.1】. Y too 【TS 24.501, §5.6.1.5】【3†L2】."
    out = clean_answer(raw)
    assert out == "X happens (TS 24.501, §A.1). Y too (TS 24.501, §5.6.1.5)."
    assert [c.section for c in extract_citations(out)] == ["A.1", "5.6.1.5"]


PASSAGES = [
    {"spec": "TS 23.501", "section": "5.15.2.1", "merged_sections": []},
    {"spec": "TS 24.501", "section": "5.5.1.2.2", "merged_sections": ["5.5.1.2.3"]},
    {"spec": "TS 24.501", "section": "5.3.6", "merged_sections": []},
]


def test_passage_numbers_become_citations() -> None:
    from src.generate.answer import clean_answer

    out = clean_answer("S-NSSAI identifies a slice [1]. The UE sends it [2][3]. Both [1, 2].", PASSAGES)
    assert out == ("S-NSSAI identifies a slice (TS 23.501, §5.15.2.1). The UE sends it (TS 24.501, §5.5.1.2.2) "
                   "(TS 24.501, §5.3.6). Both (TS 23.501, §5.15.2.1) (TS 24.501, §5.5.1.2.2).")


def test_out_of_range_passage_number_is_left_as_is() -> None:
    from src.generate.answer import clean_answer
    from src.generate.citation_check import extract_citations

    out = clean_answer("Value 7 [4].", PASSAGES)
    assert out == "Value 7 [4]."
    assert extract_citations(out) == []


def test_reference_marks_resolve_to_their_passage() -> None:
    from src.generate.answer import clean_answer

    # alone: passage 2; with a section the passage covers: that section; repeat of a citation: removed
    raw = ("X happens【2†L9-L12】. Y【2†§5.5.1.2.3】. Z【1†§9.9.9】. "
           "W (TS 24.501, §5.3.6)【3†L1-L4】. V【9†L1】.")
    assert clean_answer(raw, PASSAGES) == (
        "X happens (TS 24.501, §5.5.1.2.2). Y (TS 24.501, §5.5.1.2.3). Z (TS 23.501, §5.15.2.1). "
        "W (TS 24.501, §5.3.6). V.")


def test_section_without_spec_gets_the_only_matching_spec() -> None:
    from src.generate.answer import clean_answer

    raw = 'MICO mode [§5.3.6, "Mobile initiated connection only mode"]. Other (§4.4.4).'
    assert clean_answer(raw, PASSAGES) == "MICO mode (TS 24.501, §5.3.6). Other (§4.4.4)."
    both = PASSAGES + [{"spec": "TS 23.501", "section": "5.3.6", "merged_sections": []}]
    assert clean_answer("MICO [§5.3.6].", both) == "MICO [§5.3.6]."  # ambiguous: not guessed


def test_ollama_unreachable_is_unavailable(monkeypatch) -> None:
    import requests

    def refuse(*args, **kwargs):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "get", refuse)
    with pytest.raises(LLMUnavailable, match="not running"):
        LLM("ollama", "llama3.1:8b").check()


@pytest.mark.parametrize(
    "msg, expected",
    [
        ("Rate limit reached ... on tokens per minute (TPM). Please try again in 7.5s. Visit ...", 7.5),
        ("... Please try again in 450ms.", 0.45),
        ("... on tokens per day (TPD): Limit 200000. Please try again in 1h2m3.5s.", 3723.5),
        ("... Please try again in 12m30s.", 750.0),
        ("no hint here", None),
    ],
)
def test_retry_after_parsing(msg: str, expected) -> None:
    from src.generate.llm import retry_after_seconds

    got = retry_after_seconds(Exception(msg))
    assert got == (pytest.approx(expected) if expected is not None else None)


class _RateLimited(Exception):
    status_code = 429


def test_minute_rate_limit_waits_without_using_retries(monkeypatch) -> None:
    llm = LLM("groq", "m", max_retries=1, backoff_s=0)
    monkeypatch.setattr(llm, "check", lambda: None)
    monkeypatch.setattr("src.generate.llm.time.sleep", lambda s: None)
    calls = []

    def flaky(*args):
        calls.append(1)
        if len(calls) < 3:
            raise _RateLimited("Please try again in 2s.")
        return "ok"

    llm._call = flaky
    assert llm.generate([{"role": "user", "content": "x"}]) == "ok" and len(calls) == 3


def test_daily_rate_limit_raises_exhausted(monkeypatch) -> None:
    from src.generate.llm import RateLimitExhausted

    llm = LLM("groq", "m")
    monkeypatch.setattr(llm, "check", lambda: None)

    def daily(*args):
        raise _RateLimited("tokens per day (TPD). Please try again in 1h5m.")

    llm._call = daily
    with pytest.raises(RateLimitExhausted):
        llm.generate([{"role": "user", "content": "x"}])
