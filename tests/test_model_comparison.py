"""Model comparison: generators answer from the reference passages; embedding configs get their own index."""

from __future__ import annotations

REFUSAL = "I could not find this information in the documents."

REF = {
    "id": "S24-01", "is_refusal": False, "refusal_layer": None, "routed_specs": ["TS 24.501"], "top_score": 0.9,
    "contexts": ["[TS 24.501 | Rel-18 | §A.1 | Causes]\nCause #9 ...", "[TS 24.501 | Rel-18 | §5.6.1.5 | x]\n..."],
    "retrieved": [{"spec": "TS 24.501", "section": "A.1", "merged_sections": [], "score": 0.9},
                  {"spec": "TS 24.501", "section": "5.6.1.5", "merged_sections": [], "score": 0.5}],
}
ENTRY = {"id": "S24-01", "question": "What does cause #9 mean in 5GMM?"}


class FakeLLM:
    name = "ollama:fake"
    last_latency = 1.5

    def __init__(self, reply: str) -> None:
        self.reply, self.messages = reply, None

    def generate(self, messages):
        self.messages = messages
        return self.reply


def test_generator_gets_the_reference_passages_in_order() -> None:
    from eval.run_model_comparison import answer_with

    llm = FakeLLM("Identity not derivable [1]. Also (TS 24.501, §9.9).")
    rec = answer_with(llm, ENTRY, REF, REFUSAL)
    user = llm.messages[1]["content"]
    assert user.index("[1] [TS 24.501 | Rel-18 | §A.1") < user.index("[2] [TS 24.501 | Rel-18 | §5.6.1.5")
    assert rec["answer"].startswith("Identity not derivable (TS 24.501, §A.1)")
    assert rec["citations"] == [{"spec": "TS 24.501", "section": "A.1"}]
    assert rec["invalid_citations"] == [{"spec": "TS 24.501", "section": "9.9"}]
    assert rec["latency"] == {"llm_s": 1.5} and rec["llm"] == "ollama:fake"


def test_threshold_refusal_is_kept_without_a_model_call() -> None:
    from eval.run_model_comparison import answer_with

    llm = FakeLLM("should not be called")
    rec = answer_with(llm, ENTRY, {**REF, "is_refusal": True, "refusal_layer": "threshold"}, REFUSAL)
    assert llm.messages is None
    assert rec["is_refusal"] and rec["refusal_layer"] == "threshold" and rec["answer"] == REFUSAL


def test_model_refusal_is_normalised() -> None:
    from eval.run_model_comparison import answer_with

    rec = answer_with(FakeLLM(f'"{REFUSAL[:-1]}"'), ENTRY, REF, REFUSAL)
    assert rec["is_refusal"] and rec["refusal_layer"] == "llm" and rec["answer"] == REFUSAL


def test_embedding_config_uses_its_own_index() -> None:
    from eval.run_model_comparison import embedding_cfg
    from src.config import load_config

    cfg = load_config()
    c = embedding_cfg(cfg, "thenlper/gte-base", "")
    assert c["embeddings"]["model"] == "thenlper/gte-base" and c["embeddings"]["query_instruction"] == ""
    assert c["vector_store"]["path"] == cfg["vector_store"]["path"] + "_gte_base"
    assert cfg["embeddings"]["model"] == "BAAI/bge-base-en-v1.5"  # the main config is not modified
