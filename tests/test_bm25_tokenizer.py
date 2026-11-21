"""Tests for the BM25 tokenizer and index (src.index.bm25_index)."""

from __future__ import annotations

import pytest

from src.config import load_config, resolve_path
from src.index.bm25_index import BM25Index, load_bm25, normalise_causes, s_stem, tokenize


@pytest.mark.parametrize("identifier", ["T3510", "#9", "S-NSSAI", "5GMM", "5GSM", "N2", "PDU", "9.11.3.2"])
def test_identifiers_are_single_tokens(identifier: str) -> None:
    tokens = tokenize(f"The {identifier} is used here.")
    assert identifier.lower() in tokens


@pytest.mark.parametrize(
    "text",
    ["cause #9", "cause 9", "Cause #9", "cause value #9", "cause # 9", "#9", "5GMM cause #9."],
)
def test_cause_references_normalised(text: str) -> None:
    assert "#9" in tokenize(text)


def test_cause_normalisation_keeps_cause_word_and_other_numbers() -> None:
    assert normalise_causes("cause 9 and cause value #26") == "cause #9 and cause #26"
    assert "because" in normalise_causes("because 9 timers")  # "because" is not "cause"
    assert tokenize("cause #9") == ["cause", "#9"]


def test_compound_parts_emitted_but_section_numbers_not_split() -> None:
    tokens = tokenize("S-NSSAI and 5GMM-REGISTERED.NORMAL-SERVICE in clause 5.5.1.2.2")
    assert "s-nssai" in tokens and "nssai" in tokens
    assert "5gmm-registered.normal-service" in tokens and "normal" in tokens
    assert "5.5.1.2.2" in tokens
    assert "5" not in tokens and "1" not in tokens


def test_stopwords_and_punctuation_removed() -> None:
    tokens = tokenize("What is the role of the NRF? (see clause 6.2.6).")
    assert "the" not in tokens and "is" not in tokens and "of" not in tokens
    assert "nrf" in tokens and "role" in tokens and "6.2.6" in tokens
    assert all(not t.endswith((".", ")", "?")) for t in tokens)


def test_negation_kept() -> None:
    assert "not" in tokenize("5GS services not allowed")


def test_light_stemming_only_on_plain_words() -> None:
    assert s_stem("messages") == "message"
    assert s_stem("procedures") == "procedure"
    assert s_stem("entities") == "entity"
    assert s_stem("address") == "address"
    assert s_stem("status") == "status"
    tokens = tokenize("REGISTRATION REQUEST messages for AMFs and UEs with T3510s")
    assert "message" in tokens
    assert "t3510s" in tokens  # identifiers are never stemmed
    assert tokenize("timers", stem=False) == ["timers"]


def _chunk(cid: str, spec: str, text: str) -> dict:
    return {"chunk_id": cid, "spec": spec, "section": "1", "title": "t", "text": text, "text_for_index": text}


def test_bm25_index_query_and_spec_filter() -> None:
    chunks = [
        _chunk("a", "TS 24.501", "Timer T3510 is started when the UE sends a REGISTRATION REQUEST message."),
        _chunk("b", "TS 24.501", "Cause #9 UE identity cannot be derived by the network."),
        _chunk("c", "TS 23.501", "The NRF supports service discovery for network functions."),
        _chunk("d", "TS 23.501", "The AMF terminates the N1 and N2 reference points."),
    ]
    index = BM25Index(chunks)
    assert index.query("T3510")[0].chunk_id == "a"
    assert index.query("what does cause 9 mean")[0].chunk_id == "b"
    hits = index.query("NRF", specs=["TS 24.501"])
    assert all(h.chunk["spec"] == "TS 24.501" for h in hits)
    assert index.query("NRF", specs=["TS 23.501"])[0].chunk_id == "c"
    assert [h.rank for h in index.query("network")] == list(range(1, len(index.query("network")) + 1))
    assert index.query("the of and") == []  # only stopwords


def test_bm25_index_save_and_load(tmp_path) -> None:
    index = BM25Index(
        [
            _chunk("a", "TS 24.501", "Timer T3510 value"),
            _chunk("b", "TS 23.501", "The NRF supports discovery"),
            _chunk("c", "TS 23.501", "The AMF handles mobility"),
        ]
    )
    index.fingerprint = {"x": 1}
    path = tmp_path / "bm25.pkl"
    index.save(path)
    loaded = BM25Index.load(path)
    assert loaded.fingerprint == {"x": 1}
    assert loaded.query("T3510")[0].chunk_id == "a"


# --------------------------------------------------------------------------- real index


@pytest.fixture(scope="module")
def real_index() -> BM25Index:
    if not resolve_path(load_config()["bm25"]["path"]).exists():
        pytest.skip("BM25 index not built (run `make index`)")
    return load_bm25()


def test_real_t3510_returns_24501(real_index: BM25Index) -> None:
    hits = real_index.query("T3510", k=5)
    assert hits and all(h.chunk["spec"] == "TS 24.501" for h in hits)


def test_real_cause_9_finds_annex_a1(real_index: BM25Index) -> None:
    hits = real_index.query("What does cause #9 mean in 5GMM?", k=10)
    assert any(h.chunk["section"] == "A.1" for h in hits)
