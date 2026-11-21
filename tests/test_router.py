"""Tests for the spec router (src.retrieve.router) and acronym expansion (src.retrieve.acronyms)."""

from __future__ import annotations

import pytest

from src.config import load_config
from src.retrieve.acronyms import AcronymExpander, acronyms_path, load_acronyms
from src.retrieve.router import BOTH, SPEC_23501, SPEC_24501, Router, parse_llm_label, route

# --------------------------------------------------------------------------- plan examples


def test_plan_example_cause_9_goes_to_24501() -> None:
    d = route("What does cause #9 mean in 5GMM?")
    assert d.specs == [SPEC_24501]
    assert "5GMM" in d.reason and "cause #" in d.reason
    assert d.method == "rules"


def test_plan_example_nrf_goes_to_23501() -> None:
    d = route("What is the role of the NRF?")
    assert d.specs == [SPEC_23501]
    assert "NF name" in d.reason


def test_plan_example_registration_nas_goes_to_both() -> None:
    d = route("Describe the registration procedure and the NAS messages exchanged between the UE and the AMF.")
    assert d.specs == BOTH
    assert d.reason.startswith("cross-spec")


# --------------------------------------------------------------------------- more cases


@pytest.mark.parametrize(
    "query",
    [
        "What is the purpose of timer T3510?",
        "What does the 5GS mobile identity IE contain?",
        "Which information elements are in the REGISTRATION REQUEST message?",
        "What are the 5GMM states of the UE?",
        "What does 5GSM cause #26 mean?",
        "What is the purpose of the security mode control procedure?",
    ],
)
def test_nas_questions_go_to_24501(query: str) -> None:
    assert route(query).specs == [SPEC_24501]


@pytest.mark.parametrize(
    "query",
    [
        "What is the role of the AMF in the 5G architecture?",
        "What does the N4 reference point connect?",
        "What is a 5QI?",
        "What are the SSC modes?",
        "What is the role of the NSSF in network slicing?",
        "Which functions does the PCF provide?",
    ],
)
def test_architecture_questions_go_to_23501(query: str) -> None:
    assert route(query).specs == [SPEC_23501]


@pytest.mark.parametrize(
    "query",
    [
        "Describe PDU session establishment: the role of the SMF and the 5GSM messages exchanged.",
        "Explain the end-to-end UE-initiated deregistration.",
        "How is the S-NSSAI used in the architecture and which IE carries the requested NSSAI in NAS signalling?",
        "Compare TS 23.501 and TS 24.501 on the service request.",
    ],
)
def test_cross_spec_questions_go_to_both(query: str) -> None:
    assert route(query).specs == BOTH


def test_explicit_spec_mention_wins() -> None:
    assert route("According to TS 23.501, what is a QoS flow?").specs == [SPEC_23501]
    assert route("In 24.501, how is the QoS flow description coded?").specs == [SPEC_24501]


def test_no_signal_defaults_to_both_without_llm() -> None:
    d = route("Tell me something interesting.")
    assert d.specs == BOTH and d.method == "default"
    assert "no clear signal" in d.reason


def test_tie_uses_llm_classifier() -> None:
    calls = []

    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        return "24501"

    d = Router(llm_classifier=fake_llm).route("Tell me something interesting.")
    assert d.specs == [SPEC_24501] and d.method == "llm"
    assert len(calls) == 1 and "Tell me something interesting." in calls[0]


def test_llm_not_called_when_rules_decide() -> None:
    def failing_llm(prompt: str) -> str:
        raise AssertionError("LLM must not be called")

    assert Router(failing_llm).route("What is the purpose of timer T3510?").specs == [SPEC_24501]


def test_llm_failure_or_garbage_falls_back_to_both() -> None:
    def broken(prompt: str) -> str:
        raise RuntimeError("API down")

    assert Router(broken).route("Tell me something interesting.").specs == BOTH
    assert Router(lambda p: "no idea").route("Tell me something interesting.").specs == BOTH


@pytest.mark.parametrize(
    "answer, expected",
    [("23501", [SPEC_23501]), ("24501.", [SPEC_24501]), ("Both", BOTH), ("TS 24.501", [SPEC_24501]),
     ("23.501 and 24.501", BOTH), ("unknown", None)],
)
def test_parse_llm_label(answer: str, expected: list[str] | None) -> None:
    assert parse_llm_label(answer) == expected


# --------------------------------------------------------------------------- acronyms


def test_acronym_expander_basic() -> None:
    exp = AcronymExpander({"NRF": ["Network Repository Function"], "5G-GUTI": ["5G Globally Unique Temporary Identifier"],
                           "GUTI": ["Globally Unique Temporary Identifier"], "5G LAN": ["5G Local Area Network"]})
    q, found = exp.expand("What is the role of the NRF?")
    assert found == ["NRF"]
    assert q == "What is the role of the NRF? (NRF: Network Repository Function)"
    assert exp.find("Is the 5G-GUTI sent?") == ["5G-GUTI"]  # longest match, not GUTI
    assert exp.find("5G LAN group") == ["5G LAN"]
    assert exp.find("NRFs and nrf") == []  # whole, case-sensitive tokens only
    assert exp.expand("no acronym here") == ("no acronym here", [])


@pytest.fixture(scope="module")
def real_acronyms() -> dict[str, list[str]]:
    cfg = load_config()
    if not acronyms_path(cfg).exists():
        pytest.skip("acronyms.json not built (run `make ingest`)")
    return load_acronyms(cfg)


def test_real_acronym_dictionary(real_acronyms: dict[str, list[str]]) -> None:
    assert real_acronyms["NRF"] == ["Network Repository Function"]
    assert real_acronyms["5GMM"] == ["5GS Mobility Management"]
    assert "5G LAN" in real_acronyms  # multi-word acronym kept thanks to the tab separator
    assert len(real_acronyms) > 250
