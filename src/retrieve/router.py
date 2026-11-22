"""Query routing by specification: TS 23.501, TS 24.501 or both.

1. Rule-based scoring with weighted keyword / regex signals for each spec, plus cross-spec
   signals (e.g. "procedure ... messages", "end-to-end", "both").
2. Decision: a cross-spec signal, or signals for both specs (any score on each side) -> both;
   otherwise the only spec with signals. With no signal at all an optional LLM classifier is asked
   (returns ``23501`` / ``24501`` / ``both``); if it is unavailable or fails, both specs are used.
   Routing errs towards both specs: a missed spec loses recall that the reranker cannot recover,
   while an extra spec only costs some precision.

Spec-specific signals are terms that occur mostly in one spec, e.g. RM-/CM- states occur 65 and
328 times in TS 23.501 and 0 and 5 times in TS 24.501 (5GMM-IDLE: 0 vs 280). Terms used heavily
in both specs (MICO, periodic registration update, Requested NSSAI, PDU session) are not signals.
3. The decision comes with a short human-readable reason (shown in the UI), e.g.
   ``Routed to: TS 24.501 (keywords: 5GMM, cause #)``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

logger = logging.getLogger(__name__)

SPEC_23501 = "TS 23.501"
SPEC_24501 = "TS 24.501"
BOTH = [SPEC_23501, SPEC_24501]

# (label shown in the reason, compiled regex, weight). Case-insensitive unless noted.
Signal = tuple[str, re.Pattern[str], float]


def _ci(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


NAS_MESSAGES = [
    "REGISTRATION REQUEST", "REGISTRATION ACCEPT", "REGISTRATION REJECT", "REGISTRATION COMPLETE",
    "SERVICE REQUEST", "SERVICE ACCEPT", "SERVICE REJECT",
    "DEREGISTRATION REQUEST", "DEREGISTRATION ACCEPT",
    "PDU SESSION ESTABLISHMENT REQUEST", "PDU SESSION ESTABLISHMENT ACCEPT", "PDU SESSION ESTABLISHMENT REJECT",
    "PDU SESSION MODIFICATION REQUEST", "PDU SESSION MODIFICATION COMMAND", "PDU SESSION MODIFICATION COMPLETE",
    "PDU SESSION RELEASE REQUEST", "PDU SESSION RELEASE COMMAND", "PDU SESSION RELEASE COMPLETE",
    "SECURITY MODE COMMAND", "SECURITY MODE COMPLETE", "SECURITY MODE REJECT",
    "AUTHENTICATION REQUEST", "AUTHENTICATION RESPONSE", "AUTHENTICATION REJECT", "AUTHENTICATION FAILURE",
    "AUTHENTICATION RESULT", "IDENTITY REQUEST", "IDENTITY RESPONSE",
    "CONFIGURATION UPDATE COMMAND", "CONFIGURATION UPDATE COMPLETE",
    "DL NAS TRANSPORT", "UL NAS TRANSPORT", "NOTIFICATION RESPONSE", "5GMM STATUS", "5GSM STATUS",
]
# Uppercase NAS message names (case-sensitive: "service request procedure" in lowercase is generic).
_NAS_MESSAGE_RE = re.compile(r"\b(?:" + "|".join(re.escape(m) for m in NAS_MESSAGES) + r")\b")

SIGNALS_24501: list[Signal] = [
    ("TS 24.501", _ci(r"\b(?:TS\s*)?24\.501\b"), 5.0),
    ("NAS", re.compile(r"\bNAS\b"), 2.0),
    ("5GMM", _ci(r"\b5GMM\b"), 2.0),
    ("5GSM", _ci(r"\b5GSM\b"), 2.0),
    ("cause #", _ci(r"\bcause\s*(?:value\s*)?#?\s*\d+|#\s*\d+\b"), 2.0),
    ("cause value", _ci(r"\bcause values?\b"), 2.0),
    ("timer T3xxx", _ci(r"\bT3\d{3}\b"), 2.0),
    ("information element", _ci(r"\binformation elements?\b"), 2.0),
    ("IE", re.compile(r"\bIEs?\b"), 2.0),
    ("IEI", re.compile(r"\bIEI\b"), 2.0),
    ("NAS message name", _NAS_MESSAGE_RE, 2.0),
    ("message type", _ci(r"\bmessage types?\b"), 2.0),
    ("5GMM state", _ci(r"\b5GMM[- ](?:state|NULL|DEREGISTERED|REGISTERED|IDLE|CONNECTED)\b"), 2.0),
    ("coding", _ci(r"\b(?:coded|coding|octets?|bits? \d)\b"), 1.0),
    ("ngKSI", _ci(r"\bngKSI\b"), 2.0),
    ("security mode control", _ci(r"\bsecurity mode control\b"), 2.0),
    ("mobile identity", _ci(r"\bmobile identity\b"), 2.0),
    ("abnormal case", _ci(r"\babnormal cases?\b"), 1.0),
    ("timer", _ci(r"\btimers?\b"), 1.0),
    ("UE behaviour", _ci(r"\bUE shall\b|\bupon (?:receipt|receiving)\b"), 1.0),
    ("registration type", _ci(r"\bregistration types?\b"), 2.0),
    ("QoS rules IE", _ci(r"\bQoS rules?\b"), 1.0),
]

SIGNALS_23501: list[Signal] = [
    ("TS 23.501", _ci(r"\b(?:TS\s*)?23\.501\b"), 5.0),
    ("NF name", re.compile(r"\b(?:NRF|NSSF|PCF|UDM|UDR|AUSF|NEF|NWDAF|SCP|SEPP|CHF|NSACF|UCMF|LMF|BSF|5G-EIR|N3IWF|TNGF)\b"), 2.0),
    ("NF name (shared)", re.compile(r"\b(?:AMF|SMF|UPF)\b"), 1.0),
    ("NF selection", _ci(r"\b(?:AMF|SMF|UPF|PCF|NRF|NF) (?:discovery|selection)\b|\bselects? (?:an? |the )?(?:AMF|SMF|UPF|PCF)\b"), 2.0),
    ("reference point", _ci(r"\breference points?\b"), 2.0),
    ("Nx interface", re.compile(r"\bN(?:[1-9]|1[0-5])\b"), 1.0),
    ("service-based", _ci(r"\bservice[- ]based\b"), 2.0),
    ("architecture", _ci(r"\barchitectur(?:e|al)\b"), 2.0),
    ("network function", _ci(r"\bnetwork functions?\b"), 1.0),
    ("role / functionality", _ci(r"\b(?:role|roles|functionalit(?:y|ies)|responsib(?:le|ility))\b"), 1.0),
    ("QoS model", _ci(r"\bQoS (?:model|flows?|characteristics|parameters)\b"), 2.0),
    ("5QI", _ci(r"\b5QI\b"), 2.0),
    ("network slicing", _ci(r"\b(?:network slic(?:e|es|ing))\b"), 2.0),
    ("S-NSSAI / NSSAI", _ci(r"\b(?:S-)?NSSAI\b"), 1.0),
    ("SSC mode", _ci(r"\bSSC modes?\b"), 2.0),
    ("edge computing", _ci(r"\bedge computing\b"), 2.0),
    ("roaming", _ci(r"\broaming\b|\bhome[- ]routed\b|\bLBO\b"), 2.0),
    ("non-public network", _ci(r"\bnon-public networks?\b|\bNPN\b|\bSNPN\b"), 1.0),
    ("deployment", _ci(r"\bdeployments?\b"), 1.0),
    ("RM / CM state", re.compile(r"\b(?:RM|CM)[- ](?:states?|REGISTERED|DEREGISTERED|IDLE|CONNECTED)\b"), 2.0),
    ("reachability", _ci(r"\breachab(?:le|ility)\b"), 1.0),
]

CROSS_SIGNALS: list[Signal] = [
    ("both specs", _ci(r"\bboth (?:specs|specifications|documents)\b"), 3.0),
    ("23.501 and 24.501", _ci(r"23\.501.*24\.501|24\.501.*23\.501"), 3.0),
    ("procedure ... messages", _ci(r"\bprocedures?\b.*\bmessages?\b"), 3.0),
    ("end-to-end", _ci(r"\bend[- ]to[- ]end\b"), 3.0),
    ("and the NAS", _ci(r"\band the NAS\b"), 3.0),
    ("exchanged between", _ci(r"\bexchanged between\b"), 3.0),
    ("architecture and signalling", _ci(r"\barchitecture\b.*\bsignall?ing\b|\bsignall?ing\b.*\barchitecture\b"), 3.0),
]


@dataclass
class RouteDecision:
    """Routing result."""

    specs: list[str]
    reason: str
    method: str  # "rules" | "llm" | "default"
    scores: dict[str, float] = field(default_factory=dict)
    matched: dict[str, list[str]] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return " + ".join(self.specs)


def _score(query: str, signals: Sequence[Signal]) -> tuple[float, list[str]]:
    total, labels = 0.0, []
    for label, regex, weight in signals:
        if regex.search(query):
            total += weight
            labels.append(label)
    return total, labels


def parse_llm_label(text: str) -> list[str] | None:
    """Map an LLM answer to a spec list; ``None`` if it cannot be understood."""
    t = text.strip().lower()
    has23 = "23501" in t.replace(".", "") or "23.501" in t
    has24 = "24501" in t.replace(".", "") or "24.501" in t
    if "both" in t or (has23 and has24):
        return list(BOTH)
    if has23:
        return [SPEC_23501]
    if has24:
        return [SPEC_24501]
    return None


ROUTER_PROMPT = (
    "Classify which 3GPP specification is needed to answer the question.\n"
    "- 23501: TS 23.501, 5G System architecture (network functions, reference points, QoS model,"
    " network slicing, sessions and mobility concepts).\n"
    "- 24501: TS 24.501, NAS protocol between UE and network (5GMM/5GSM procedures, NAS messages,"
    " information elements, timers, cause values).\n"
    "- both: the question needs the architecture and the NAS signalling.\n"
    "Answer with exactly one word: 23501, 24501 or both.\n\nQuestion: {question}"
)


class Router:
    """Rule-based router with an optional LLM fallback for ties."""

    def __init__(self, llm_classifier: Callable[[str], str] | None = None) -> None:
        self.llm_classifier = llm_classifier

    def route(self, query: str) -> RouteDecision:
        s23, m23 = _score(query, SIGNALS_23501)
        s24, m24 = _score(query, SIGNALS_24501)
        sx, mx = _score(query, CROSS_SIGNALS)
        scores = {SPEC_23501: s23, SPEC_24501: s24, "cross": sx}
        matched = {SPEC_23501: m23, SPEC_24501: m24, "cross": mx}

        def decision(specs: list[str], why: str, method: str = "rules") -> RouteDecision:
            return RouteDecision(specs, why, method, scores, matched)

        if sx > 0:
            return decision(list(BOTH), f"cross-spec ({', '.join(mx)})")
        # A spec named explicitly ("In 24.501, ...") decides, unless both are named.
        named = [spec for spec, m in ((SPEC_23501, m23), (SPEC_24501, m24)) if spec in m]
        if len(named) == 1:
            return decision(named, f"spec named in the question ({named[0]})")
        if s23 > 0 and s24 > 0:
            return decision(list(BOTH), f"signals for both ({', '.join(m23 + m24)})")
        if s23 > 0:
            return decision([SPEC_23501], f"keywords: {', '.join(m23)}")
        if s24 > 0:
            return decision([SPEC_24501], f"keywords: {', '.join(m24)}")

        # No signal at all: ask the LLM if available, otherwise search both specs.
        if self.llm_classifier is not None:
            try:
                answer = self.llm_classifier(ROUTER_PROMPT.format(question=query))
                specs = parse_llm_label(answer)
                if specs:
                    return decision(specs, f"LLM classification ({answer.strip()[:20]})", "llm")
                logger.warning("Router: unparseable LLM answer %r", answer)
            except Exception as exc:  # LLM unavailable: fall back to both
                logger.warning("Router: LLM classification failed (%s)", exc)
        why = "no clear signal: both specs"
        return decision(list(BOTH), why, "default")


def route(query: str, llm_classifier: Callable[[str], str] | None = None) -> RouteDecision:
    """Convenience function: route a query with the rule-based router."""
    return Router(llm_classifier).route(query)
