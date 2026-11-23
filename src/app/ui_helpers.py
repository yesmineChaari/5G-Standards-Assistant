"""Pure helpers for the Streamlit app (no Streamlit import, so they can be unit tested)."""

from __future__ import annotations

import html
import re
from typing import Any, Iterable, Sequence

from src.generate.citation_check import citation_spans

# Characters that Streamlit markdown would interpret ($ starts LaTeX).
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]<>#|~$])")

SPEC_FILTERS = {
    "Auto (router)": None,
    "TS 23.501 only": ["TS 23.501"],
    "TS 24.501 only": ["TS 24.501"],
    "Both specs": ["TS 23.501", "TS 24.501"],
}

# Suggested models per provider (the selector also lists the models a Groq key / Ollama server offers).
DEFAULT_MODELS = {
    "groq": ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"],
    "gemini": ["gemini-3.1-flash-lite", "gemini-3.6-flash"],
    "openai": ["gpt-4o-mini", "gpt-4o"],
    "ollama": ["llama3.1:8b", "qwen2.5:7b", "mistral:7b"],
}

# Groq also serves speech, TTS and guard models; they cannot answer questions.
_NON_CHAT = re.compile(r"whisper|orpheus|guard|tts|playai|distil", re.I)


def escape_md(text: str) -> str:
    """Show plain spec text literally in ``st.markdown`` (keeps line breaks)."""
    return _MD_SPECIAL.sub(r"\\\1", text).replace("\n", "  \n")


def escape_dollars(text: str) -> str:
    """LLM answers are markdown on purpose; only ``$`` must be escaped (it would start LaTeX)."""
    return text.replace("$", r"\$")


def table_markdown(text: str) -> str:
    """Chunk text of a table chunk, ready for ``st.markdown`` (the markdown table is kept)."""
    return text.replace("$", r"\$").replace("<br>", " / ")


def chat_models(model_ids: Iterable[str]) -> list[str]:
    """Drop the non-chat models from a provider's model list."""
    return sorted(m for m in model_ids if not _NON_CHAT.search(m))


def model_options(provider: str, discovered: Sequence[str] = (), current: str | None = None) -> list[str]:
    """Discovered models (if any) else the defaults; the configured model is always included first."""
    options = list(discovered) or list(DEFAULT_MODELS.get(provider, []))
    if current and current not in options:
        options.insert(0, current)
    elif current:
        options.remove(current)
        options.insert(0, current)
    return options


def chunk_sections(chunk: dict[str, Any]) -> set[str]:
    return {chunk["section"], *(chunk.get("merged_sections") or [])}


def is_cited(chunk: dict[str, Any], citations: Sequence[dict[str, str]]) -> bool:
    """True if a valid citation of the answer points to this chunk (its section or a merged one)."""
    secs = chunk_sections(chunk)
    return any(c["spec"] == chunk["spec"] and c["section"] in secs for c in citations)


def highlight_citations(answer: str, invalid: Sequence[dict[str, str]]) -> str:
    """Colour citations in the answer: valid ones in the theme colour, invalid ones red and flagged.

    Text is never removed.
    """
    bad = {(c["spec"], c["section"]) for c in invalid}
    out, pos = [], 0
    for start, end, cits in citation_spans(answer):
        out.append(answer[pos:start])
        label = answer[start:end]
        if any((c.spec, c.section) in bad for c in cits):
            out.append(f":red[**{label}**, not in the sources]")
        else:
            out.append(f":primary[**{label}**]")
        pos = end
    out.append(answer[pos:])
    return "".join(out)


def refusal_explanation(layer: str | None, top_score: float | None, threshold: float) -> str:
    if layer == "threshold" and top_score is not None:
        return (f"Refused before calling the LLM: the best reranker score ({top_score:.3f}) is below the "
                f"refusal threshold ({threshold:.2f}), so no passage looks relevant.")
    if layer == "llm":
        return "The LLM found no answer in the retrieved passages (prompt rule)."
    if layer == "no_context":
        return "No passage was retrieved."
    return "Refusal."


def format_latency(latency: dict[str, float]) -> str:
    names = {"retrieval_s": "retrieval", "llm_s": "LLM", "total_s": "total"}
    return " · ".join(f"{names.get(k, k)} {v:.1f} s" for k, v in latency.items())


# --------------------------------------------------------------------------- HTML fragments
# Every value inserted below goes through html.escape, so passage or answer text cannot inject markup.

_FLAG_NAMES = {
    "use_hybrid": "hybrid",
    "use_reranker": "reranker",
    "use_routing": "routing",
    "use_acronym_expansion": "acronyms",
}


def flags_summary(flags: dict[str, Any]) -> str:
    """``hybrid, reranker, routing on · acronyms off`` (only the parts that apply)."""
    on = [_FLAG_NAMES.get(k, k) for k, v in flags.items() if v]
    off = [_FLAG_NAMES.get(k, k) for k, v in flags.items() if not v]
    parts = []
    if on:
        parts.append(", ".join(on) + " on")
    if off:
        parts.append(", ".join(off) + " off")
    return " · ".join(parts)


def chip(text: str, kind: str = "") -> str:
    cls = f"chip {kind}".strip()
    return f'<span class="{cls}">{html.escape(text)}</span>'


def meta_html(specs: Sequence[str], reason: str, top_score: float | None = None,
              total_s: float | None = None) -> str:
    """Route, best score and latency as a row of small labels."""
    parts = [chip(s, "spec") for s in specs]
    if top_score is not None:
        parts.append(chip(f"top score {top_score:.2f}"))
    if total_s is not None:
        parts.append(chip(f"{total_s:.1f} s"))
    if reason:
        parts.append(f'<span class="meta-note">{html.escape(reason)}</span>')
    return f'<div class="meta">{"".join(parts)}</div>'


def notice_html(title: str, body: str, detail: str = "", kind: str = "") -> str:
    """A bordered notice (refusal, warning, missing LLM). ``body`` may be bold as a whole."""
    cls = f"notice {kind}".strip()
    out = (f'<div class="{cls}"><p class="notice-title">{html.escape(title)}</p>'
           f'<p class="notice-body">{html.escape(body)}</p>')
    if detail:
        out += f'<p class="notice-detail">{html.escape(detail)}</p>'
    return out + "</div>"


def question_html(question: str) -> str:
    return f'<p class="q-label">Question</p><p class="q-text">{html.escape(question)}</p>'


def source_label(rank: int, spec: str, section: str, title: str, score: str, cited: bool) -> str:
    """Expander label of a source passage (plain text: Streamlit renders it as markdown)."""
    mark = " · cited" if cited else ""
    return f"{rank} · **{spec} §{section}** {escape_md(title)} · {score}{mark}"
