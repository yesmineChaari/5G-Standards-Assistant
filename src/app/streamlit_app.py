"""Streamlit chat interface for the 3GPP 5G RAG assistant.

Run with ``make app`` (``streamlit run src/app/streamlit_app.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # streamlit puts only the script folder on sys.path
    sys.path.insert(0, str(ROOT))

import html  # noqa: E402
import os  # noqa: E402
from typing import Any  # noqa: E402

import streamlit as st  # noqa: E402

from src.app.ui_helpers import (  # noqa: E402
    SPEC_FILTERS,
    chat_models,
    escape_dollars,
    escape_md,
    flags_summary,
    format_latency,
    highlight_citations,
    is_cited,
    meta_html,
    model_options,
    normalize_display,
    notice_html,
    question_html,
    refusal_explanation,
    source_label,
    table_markdown,
)
from src.config import load_config  # noqa: E402
from src.generate.answer import AnswerResult, Answerer  # noqa: E402
from src.generate.llm import PROVIDERS, LLM, LLMUnavailable, get_llm  # noqa: E402
from src.retrieve.pipeline import Pipeline, RetrievalResult  # noqa: E402

st.set_page_config(page_title="5G Standards Assistant", page_icon=":material/cell_tower:", layout="centered")

CFG = load_config()
STYLE = (Path(__file__).parent / "style.css").read_text(encoding="utf-8")
EXAMPLES = [
    "What does cause #9 mean in 5GMM?",
    "What is the role of the NRF?",
    "What happens when timer T3510 expires?",
    "How does the AMF handle a UE registration request and which network functions are involved?",
]
PROVIDER_NAMES = {"groq": "Groq", "gemini": "Gemini", "openai": "OpenAI", "ollama": "Ollama"}
OTHER_MODEL = "Other model..."


def html_block(fragment: str) -> None:
    st.markdown(fragment, unsafe_allow_html=True)


# --------------------------------------------------------------------------- cached resources


@st.cache_resource(show_spinner="Loading indexes and models (first run only)...")
def load_pipeline() -> Pipeline:
    pipeline = Pipeline(CFG)
    _ = pipeline.vector_store.embedder.model  # load both models now, not on the first question
    _ = pipeline.reranker.model
    return pipeline


@st.cache_resource(show_spinner=False)
def load_llm(provider: str, model: str) -> LLM:
    return get_llm(provider, model, CFG)


@st.cache_resource(show_spinner=False)
def load_answerer() -> Answerer:
    # The LLM is passed per question; this one only fills the constructor.
    return Answerer(CFG, llm=load_llm(CFG["llm"]["provider"], CFG["llm"]["model"]), pipeline=load_pipeline())


@st.cache_data(ttl=3600, show_spinner=False)
def discover_models(provider: str) -> list[str]:
    """Models offered by the Groq key or the local Ollama server ([] if they cannot be listed)."""
    try:
        if provider == "groq":
            from groq import Groq

            from src.config import get_api_key

            key = get_api_key("groq")
            return chat_models(m.id for m in Groq(api_key=key).models.list().data) if key else []
        if provider == "ollama":
            import ollama

            client = ollama.Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
            return sorted(m.model for m in client.list().models)
    except Exception:  # offline, no key, server not running
        return []
    return []


# --------------------------------------------------------------------------- layout


def header() -> None:
    specs = CFG["corpus"]["specs"]
    eyebrow = f"3GPP {specs[0]['release']} · " + " · ".join(s["id"] for s in specs)
    html_block(
        '<div class="app-head">'
        f'<p class="app-eyebrow">{html.escape(eyebrow)}</p>'
        '<p class="app-title">5G Standards Assistant</p>'
        '<p class="app-sub">Answers drawn only from the 5G system architecture (TS 23.501) and the NAS '
        'protocol (TS 24.501), with the section behind every statement. When the documents do not '
        'hold the answer, it says so. Questions in any language, answers in English.</p>'
        "</div>"
    )


def sidebar() -> dict[str, Any]:
    r = CFG["retrieval"]
    with st.sidebar:
        html_block('<p class="side-title">Settings</p>')

        html_block('<p class="side-label first">Retrieval</p>')
        use_hybrid = st.toggle("Hybrid search (dense + BM25)", value=r["use_hybrid"],
                               help="Off: dense (embedding) search only.")
        use_reranker = st.toggle("Reranker (cross-encoder)", value=r["use_reranker"],
                                 help="Off: no reranking and no score-based refusal (layer 1).")
        use_routing = st.toggle("Routing to the right spec", value=r["use_routing"],
                                help="Off: one search over the whole corpus.")
        use_acronyms = st.toggle("Acronym expansion", value=r["use_acronym_expansion"],
                                 help="Adds the full names of acronyms (clause 3.2) to the search query.")

        html_block('<p class="side-label">Scope</p>')
        spec_label = st.radio("Specifications", list(SPEC_FILTERS), index=0,
                              help="A manual filter overrides the router.")
        top_k = st.slider("Passages given to the model", 1, 15, r["final_k"])

        html_block('<p class="side-label">Language model</p>')
        providers = list(PROVIDERS)
        provider = st.selectbox(
            "Provider", providers, index=providers.index(CFG["llm"]["provider"]),
            format_func=lambda p: PROVIDER_NAMES.get(p, p) + (" · local" if p == "ollama" else " · API"),
        )
        current = CFG["llm"]["model"] if provider == CFG["llm"]["provider"] else None
        options = model_options(provider, discover_models(provider), current)
        model = st.selectbox("Model", options + [OTHER_MODEL])
        if model == OTHER_MODEL:
            model = st.text_input("Model name", value=options[0] if options else "")

        html_block('<p class="side-label">Corpus</p>')
        lines = "<br>".join(
            f"<b>{html.escape(s['id'])}</b> v{html.escape(str(s['version']))} · {html.escape(str(s['release']))}"
            for s in CFG["corpus"]["specs"]
        )
        html_block(f'<div class="corpus">{lines}</div>')
        st.write("")
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    return {
        "flags": {"use_hybrid": use_hybrid, "use_reranker": use_reranker, "use_routing": use_routing,
                  "use_acronym_expansion": use_acronyms},
        "specs": SPEC_FILTERS[spec_label],
        "spec_label": spec_label,
        "top_k": top_k,
        "provider": provider,
        "model": model,
    }


# --------------------------------------------------------------------------- rendering


def render_sources(chunks: list, citations: list[dict[str, str]]) -> None:
    n_cited = sum(is_cited(c.chunk, citations) for c in chunks)
    count = f"{len(chunks)} passages" + (f" · {n_cited} cited" if citations else "")
    html_block(f'<div class="sources-head"><span class="section-label">Sources</span>'
               f'<span class="meta-note">{count}</span></div>')
    for cand in chunks:
        ch = cand.chunk
        score = (f"{cand.rerank_score:.3f}" if cand.rerank_score is not None
                 else f"RRF {cand.fused_score:.4f}")
        label = source_label(cand.final_rank, ch["spec"], ch["section"], ch["title"], score,
                             is_cited(ch, citations))
        with st.expander(label):
            meta = [f"{ch['spec']} v{ch.get('version', '')} ({ch.get('release', '')})",
                    f"§{ch['section']}", ch.get("chunk_type", "text")]
            merged = [s for s in ch.get("merged_sections") or [] if s != ch["section"]]
            if merged:
                meta.append("also covers §" + ", §".join(merged))
            ranks = []
            if cand.dense_rank:
                ranks.append(f"dense #{cand.dense_rank}")
            if cand.bm25_rank:
                ranks.append(f"BM25 #{cand.bm25_rank}")
            if cand.rerank_score is not None:
                ranks.append(f"reranker {cand.rerank_score:.3f}")
            path = ch.get("section_path")
            if path:
                ranks.append(" > ".join(path) if isinstance(path, list) else str(path))
            html_block('<p class="src-meta">' + html.escape(" · ".join(meta)) + "<br>"
                       + html.escape(" · ".join(ranks)) + "</p>")
            if ch.get("chunk_type") == "table":
                st.markdown(table_markdown(ch["text"]))
            else:
                st.markdown(escape_md(ch["text"]))


def render_answer(res: AnswerResult) -> None:
    html_block('<p class="a-label">Answer</p>')
    html_block(meta_html(res.routed_specs, res.route_reason, res.top_score, res.latency.get("total_s")))
    if res.is_refusal:
        detail = refusal_explanation(res.refusal_layer, res.top_score, CFG["reranker"]["refusal_threshold"])
        html_block(notice_html("Not in the documents", res.answer, detail))
    else:
        st.markdown(highlight_citations(escape_dollars(normalize_display(res.answer)), res.invalid_citations))
        if res.invalid_citations:
            labels = ", ".join(f"({c['spec']}, §{c['section']})" for c in res.invalid_citations)
            html_block(notice_html("Citation check",
                                   f"Not among the retrieved passages: {labels}.",
                                   "Verify these statements against the specification.", kind="warn"))
        elif not res.citations:
            html_block(notice_html("No citation",
                                   "This answer cites no section. Check it against the sources below."))
    run = f"{res.llm} · {flags_summary(res.flags)} · {format_latency(res.latency)}"
    html_block(f'<p class="run-info">{html.escape(run)}</p>')
    render_sources(res.chunks, res.citations)


def render_retrieval_only(ret: RetrievalResult, error: str) -> None:
    html_block('<p class="a-label">Retrieved passages</p>')
    html_block(meta_html(ret.routed_specs, ret.route_reason, ret.top_score))
    html_block(notice_html("Language model unavailable", error,
                           "Showing the retrieved passages only.", kind="warn"))
    render_sources(ret.results, [])


def render_message(msg: dict[str, Any]) -> None:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            html_block(question_html(msg["content"]))
        elif "result" in msg:
            render_answer(msg["result"])
        elif "retrieval" in msg:
            render_retrieval_only(msg["retrieval"], msg["error"])
        else:
            html_block(notice_html("Request failed", msg["content"], kind="warn"))


def examples() -> str | None:
    html_block('<p class="section-label">Example questions</p>')
    picked = None
    with st.container(key="examples"):
        for row in range(0, len(EXAMPLES), 2):
            cols = st.columns(2)
            for col, ex in zip(cols, EXAMPLES[row:row + 2]):
                if col.button(ex, use_container_width=True):
                    picked = ex
    return picked


# --------------------------------------------------------------------------- main


def ask(question: str, settings: dict[str, Any]) -> dict[str, Any]:
    answerer = load_answerer()
    llm = load_llm(settings["provider"], settings["model"])
    kwargs = dict(**settings["flags"], specs=settings["specs"], final_k=settings["top_k"])
    try:
        llm.check()
    except LLMUnavailable as exc:
        answerer.pipeline.router.llm_classifier = None
        ret = answerer.pipeline.retrieve(question, **kwargs)
        return {"role": "assistant", "retrieval": ret, "error": str(exc)}
    answerer.pipeline.router.llm_classifier = llm.classify  # routing ties go to the selected LLM
    try:
        res = answerer.answer(question, llm=llm, **kwargs)
    except LLMUnavailable as exc:  # e.g. Ollama model not pulled
        answerer.pipeline.router.llm_classifier = None
        return {"role": "assistant", "retrieval": answerer.pipeline.retrieve(question, **kwargs), "error": str(exc)}
    except Exception as exc:  # rate limit, network, wrong model name
        return {"role": "assistant", "content": f"The LLM call failed: {exc}"}
    return {"role": "assistant", "result": res}


def main() -> None:
    html_block(f"<style>{STYLE}</style>")
    settings = sidebar()
    header()

    load_pipeline()
    st.session_state.setdefault("history", [])
    for msg in st.session_state.history:
        render_message(msg)

    pending = examples() if not st.session_state.history else None
    question = st.chat_input("Ask about TS 23.501 or TS 24.501") or pending
    if not question:
        return

    user_msg = {"role": "user", "content": question}
    st.session_state.history.append(user_msg)
    render_message(user_msg)
    with st.chat_message("assistant"):
        with st.spinner("Searching the specifications..."):
            reply = ask(question, settings)
    st.session_state.history.append(reply)
    st.rerun()


main()
