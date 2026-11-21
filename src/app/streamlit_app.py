"""Streamlit chat interface for the 3GPP 5G RAG assistant.

Run with ``make app`` (``streamlit run src/app/streamlit_app.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # streamlit puts only the script folder on sys.path
    sys.path.insert(0, str(ROOT))

import os  # noqa: E402
from typing import Any  # noqa: E402

import streamlit as st  # noqa: E402

from src.app.ui_helpers import (  # noqa: E402
    SPEC_FILTERS,
    chat_models,
    escape_dollars,
    escape_md,
    format_latency,
    highlight_citations,
    is_cited,
    model_options,
    refusal_explanation,
    table_markdown,
)
from src.config import load_config  # noqa: E402
from src.generate.answer import AnswerResult, Answerer  # noqa: E402
from src.generate.llm import PROVIDERS, LLM, LLMUnavailable, get_llm  # noqa: E402
from src.retrieve.pipeline import Pipeline, RetrievalResult  # noqa: E402

st.set_page_config(page_title="5G Specs Assistant", page_icon="📡", layout="wide")

CFG = load_config()
EXAMPLES = [
    "What does cause #9 mean in 5GMM?",
    "What is the role of the NRF?",
    "What happens when timer T3510 expires?",
    "How does the AMF handle a UE registration request and which network functions are involved?",
]


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


# --------------------------------------------------------------------------- sidebar


def sidebar() -> dict[str, Any]:
    r = CFG["retrieval"]
    with st.sidebar:
        st.header("Settings")

        st.subheader("Retrieval")
        use_hybrid = st.toggle("Hybrid search (dense + BM25)", value=r["use_hybrid"],
                               help="Off: dense (embedding) search only.")
        use_reranker = st.toggle("Reranker (cross-encoder)", value=r["use_reranker"],
                                 help="Off: no reranking and no score-based refusal (layer 1).")
        use_routing = st.toggle("Routing to the right spec", value=r["use_routing"],
                                help="Off: one search over the whole corpus.")
        use_acronyms = st.toggle("Acronym expansion", value=r["use_acronym_expansion"],
                                 help="Adds the full names of acronyms (clause 3.2) to the search query.")
        spec_label = st.radio("Spec filter", list(SPEC_FILTERS), index=0,
                              help="A manual filter overrides the router.")
        top_k = st.slider("Passages given to the LLM (top-k)", 1, 15, r["final_k"])

        st.subheader("LLM")
        providers = list(PROVIDERS)
        provider = st.selectbox("Provider", providers, index=providers.index(CFG["llm"]["provider"]),
                                format_func=lambda p: f"{p} (local)" if p == "ollama" else f"{p} (API)")
        current = CFG["llm"]["model"] if provider == CFG["llm"]["provider"] else None
        options = model_options(provider, discover_models(provider), current)
        model = st.selectbox("Model", options + ["Other..."])
        if model == "Other...":
            model = st.text_input("Model name", value=options[0] if options else "")

        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.history = []
            st.rerun()
        specs = CFG["corpus"]["specs"]
        corpus = ", ".join(f"{s['id']} v{s['version']}" for s in specs)
        st.caption(f"Corpus: {corpus} ({specs[0]['release']})")

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
    st.markdown(f"**Sources** ({len(chunks)} passages)")
    for cand in chunks:
        ch = cand.chunk
        cited = is_cited(ch, citations)
        score = (f"reranker {cand.rerank_score:.3f}" if cand.rerank_score is not None
                 else f"RRF {cand.fused_score:.4f}")
        mark = "✅ cited · " if cited else ""
        label = f"[{cand.final_rank}] {mark}{ch['spec']} §{ch['section']} – {ch['title']} · {score}"
        with st.expander(label):
            meta = [
                f"**Spec:** {ch['spec']} v{ch.get('version', '')} ({ch.get('release', '')})",
                f"**Section:** §{ch['section']}",
                f"**Title:** {escape_md(ch['title'])}",
                f"**Type:** {ch.get('chunk_type', 'text')}",
            ]
            merged = [s for s in ch.get("merged_sections") or [] if s != ch["section"]]
            if merged:
                meta.append(f"**Also covers:** §{', §'.join(merged)}")
            st.markdown(" · ".join(meta))
            ranks = []
            if cand.dense_rank:
                ranks.append(f"dense #{cand.dense_rank}")
            if cand.bm25_rank:
                ranks.append(f"BM25 #{cand.bm25_rank}")
            if cand.rerank_score is not None:
                ranks.append(f"reranker {cand.rerank_score:.3f}")
            if ch.get("section_path"):
                ranks.append("path: " + " > ".join(ch["section_path"]) if isinstance(ch["section_path"], list)
                             else f"path: {ch['section_path']}")
            st.caption(" · ".join(ranks))
            if ch.get("chunk_type") == "table":
                st.markdown(table_markdown(ch["text"]))
            else:
                st.markdown(escape_md(ch["text"]))


def render_answer(res: AnswerResult) -> None:
    st.caption(f"🧭 {res.retrieval.route_label if res.retrieval else res.route_reason}")
    if res.is_refusal:
        st.error(f"**{res.answer}**  \n{refusal_explanation(res.refusal_layer, res.top_score, CFG['reranker']['refusal_threshold'])}",
                 icon="🚫")
    else:
        st.markdown(highlight_citations(escape_dollars(res.answer), res.invalid_citations))
        if res.invalid_citations:
            labels = ", ".join(f"({c['spec']}, §{c['section']})" for c in res.invalid_citations)
            st.warning(f"Invalid citations detected (not in the retrieved passages): {labels}", icon="⚠️")
        elif not res.citations:
            st.info("This answer contains no citation; check it against the sources below.", icon="ℹ️")
    flags = ", ".join(k.replace("use_", "").replace("_", " ") + (" on" if v else " off")
                      for k, v in res.flags.items())
    top = f" · top reranker score {res.top_score:.3f}" if res.top_score is not None else ""
    st.caption(f"{res.llm} · {flags}{top} · {format_latency(res.latency)}")
    render_sources(res.chunks, res.citations)


def render_retrieval_only(ret: RetrievalResult, error: str) -> None:
    st.caption(f"🧭 {ret.route_label}")
    st.error(f"LLM not available: {error}  \nShowing the retrieved passages only.", icon="🔌")
    render_sources(ret.results, [])


def render_message(msg: dict[str, Any]) -> None:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(escape_md(msg["content"]))
        elif "result" in msg:
            render_answer(msg["result"])
        elif "retrieval" in msg:
            render_retrieval_only(msg["retrieval"], msg["error"])
        else:
            st.error(msg["content"])


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
    settings = sidebar()
    st.title("📡 5G Specs Assistant")
    st.caption("Answers from 3GPP TS 23.501 and TS 24.501 (Release 18) only, with section citations. "
               "Questions in any language; answers in English.")

    load_pipeline()
    st.session_state.setdefault("history", [])
    for msg in st.session_state.history:
        render_message(msg)

    pending = None
    if not st.session_state.history:
        st.markdown("**Try an example:**")
        cols = st.columns(len(EXAMPLES))
        for col, ex in zip(cols, EXAMPLES):
            if col.button(ex, use_container_width=True):
                pending = ex
    question = st.chat_input("Ask a question about TS 23.501 / TS 24.501") or pending
    if not question:
        return

    user_msg = {"role": "user", "content": question}
    st.session_state.history.append(user_msg)
    render_message(user_msg)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving passages and generating the answer..."):
            reply = ask(question, settings)
    st.session_state.history.append(reply)
    st.rerun()


main()
