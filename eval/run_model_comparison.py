"""Model comparison (bonus): generators (API vs local) and embedding models.

Usage:
    python -m eval.run_model_comparison llm          # other generators on the passages of `full`
    python -m eval.run_model_comparison embeddings   # other embedding models, retrieval only
    python -m eval.run_model_comparison report       # rewrite the report from the raw files
    options: [--limit N] [--no-ragas]

**Generators.** Each model in ``evaluation.compare_llms`` answers every question from exactly the
six passages that ``full`` gave the main generator (read from ``generation_raw.jsonl``), with the
same prompt. Retrieval is held fixed, so differences come from the generator only. A question
refused by the reranker threshold (layer 1) is refused for every generator, without a model
call. Metrics are those of the generation evaluation (refusals, citations, RAGAS with the same
judge) plus the generator's latency. Model calls are recorded like the main evaluation's
(``eval/llm_cache.py``), so the comparison replays without the local model.

**Embedding models.** Each model in ``evaluation.compare_embeddings`` gets its own Chroma index
(``data/indexes/chroma_<model>``) and the retrieval evaluation runs on
``evaluation.embedding_compare_configs``. Everything else (chunks, BM25, reranker, router) is
unchanged, so differences come from the embedding model only.

Both comparisons give paired bootstrap 95% CIs of the difference with the main model. Writes
``eval/results/model_comparison.md`` and the raw files ``model_comparison_llm_raw.jsonl`` and
``model_comparison_embeddings_raw.jsonl``.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import re
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

from eval.llm_cache import CachedLLM
from eval.make_tables import paired_delta
from eval.metrics import load_testset, mean, question_group, question_metrics, unverified_ids
from eval.run_generation_eval import aggregate as gen_aggregate
from eval.run_generation_eval import apply_ragas, citation_stats
from eval.run_retrieval_eval import CONFIGS, GROUPS, RetrievalRunner, flag_text, fmt, md_table
from eval.run_retrieval_eval import aggregate as ret_aggregate
from src.config import load_config, resolve_path
from src.generate.answer import clean_answer, is_refusal_text
from src.generate.citation_check import check_citations
from src.generate.llm import LLMUnavailable, RateLimitExhausted, get_llm
from src.generate.prompt import build_messages

logger = logging.getLogger("eval.models")

RESULTS_DIR = "eval/results"
LLM_RAW = "model_comparison_llm_raw.jsonl"
EMB_RAW = "model_comparison_embeddings_raw.jsonl"
REPORT = "model_comparison.md"
REFERENCE_CONFIG = "full"
LAYER1 = ("threshold", "no_context")  # refusals decided before the model is called


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --------------------------------------------------------------------------- generators


def passages(rec: dict[str, Any]) -> list[dict[str, Any]]:
    """The passages of a recorded answer, in the order they were given to the model."""
    return [{"spec": r["spec"], "section": r["section"], "merged_sections": r.get("merged_sections") or [],
             "text_for_index": text} for r, text in zip(rec["retrieved"], rec["contexts"])]


def answer_with(llm: CachedLLM, entry: dict[str, Any], ref: dict[str, Any], refusal: str) -> dict[str, Any]:
    """Answer ``entry`` with ``llm`` from the passages of the reference answer ``ref``."""
    base = {"config": REFERENCE_CONFIG, "id": entry["id"], "llm": llm.name, "contexts": ref["contexts"],
            "retrieved": ref["retrieved"], "routed_specs": ref["routed_specs"], "top_score": ref["top_score"]}
    if ref["is_refusal"] and ref["refusal_layer"] in LAYER1:
        return {**base, "answer": refusal, "is_refusal": True, "refusal_layer": ref["refusal_layer"],
                "citations": [], "invalid_citations": [], "latency": {}}
    chunks = passages(ref)
    text = clean_answer(llm.generate(build_messages(entry["question"], chunks, refusal)), chunks)
    latency = {"llm_s": llm.last_latency}
    if is_refusal_text(text, refusal):
        return {**base, "answer": refusal, "is_refusal": True, "refusal_layer": "llm",
                "citations": [], "invalid_citations": [], "latency": latency}
    report = check_citations(text, chunks)
    return {**base, "answer": text, "is_refusal": False, "refusal_layer": None,
            "citations": [{"spec": c.spec, "section": c.section} for c in report.valid],
            "invalid_citations": [{"spec": c.spec, "section": c.section} for c in report.invalid],
            "latency": latency}


def run_llms(cfg: dict[str, Any], entries: list[dict[str, Any]], use_ragas: bool) -> int:
    ev = cfg["evaluation"]
    out = resolve_path(RESULTS_DIR)
    refs = {r["id"]: r for r in _read_jsonl(out / "generation_raw.jsonl") if r["config"] == REFERENCE_CONFIG}
    if not refs:
        print("generation_raw.jsonl has no `full` answers: run `make eval-generation` first.")
        return 1
    judge = None
    if use_ragas:
        from eval.judge import RagasJudge

        judge = RagasJudge(cfg)
        try:
            judge.check()
        except Exception as exc:
            logger.warning("RAGAS judge %s unavailable (%s); recorded scores only", judge.name, exc)
            judge.exhausted = True

    raw = [r for r in _read_jsonl(out / LLM_RAW) if r["llm"] not in ev.get("compare_llms", [])]
    status = 0
    for spec in ev.get("compare_llms", []):
        provider, _, model = spec.partition(":")
        llm = CachedLLM(get_llm(provider=provider, model=model, cfg=cfg))
        try:
            llm.check()
        except LLMUnavailable as exc:
            logger.error("%s unavailable (%s): skipped", spec, exc)
            status = 3
            continue
        for i, e in enumerate(entries, start=1):
            ref = refs.get(e["id"])
            if ref is None:
                continue
            t0 = time.perf_counter()
            try:
                rec = answer_with(llm, e, ref, cfg["refusal_message"])
            except (RateLimitExhausted, LLMUnavailable) as exc:
                logger.error("Stopping %s: %s", spec, exc)
                status = 3
                break
            rec["ragas"] = apply_ragas(judge, rec, e) if judge else {"faithfulness": None, "answer_correctness": None}
            raw.append(rec)
            logger.info("[%s] %d/%d %s%s (%.1f s)", spec, i, len(entries), e["id"],
                        f" refusal:{rec['refusal_layer']}" if rec["is_refusal"] else "", time.perf_counter() - t0)
        logger.info("%s: %d new model calls, %d replayed", spec, llm.misses, llm.hits)
    _write_jsonl(out / LLM_RAW, raw)
    if judge and judge.missing:
        status = 3
    write_report(cfg, entries)
    return status


# --------------------------------------------------------------------------- embedding models


def embedding_cfg(cfg: dict[str, Any], model: str, query_instruction: str) -> dict[str, Any]:
    c = copy.deepcopy(cfg)
    c["embeddings"]["model"] = model
    c["embeddings"]["query_instruction"] = query_instruction
    c["vector_store"]["path"] = f"{cfg['vector_store']['path']}_{slug(model.split('/')[-1])}"
    return c


def retrieval_rows(runner: RetrievalRunner, config: str, model: str, entries) -> list[dict[str, Any]]:
    rows = []
    for e, rec in zip(entries, runner.run(config, entries)):
        rows.append({"model": model, "config": config, "id": e["id"], "type": question_group(e),
                     "metrics": question_metrics(rec["chunks"], e), "latency_s": rec["latency_s"],
                     "retrieved": [f"{c['spec']} §{c['section']}" for c in rec["chunks"]]})
    return rows


def run_embeddings(cfg: dict[str, Any], entries: list[dict[str, Any]]) -> int:
    from src.index.vector_store import build_index

    ev = cfg["evaluation"]
    configs = ev.get("embedding_compare_configs", ["dense_only", "full"])
    in_corpus = [e for e in entries if e["type"] != "out_of_corpus"]
    models = [(cfg["embeddings"]["model"], cfg["embeddings"].get("query_instruction", ""), cfg)]
    for m in ev.get("compare_embeddings", []):
        models.append((m["model"], m.get("query_instruction", ""),
                       embedding_cfg(cfg, m["model"], m.get("query_instruction", ""))))
    rows, index_s = [], {}
    for model, _, mcfg in models:
        t0 = time.perf_counter()
        build_index(mcfg)  # skipped when the index is up to date (fingerprint: model + chunks)
        index_s[model] = time.perf_counter() - t0
        runner = RetrievalRunner(mcfg)
        for config in configs:
            rows += retrieval_rows(runner, config, model, in_corpus)
        del runner
        gc.collect()
    out = resolve_path(RESULTS_DIR)
    _write_jsonl(out / EMB_RAW, rows)
    write_report(cfg, entries)
    return 0


# --------------------------------------------------------------------------- report


def _delta_cell(d) -> str:
    value, ci, n = d
    if value is None:
        return "-"
    sig = " *" if ci and not (ci[0] <= 0 <= ci[1]) else ""
    return f"{value:+.3f} [{ci[0]:+.2f}, {ci[1]:+.2f}]{sig}" if ci else f"{value:+.3f}"


def _gen_rows(records: list[dict[str, Any]], by_id: dict[str, dict[str, Any]], refusal: str) -> dict[str, dict]:
    rows = {}
    for r in records:
        e = by_id.get(r["id"])
        if e is None:
            continue
        rows[r["id"]] = {**r, "type": question_group(e), "cit": citation_stats(r, e),
                         "exact_refusal": r["answer"] == refusal, "manual": None,
                         "ragas": r.get("ragas") or {"faithfulness": None, "answer_correctness": None}}
    return rows


def _per_question(rows: dict[str, dict], key: str) -> dict[str, float]:
    out = {}
    for qid, r in rows.items():
        if r["type"] == "out_of_corpus":
            continue
        if key == "false_refusal":
            out[qid] = float(r["is_refusal"])
        elif key == "faithfulness":
            if not r["is_refusal"] and r["ragas"]["faithfulness"] is not None:
                out[qid] = r["ragas"]["faithfulness"]
        elif r["ragas"]["answer_correctness"] is not None:
            out[qid] = r["ragas"]["answer_correctness"]
    return out


def llm_section(cfg, entries) -> list[str]:
    out = resolve_path(RESULTS_DIR)
    refusal = cfg["refusal_message"]
    by_id = {e["id"]: e for e in entries}
    ref_records = [r for r in _read_jsonl(out / "generation_raw.jsonl") if r["config"] == REFERENCE_CONFIG]
    other = _read_jsonl(out / LLM_RAW)
    if not ref_records or not other:
        return ["## Generators\n", "Not run yet (`python -m eval.run_model_comparison llm`).\n"]
    gens = {ref_records[0]["llm"]: _gen_rows(ref_records, by_id, refusal)}
    for name in dict.fromkeys(r["llm"] for r in other):
        gens[name] = _gen_rows([r for r in other if r["llm"] == name], by_id, refusal)
    main = next(iter(gens))
    parts = ["## Generators\n",
             f"Every generator answers from the same six passages (configuration `{REFERENCE_CONFIG}`) with the "
             "same prompt; questions refused by the reranker threshold are refused for all of them without a "
             f"model call. RAGAS judge: `{cfg['evaluation']['judge_provider']}:{cfg['evaluation']['judge_model']}`. "
             "LLM latency is the model time per answered question (local model: on this machine's GPU).\n"]
    table = []
    for name, rows in gens.items():
        a = gen_aggregate(list(rows.values()))
        lat = mean(r["latency"].get("llm_s") for r in rows.values() if r["latency"].get("llm_s") is not None)
        table.append([f"`{name}`" + (" (main)" if name == main else ""), f"{len(rows)}/{len(entries)}",
                      fmt(a["refusal_acc"], 2), fmt(a["false_refusal"], 2), fmt(a["with_citation"], 2),
                      fmt(a["cit_validity"], 2), fmt(a["cit_correctness"], 2),
                      fmt(a["faithfulness"], 2) + (f" (n={a['n_faith']})" if a["faithfulness"] is not None else ""),
                      fmt(a["answer_correctness"], 2) + (f" (n={a['n_ac']})" if a["answer_correctness"] is not None else ""),
                      fmt(lat, 1)])
    parts.append(md_table(["Generator", "Answered", "Refusal acc. (OOC)", "False refusals", "With citation",
                           "Cit. validity", "Cit. correctness", "Faithfulness", "Answer corr.", "LLM (s)"], table) + "\n")
    deltas = []
    for name, rows in list(gens.items())[1:]:
        deltas.append([f"`{name}`"] + [_delta_cell(paired_delta(_per_question(gens[main], k), _per_question(rows, k)))
                                       for k in ("false_refusal", "faithfulness", "answer_correctness")])
    parts.append(f"Paired difference with `{main}` (other - main), bootstrap 95% CI; `*` = the CI excludes 0:\n")
    parts.append(md_table(["Generator", "Δ False refusals", "Δ Faithfulness", "Δ Answer corr."], deltas) + "\n")
    return parts


def embedding_section(cfg, entries) -> list[str]:
    rows = _read_jsonl(resolve_path(RESULTS_DIR) / EMB_RAW)
    if not rows:
        return ["## Embedding models\n", "Not run yet (`python -m eval.run_model_comparison embeddings`).\n"]
    main = cfg["embeddings"]["model"]
    models = list(dict.fromkeys(r["model"] for r in rows))
    configs = list(dict.fromkeys(r["config"] for r in rows))
    parts = ["## Embedding models\n",
             "Same chunks, BM25 index, reranker and router; only the embedding model (and its Chroma index) "
             "changes. In-corpus questions; Recall@5 and MRR with bootstrap 95% CIs.\n"]
    for config in configs:
        parts.append(f"### `{config}` ({flag_text(config)})\n")
        table, deltas = [], []
        base = {r["id"]: r for r in rows if r["model"] == main and r["config"] == config}
        for model in models:
            sel = [r for r in rows if r["model"] == model and r["config"] == config]
            a = ret_aggregate([{**r, "group": r["type"]} for r in sel])
            cross = [r for r in sel if r["type"] == "cross_spec"]
            table.append([f"`{model}`" + (" (main)" if model == main else ""), fmt(a["recall@1"]),
                          fmt(a["recall@5"]), f"{a['ci_recall@5'][0]:.2f}–{a['ci_recall@5'][1]:.2f}",
                          fmt(a["recall@10"]), fmt(a["rr"]),
                          fmt(mean(r["metrics"]["recall@5"] for r in cross)), fmt(a["latency_s"], 2)])
            if model != main:
                cur = {r["id"]: r for r in sel}
                deltas.append([f"`{model}`"] + [
                    _delta_cell(paired_delta({i: base[i]["metrics"][k] for i in base},
                                             {i: cur[i]["metrics"][k] for i in cur}))
                    for k in ("recall@5", "rr")])
        parts.append(md_table(["Model", "R@1", "R@5", "R@5 95% CI", "R@10", "MRR", "R@5 cross-spec",
                               "Latency (s)"], table) + "\n")
        parts.append(md_table(["Model", "Δ Recall@5 vs main", "Δ MRR vs main"], deltas) + "\n")
    return parts


def write_report(cfg: dict[str, Any], entries: list[dict[str, Any]]) -> None:
    bad = unverified_ids(entries)
    parts = ["# Model comparison\n",
             f"Generated by `eval/run_model_comparison.py` on {date.today().isoformat()}.\n",
             (f"> **WARNING: {len(bad)}/{len(entries)} test-set entries are not verified;** these results are "
              "provisional.\n") if bad else ""]
    parts += llm_section(cfg, entries) + embedding_section(cfg, entries)
    path = resolve_path(RESULTS_DIR) / REPORT
    path.write_text("\n".join(p for p in parts if p), encoding="utf-8")
    print(f"Wrote {path}")


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Model comparison: generators and embedding models")
    parser.add_argument("what", choices=["llm", "embeddings", "report"])
    parser.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    parser.add_argument("--no-ragas", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("httpx", "sentence_transformers", "chromadb", "urllib3", "openai", "src.generate.citation_check"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    entries = load_testset()
    if args.limit:
        entries = entries[: args.limit]
    if args.what == "llm":
        return run_llms(cfg, entries, use_ragas=not args.no_ragas)
    if args.what == "embeddings":
        return run_embeddings(cfg, entries)
    write_report(cfg, entries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
