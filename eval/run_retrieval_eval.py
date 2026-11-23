"""Retrieval evaluation: Recall@k, Hit@k, MRR per configuration, per question type, and routing accuracy.

Usage:
    python -m eval.run_retrieval_eval [--configs full dense_only ...] [--limit N] [--no-cache]
                                      [--no-llm-routing]

Configurations (ablations, plan section 7.2). Acronym expansion is on in every configuration (the
system default) except ``full_no_acronyms``, an extra ablation of that component.

    name               hybrid  reranker  routing
    dense_only         no      no        no
    bm25_only          (BM25 alone)      no
    hybrid             yes     no        no
    hybrid_rerank      yes     yes       no
    full               yes     yes       yes
    full_no_hybrid     no      yes       yes
    full_no_rerank     yes     no        yes
    full_no_acronyms   yes     yes       yes   (acronym expansion off)

Every configuration returns its top 10 chunks; metrics are computed on in-corpus questions only.
Retrieval outputs are cached in ``eval/cache/retrieval/`` (keyed by question, configuration, chunks
file and retrieval settings), so re-runs only recompute what changed.

Writes ``eval/results/retrieval_results.md``, ``routing_results.md`` and ``retrieval_raw.jsonl``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

from eval.metrics import (
    K_VALUES,
    bootstrap_ci,
    load_testset,
    mean,
    question_group,
    question_metrics,
    route_scores,
    unverified_ids,
)
from src.config import load_config, resolve_path
from src.index.common import file_sha256

logger = logging.getLogger("eval.retrieval")

CONFIGS: dict[str, dict[str, Any]] = {
    "dense_only": {"use_hybrid": False, "use_reranker": False, "use_routing": False},
    "bm25_only": {"bm25_only": True, "use_routing": False},
    "hybrid": {"use_hybrid": True, "use_reranker": False, "use_routing": False},
    "hybrid_rerank": {"use_hybrid": True, "use_reranker": True, "use_routing": False},
    "full": {"use_hybrid": True, "use_reranker": True, "use_routing": True},
    "full_no_hybrid": {"use_hybrid": False, "use_reranker": True, "use_routing": True},
    "full_no_rerank": {"use_hybrid": True, "use_reranker": False, "use_routing": True},
    "full_no_acronyms": {"use_hybrid": True, "use_reranker": True, "use_routing": True,
                         "use_acronym_expansion": False},
}
GROUPS = [("all", "All in-corpus"), ("simple_23501", "Simple TS 23.501"),
          ("simple_24501", "Simple TS 24.501"), ("cross_spec", "Cross-spec")]
K_MAX = 10
RESULTS_DIR = "eval/results"
CACHE_DIR = "eval/cache/retrieval"


RETRIEVAL_CODE = ["src/retrieve/router.py", "src/retrieve/pipeline.py", "src/retrieve/hybrid.py",
                  "src/retrieve/reranker.py", "src/retrieve/acronyms.py", "src/index/bm25_index.py",
                  "src/index/vector_store.py", "src/index/embed.py"]


def retrieval_code_hash() -> str:
    """Changes when the retrieval code changes, so cached results are not reused after a code change."""
    h = hashlib.sha256()
    for rel in RETRIEVAL_CODE:
        h.update(resolve_path(rel).read_bytes())
    return h.hexdigest()[:16]


def flag_text(name: str) -> str:
    c = CONFIGS[name]
    if c.get("bm25_only"):
        return "BM25 only"
    parts = [("hybrid" if c["use_hybrid"] else "dense"), ("rerank" if c["use_reranker"] else "no rerank"),
             ("routing" if c["use_routing"] else "no routing")]
    if c.get("use_acronym_expansion") is False:
        parts.append("no acronyms")
    return ", ".join(parts)


# --------------------------------------------------------------------------- running


class RetrievalRunner:
    def __init__(self, cfg: dict[str, Any], use_cache: bool = True, llm_routing: bool = True) -> None:
        from src.retrieve.pipeline import Pipeline

        self.cfg = cfg
        classifier, self.router_llm = None, None
        if llm_routing:
            try:  # recorded calls (shared with the generation evaluation) are replayed without a key
                from eval.llm_cache import CachedLLM
                from src.generate.llm import get_llm

                llm = CachedLLM(get_llm(cfg=cfg))
                llm.check()
                classifier, self.router_llm = llm.classify, llm.name
            except Exception as exc:  # no key and nothing recorded: ties search both specs
                logger.warning("No LLM for routing ties (%s); ties search both specs", exc)
        self.pipeline = Pipeline(cfg, llm_classifier=classifier)
        self.use_cache = use_cache
        self.cache_dir = resolve_path(CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        settings = {k: cfg[k] for k in ("retrieval", "reranker", "embeddings", "bm25", "chunking")}
        self.base_key = json.dumps(
            {"chunks": file_sha256(resolve_path(cfg["paths"]["chunks_file"])), "settings": settings,
             "router_llm": self.router_llm, "k": K_MAX, "code": retrieval_code_hash()},
            sort_keys=True,
        )

    def _key(self, config: str, question: str) -> str:
        raw = json.dumps({"base": self.base_key, "config": CONFIGS[config], "q": question}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:20]

    def _cache_file(self, config: str) -> Path:
        return self.cache_dir / f"{config}.jsonl"

    def load_cache(self, config: str) -> dict[str, dict[str, Any]]:
        path = self._cache_file(config)
        if not self.use_cache or not path.exists():
            return {}
        return {r["key"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if r}

    def retrieve(self, config: str, question: str) -> dict[str, Any]:
        c = CONFIGS[config]
        if c.get("bm25_only"):
            query, _ = self.pipeline.expander.expand(question)
            t0 = time.perf_counter()
            hits = self.pipeline.bm25.query(query, k=K_MAX)
            elapsed = time.perf_counter() - t0
            chunks = [h.chunk for h in hits]
            scores = [h.score for h in hits]
            routed, reason, method = ["TS 23.501", "TS 24.501"], "routing off: whole corpus", None
        else:
            res = self.pipeline.retrieve(
                question, use_hybrid=c["use_hybrid"], use_reranker=c["use_reranker"],
                use_routing=c["use_routing"], use_acronym_expansion=c.get("use_acronym_expansion"),
                final_k=K_MAX,
            )
            elapsed = sum(res.timings.values())
            chunks = [cand.chunk for cand in res.results]
            scores = [cand.rerank_score if cand.rerank_score is not None else cand.fused_score
                      for cand in res.results]
            routed, reason = res.routed_specs, res.route_reason
            method = res.route.method if res.route else None
        return {
            "chunks": [{k: ch[k] for k in ("chunk_id", "spec", "section", "title")}
                       | {"merged_sections": ch.get("merged_sections") or []} for ch in chunks],
            "scores": scores,
            "routed_specs": routed,
            "route_reason": reason,
            "route_method": method,
            "latency_s": elapsed,
        }

    def run(self, config: str, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cache = self.load_cache(config)
        out, new = [], []
        for i, e in enumerate(entries, start=1):
            key = self._key(config, e["question"])
            if key in cache:
                rec = cache[key]
            else:
                rec = {"key": key, "id": e["id"], **self.retrieve(config, e["question"])}
                new.append(rec)
                logger.info("[%s] %d/%d %s (%.1f s)", config, i, len(entries), e["id"], rec["latency_s"])
            out.append(rec)
        if new and self.use_cache:
            with self._cache_file(config).open("a", encoding="utf-8") as f:
                for rec in new:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info("[%s] done: %d computed, %d from cache", config, len(new), len(entries) - len(new))
        return out


# --------------------------------------------------------------------------- aggregation


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [f"recall@{k}" for k in K_VALUES] + [f"hit@{k}" for k in K_VALUES] + ["rr"]
    agg = {k: mean(r["metrics"][k] for r in rows) for k in keys}
    cross = [r for r in rows if r["group"] == "cross_spec"]
    for k in ("both_specs@5", "both_specs@10", "both_relevant@5", "both_relevant@10"):
        agg[k] = mean(r["metrics"][k] for r in cross) if cross else None
    agg["latency_s"] = mean(r["latency_s"] for r in rows)
    agg["n"] = len(rows)
    agg["ci_recall@5"] = bootstrap_ci([r["metrics"]["recall@5"] for r in rows])
    agg["ci_rr"] = bootstrap_ci([r["metrics"]["rr"] for r in rows])
    return agg


def fmt(v: Any, digits: int = 3) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def fmt_ci(ci: tuple[float, float] | None) -> str:
    return f"{ci[0]:.2f}–{ci[1]:.2f}" if ci else "-"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def warning_block(entries: list[dict[str, Any]]) -> str:
    bad = unverified_ids(entries)
    if not bad:
        return "All test-set entries are verified.\n"
    return (f"> **WARNING: {len(bad)}/{len(entries)} test-set entries are not verified.** Gold sections "
            "were proposed with `scripts/find_sections.py` and have not yet been checked by a person; "
            "these results are provisional.\n")


def write_retrieval_md(path: Path, entries, configs, per_config, cfg, router_llm) -> None:
    in_corpus = [e for e in entries if e["type"] != "out_of_corpus"]
    parts = [
        "# Retrieval evaluation\n",
        f"Generated by `eval/run_retrieval_eval.py` on {date.today().isoformat()}.\n",
        warning_block(entries),
        f"- Questions: {len(in_corpus)} in-corpus (out-of-corpus questions have no gold sections and are "
        "excluded from retrieval metrics).",
        f"- Every configuration returns its top {K_MAX} chunks; reranked configurations rerank the top "
        f"{cfg['retrieval']['rerank_candidates']} fused candidates with `{cfg['reranker']['fallback_model'] if cfg['reranker'].get('use_fallback_on_cpu') else cfg['reranker']['model']}`.",
        f"- Acronym expansion is on in every configuration except `full_no_acronyms`. Routing ties: "
        f"{router_llm or 'no LLM (both specs searched)'}.",
        "- **Matching rule:** a retrieved chunk is relevant if it has the same spec as a gold section and its "
        "section equals the gold section, is a sub-section of it (`5.5.1.2.2` matches gold `5.5.1.2`), "
        "or the gold section is in its `merged_sections`.",
        "- **Recall@k:** fraction of a question's gold sections found in the top k, averaged over questions. "
        "**Hit@k:** at least one gold section in the top k. **MRR:** mean of 1/rank of the first relevant "
        "chunk (0 if none in the top 10).",
        "- Cross-spec only: **both specs@k** = chunks from both specs in the top k; **both relevant@k** = a "
        "relevant chunk from each spec in the top k.",
        f"- **Sample size:** with {len(in_corpus)} questions, one question moves an overall Hit@k by "
        f"{1 / max(1, len(in_corpus)):.3f} (0.1 in a 10-question group, 0.125 in the cross-spec group). "
        "95% confidence intervals: percentile bootstrap over questions (2000 resamples, seed 42). Where the "
        "intervals overlap, the difference between configurations is not significant.",
        "- **Latency** is the retrieval time per question on a laptop CPU (no GPU), dominated by the reranker; "
        "it depends on the machine's load and varied between runs of the same code (e.g. 14 s to 45 s per "
        "reranked query), so it is indicative only. Cached results keep the latency of the run that computed them.\n",
        "## Overall (all in-corpus questions)\n",
    ]
    headers = ["Config", "Components", "R@1", "R@3", "R@5", "R@5 95% CI", "R@10", "Hit@1", "Hit@5", "MRR",
               "MRR 95% CI", "Latency (s)"]
    rows = []
    for name in configs:
        a = per_config[name]["agg"]["all"]
        rows.append([f"`{name}`", flag_text(name), fmt(a["recall@1"]), fmt(a["recall@3"]), fmt(a["recall@5"]),
                     fmt_ci(a["ci_recall@5"]), fmt(a["recall@10"]), fmt(a["hit@1"]), fmt(a["hit@5"]),
                     fmt(a["rr"]), fmt_ci(a["ci_rr"]), fmt(a["latency_s"], 2)])
    parts.append(md_table(headers, rows) + "\n")

    for group, title in GROUPS[1:]:
        n = per_config[configs[0]]["agg"][group]["n"]
        parts.append(f"## {title} ({n} questions)\n")
        headers = ["Config", "R@1", "R@3", "R@5", "R@10", "Hit@5", "MRR"]
        if group == "cross_spec":
            headers += ["Both specs@5", "Both specs@10", "Both relevant@5", "Both relevant@10"]
        rows = []
        for name in configs:
            a = per_config[name]["agg"][group]
            row = [f"`{name}`", fmt(a["recall@1"]), fmt(a["recall@3"]), fmt(a["recall@5"]), fmt(a["recall@10"]),
                   fmt(a["hit@5"]), fmt(a["rr"])]
            if group == "cross_spec":
                row += [fmt(a["both_specs@5"]), fmt(a["both_specs@10"]), fmt(a["both_relevant@5"]),
                        fmt(a["both_relevant@10"])]
            rows.append(row)
        parts.append(md_table(headers, rows) + "\n")

    parts.append("## Per question: rank of the first relevant chunk (top 10, `-` = not found)\n")
    headers = ["Question", "Type"] + [f"`{c}`" for c in configs]
    rows = []
    for e in in_corpus:
        row = [e["id"], question_group(e)]
        for name in configs:
            r = per_config[name]["rows_by_id"][e["id"]]
            row.append(fmt(r["metrics"]["first_rank"]))
        rows.append(row)
    parts.append(md_table(headers, rows) + "\n")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_routing_md(path: Path, entries, records, router_llm) -> None:
    in_corpus = [e for e in entries if e["type"] != "out_of_corpus"]
    scored = [(e, records[e["id"]], route_scores(records[e["id"]]["routed_specs"], e["expected_spec"]))
              for e in in_corpus]
    parts = ["# Routing evaluation\n",
             f"Generated by `eval/run_retrieval_eval.py` on {date.today().isoformat()}.\n",
             warning_block(entries),
             "Router output (configuration `full`) compared with `expected_spec`. **Exact:** same set of specs. "
             "**Covers:** every expected spec was searched (routing a simple question to both specs loses no "
             f"recall, only some precision). Routing ties: {router_llm or 'no LLM (both specs searched)'}.\n",
             "## Accuracy\n"]
    rows = []
    for group, title in GROUPS:
        sel = [s for s in scored if group == "all" or question_group(s[0]) == group]
        rows.append([title, str(len(sel)), fmt(mean(float(s[2]["exact"]) for s in sel)),
                     fmt(mean(float(s[2]["covers"]) for s in sel))])
    parts.append(md_table(["Questions", "n", "Exact", "Covers"], rows) + "\n")
    methods: dict[str, int] = {}
    for e in entries:
        m = records[e["id"]].get("route_method") or "-"
        methods[m] = methods.get(m, 0) + 1
    parts.append("Decision method counts (all 31 questions): "
                 + ", ".join(f"{k}: {v}" for k, v in sorted(methods.items())) + "\n")
    parts.append("## Per question\n")
    rows = []
    for e in entries:
        rec = records[e["id"]]
        if e["type"] == "out_of_corpus":
            verdict = "n/a (out of corpus)"
        else:
            s = route_scores(rec["routed_specs"], e["expected_spec"])
            verdict = "exact" if s["exact"] else ("covers" if s["covers"] else "**wrong**")
        rows.append([e["id"], " + ".join(e["expected_spec"]) or "-", " + ".join(rec["routed_specs"]),
                     rec.get("route_method") or "-", rec["route_reason"].replace("|", "/"), verdict])
    parts.append(md_table(["Question", "Expected", "Routed", "Method", "Reason", "Result"], rows) + "\n")
    path.write_text("\n".join(parts), encoding="utf-8")


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retrieval evaluation")
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--testset", default=None)
    parser.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-llm-routing", action="store_true", help="routing ties search both specs")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("httpx", "sentence_transformers", "chromadb", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    sys.stdout.reconfigure(encoding="utf-8")

    cfg = load_config()
    entries = load_testset(args.testset)
    if args.limit:
        entries = entries[: args.limit]
    in_corpus = [e for e in entries if e["type"] != "out_of_corpus"]
    runner = RetrievalRunner(cfg, use_cache=not args.no_cache, llm_routing=not args.no_llm_routing)

    configs = args.configs
    if "full" not in configs:
        configs = ["full", *configs]  # routing results come from the full configuration
    per_config: dict[str, dict[str, Any]] = {}
    raw_lines = []
    for name in configs:
        records = {r["id"]: r for r in runner.run(name, entries)}
        rows = []
        for e in in_corpus:
            rec = records[e["id"]]
            row = {"id": e["id"], "group": question_group(e), "latency_s": rec["latency_s"],
                   "metrics": question_metrics(rec["chunks"], e)}
            rows.append(row)
            raw_lines.append({"config": name, "id": e["id"], "type": question_group(e),
                              "question": e["question"], "routed_specs": rec["routed_specs"],
                              "route_reason": rec["route_reason"], "latency_s": rec["latency_s"],
                              "metrics": row["metrics"],
                              "retrieved": [f"{c['spec']} §{c['section']}" for c in rec["chunks"]],
                              "scores": rec["scores"]})
        agg = {g: aggregate([r for r in rows if g == "all" or r["group"] == g]) for g, _ in GROUPS}
        per_config[name] = {"records": records, "rows_by_id": {r["id"]: r for r in rows}, "agg": agg}
        a = agg["all"]
        logger.info("%-17s R@5 %.3f  R@10 %.3f  MRR %.3f", name, a["recall@5"], a["recall@10"], a["rr"])

    out = resolve_path(RESULTS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    order = [c for c in CONFIGS if c in configs]
    write_retrieval_md(out / "retrieval_results.md", entries, order, per_config, cfg, runner.router_llm)
    write_routing_md(out / "routing_results.md", entries, per_config["full"]["records"], runner.router_llm)
    with (out / "retrieval_raw.jsonl").open("w", encoding="utf-8") as f:
        for line in raw_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"\nWrote {out / 'retrieval_results.md'}, {out / 'routing_results.md'}, {out / 'retrieval_raw.jsonl'}")
    if unverified_ids(entries):
        print(f"WARNING: {len(unverified_ids(entries))} test-set entries are not verified; results are provisional.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
