"""Generation evaluation: refusals, citations, RAGAS faithfulness / answer correctness, latency.

Usage:
    python -m eval.run_generation_eval [--configs full dense_only ...] [--ragas-configs full ...]
                                       [--limit N] [--no-ragas]

For each configuration (``evaluation.generation_configs``, default: full, hybrid_rerank,
full_no_hybrid, full_no_rerank, dense_only) every test question goes through the real answer
pipeline (retrieval, refusal layer 1, LLM, refusal layer 2, citation check).

Metrics:
  * refusal accuracy: out-of-corpus questions answered with the exact refusal message;
  * false refusal rate: in-corpus questions refused;
  * citations (in-corpus, non-refused answers): share of answers with at least one citation;
    validity = cited (spec, section) in the retrieved context / all citations; correctness =
    citations matching a gold section (same rule as the retrieval evaluation) / all citations;
  * RAGAS faithfulness (answer vs retrieved contexts, non-refused in-corpus answers) and answer
    correctness (answer vs reference answer; a refusal of an in-corpus question scores 0), judged
    by ``evaluation.judge_model``, on ``evaluation.ragas_configs`` (cost: ~8k judge tokens per answer);
  * manual score (0 wrong / 1 partial / 2 correct) read back from ``eval/results/manual_grading.csv``;
  * latency (retrieval, LLM, total). LLM latency of a cached call is the latency of the original call.

Caching: LLM calls by (model, exact messages) in ``eval/cache/generation/llm_calls.jsonl`` (identical
prompts from different configurations share one call), whole answers by (configuration, question,
code, settings), RAGAS scores in ``eval/cache/ragas.jsonl``. If the provider's daily quota is reached
the run stops cleanly, writes partial results and can be resumed later with the same command.

Writes ``eval/results/generation_raw.jsonl``, ``generation_results.md`` and ``manual_grading.csv``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

from eval.metrics import is_relevant, load_testset, mean, question_group, unverified_ids
from eval.run_retrieval_eval import CONFIGS, flag_text, fmt, md_table, retrieval_code_hash
from src.config import load_config, resolve_path
from src.generate.llm import LLM, RateLimitExhausted, get_llm
from src.index.common import file_sha256

logger = logging.getLogger("eval.generation")

RESULTS_DIR = "eval/results"
CACHE_DIR = "eval/cache/generation"
MANUAL_CSV = "manual_grading.csv"
MANUAL_CONFIG = "full"
GENERATION_CODE = ["src/generate/answer.py", "src/generate/prompt.py", "src/generate/citation_check.py"]


def _hash(obj: Any, n: int = 24) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:n]


def _code_hash() -> str:
    h = hashlib.sha256(retrieval_code_hash().encode())
    for rel in GENERATION_CODE:
        h.update(resolve_path(rel).read_bytes())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- cached LLM


class CachedLLM:
    """Wraps an :class:`LLM`; identical (model, messages, settings) calls are answered from disk."""

    def __init__(self, llm: LLM, path: Path) -> None:
        self.llm = llm
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, dict[str, Any]] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.cache[rec["key"]] = rec
        self.hits = self.misses = 0
        self.last_latency = 0.0

    @property
    def name(self) -> str:
        return self.llm.name

    def check(self) -> None:
        self.llm.check()

    def generate(self, messages, temperature=None, max_tokens=None) -> str:
        key = _hash([self.llm.name, self.llm.reasoning_effort, messages, temperature, max_tokens,
                     self.llm.temperature, self.llm.max_tokens])
        if key in self.cache:
            self.hits += 1
            self.last_latency = self.cache[key]["latency_s"]
            return self.cache[key]["text"]
        t0 = time.perf_counter()
        text = self.llm.generate(messages, temperature=temperature, max_tokens=max_tokens)
        # model time only: waiting for the provider's rate limit is not latency of the system
        self.last_latency = time.perf_counter() - t0 - getattr(self.llm, "last_wait_s", 0.0)
        self.misses += 1
        rec = {"key": key, "text": text, "latency_s": self.last_latency, "model": self.llm.name}
        self.cache[key] = rec
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return text

    def classify(self, prompt: str) -> str:
        return self.generate([{"role": "user", "content": prompt}], temperature=0.0,
                             max_tokens=self.llm.classify_max_tokens)


# --------------------------------------------------------------------------- generation


class GenerationRunner:
    def __init__(self, cfg: dict[str, Any], use_cache: bool = True) -> None:
        from src.generate.answer import Answerer
        from src.retrieve.pipeline import Pipeline

        self.cfg = cfg
        base_llm = get_llm(cfg=cfg)
        base_llm.check()
        self.llm = CachedLLM(base_llm, resolve_path(CACHE_DIR) / "llm_calls.jsonl")
        self.pipeline = Pipeline(cfg, llm_classifier=self.llm.classify)
        self.answerer = Answerer(cfg, llm=self.llm, pipeline=self.pipeline)
        self.use_cache = use_cache
        self.base_key = _hash({
            "chunks": file_sha256(resolve_path(cfg["paths"]["chunks_file"])),
            "settings": {k: cfg[k] for k in ("retrieval", "reranker", "embeddings", "bm25", "llm", "refusal_message")},
            "code": _code_hash(),
        })

    def _cache_file(self, config: str) -> Path:
        return resolve_path(CACHE_DIR) / f"{config}.jsonl"

    def _load(self, config: str) -> dict[str, dict[str, Any]]:
        path = self._cache_file(config)
        if not self.use_cache or not path.exists():
            return {}
        return {r["key"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if r}

    def answer_one(self, config: str, entry: dict[str, Any]) -> dict[str, Any]:
        c = CONFIGS[config]
        res = self.answerer.answer(
            entry["question"], use_hybrid=c["use_hybrid"], use_reranker=c["use_reranker"],
            use_routing=c["use_routing"], use_acronym_expansion=c.get("use_acronym_expansion"),
        )
        latency = dict(res.latency)
        if "llm_s" in latency:  # time of the original (possibly cached) call
            latency["total_s"] = latency["total_s"] - latency["llm_s"] + self.llm.last_latency
            latency["llm_s"] = self.llm.last_latency
        return {
            "answer": res.answer,
            "is_refusal": res.is_refusal,
            "refusal_layer": res.refusal_layer,
            "routed_specs": res.routed_specs,
            "route_reason": res.route_reason,
            "top_score": res.top_score,
            "citations": res.citations,
            "invalid_citations": res.invalid_citations,
            "contexts": [cand.chunk["text_for_index"] for cand in res.chunks],
            "retrieved": [{"spec": cand.chunk["spec"], "section": cand.chunk["section"],
                           "merged_sections": cand.chunk.get("merged_sections") or [],
                           "score": cand.rerank_score if cand.rerank_score is not None else cand.fused_score}
                          for cand in res.chunks],
            "latency": latency,
            "llm": res.llm,
        }

    def run(self, config: str, entries: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], bool]:
        """Returns (records by question id, complete?)."""
        cache = self._load(config)
        out: dict[str, dict[str, Any]] = {}
        for i, e in enumerate(entries, start=1):
            key = _hash([self.base_key, config, CONFIGS[config], e["question"]])
            if key in cache:
                out[e["id"]] = cache[key]
                continue
            try:
                rec = {"key": key, "id": e["id"], "config": config, **self.answer_one(config, e)}
            except RateLimitExhausted as exc:
                logger.error("Stopping: %s", exc)
                return out, False
            out[e["id"]] = rec
            with self._cache_file(config).open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            layer = f" refusal:{rec['refusal_layer']}" if rec["is_refusal"] else ""
            logger.info("[%s] %d/%d %s%s (%.1f s)", config, i, len(entries), e["id"], layer,
                        rec["latency"].get("total_s", 0.0))
        return out, True


# --------------------------------------------------------------------------- metrics


def citation_stats(rec: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    cits = rec["citations"] + rec["invalid_citations"]
    golds = entry["gold_sections"]
    correct = sum(is_relevant({"spec": c["spec"], "section": c["section"], "merged_sections": []}, golds)
                  for c in cits)
    return {"n": len(cits), "valid": len(rec["citations"]), "correct": correct}


def apply_ragas(judge, rec: dict[str, Any], entry: dict[str, Any]) -> dict[str, float | None]:
    scores: dict[str, float | None] = {"faithfulness": None, "answer_correctness": None}
    if entry["type"] == "out_of_corpus":
        return scores
    if rec["is_refusal"]:
        scores["answer_correctness"] = 0.0  # a refusal of an in-corpus question answers nothing
        return scores
    scores["faithfulness"] = judge.score("faithfulness", entry["question"], rec["answer"], rec["contexts"])
    scores["answer_correctness"] = judge.score("answer_correctness", entry["question"], rec["answer"],
                                               rec["contexts"], entry["reference_answer"])
    return scores


def read_manual(path: Path) -> dict[tuple[str, str], float]:
    if not path.exists():
        return {}
    out = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            val = (row.get("manual_score") or "").strip()
            if val in ("0", "1", "2"):
                out[(row["config"], row["id"])] = float(val)
    return out


def write_manual(path: Path, entries, records: dict[str, dict[str, Any]], config: str) -> None:
    """Grading sheet for one configuration; scores already filled in are kept."""
    existing: dict[tuple[str, str], dict[str, str]] = {}
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as f:
            existing = {(r["config"], r["id"]): r for r in csv.DictReader(f)}
    fields = ["config", "id", "type", "question", "reference_answer", "answer", "citations",
              "invalid_citations", "manual_score", "grader_notes"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in entries:
            rec = records.get(e["id"])
            if rec is None:
                continue
            old = existing.get((config, e["id"]), {})
            same_answer = old.get("answer") == rec["answer"]
            w.writerow({
                "config": config, "id": e["id"], "type": question_group(e), "question": e["question"],
                "reference_answer": e["reference_answer"], "answer": rec["answer"],
                "citations": "; ".join(f"{c['spec']} §{c['section']}" for c in rec["citations"]),
                "invalid_citations": "; ".join(f"{c['spec']} §{c['section']}" for c in rec["invalid_citations"]),
                # a grade is kept only if the answer it was given for has not changed
                "manual_score": old.get("manual_score", "") if same_answer else "",
                "grader_notes": old.get("grader_notes", "") if same_answer else "",
            })


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    inc = [r for r in rows if r["type"] != "out_of_corpus"]
    ooc = [r for r in rows if r["type"] == "out_of_corpus"]
    answered = [r for r in inc if not r["is_refusal"]]
    n_cit = sum(r["cit"]["n"] for r in answered)
    return {
        "n": len(rows), "n_in": len(inc), "n_ooc": len(ooc),
        "refusal_acc": mean(float(r["exact_refusal"]) for r in ooc) if ooc else None,
        "false_refusal": mean(float(r["is_refusal"]) for r in inc) if inc else None,
        "with_citation": mean(float(r["cit"]["n"] > 0) for r in answered) if answered else None,
        "cit_validity": (sum(r["cit"]["valid"] for r in answered) / n_cit) if n_cit else None,
        "cit_correctness": (sum(r["cit"]["correct"] for r in answered) / n_cit) if n_cit else None,
        "faithfulness": mean(r["ragas"]["faithfulness"] for r in answered),
        "answer_correctness": mean(r["ragas"]["answer_correctness"] for r in inc),
        "manual": mean(r["manual"] for r in inc),
        "n_manual": sum(r["manual"] is not None for r in inc),
        "n_faith": sum(r["ragas"]["faithfulness"] is not None for r in answered),
        "n_ac": sum(r["ragas"]["answer_correctness"] is not None for r in inc),
        "lat_total": mean(r["latency"].get("total_s") for r in rows),
        "lat_llm": mean(r["latency"].get("llm_s") for r in rows),
    }


# --------------------------------------------------------------------------- report


def write_md(path: Path, entries, configs, per_config, complete, judge_name, ragas_configs, cfg,
             ragas_missing: int = 0) -> None:
    bad = unverified_ids(entries)
    parts = [
        "# Generation evaluation\n",
        f"Generated by `eval/run_generation_eval.py` on {date.today().isoformat()}.\n",
        (f"> **WARNING: {len(bad)}/{len(entries)} test-set entries are not verified.** Reference answers and "
         "gold sections have not yet been checked by a person; these results are provisional.\n") if bad else "",
        "" if all(complete.values()) else
        ("> **INCOMPLETE:** the LLM provider's quota was reached; configurations marked incomplete were not "
         "fully answered. Re-run the same command later to resume (cached answers are reused).\n"),
        (f"> **INCOMPLETE RAGAS:** {ragas_missing} judge score(s) could not be computed (judge quota); the "
         "`n=` next to each RAGAS mean is the number of answers actually scored. Re-run later to complete.\n")
        if ragas_missing else "",
        f"- Generator: `{cfg['llm']['provider']}:{cfg['llm']['model']}` (temperature {cfg['llm']['temperature']}, "
        f"top-{cfg['retrieval']['final_k']} chunks). RAGAS judge: `{judge_name}`, on "
        f"{', '.join(f'`{c}`' for c in ragas_configs) or 'no configuration'}. The judge is a different model "
        "than the generator; if both come from the same family, some self-preference bias cannot be excluded.",
        f"- Refusal threshold (layer 1, reranker score): {cfg['reranker']['refusal_threshold']}; only active in "
        "configurations with the reranker. Layer 2 is the prompt rule.",
        "- **Refusal accuracy:** out-of-corpus questions answered with the exact refusal message. **False "
        "refusals:** in-corpus questions refused.",
        "- **Citations** (in-corpus, non-refused answers): **with citation** = share of answers with at least "
        "one citation; **validity** = cited (spec, section) present in the retrieved context / all citations; "
        "**correctness** = citations matching a gold section (same rule as the retrieval evaluation) / all citations.",
        "- **Faithfulness** (RAGAS): share of the answer's claims supported by the retrieved contexts "
        "(non-refused in-corpus answers). **Answer correctness** (RAGAS): agreement with the reference answer "
        "(factual F1 + semantic similarity); a refusal of an in-corpus question scores 0.",
        "- **Manual score:** mean of the team's grades in `manual_grading.csv` (0 wrong, 1 partial, 2 correct), "
        "configuration `full`.",
        "- LLM latency of a cached call is the latency measured when the call was made. Calls made before "
        "rate-limit waits were excluded from the measurement (configurations `full_no_rerank` and "
        "`dense_only` in this run) include time spent waiting for the free-tier per-minute token limit, so "
        "their LLM and total latency are overestimated; the model itself answers in about 2-4 s.\n",
        "## By configuration\n",
    ]
    headers = ["Config", "Components", "Answered", "Refusal acc. (OOC)", "False refusals", "With citation",
               "Cit. validity", "Cit. correctness", "Faithfulness", "Answer corr.", "Manual (0-2)",
               "LLM (s)", "Total (s)"]
    rows = []
    for name in configs:
        a = per_config[name]["agg"]["all"]
        status = f"{a['n']}/{len(entries)}" + ("" if complete[name] else " (incomplete)")
        rows.append([f"`{name}`", flag_text(name), status, fmt(a["refusal_acc"], 2), fmt(a["false_refusal"], 2),
                     fmt(a["with_citation"], 2), fmt(a["cit_validity"], 2), fmt(a["cit_correctness"], 2),
                     fmt(a["faithfulness"], 2) + (f" (n={a['n_faith']})" if a["faithfulness"] is not None else ""),
                     fmt(a["answer_correctness"], 2) + (f" (n={a['n_ac']})" if a["answer_correctness"] is not None else ""),
                     fmt(a["manual"], 2) + (f" (n={a['n_manual']})" if a["manual"] is not None else ""),
                     fmt(a["lat_llm"], 1), fmt(a["lat_total"], 1)])
    parts.append(md_table(headers, rows) + "\n")

    groups = [("simple_23501", "Simple TS 23.501"), ("simple_24501", "Simple TS 24.501"),
              ("cross_spec", "Cross-spec"), ("out_of_corpus", "Out of corpus")]
    for name in configs:
        if name != MANUAL_CONFIG and name not in ragas_configs:
            continue
        parts.append(f"## `{name}` by question type\n")
        rows = []
        for g, title in groups:
            a = per_config[name]["agg"].get(g)
            if not a or not a["n"]:
                continue
            rows.append([title, str(a["n"]), fmt(a["refusal_acc"], 2), fmt(a["false_refusal"], 2),
                         fmt(a["with_citation"], 2), fmt(a["cit_validity"], 2), fmt(a["cit_correctness"], 2),
                         fmt(a["faithfulness"], 2), fmt(a["answer_correctness"], 2), fmt(a["manual"], 2)])
        parts.append(md_table(["Type", "n", "Refusal acc.", "False ref.", "With cit.", "Validity", "Correctness",
                               "Faithfulness", "Answer corr.", "Manual"], rows) + "\n")

    for name in configs:
        if name != MANUAL_CONFIG:
            continue
        parts.append(f"## `{name}` per question\n")
        rows = []
        for e in entries:
            r = per_config[name]["rows"].get(e["id"])
            if r is None:
                rows.append([e["id"], "(not answered yet)"] + [""] * 6)
                continue
            outcome = (f"refused ({r['refusal_layer']})" if r["is_refusal"] else "answered")
            rows.append([e["id"], outcome, f"{r['cit']['valid']}/{r['cit']['n']}", str(r["cit"]["correct"]),
                         fmt(r["ragas"]["faithfulness"], 2), fmt(r["ragas"]["answer_correctness"], 2),
                         fmt(r["manual"], 0), fmt(r["latency"].get("total_s"), 1)])
        parts.append(md_table(["Question", "Outcome", "Valid/all cit.", "Correct cit.", "Faithfulness",
                               "Answer corr.", "Manual", "Total (s)"], rows) + "\n")
    path.write_text("\n".join(p for p in parts if p is not None), encoding="utf-8")


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    ev = cfg.get("evaluation", {})
    parser = argparse.ArgumentParser(description="Generation evaluation")
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=ev.get("generation_configs"))
    parser.add_argument("--ragas-configs", nargs="*", choices=list(CONFIGS), default=ev.get("ragas_configs", ["full"]))
    parser.add_argument("--no-ragas", action="store_true")
    parser.add_argument("--testset", default=None)
    parser.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    parser.add_argument("--no-cache", action="store_true", help="recompute answers (LLM calls stay cached)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("httpx", "sentence_transformers", "chromadb", "urllib3", "openai", "src.generate.citation_check"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    sys.stdout.reconfigure(encoding="utf-8")

    entries = load_testset(args.testset)
    if args.limit:
        entries = entries[: args.limit]
    by_id = {e["id"]: e for e in entries}
    runner = GenerationRunner(cfg, use_cache=not args.no_cache)
    out = resolve_path(RESULTS_DIR)
    out.mkdir(parents=True, exist_ok=True)

    records: dict[str, dict[str, dict[str, Any]]] = {}
    complete: dict[str, bool] = {}
    for name in args.configs:
        records[name], complete[name] = runner.run(name, entries)
        logger.info("[%s] %d/%d answered%s", name, len(records[name]), len(entries),
                    "" if complete[name] else " (stopped: quota)")
        if not complete[name]:  # quota reached: the remaining configurations are not started
            for rest in args.configs[args.configs.index(name) + 1:]:
                records[rest], complete[rest] = {}, False
            break
    logger.info("LLM calls: %d new, %d from cache", runner.llm.misses, runner.llm.hits)

    judge = None
    ragas_configs = [] if args.no_ragas else [c for c in args.ragas_configs if c in records]
    if ragas_configs:
        from eval.judge import RagasJudge

        judge = RagasJudge(cfg)
        judge.check()

    manual = read_manual(out / MANUAL_CSV)
    per_config: dict[str, dict[str, Any]] = {}
    raw_lines = []
    for name in args.configs:
        rows: dict[str, dict[str, Any]] = {}
        for qid, rec in records.get(name, {}).items():
            e = by_id[qid]
            ragas = {"faithfulness": None, "answer_correctness": None}
            if judge and name in ragas_configs:
                ragas = apply_ragas(judge, rec, e)
            row = {**rec, "type": question_group(e), "cit": citation_stats(rec, e), "ragas": ragas,
                   "exact_refusal": rec["answer"] == cfg["refusal_message"],
                   "manual": manual.get((name, qid))}
            rows[qid] = row
            raw_lines.append({"config": name, "id": qid, "type": row["type"], "question": e["question"],
                              "reference_answer": e["reference_answer"], "answer": rec["answer"],
                              "is_refusal": rec["is_refusal"], "refusal_layer": rec["refusal_layer"],
                              "routed_specs": rec["routed_specs"], "top_score": rec["top_score"],
                              "contexts": rec["contexts"], "retrieved": rec["retrieved"],
                              "citations": rec["citations"], "invalid_citations": rec["invalid_citations"],
                              "citation_stats": row["cit"], "ragas": ragas, "manual_score": row["manual"],
                              "latency": rec["latency"], "llm": rec["llm"]})
        groups = ["all", "simple_23501", "simple_24501", "cross_spec", "out_of_corpus"]
        agg = {g: aggregate([r for r in rows.values() if g == "all" or r["type"] == g]) for g in groups}
        per_config[name] = {"rows": rows, "agg": agg}
        if name == MANUAL_CONFIG and rows:
            write_manual(out / MANUAL_CSV, entries, records[name], name)

    with (out / "generation_raw.jsonl").open("w", encoding="utf-8") as f:
        for line in raw_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    write_md(out / "generation_results.md", entries, args.configs, per_config, complete,
             judge.name if judge else "-", ragas_configs, cfg, judge.missing if judge else 0)
    print(f"\nWrote {out / 'generation_results.md'}, {out / 'generation_raw.jsonl'}, {out / MANUAL_CSV}")
    if unverified_ids(entries):
        print(f"WARNING: {len(unverified_ids(entries))} test-set entries are not verified; results are provisional.")
    if not all(complete.values()) or (judge and judge.missing):
        print("INCOMPLETE: provider quota reached (answers and/or RAGAS scores missing). "
              "Re-run the same command later to resume; everything computed so far is cached.")
        return 3
    return 0


if __name__ == "__main__":
    import os

    code = main()
    # All results are written and closed at this point. On Windows, interpreter shutdown after RAGAS
    # (async HTTP clients) and torch can crash with a segmentation fault, which would turn a successful
    # run into a failed `make` target; exit directly instead.
    sys.stdout.flush()
    sys.stderr.flush()
    logging.shutdown()
    os._exit(code)
