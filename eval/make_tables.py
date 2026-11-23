"""Ablation summary table and charts from the raw evaluation outputs (no models, no API calls).

Usage:
    python -m eval.make_tables

Reads ``eval/results/retrieval_raw.jsonl`` and ``generation_raw.jsonl`` (written by
``run_retrieval_eval`` and ``run_generation_eval``) and writes, in ``eval/results/``:

  * ``ablation_summary.md``: for each component, the configuration without it compared with ``full``
    (delta and a paired bootstrap 95% CI over questions: the same resampled questions are used for
    both configurations, so the CI is that of the per-question difference);
  * ``chart_retrieval.png``: Recall@5 and MRR per configuration, with 95% CIs;
  * ``chart_ablation.png``: change in Recall@5 and MRR when each component is removed;
  * ``chart_citations.png``: answers with a citation, citation validity and correctness per configuration.

``retrieval_results.md``, ``routing_results.md`` and ``generation_results.md`` are written by the
evaluation scripts themselves.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any

from eval.metrics import bootstrap_ci, mean
from eval.run_retrieval_eval import CONFIGS, flag_text, fmt, md_table
from src.config import load_config, resolve_path

logger = logging.getLogger("eval.tables")

RESULTS_DIR = "eval/results"

# (component, configuration without it). Each is compared with `full`.
ABLATIONS = [
    ("Hybrid search (BM25 + RRF)", "full_no_hybrid"),
    ("Reranker", "full_no_rerank"),
    ("Routing", "hybrid_rerank"),
    ("Acronym expansion", "full_no_acronyms"),
    ("All three (hybrid, reranker, routing)", "dense_only"),
]

# Chart style: validated categorical slots 1-3 (light), diverging blue/red, recessive chrome.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
POSITIVE, NEGATIVE = "#2a78d6", "#e34948"


# --------------------------------------------------------------------------- data


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def by_config(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(r["config"], {})[r["id"]] = r
    return out


def paired_delta(a: dict[str, float], b: dict[str, float]) -> tuple[float | None, tuple[float, float] | None, int]:
    """Mean of (b - a) over the questions both have, and its bootstrap 95% CI."""
    ids = sorted(set(a) & set(b))
    diffs = [b[i] - a[i] for i in ids if a[i] is not None and b[i] is not None]
    if not diffs:
        return None, None, 0
    return sum(diffs) / len(diffs), bootstrap_ci(diffs), len(diffs)


def retrieval_metric(rows: dict[str, dict[str, Any]], key: str, types: set[str] | None = None) -> dict[str, float]:
    return {qid: r["metrics"][key] for qid, r in rows.items()
            if key in r["metrics"] and (types is None or r["type"] in types)}


def generation_metrics(rows: dict[str, dict[str, Any]], refusal: str) -> dict[str, dict[str, float]]:
    """Per-question generation values (keyed like the retrieval metrics)."""
    m: dict[str, dict[str, float]] = {"false_refusal": {}, "ooc_refused": {}, "with_citation": {},
                                      "cit_validity": {}, "cit_correctness": {}, "lat_total": {}}
    for qid, r in rows.items():
        if r["type"] == "out_of_corpus":
            m["ooc_refused"][qid] = float(r["answer"] == refusal)
            continue
        m["false_refusal"][qid] = float(r["is_refusal"])
        if not r["is_refusal"]:
            s = r["citation_stats"]
            m["with_citation"][qid] = float(s["n"] > 0)
            if s["n"]:
                m["cit_validity"][qid] = s["valid"] / s["n"]
                m["cit_correctness"][qid] = s["correct"] / s["n"]
        if r["latency"].get("total_s") is not None:
            m["lat_total"][qid] = r["latency"]["total_s"]
    return m


def micro(rows: dict[str, dict[str, Any]], field: str) -> float | None:
    """Citation validity / correctness pooled over all citations (as in generation_results.md)."""
    n = sum(r["citation_stats"]["n"] for r in rows.values() if r["type"] != "out_of_corpus" and not r["is_refusal"])
    if not n:
        return None
    return sum(r["citation_stats"][field] for r in rows.values()
               if r["type"] != "out_of_corpus" and not r["is_refusal"]) / n


# --------------------------------------------------------------------------- markdown


def fmt_delta(d: float | None, ci: tuple[float, float] | None, digits: int = 3) -> str:
    if d is None:
        return "-"
    text = f"{d:+.{digits}f}"
    if ci:
        sig = "" if ci[0] <= 0 <= ci[1] else " *"
        text += f" [{ci[0]:+.2f}, {ci[1]:+.2f}]{sig}"
    return text


def write_summary(path: Path, ret: dict, gen: dict, refusal: str, unverified: list[str]) -> list[dict[str, Any]]:
    """Writes ablation_summary.md; returns the retrieval deltas used by the ablation chart."""
    full_r, full_g = ret.get("full", {}), gen.get("full", {})
    cross = {"cross_spec"}
    in_corpus = {"simple_23501", "simple_24501", "cross_spec"}
    parts = [
        "# Ablation summary\n",
        f"Generated by `eval/make_tables.py` on {date.today().isoformat()} from `retrieval_raw.jsonl` and "
        "`generation_raw.jsonl`.\n",
        (f"> **WARNING: {len(unverified)} test-set entries are not verified;** these results are provisional.\n"
         if unverified else ""),
        "Each row removes one component from the full system (`full` = hybrid search + reranker + routing + "
        "acronym expansion) and reports **(configuration without it) - full**. A negative retrieval delta means "
        "the component helps. In brackets: paired bootstrap 95% CI over questions (2000 resamples, seed 42); "
        "`*` = the CI excludes 0. With 28 in-corpus questions (8 cross-spec), most differences are within noise.\n",
        "## Retrieval\n",
    ]
    rows, deltas = [], []
    for component, cfg in ABLATIONS:
        r = ret.get(cfg)
        if not r or not full_r:
            continue
        d5 = paired_delta(retrieval_metric(full_r, "recall@5", in_corpus), retrieval_metric(r, "recall@5", in_corpus))
        dm = paired_delta(retrieval_metric(full_r, "rr", in_corpus), retrieval_metric(r, "rr", in_corpus))
        dh = paired_delta(retrieval_metric(full_r, "hit@5", in_corpus), retrieval_metric(r, "hit@5", in_corpus))
        db = paired_delta(retrieval_metric(full_r, "both_specs@5", cross), retrieval_metric(r, "both_specs@5", cross))
        lat_full = mean(x["latency_s"] for x in full_r.values())
        lat = mean(x["latency_s"] for x in r.values())
        rows.append([component, f"`{cfg}`", fmt_delta(*d5[:2]), fmt_delta(*dm[:2]), fmt_delta(*dh[:2]),
                     fmt_delta(*db[:2]), f"{lat - lat_full:+.1f}" if lat is not None and lat_full is not None else "-"])
        deltas.append({"component": component, "config": cfg, "recall@5": d5, "mrr": dm})
    parts.append(md_table(["Component removed", "Config", "Δ Recall@5", "Δ MRR", "Δ Hit@5",
                           "Δ Both specs@5 (cross-spec)", "Δ Latency (s)"], rows) + "\n")
    parts.append("Latency is indicative only (laptop CPU, load-dependent; see `retrieval_results.md`).\n")

    parts.append("## Generation\n")
    if not full_g:
        parts.append("No generation results yet (`make eval-generation`).\n")
    else:
        fg = generation_metrics(full_g, refusal)
        rows = []
        for component, cfg in ABLATIONS:
            g = gen.get(cfg)
            if not g:
                continue
            mg = generation_metrics(g, refusal)
            fr = paired_delta(fg["false_refusal"], mg["false_refusal"])
            oc = paired_delta(fg["ooc_refused"], mg["ooc_refused"])
            wc = paired_delta(fg["with_citation"], mg["with_citation"])
            cv, cf = micro(full_g, "valid"), micro(g, "valid")
            cc, cg = micro(full_g, "correct"), micro(g, "correct")
            rows.append([component, f"`{cfg}`", fmt_delta(*fr[:2]), fmt_delta(*oc[:2], digits=2),
                         fmt_delta(*wc[:2]),
                         f"{cf - cv:+.3f}" if cf is not None and cv is not None else "-",
                         f"{cg - cc:+.3f}" if cg is not None and cc is not None else "-"])
        parts.append(md_table(["Component removed", "Config", "Δ False refusals", "Δ Refusal acc. (OOC)",
                               "Δ With citation", "Δ Cit. validity", "Δ Cit. correctness"], rows) + "\n")
        parts.append("For false refusals a positive delta is worse (more in-corpus questions refused). "
                     "Citation validity and correctness are pooled over all citations (no per-question CI). "
                     "\"With citation\" compares only questions answered (not refused) by both configurations. "
                     "RAGAS scores exist for `full` only, so they have no ablation delta.\n")

    parts.append("## Charts\n")
    parts.append("- `chart_retrieval.png`: Recall@5 and MRR per configuration with 95% CIs.")
    parts.append("- `chart_ablation.png`: Δ Recall@5 and Δ MRR when each component is removed (paired 95% CIs).")
    parts.append("- `chart_citations.png`: answers with a citation, citation validity and correctness per configuration.\n")
    path.write_text("\n".join(p for p in parts if p is not None), encoding="utf-8")
    return deltas


# --------------------------------------------------------------------------- charts


def _style(ax, xlabel: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="x", colors=MUTED, labelsize=9, length=0)
    ax.tick_params(axis="y", colors=INK_2, labelsize=10, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)


def _header(fig, title: str, subtitle: str, height: float, legend: bool) -> float:
    """Title and subtitle at fixed distances (inches) from the top; returns the top of the plot area."""
    fig.text(0.01, 1 - 0.12 / height, title, ha="left", va="top", fontsize=12, color=INK, fontweight="semibold")
    fig.text(0.01, 1 - 0.42 / height, subtitle, ha="left", va="top", fontsize=8.5, color=INK_2)
    return 1 - (0.62 if legend else 0.7) / height  # tight_layout already makes room for the legend


def _figure(width: float, height: float):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans", "sans-serif"]
    fig, ax = plt.subplots(figsize=(width, height), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax, plt


def grouped_hbars(path: Path, title: str, subtitle: str, labels: list[str], series: list[tuple[str, list]],
                  cis: list[list] | None = None, xlabel: str = "Score (0 to 1)") -> None:
    """Horizontal grouped bars: one group per configuration, one bar per metric (same 0-1 scale)."""
    height = 0.55 * len(labels) * max(2, len(series)) / 2 + 1.5
    fig, ax, plt = _figure(9, height)
    n = len(series)
    band = 0.7
    h = band / n
    for j, (name, values) in enumerate(series):
        ys = [i + (j - (n - 1) / 2) * h for i in range(len(labels))]
        vals = [v if v is not None else 0 for v in values]
        ax.barh(ys, vals, height=h * 0.92, color=SERIES[j], edgecolor=SURFACE, linewidth=1.5, label=name, zorder=2)
        for y, v, raw in zip(ys, vals, values):
            text = "n/a" if raw is None else f"{raw:.2f}"
            ci = cis[j][ys.index(y)] if cis and cis[j] else None
            x = (ci[1] if ci else v) + 0.012
            ax.text(x, y, text, va="center", ha="left", fontsize=8, color=INK_2, zorder=3)
        if cis and cis[j]:
            for y, ci in zip(ys, cis[j]):
                if ci:
                    ax.plot([ci[0], ci[1]], [y, y], color=INK_2, linewidth=1, zorder=3)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.1)
    _style(ax, xlabel)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=n, frameon=False, fontsize=9,
              labelcolor=INK_2, handlelength=1.2, borderaxespad=0.2)
    top = _header(fig, title, subtitle, height, legend=True)
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def ablation_chart(path: Path, deltas: list[dict[str, Any]]) -> None:
    """Two panels (Δ Recall@5, Δ MRR), diverging colour by sign, paired CIs as whiskers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans", "sans-serif"]
    height = 0.55 * len(deltas) + 1.7
    fig, axes = plt.subplots(1, 2, figsize=(10, height), dpi=200, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    labels = [d["component"] for d in deltas]
    lo = min([d[k][1][0] for d in deltas for k in ("recall@5", "mrr") if d[k][1]] + [-0.1])
    hi = max([d[k][1][1] for d in deltas for k in ("recall@5", "mrr") if d[k][1]] + [0.1])
    pad = 0.04
    for ax, key, name in ((axes[0], "recall@5", "Δ Recall@5"), (axes[1], "mrr", "Δ MRR")):
        for i, d in enumerate(deltas):
            val, ci, _ = d[key]
            if val is None:
                continue
            ax.barh(i, val, height=0.5, color=POSITIVE if val >= 0 else NEGATIVE, edgecolor=SURFACE,
                    linewidth=1.5, zorder=2)
            if ci:
                ax.plot([ci[0], ci[1]], [i, i], color=INK_2, linewidth=1, zorder=3)
            edge = (ci[1] if ci else val) if val >= 0 else (ci[0] if ci else val)
            ax.text(edge + (0.01 if val >= 0 else -0.01), i, f"{val:+.3f}", va="center",
                    ha="left" if val >= 0 else "right", fontsize=8, color=INK_2, zorder=3)
        ax.axvline(0, color=BASELINE, linewidth=1, zorder=1)
        ax.set_xlim(lo - pad - 0.06, hi + pad + 0.06)
        _style(ax, "without the component − full")
        ax.set_title(name, loc="left", fontsize=10, color=INK)
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].invert_yaxis()
    top = _header(fig, "What each component contributes to retrieval",
                  "Red, left of zero: removing the component lowers the score, so the component helps. "
                  "Whiskers: paired bootstrap 95% CI over the 28 in-corpus questions.", height, legend=False)
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------- main


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    out = resolve_path(RESULTS_DIR)
    ret = by_config(load_jsonl(out / "retrieval_raw.jsonl"))
    gen = by_config(load_jsonl(out / "generation_raw.jsonl"))
    if not ret:
        print("No retrieval results: run `make eval-retrieval` first.")
        return 1
    from eval.metrics import load_testset, unverified_ids

    unverified = unverified_ids(load_testset(warn=False))
    deltas = write_summary(out / "ablation_summary.md", ret, gen, cfg["refusal_message"], unverified)

    order = [c for c in CONFIGS if c in ret]
    in_corpus = {"simple_23501", "simple_24501", "cross_spec"}
    r5 = [retrieval_metric(ret[c], "recall@5", in_corpus) for c in order]
    mrr = [retrieval_metric(ret[c], "rr", in_corpus) for c in order]
    grouped_hbars(
        out / "chart_retrieval.png",
        "Retrieval quality per configuration",
        f"{len(r5[0])} in-corpus questions · whiskers: 95% bootstrap CI · full = hybrid + reranker + routing"
        + (" · provisional (test set not verified)" if unverified else ""),
        [f"{c}  ({flag_text(c)})" for c in order],
        [("Recall@5", [mean(v.values()) for v in r5]), ("MRR", [mean(v.values()) for v in mrr])],
        cis=[[bootstrap_ci(list(v.values())) for v in r5], [bootstrap_ci(list(v.values())) for v in mrr]],
    )
    ablation_chart(out / "chart_ablation.png", deltas)

    if gen:
        gorder = [c for c in CONFIGS if c in gen]
        with_cit, validity, correctness = [], [], []
        for c in gorder:
            m = generation_metrics(gen[c], cfg["refusal_message"])
            with_cit.append(mean(m["with_citation"].values()))
            validity.append(micro(gen[c], "valid"))
            correctness.append(micro(gen[c], "correct"))
        grouped_hbars(
            out / "chart_citations.png",
            "Citations in the generated answers",
            "In-corpus answers that were not refused · validity: cited section was retrieved · "
            "correctness: cited section is a gold section" + (" · provisional" if unverified else ""),
            [f"{c}  ({flag_text(c)})" for c in gorder],
            [("Answers with a citation", with_cit), ("Citation validity", validity),
             ("Citation correctness", correctness)],
            xlabel="Share (0 to 1)",
        )
    print(f"Wrote {out / 'ablation_summary.md'} and charts chart_retrieval.png, chart_ablation.png"
          + (", chart_citations.png" if gen else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
