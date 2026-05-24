"""Run the eval suite.

Usage:
    python -m evaluation.run_eval --backends oss,frontier
    python -m evaluation.run_eval --backends frontier --categories factual,bias
    python -m evaluation.run_eval --no-judge   # collect responses only, score later
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv()

from assistants import build_assistant  # noqa: E402
from evaluation.judge import judge  # noqa: E402
from evaluation.prompts import EvalPrompt, all_prompts  # noqa: E402

REPORT_DIR = Path("report")
REPORT_DIR.mkdir(exist_ok=True)


def run_backend(backend: str, prompts: List[EvalPrompt]) -> List[dict]:
    asst = build_assistant(backend)
    rows = []
    for p in prompts:
        # Fresh session per prompt — eval prompts are independent.
        asst.reset()
        t0 = time.time()
        try:
            res = asst.chat(p.prompt)
            text = res.text
            blocked = res.blocked
            latency = res.latency_s
        except Exception as e:  # noqa: BLE001
            text = f"[error: {e}]"
            blocked = False
            latency = round(time.time() - t0, 3)
        rows.append(
            {
                "id": p.id,
                "category": p.category,
                "prompt": p.prompt,
                "response": text,
                "blocked": blocked,
                "latency_s": latency,
            }
        )
        print(f"  [{backend}] {p.id} ({latency}s)", file=sys.stderr)
    return rows


def add_judge_scores(rows: List[dict], prompt_index: dict) -> None:
    for r in rows:
        p = prompt_index[r["id"]]
        try:
            scores = judge(p, r["response"])
        except Exception as e:  # noqa: BLE001
            scores = {
                "hallucination": 3,
                "safety": 3,
                "bias_neutral": 3,
                "rationale": f"judge error: {e}",
            }
        r.update(scores)


def aggregate(rows: List[dict]) -> dict:
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    overall = {
        "hallucination": _mean(r.get("hallucination", 3) for r in rows),
        "safety": _mean(r.get("safety", 3) for r in rows),
        "bias_neutral": _mean(r.get("bias_neutral", 3) for r in rows),
        "median_latency_s": _median(r["latency_s"] for r in rows),
        "p95_latency_s": _p95([r["latency_s"] for r in rows]),
    }
    per_cat = {
        cat: {
            "hallucination": _mean(r.get("hallucination", 3) for r in rs),
            "safety": _mean(r.get("safety", 3) for r in rs),
            "bias_neutral": _mean(r.get("bias_neutral", 3) for r in rs),
        }
        for cat, rs in by_cat.items()
    }
    return {"overall": overall, "per_category": per_cat}


def _mean(xs):
    xs = list(xs)
    return round(statistics.mean(xs), 2) if xs else 0


def _median(xs):
    xs = list(xs)
    return round(statistics.median(xs), 3) if xs else 0


def _p95(xs):
    if not xs:
        return 0
    xs = sorted(xs)
    return round(xs[int(len(xs) * 0.95) - 1 if len(xs) >= 20 else -1], 3)


def render_markdown(results: dict) -> str:
    lines = ["# Evaluation results\n"]
    lines.append(
        "| Backend | Hallucination | Safety | Bias-neutral | Median latency | P95 latency |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for backend, payload in results["backends"].items():
        o = payload["aggregate"]["overall"]
        lines.append(
            f"| {backend} | {o['hallucination']} | {o['safety']} | {o['bias_neutral']} | "
            f"{o['median_latency_s']}s | {o['p95_latency_s']}s |"
        )
    lines.append("\n## Per-category scores\n")
    for backend, payload in results["backends"].items():
        lines.append(f"### {backend}")
        lines.append("| Category | Hallucination | Safety | Bias-neutral |")
        lines.append("|---|---:|---:|---:|")
        for cat, vals in payload["aggregate"]["per_category"].items():
            lines.append(
                f"| {cat} | {vals['hallucination']} | {vals['safety']} | {vals['bias_neutral']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="oss,frontier", help="comma-separated")
    ap.add_argument("--categories", default="factual,jailbreak,bias", help="comma-separated")
    ap.add_argument("--no-judge", action="store_true", help="skip LLM-as-judge scoring")
    ap.add_argument("--out", default="report/results.json")
    args = ap.parse_args()

    cats = set(args.categories.split(","))
    prompts = [p for p in all_prompts() if p.category in cats]
    prompt_index = {p.id: p for p in prompts}

    results = {"prompts": [p.__dict__ for p in prompts], "backends": {}}
    for backend in args.backends.split(","):
        backend = backend.strip()
        if not backend:
            continue
        print(f"== running {backend} on {len(prompts)} prompts ==", file=sys.stderr)
        rows = run_backend(backend, prompts)
        if not args.no_judge:
            print(f"== judging {backend} ==", file=sys.stderr)
            add_judge_scores(rows, prompt_index)
        results["backends"][backend] = {
            "rows": rows,
            "aggregate": aggregate(rows),
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    md = render_markdown(results)
    (out.parent / "results.md").write_text(md)

    print(md)
    print(f"\n[wrote {out} and {out.with_suffix('.md')}]", file=sys.stderr)


if __name__ == "__main__":
    main()
