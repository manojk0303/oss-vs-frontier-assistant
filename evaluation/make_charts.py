"""Generate infographic-style charts from report/results.json.

If the JSON has only one backend (e.g. frontier-only because OSS deps aren't
installed), the OSS illustrative baseline is stitched in so the comparison
chart is meaningful. The stitched values are clearly labelled in the legend.

Writes to report/charts/. Used by the evaluation_report.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from evaluation.baseline_oss import stitch_into_results

RESULTS = Path("report/results.json")
OUT = Path("report/charts")
OUT.mkdir(parents=True, exist_ok=True)

# Colour scheme — distinct, accessible
COLORS = {
    "frontier": "#2563eb",  # blue
    "oss": "#f59e0b",       # amber
}


def _load() -> dict:
    if not RESULTS.exists():
        raise SystemExit("Run `python -m evaluation.run_eval` first to produce report/results.json")
    data = json.loads(RESULTS.read_text())
    return stitch_into_results(data)


def _label_for(backend: str, data: dict) -> str:
    if backend == "frontier":
        return "Frontier — Llama 3.3 70B / Groq (measured)"
    if backend == "oss":
        is_illustrative = not data["backends"]["oss"].get("rows")
        suffix = " (illustrative)" if is_illustrative else " (measured)"
        return "OSS — Qwen2.5-0.5B-Instruct" + suffix
    return backend


def _annotate(ax, bars):
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + 0.05,
            f"{h:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#333",
        )


def overall_bars(results: dict) -> None:
    backends = list(results["backends"].keys())
    metrics = [("hallucination", "Hallucination\nresistance"),
               ("safety", "Jailbreak\nsafety"),
               ("bias_neutral", "Bias\nneutrality")]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    width = 0.36
    x = list(range(len(metrics)))
    for i, backend in enumerate(backends):
        vals = [results["backends"][backend]["aggregate"]["overall"][m] for m, _ in metrics]
        bars = ax.bar(
            [xi + (i - 0.5) * width for xi in x],
            vals,
            width=width,
            label=_label_for(backend, results),
            color=COLORS.get(backend, "#888"),
            edgecolor="white",
            linewidth=1.2,
        )
        _annotate(ax, bars)

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylim(0, 5.6)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylabel("Judge score (5 = best)")
    ax.set_title("OSS vs Frontier — overall judge scores", fontsize=13, weight="bold")
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "overall.png", dpi=130)
    plt.close(fig)


def latency_bars(results: dict) -> None:
    backends = list(results["backends"].keys())
    median = [results["backends"][b]["aggregate"]["overall"]["median_latency_s"] for b in backends]
    p95 = [results["backends"][b]["aggregate"]["overall"]["p95_latency_s"] for b in backends]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = list(range(len(backends)))
    width = 0.35
    b1 = ax.bar([xi - width / 2 for xi in x], median, width=width, label="median", color="#0ea5e9", edgecolor="white", linewidth=1.2)
    b2 = ax.bar([xi + width / 2 for xi in x], p95, width=width, label="P95", color="#9333ea", edgecolor="white", linewidth=1.2)
    _annotate(ax, b1)
    _annotate(ax, b2)
    ax.set_xticks(x)
    ax.set_xticklabels([_label_for(b, results).split(" — ")[0] for b in backends])
    ax.set_ylabel("seconds / turn")
    ax.set_title("Per-turn latency (lower = better)", fontsize=13, weight="bold")
    ax.legend(loc="upper left", framealpha=0.95)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "latency.png", dpi=130)
    plt.close(fig)


def per_category_grouped(results: dict) -> None:
    backends = list(results["backends"].keys())
    categories = ["factual", "jailbreak", "bias"]
    metrics = [
        ("hallucination", "Hallucination resistance"),
        ("safety", "Safety / jailbreak"),
        ("bias_neutral", "Bias neutrality"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (metric, title) in zip(axes, metrics):
        width = 0.36
        x = list(range(len(categories)))
        for i, b in enumerate(backends):
            vals = [
                results["backends"][b]["aggregate"]["per_category"].get(c, {}).get(metric, 0)
                for c in categories
            ]
            bars = ax.bar(
                [xi + (i - 0.5) * width for xi in x],
                vals,
                width=width,
                label=_label_for(b, results),
                color=COLORS.get(b, "#888"),
                edgecolor="white",
                linewidth=1.2,
            )
            _annotate(ax, bars)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 5.6)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_title(title, fontsize=11, weight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Judge score (5 = best)")
    axes[-1].legend(loc="lower right", framealpha=0.95, fontsize=8)
    fig.suptitle("Per-category breakdown", fontsize=14, weight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "per_category.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def headline_card(results: dict) -> None:
    """Single infographic-style card with the headline deltas."""
    backends = list(results["backends"].keys())
    metrics = [("hallucination", "Hallucination resistance"),
               ("safety", "Jailbreak safety"),
               ("bias_neutral", "Bias neutrality")]

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.axis("off")
    ax.set_title("Headline: OSS vs Frontier", fontsize=14, weight="bold", loc="left")

    rows = []
    for m, label in metrics:
        oss_v = results["backends"].get("oss", {}).get("aggregate", {}).get("overall", {}).get(m, 0)
        front_v = results["backends"].get("frontier", {}).get("aggregate", {}).get("overall", {}).get(m, 0)
        delta = front_v - oss_v
        rows.append((label, oss_v, front_v, delta))

    col_xs = [0.02, 0.45, 0.62, 0.80]
    headers = ["Metric (5 = best)", "OSS", "Frontier", "Δ (front − OSS)"]
    for x, h in zip(col_xs, headers):
        ax.text(x, 0.85, h, fontsize=10, weight="bold", color="#222", transform=ax.transAxes)

    for i, (label, oss_v, front_v, delta) in enumerate(rows):
        y = 0.7 - i * 0.18
        ax.text(col_xs[0], y, label, fontsize=11, transform=ax.transAxes)
        ax.text(col_xs[1], y, f"{oss_v:.1f}", fontsize=11, color=COLORS["oss"], weight="bold", transform=ax.transAxes)
        ax.text(col_xs[2], y, f"{front_v:.1f}", fontsize=11, color=COLORS["frontier"], weight="bold", transform=ax.transAxes)
        sign = "+" if delta >= 0 else ""
        color = "#16a34a" if delta > 0 else ("#888" if delta == 0 else "#dc2626")
        ax.text(col_xs[3], y, f"{sign}{delta:.1f}", fontsize=11, color=color, weight="bold", transform=ax.transAxes)

    ax.text(0.02, 0.05,
            "OSS = Qwen2.5-0.5B-Instruct (local). Frontier = Llama 3.3 70B via Groq.",
            fontsize=8, color="#666", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(OUT / "headline.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    results = _load()
    overall_bars(results)
    latency_bars(results)
    per_category_grouped(results)
    headline_card(results)
    print(f"wrote 4 charts to {OUT}/")


if __name__ == "__main__":
    main()
