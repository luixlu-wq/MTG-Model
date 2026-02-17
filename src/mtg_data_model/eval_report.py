from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from statistics import mean, median
from typing import Any

from .config import settings


@dataclass
class EvalSummary:
    metrics_file: str
    report_json: str
    report_markdown: str
    charts: list[str]
    rows_total: int
    avg_latency_ms: float | None
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    avg_tokens_per_sec: float | None
    avg_prompt_tokens: float | None
    avg_output_tokens: float | None
    exact_match_rate: float | None
    avg_char_f1: float | None
    avg_seq_ratio: float | None
    recommendations: list[str]


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _find_latest_eval_metrics(log_dir: str) -> str:
    files = []
    for name in os.listdir(log_dir):
        if re.match(r"^eval-\d{14}\.metrics\.jsonl$", name):
            files.append(os.path.join(log_dir, name))
    if not files:
        raise FileNotFoundError(f"No eval-<timestamp>.metrics.jsonl found in {log_dir}")
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def _load_rows(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    idx = int((pct / 100.0) * (len(s) - 1))
    return s[idx]


def _round_or_none(value: float | None, ndigits: int = 6) -> float | None:
    return None if value is None else round(value, ndigits)


def _recommend(avg_latency_ms: float | None, p95_latency_ms: float | None, avg_char_f1: float | None, exact_rate: float | None) -> list[str]:
    recs: list[str] = []
    if avg_latency_ms is not None and avg_latency_ms > 3000:
        recs.append("Average eval latency is high; reduce max_new_tokens or use a smaller generation batch.")
    if p95_latency_ms is not None and avg_latency_ms is not None and p95_latency_ms > (avg_latency_ms * 2.0):
        recs.append("Latency variance is high; check CPU offload or unstable I/O during generation.")
    if avg_char_f1 is not None and avg_char_f1 < 0.45:
        recs.append("Similarity to teacher is low; consider more SFT epochs or stronger data filtering.")
    if exact_rate is not None and exact_rate < 0.02:
        recs.append("Exact match rate is very low; this can be normal for long-form generation, use semantic metrics too.")
    if not recs:
        recs.append("Eval trend looks stable; run qualitative spot checks on difficult tourism routes and logistics prompts.")
    return recs


def _make_charts(rows: list[dict[str, Any]], out_dir: str, prefix: str) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    x = list(range(1, len(rows) + 1))
    lat = [_to_float(r.get("latency_ms")) for r in rows]
    tps = [_to_float(r.get("tokens_per_sec")) for r in rows]
    out_tok = [_to_float(r.get("output_tokens")) for r in rows]
    char_f1 = [_to_float(r.get("char_f1")) for r in rows]
    seq_ratio = [_to_float(r.get("seq_ratio")) for r in rows]

    charts: list[str] = []

    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax1.plot(x, lat, linewidth=1.0, color="#1f77b4")
    ax1.set_ylabel("Latency (ms)")
    ax1.grid(alpha=0.25)
    ax1.set_title("Eval Runtime")
    ax2.plot(x, tps, linewidth=1.0, color="#2ca02c", label="tokens/sec")
    ax2.plot(x, out_tok, linewidth=1.0, color="#ff7f0e", alpha=0.75, label="output tokens")
    ax2.set_xlabel("Sample Index")
    ax2.grid(alpha=0.25)
    ax2.legend()
    p1 = os.path.join(out_dir, f"{prefix}.runtime.png")
    fig1.tight_layout()
    fig1.savefig(p1, dpi=150)
    plt.close(fig1)
    charts.append(p1)

    has_quality = any(v is not None for v in char_f1) or any(v is not None for v in seq_ratio)
    if has_quality:
        fig2, ax3 = plt.subplots(figsize=(11, 4))
        ax3.plot(x, char_f1, linewidth=1.0, color="#d62728", label="char_f1")
        ax3.plot(x, seq_ratio, linewidth=1.0, color="#9467bd", label="seq_ratio")
        ax3.set_xlabel("Sample Index")
        ax3.set_ylabel("Similarity")
        ax3.set_ylim(0.0, 1.0)
        ax3.grid(alpha=0.25)
        ax3.legend()
        ax3.set_title("Eval Similarity vs Teacher")
        p2 = os.path.join(out_dir, f"{prefix}.quality.png")
        fig2.tight_layout()
        fig2.savefig(p2, dpi=150)
        plt.close(fig2)
        charts.append(p2)

    return charts


def _write_markdown(summary: EvalSummary) -> None:
    lines = [
        "# Eval Report",
        "",
        f"- Metrics file: `{summary.metrics_file}`",
        "",
        "## Key Metrics",
        f"- Rows: {summary.rows_total}",
        f"- Latency (ms): avg={summary.avg_latency_ms}, p50={summary.p50_latency_ms}, p95={summary.p95_latency_ms}",
        f"- Tokens/sec: avg={summary.avg_tokens_per_sec}",
        f"- Prompt tokens avg: {summary.avg_prompt_tokens}",
        f"- Output tokens avg: {summary.avg_output_tokens}",
        f"- Exact match rate: {summary.exact_match_rate}",
        f"- Char F1 avg: {summary.avg_char_f1}",
        f"- Seq ratio avg: {summary.avg_seq_ratio}",
        "",
        "## Recommendations",
    ]
    lines.extend([f"- {item}" for item in summary.recommendations])
    if summary.charts:
        lines.extend(["", "## Charts"])
        lines.extend([f"- `{path}`" for path in summary.charts])
    with open(summary.report_markdown, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze eval metrics and generate report + charts.")
    ap.add_argument("--metrics", default="", help="Path to eval-<timestamp>.metrics.jsonl")
    ap.add_argument("--log-dir", default=settings.log_dir, help="Folder used to locate latest eval metrics.")
    ap.add_argument("--out-dir", default=os.path.join(settings.data_dir, "report"), help="Output folder for report files.")
    ap.add_argument(
        "--diagrams-dir",
        default=os.path.join(settings.data_dir, "diagrams"),
        help="Output folder for generated chart images.",
    )
    ap.add_argument("--latest", action="store_true", help="Use latest eval metrics from --log-dir when --metrics is not set.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.diagrams_dir, exist_ok=True)

    metrics_path = args.metrics.strip()
    if not metrics_path:
        if args.latest:
            metrics_path = _find_latest_eval_metrics(args.log_dir)
        else:
            raise ValueError("Provide --metrics or use --latest.")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Eval metrics not found: {metrics_path}")

    rows = _load_rows(metrics_path)
    if not rows:
        raise RuntimeError("Eval metrics file is empty.")

    lat = [_to_float(r.get("latency_ms")) for r in rows if _to_float(r.get("latency_ms")) is not None]
    tps = [_to_float(r.get("tokens_per_sec")) for r in rows if _to_float(r.get("tokens_per_sec")) is not None]
    p_tok = [_to_float(r.get("prompt_tokens")) for r in rows if _to_float(r.get("prompt_tokens")) is not None]
    o_tok = [_to_float(r.get("output_tokens")) for r in rows if _to_float(r.get("output_tokens")) is not None]
    c_f1 = [_to_float(r.get("char_f1")) for r in rows if _to_float(r.get("char_f1")) is not None]
    s_ratio = [_to_float(r.get("seq_ratio")) for r in rows if _to_float(r.get("seq_ratio")) is not None]
    exacts = [_to_bool(r.get("exact_match")) for r in rows if _to_bool(r.get("exact_match")) is not None]

    avg_latency = mean(lat) if lat else None
    p50_latency = _percentile(lat, 50.0)
    p95_latency = _percentile(lat, 95.0)
    avg_tps = mean(tps) if tps else None
    avg_prompt_tokens = mean(p_tok) if p_tok else None
    avg_output_tokens = mean(o_tok) if o_tok else None
    exact_rate = (sum(1 for v in exacts if v) / len(exacts)) if exacts else None
    avg_char_f1 = mean(c_f1) if c_f1 else None
    avg_seq_ratio = mean(s_ratio) if s_ratio else None

    recs = _recommend(avg_latency, p95_latency, avg_char_f1, exact_rate)

    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    prefix = f"eval-report-{stamp}"
    report_json = os.path.join(args.out_dir, f"{prefix}.json")
    report_markdown = os.path.join(args.out_dir, f"{prefix}.md")
    charts = _make_charts(rows, args.diagrams_dir, prefix)

    summary = EvalSummary(
        metrics_file=metrics_path,
        report_json=report_json,
        report_markdown=report_markdown,
        charts=charts,
        rows_total=len(rows),
        avg_latency_ms=_round_or_none(avg_latency, 3),
        p50_latency_ms=_round_or_none(p50_latency, 3),
        p95_latency_ms=_round_or_none(p95_latency, 3),
        avg_tokens_per_sec=_round_or_none(avg_tps, 3),
        avg_prompt_tokens=_round_or_none(avg_prompt_tokens, 3),
        avg_output_tokens=_round_or_none(avg_output_tokens, 3),
        exact_match_rate=_round_or_none(exact_rate, 6),
        avg_char_f1=_round_or_none(avg_char_f1, 6),
        avg_seq_ratio=_round_or_none(avg_seq_ratio, 6),
        recommendations=recs,
    )

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(summary.__dict__, f, ensure_ascii=False, indent=2)
    _write_markdown(summary)

    print(f"eval_report_json={report_json}")
    print(f"eval_report_md={report_markdown}")
    if charts:
        print("eval_report_charts=" + ",".join(charts))
    else:
        print("eval_report_charts=none (install matplotlib to enable chart generation)")


if __name__ == "__main__":
    main()
