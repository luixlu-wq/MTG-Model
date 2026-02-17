from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from statistics import mean, median
from typing import Any

from .config import settings


@dataclass
class Summary:
    train_log: str
    metrics_file: str | None
    report_json: str
    report_markdown: str
    charts: list[str]
    rows_total: int
    train_rows: int
    eval_rows: int
    steps_completed: int
    epochs_completed: float
    start_loss_avg: float | None
    end_loss_avg: float | None
    loss_reduction_ratio: float | None
    best_eval_loss: float | None
    best_eval_step: int | None
    final_eval_loss: float | None
    final_eval_ppl: float | None
    grad_norm_median: float | None
    grad_norm_max: float | None
    lr_start: float | None
    lr_end: float | None
    step_time_head_avg_s: float | None
    step_time_tail_avg_s: float | None
    step_time_median_s: float | None
    verdict: str
    recommendations: list[str]


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text.lower() in {"none", "nan", ""}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _to_int(value: Any) -> int | None:
    f = _to_float(value)
    if f is None:
        return None
    return int(f)


def _parse_ts(line: str) -> datetime | None:
    match = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})", line)
    if not match:
        return None
    return datetime.strptime(f"{match.group(1)}.{match.group(2)}", "%Y-%m-%d %H:%M:%S.%f")


def _find_latest_train_log(log_dir: str) -> str:
    files = []
    for name in os.listdir(log_dir):
        if re.match(r"^train-\d{14}\.log$", name):
            files.append(os.path.join(log_dir, name))
    if not files:
        raise FileNotFoundError(f"No train-<timestamp>.log found in {log_dir}")
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def _infer_metrics_path(train_log_path: str) -> str | None:
    if not train_log_path.endswith(".log"):
        return None
    candidate = train_log_path[:-4] + ".metrics.jsonl"
    return candidate if os.path.exists(candidate) else None


def _load_metrics(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _load_train_log_rows(path: str) -> tuple[list[dict[str, Any]], list[tuple[int, datetime]]]:
    rows: list[dict[str, Any]] = []
    timed_steps: list[tuple[int, datetime]] = []
    pair_pat = re.compile(r"(\w+)=([^\s]+)")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if "train_log step=" not in line:
                continue
            ts = _parse_ts(line)
            tail = line.split("train_log ", 1)[1]
            pairs = dict(pair_pat.findall(tail))
            step = _to_int(pairs.get("step"))
            epoch = _to_float(pairs.get("epoch"))
            loss = _to_float(pairs.get("loss"))
            eval_loss = _to_float(pairs.get("eval_loss"))
            eval_ppl = _to_float(pairs.get("eval_ppl"))
            lr = _to_float(pairs.get("lr"))
            grad_norm = _to_float(pairs.get("grad_norm"))
            row = {
                "step": step,
                "epoch": epoch,
                "loss": loss,
                "eval_loss": eval_loss,
                "eval_ppl": eval_ppl,
                "learning_rate": lr,
                "grad_norm": grad_norm,
            }
            rows.append(row)
            if ts is not None and step is not None and loss is not None:
                timed_steps.append((step, ts))
    return rows, timed_steps


def _safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _safe_median(values: list[float]) -> float | None:
    return median(values) if values else None


def _build_step_time_stats(timed_steps: list[tuple[int, datetime]]) -> tuple[float | None, float | None, float | None]:
    if len(timed_steps) < 2:
        return None, None, None
    deltas: list[float] = []
    for i in range(1, len(timed_steps)):
        prev_step, prev_ts = timed_steps[i - 1]
        curr_step, curr_ts = timed_steps[i]
        if curr_step == prev_step + 1:
            deltas.append((curr_ts - prev_ts).total_seconds())
    if not deltas:
        return None, None, None
    head = deltas[: min(30, len(deltas))]
    tail = deltas[-min(30, len(deltas)) :]
    return _safe_mean(head), _safe_mean(tail), _safe_median(deltas)


def _round_or_none(value: float | None, ndigits: int = 6) -> float | None:
    return None if value is None else round(value, ndigits)


def _evaluate_quality(
    start_loss_avg: float | None,
    end_loss_avg: float | None,
    final_eval_loss: float | None,
    first_eval_loss: float | None,
    tail_step_time: float | None,
    head_step_time: float | None,
) -> tuple[str, list[str], float | None]:
    recs: list[str] = []
    reduction_ratio = None
    if start_loss_avg and end_loss_avg and start_loss_avg > 0:
        reduction_ratio = (start_loss_avg - end_loss_avg) / start_loss_avg

    if final_eval_loss is not None:
        if final_eval_loss <= 0.2:
            verdict = "good"
        elif final_eval_loss <= 0.35:
            verdict = "acceptable"
        else:
            verdict = "weak"
    elif reduction_ratio is not None and reduction_ratio >= 0.5:
        verdict = "good"
    elif reduction_ratio is not None and reduction_ratio >= 0.2:
        verdict = "acceptable"
    else:
        verdict = "weak"

    if first_eval_loss is not None and final_eval_loss is not None:
        if final_eval_loss > first_eval_loss:
            recs.append("Eval loss regressed; lower learning rate or increase regularization.")
        elif first_eval_loss - final_eval_loss < 0.01:
            recs.append("Eval improvement is small; extend epochs or increase training data variety.")

    if tail_step_time and head_step_time and tail_step_time > (head_step_time * 3.0):
        recs.append("Step latency increased significantly; reduce max_len or check I/O and thermal throttling.")

    if not recs:
        recs.append("Training trend is stable. Run full eval on held-out prompts before packaging.")

    return verdict, recs, reduction_ratio


def _make_charts(rows: list[dict[str, Any]], out_dir: str, prefix: str) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    train_rows = [r for r in rows if _to_float(r.get("loss")) is not None and _to_int(r.get("step")) is not None]
    eval_rows = [r for r in rows if _to_float(r.get("eval_loss")) is not None and _to_int(r.get("step")) is not None]
    if not train_rows:
        return []

    train_steps = [_to_int(r["step"]) for r in train_rows]
    train_loss = [_to_float(r["loss"]) for r in train_rows]
    lr_rows = [r for r in train_rows if _to_float(r.get("learning_rate")) is not None]
    gn_rows = [r for r in train_rows if _to_float(r.get("grad_norm")) is not None]
    charts: list[str] = []

    # Loss curve
    fig1, ax1 = plt.subplots(figsize=(11, 4))
    ax1.plot(train_steps, train_loss, linewidth=1.0, color="#1f77b4", label="train_loss")
    if eval_rows:
        ax1.plot(
            [_to_int(r["step"]) for r in eval_rows],
            [_to_float(r["eval_loss"]) for r in eval_rows],
            marker="o",
            linewidth=1.2,
            color="#d62728",
            label="eval_loss",
        )
    ax1.set_title("Training vs Eval Loss")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.25)
    ax1.legend()
    p1 = os.path.join(out_dir, f"{prefix}.loss.png")
    fig1.tight_layout()
    fig1.savefig(p1, dpi=150)
    plt.close(fig1)
    charts.append(p1)

    # LR and grad norm
    fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    if lr_rows:
        ax2.plot(
            [_to_int(r["step"]) for r in lr_rows],
            [_to_float(r["learning_rate"]) for r in lr_rows],
            linewidth=1.0,
            color="#2ca02c",
        )
    ax2.set_ylabel("Learning Rate")
    ax2.grid(alpha=0.25)
    if gn_rows:
        ax3.plot(
            [_to_int(r["step"]) for r in gn_rows],
            [_to_float(r["grad_norm"]) for r in gn_rows],
            linewidth=1.0,
            color="#ff7f0e",
        )
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Grad Norm")
    ax3.grid(alpha=0.25)
    fig2.suptitle("Optimizer Signals")
    p2 = os.path.join(out_dir, f"{prefix}.optim.png")
    fig2.tight_layout()
    fig2.savefig(p2, dpi=150)
    plt.close(fig2)
    charts.append(p2)
    return charts


def _write_markdown(summary: Summary) -> None:
    lines = [
        "# Train Report",
        "",
        f"- Train log: `{summary.train_log}`",
        f"- Metrics file: `{summary.metrics_file}`",
        f"- Verdict: **{summary.verdict}**",
        "",
        "## Key Metrics",
        f"- Rows: total={summary.rows_total}, train={summary.train_rows}, eval={summary.eval_rows}",
        f"- Completed: steps={summary.steps_completed}, epochs={summary.epochs_completed}",
        f"- Loss avg (start/end): {summary.start_loss_avg} / {summary.end_loss_avg}",
        f"- Loss reduction ratio: {summary.loss_reduction_ratio}",
        f"- Best eval loss: {summary.best_eval_loss} (step={summary.best_eval_step})",
        f"- Final eval: loss={summary.final_eval_loss}, ppl={summary.final_eval_ppl}",
        f"- Grad norm: median={summary.grad_norm_median}, max={summary.grad_norm_max}",
        f"- LR: start={summary.lr_start}, end={summary.lr_end}",
        f"- Step time (s): head_avg={summary.step_time_head_avg_s}, tail_avg={summary.step_time_tail_avg_s}, median={summary.step_time_median_s}",
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
    ap = argparse.ArgumentParser(description="Analyze train logs and generate quality report + charts.")
    ap.add_argument("--log", default="", help="Path to train-<timestamp>.log")
    ap.add_argument("--metrics", default="", help="Path to train-<timestamp>.metrics.jsonl")
    ap.add_argument("--log-dir", default=settings.log_dir, help="Folder used to locate latest train log.")
    ap.add_argument(
        "--out-dir",
        default=os.path.join(settings.data_dir, "report"),
        help="Output folder for report files.",
    )
    ap.add_argument(
        "--diagrams-dir",
        default=os.path.join(settings.data_dir, "diagrams"),
        help="Output folder for generated chart images.",
    )
    ap.add_argument("--latest", action="store_true", help="Use latest train log from --log-dir when --log is not set.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.diagrams_dir, exist_ok=True)

    log_path = args.log.strip()
    if not log_path:
        if args.latest:
            log_path = _find_latest_train_log(args.log_dir)
        else:
            raise ValueError("Provide --log or use --latest.")
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Train log not found: {log_path}")

    metrics_path = args.metrics.strip() or (_infer_metrics_path(log_path) or "")
    rows: list[dict[str, Any]]
    timed_steps: list[tuple[int, datetime]]

    log_rows, timed_steps = _load_train_log_rows(log_path)
    if metrics_path and os.path.exists(metrics_path):
        rows = _load_metrics(metrics_path)
    else:
        metrics_path = ""
        rows = log_rows

    if not rows:
        raise RuntimeError("No training rows found in log/metrics files.")

    train_rows = [r for r in rows if _to_float(r.get("loss")) is not None and _to_int(r.get("step")) is not None]
    eval_rows = [r for r in rows if _to_float(r.get("eval_loss")) is not None and _to_int(r.get("step")) is not None]
    if not train_rows:
        raise RuntimeError("No train loss rows found.")

    # Aggregate stats.
    sorted_train = sorted(train_rows, key=lambda r: _to_int(r.get("step")) or 0)
    sorted_eval = sorted(eval_rows, key=lambda r: _to_int(r.get("step")) or 0)
    steps_completed = _to_int(sorted_train[-1]["step"]) or 0
    epochs_completed = _to_float(sorted_train[-1].get("epoch")) or 0.0

    n_head = min(20, len(sorted_train))
    n_tail = min(20, len(sorted_train))
    start_loss_avg = _safe_mean([_to_float(r["loss"]) for r in sorted_train[:n_head] if _to_float(r["loss"]) is not None])
    end_loss_avg = _safe_mean([_to_float(r["loss"]) for r in sorted_train[-n_tail:] if _to_float(r["loss"]) is not None])

    best_eval_loss = None
    best_eval_step = None
    final_eval_loss = None
    final_eval_ppl = None
    first_eval_loss = None
    if sorted_eval:
        first_eval_loss = _to_float(sorted_eval[0].get("eval_loss"))
        best_row = min(sorted_eval, key=lambda r: _to_float(r.get("eval_loss")) or float("inf"))
        best_eval_loss = _to_float(best_row.get("eval_loss"))
        best_eval_step = _to_int(best_row.get("step"))
        final_eval_loss = _to_float(sorted_eval[-1].get("eval_loss"))
        if final_eval_loss is not None:
            final_eval_ppl = math.exp(final_eval_loss)

    grad_values = [_to_float(r.get("grad_norm")) for r in sorted_train if _to_float(r.get("grad_norm")) is not None]
    lr_values = [_to_float(r.get("learning_rate")) for r in sorted_train if _to_float(r.get("learning_rate")) is not None]
    step_time_head_avg_s, step_time_tail_avg_s, step_time_median_s = _build_step_time_stats(timed_steps)

    verdict, recommendations, reduction_ratio = _evaluate_quality(
        start_loss_avg=start_loss_avg,
        end_loss_avg=end_loss_avg,
        final_eval_loss=final_eval_loss,
        first_eval_loss=first_eval_loss,
        tail_step_time=step_time_tail_avg_s,
        head_step_time=step_time_head_avg_s,
    )

    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    prefix = f"train-report-{stamp}"
    report_json = os.path.join(args.out_dir, f"{prefix}.json")
    report_markdown = os.path.join(args.out_dir, f"{prefix}.md")
    charts = _make_charts(rows, args.diagrams_dir, prefix)

    summary = Summary(
        train_log=log_path,
        metrics_file=metrics_path or None,
        report_json=report_json,
        report_markdown=report_markdown,
        charts=charts,
        rows_total=len(rows),
        train_rows=len(train_rows),
        eval_rows=len(eval_rows),
        steps_completed=steps_completed,
        epochs_completed=round(epochs_completed, 6),
        start_loss_avg=_round_or_none(start_loss_avg, 6),
        end_loss_avg=_round_or_none(end_loss_avg, 6),
        loss_reduction_ratio=_round_or_none(reduction_ratio, 6),
        best_eval_loss=_round_or_none(best_eval_loss, 6),
        best_eval_step=best_eval_step,
        final_eval_loss=_round_or_none(final_eval_loss, 6),
        final_eval_ppl=_round_or_none(final_eval_ppl, 6),
        grad_norm_median=_round_or_none(_safe_median([v for v in grad_values if v is not None]), 6),
        grad_norm_max=_round_or_none(max(grad_values) if grad_values else None, 6),
        lr_start=_round_or_none(lr_values[0] if lr_values else None, 10),
        lr_end=_round_or_none(lr_values[-1] if lr_values else None, 10),
        step_time_head_avg_s=_round_or_none(step_time_head_avg_s, 3),
        step_time_tail_avg_s=_round_or_none(step_time_tail_avg_s, 3),
        step_time_median_s=_round_or_none(step_time_median_s, 3),
        verdict=verdict,
        recommendations=recommendations,
    )

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(summary.__dict__, f, ensure_ascii=False, indent=2)
    _write_markdown(summary)

    print(f"train_report_json={report_json}")
    print(f"train_report_md={report_markdown}")
    if charts:
        print("train_report_charts=" + ",".join(charts))
    else:
        print("train_report_charts=none (install matplotlib to enable chart generation)")


if __name__ == "__main__":
    main()
