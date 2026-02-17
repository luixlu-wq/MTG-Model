from __future__ import annotations

import argparse
import os
import subprocess
import sys
import re


def run(cmd: list[str], env: dict[str, str]) -> None:
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    return env


def _default_ckpt_from_model(student_base: str) -> str:
    # Convert model id/path to a stable folder-friendly suffix.
    base = (student_base or "").strip()
    name = base.rsplit("/", 1)[-1] if "/" in base else base
    name = name.lower()
    name = re.sub(r"[^a-z0-9._-]+", "-", name)
    name = name.strip("-._")
    if not name:
        name = "student"
    return rf"data\checkpoints\{name}-lora"


def step_prompts(args: argparse.Namespace, env: dict[str, str], python: str) -> None:
    cmd = [
        python,
        "-m",
        "mtg_data_model.build_prompts",
        "--spots",
        args.spots,
        "--out",
        args.prompts_out,
    ]
    if args.prompts_limit > 0:
        cmd.extend(["--limit", str(args.prompts_limit)])
    run(cmd, env)


def step_teacher(args: argparse.Namespace, env: dict[str, str], python: str) -> None:
    cmd = [
        python,
        "-m",
        "mtg_data_model.teacher_generate",
        "--backend",
        args.backend,
        "--ollama-url",
        args.ollama_url,
        "--ollama-num-gpu",
        str(args.ollama_num_gpu),
        "--model",
        args.teacher_model,
        "--in",
        args.prompts_out,
        "--out",
        args.teacher_out,
    ]
    if args.teacher_limit > 0:
        cmd.extend(["--limit", str(args.teacher_limit)])
    if args.overwrite_teacher:
        cmd.append("--overwrite")
    else:
        cmd.append("--resume")
        if args.resume_verify:
            cmd.append("--resume-verify")
    run(cmd, env)


def step_train(args: argparse.Namespace, env: dict[str, str], python: str) -> None:
    cmd = [
        python,
        "-m",
        "mtg_data_model.sft_train",
        "--train",
        args.teacher_out,
        "--base",
        args.student_base,
        "--out",
        args.ckpt_out,
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--batch",
        str(args.batch),
        "--grad-accum",
        str(args.grad_accum),
        "--max-len",
        str(args.max_len),
        "--save-steps",
        str(args.save_steps),
        "--logging-steps",
        str(args.logging_steps),
        "--val-ratio",
        str(args.val_ratio),
        "--seed",
        str(args.seed),
        "--device-mode",
        args.device_mode,
        "--bnb-compute-dtype",
        args.bnb_compute_dtype,
    ]
    if args.load_in_4bit:
        cmd.append("--load-in-4bit")
    if args.gradient_checkpointing:
        cmd.append("--gradient-checkpointing")
    if args.allow_cpu:
        cmd.append("--allow-cpu")
    if args.early_stop:
        cmd.extend(
            [
                "--early-stop",
                "--early-stop-patience",
                str(args.early_stop_patience),
                "--early-stop-threshold",
                str(args.early_stop_threshold),
            ]
        )
    run(cmd, env)


def step_eval(args: argparse.Namespace, env: dict[str, str], python: str) -> None:
    cmd = [
        python,
        "-m",
        "mtg_data_model.eval",
        "--model",
        args.ckpt_out,
        "--test",
        args.teacher_out,
    ]
    if args.eval_limit > 0:
        cmd.extend(["--limit", str(args.eval_limit)])
    run(cmd, env)


def step_report(args: argparse.Namespace, env: dict[str, str], python: str) -> None:
    cmd = [
        python,
        "-m",
        "mtg_data_model.train_report",
        "--log-dir",
        args.report_log_dir,
        "--out-dir",
        args.report_out_dir,
        "--diagrams-dir",
        args.report_diagrams_dir,
    ]
    if args.report_log:
        cmd.extend(["--log", args.report_log])
    elif args.report_latest or not args.report_log:
        cmd.append("--latest")
    if args.report_metrics:
        cmd.extend(["--metrics", args.report_metrics])
    run(cmd, env)


def step_eval_report(args: argparse.Namespace, env: dict[str, str], python: str) -> None:
    cmd = [
        python,
        "-m",
        "mtg_data_model.eval_report",
        "--log-dir",
        args.eval_report_log_dir,
        "--out-dir",
        args.eval_report_out_dir,
        "--diagrams-dir",
        args.eval_report_diagrams_dir,
    ]
    if args.eval_report_metrics:
        cmd.extend(["--metrics", args.eval_report_metrics])
    elif args.eval_report_latest or not args.eval_report_metrics:
        cmd.append("--latest")
    run(cmd, env)


def step_package(args: argparse.Namespace, env: dict[str, str], python: str) -> None:
    run(
        [
            python,
            "-m",
            "mtg_data_model.package",
            "--model",
            args.ckpt_out,
            "--out",
            args.export_out,
        ],
        env,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run mtg-data-model tasks separately.")
    p.add_argument(
        "task",
        choices=["prompts", "teacher", "train", "report", "eval", "eval-report", "package", "all"],
        help="Task to run.",
    )
    p.add_argument("--spots", default=r"..\maptogo-data-collector\data\curated\spots.jsonl")
    p.add_argument("--prompts-out", default=r"data\prompts.jsonl")
    p.add_argument("--teacher-out", default=r"data\teacher.jsonl")
    p.add_argument("--teacher-model", default="qwen2.5:32b-instruct-q4_K_M")
    p.add_argument("--backend", choices=["ollama", "transformers"], default="ollama")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--ollama-num-gpu", type=int, default=999)
    p.add_argument("--student-base", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--ckpt-out", default="", help="Checkpoint output dir. If omitted, derived from --student-base.")
    p.add_argument("--export-out", default=r"data\export")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--logging-steps", type=int, default=1)
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device-mode", choices=["cuda0", "auto"], default="cuda0")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--bnb-compute-dtype", choices=["float16", "bfloat16"], default="bfloat16")
    p.add_argument("--early-stop", action="store_true")
    p.add_argument("--early-stop-patience", type=int, default=3)
    p.add_argument("--early-stop-threshold", type=float, default=0.0)
    p.add_argument("--prompts-limit", type=int, default=0, help="Limit spots when building prompts (0=all).")
    p.add_argument("--teacher-limit", type=int, default=0, help="Limit rows when generating teacher outputs (0=all).")
    p.add_argument("--overwrite-teacher", action="store_true", help="Overwrite teacher output instead of resume.")
    p.add_argument("--resume-verify", action="store_true", help="Verify prompts/output alignment before resume.")
    p.add_argument("--eval-limit", type=int, default=0, help="Limit eval rows for this run (0=all remaining).")
    p.add_argument("--report-log", default="", help="Optional path to train log for report.")
    p.add_argument("--report-metrics", default="", help="Optional path to metrics jsonl for report.")
    p.add_argument("--report-log-dir", default="log", help="Folder used to locate latest train log.")
    p.add_argument("--report-out-dir", default=r"data\report", help="Output folder for report files.")
    p.add_argument("--report-diagrams-dir", default=r"data\diagrams", help="Output folder for report chart images.")
    p.add_argument("--report-latest", action="store_true", help="Use latest train log when --report-log is omitted.")
    p.add_argument("--eval-report-metrics", default="", help="Optional path to eval metrics jsonl for eval report.")
    p.add_argument("--eval-report-log-dir", default="log", help="Folder used to locate latest eval metrics.")
    p.add_argument("--eval-report-out-dir", default=r"data\report", help="Output folder for eval report files.")
    p.add_argument("--eval-report-diagrams-dir", default=r"data\diagrams", help="Output folder for eval report charts.")
    p.add_argument("--eval-report-latest", action="store_true", help="Use latest eval metrics when --eval-report-metrics is omitted.")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)
    env = build_env()
    python = sys.executable

    if not args.ckpt_out:
        args.ckpt_out = _default_ckpt_from_model(args.student_base)
        print(f"> auto ckpt-out: {args.ckpt_out}")

    if args.task in ("prompts", "all"):
        step_prompts(args, env, python)
    if args.task in ("teacher", "all"):
        step_teacher(args, env, python)
    if args.task in ("train", "all"):
        step_train(args, env, python)
    if args.task in ("report", "all"):
        step_report(args, env, python)
    if args.task in ("eval", "all"):
        step_eval(args, env, python)
    if args.task in ("eval-report", "all"):
        step_eval_report(args, env, python)
    if args.task in ("package", "all"):
        step_package(args, env, python)

    print(f"Task complete: {args.task}")


if __name__ == "__main__":
    main()
