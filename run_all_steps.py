from __future__ import annotations

import os
import subprocess
import sys


def run(cmd: list[str], env: dict[str, str]) -> None:
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)

    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    python = sys.executable

    spots_path = r"..\maptogo-data-collector\data\curated\spots.jsonl"
    prompts_path = r"data\prompts.jsonl"
    teacher_path = r"data\teacher.jsonl"

    ollama_url = "http://localhost:11434"
    ollama_num_gpu = "999"
    teacher_model = "qwen2.5:32b-instruct-q4_K_M"

    student_base = "Qwen/Qwen2.5-3B-Instruct"
    ckpt_out = r"data\checkpoints\qwen2.5-3b-lora"
    export_out = r"data\export"

    run(
        [
            python,
            "-m",
            "mtg_data_model.build_prompts",
            "--spots",
            spots_path,
            "--out",
            prompts_path,
        ],
        env,
    )

    run(
        [
            python,
            "-m",
            "mtg_data_model.teacher_generate",
            "--backend",
            "ollama",
            "--ollama-url",
            ollama_url,
            "--ollama-num-gpu",
            ollama_num_gpu,
            "--model",
            teacher_model,
            "--in",
            prompts_path,
            "--out",
            teacher_path,
        ],
        env,
    )

    run(
        [
            python,
            "-m",
            "mtg_data_model.sft_train",
            "--train",
            teacher_path,
            "--base",
            student_base,
            "--out",
            ckpt_out,
            "--epochs",
            "2",
            "--lr",
            "1e-4",
            "--batch",
            "1",
            "--grad-accum",
            "8",
            "--max-len",
            "2048",
        ],
        env,
    )

    run(
        [
            python,
            "-m",
            "mtg_data_model.train_report",
            "--out-dir",
            r"data\report",
            "--diagrams-dir",
            r"data\diagrams",
            "--latest",
        ],
        env,
    )

    run(
        [
            python,
            "-m",
            "mtg_data_model.eval",
            "--model",
            ckpt_out,
            "--test",
            teacher_path,
        ],
        env,
    )

    run(
        [
            python,
            "-m",
            "mtg_data_model.eval_report",
            "--out-dir",
            r"data\report",
            "--diagrams-dir",
            r"data\diagrams",
            "--latest",
        ],
        env,
    )

    run(
        [
            python,
            "-m",
            "mtg_data_model.package",
            "--model",
            ckpt_out,
            "--out",
            export_out,
        ],
        env,
    )

    print("Pipeline finished.")


if __name__ == "__main__":
    main()
