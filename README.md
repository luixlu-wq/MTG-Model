# mtg-data-model

Distillation pipeline for a local China 5A tourism model:

- Teacher generation (Qwen/Ollama) -> `teacher.jsonl`
- Student SFT (LoRA) -> checkpoint
- Train/Eval reporting -> JSON/Markdown/diagrams

This project is designed to work with data produced by `maptogo-data-collector`.

## Purpose of Each Stage

1) Build prompts  
Purpose: convert curated spot records into structured training prompts.

2) Teacher generation  
Purpose: ask a stronger model to produce high-quality target answers for SFT.

3) SFT training  
Purpose: train LoRA adapters on teacher outputs.

4) Train report  
Purpose: analyze training behavior (loss, eval loss, speed, gradients) and generate diagrams.

5) Eval generation  
Purpose: run the trained model on test prompts and collect per-row runtime/quality signals.

6) Eval report  
Purpose: summarize eval metrics and generate runtime/quality diagrams.

7) Package  
Purpose: export/package checkpoint artifacts (currently placeholder export marker).

## Directory Roles

- `src/mtg_data_model/` -> pipeline code
- `data/prompts.jsonl` -> generated prompts
- `data/teacher.jsonl` -> teacher outputs used for SFT
- `data/checkpoints/...` -> LoRA checkpoints
- `data/eval_outputs.jsonl` -> eval generations
- `data/report/` -> train/eval report JSON+MD
- `data/diagrams/` -> charts from train/eval report
- `log/` -> run logs and metrics JSONL files

## Environment Setup

```powershell
cd mtg-data-model
.\venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

Why `-e .`: lets you run `python -m mtg_data_model.*` without manually setting `PYTHONPATH`.

## Quick Command Cheat Sheet

```powershell
# Full pipeline (all steps)
python run_tasks.py all

# 1) Build prompts
python run_tasks.py prompts

# 2) Generate teacher outputs
python run_tasks.py teacher --resume-verify

# 3) Train LoRA student
python run_tasks.py train --student-base Qwen/Qwen2.5-32B-Instruct --device-mode cuda0 --load-in-4bit --gradient-checkpointing

# 4) Training quality report + diagrams
python run_tasks.py report

# 5) Eval generation
python run_tasks.py eval

# 6) Eval quality report + diagrams
python run_tasks.py eval-report

# 7) Package/export
python run_tasks.py package
```

## Step-by-Step Commands

## Step 3: Build prompts
Purpose: transform each input spot into multiple prompt types (general, itinerary, bilingual).

Input:
- `..\maptogo-data-collector\data\curated\spots.jsonl`

Output:
- `data\prompts.jsonl`

```powershell
python -m mtg_data_model.build_prompts --spots ..\maptogo-data-collector\data\curated\spots.jsonl --out data\prompts.jsonl
```

Useful flag:
- `--limit N` to iterate quickly.

## Step 4: Teacher generation
Purpose: generate assistant answers for each prompt row.

Input:
- `data\prompts.jsonl`

Output:
- `data\teacher.jsonl`

Ollama (recommended for local 32B inference):
```powershell
python -m mtg_data_model.teacher_generate --backend ollama --ollama-url http://localhost:11434 --ollama-num-gpu 999 --model qwen2.5:32b-instruct-q4_K_M --in data\prompts.jsonl --out data\teacher.jsonl --resume
```

Resume safety check:
```powershell
python -m mtg_data_model.teacher_generate --backend ollama --ollama-url http://localhost:11434 --ollama-num-gpu 999 --model qwen2.5:32b-instruct-q4_K_M --in data\prompts.jsonl --out data\teacher.jsonl --resume --resume-verify
```

Force restart:
```powershell
python -m mtg_data_model.teacher_generate --backend ollama --ollama-url http://localhost:11434 --ollama-num-gpu 999 --model qwen2.5:32b-instruct-q4_K_M --in data\prompts.jsonl --out data\teacher.jsonl --overwrite
```

Logs:
- `log/teacher-<timestamp>.log`

## Step 5: SFT training (LoRA)
Purpose: train LoRA adapter using teacher rows.

Input:
- `data\teacher.jsonl`

Output:
- `data\checkpoints\qwen2.5-32b-lora`

```powershell
python run_tasks.py train --student-base Qwen/Qwen2.5-32B-Instruct --ckpt-out data\checkpoints\qwen2.5-32b-lora --device-mode cuda0 --load-in-4bit --gradient-checkpointing --bnb-compute-dtype bfloat16 --early-stop --early-stop-patience 3 --early-stop-threshold 0.0001
```

Training logs:
- `log/train-<timestamp>.log`
- `log/train-<timestamp>.metrics.jsonl`

## Step 6: Train report (quality + diagrams)
Purpose: summarize training quality and performance from train logs.

Output:
- `data/report/train-report-<timestamp>.json`
- `data/report/train-report-<timestamp>.md`
- `data/diagrams/train-report-<timestamp>.loss.png`
- `data/diagrams/train-report-<timestamp>.optim.png`

```powershell
python -m mtg_data_model.train_report --out-dir data\report --diagrams-dir data\diagrams --latest
```

## Step 7: Eval generation
Purpose: run the trained model on test rows and capture detailed row-level metrics.

Input:
- model checkpoint dir
- `data\teacher.jsonl` (or any compatible test JSONL)

Output:
- `data\eval_outputs.jsonl`
- `log/eval-<timestamp>.log`
- `log/eval-<timestamp>.metrics.jsonl`

```powershell
python -m mtg_data_model.eval --model data\checkpoints\qwen2.5-32b-lora --test data\teacher.jsonl --log-every 1
```

Common flags:
- `--limit N` (default `0`, means all remaining rows)
- `--overwrite` (restart instead of resume)
- `--device-mode cuda0|auto|cpu`
- `--load-in-4bit / --no-load-in-4bit`

## Step 8: Eval report (quality + diagrams)
Purpose: summarize eval metrics (latency, throughput, similarity, exact-match) and chart trends.

Output:
- `data/report/eval-report-<timestamp>.json`
- `data/report/eval-report-<timestamp>.md`
- `data/diagrams/eval-report-<timestamp>.runtime.png`
- `data/diagrams/eval-report-<timestamp>.quality.png`

```powershell
python -m mtg_data_model.eval_report --out-dir data\report --diagrams-dir data\diagrams --latest
```

## Step 9: Package
Purpose: package/export model artifacts for downstream usage.

```powershell
python -m mtg_data_model.package --model data\checkpoints\qwen2.5-32b-lora --out data\export
```

Note: current package step writes a marker README (placeholder export flow).

## run_tasks.py Tasks and Purpose

`run_tasks.py` is the orchestrator for single-step or full pipeline execution.

- `prompts` -> build prompts
- `teacher` -> teacher generation
- `train` -> SFT
- `report` -> train report
- `eval` -> eval generation
- `eval-report` -> eval report
- `package` -> package output
- `all` -> run full sequence in order

Examples:
```powershell
python run_tasks.py prompts
python run_tasks.py teacher --resume-verify
python run_tasks.py train --student-base Qwen/Qwen2.5-32B-Instruct --device-mode cuda0
python run_tasks.py report
python run_tasks.py eval
python run_tasks.py eval-report
python run_tasks.py package
python run_tasks.py all
```

## Function-Level Reference (Main Modules)

- `build_prompts.py`
  - Purpose: convert each spot record into multiple supervised prompt styles so the student sees varied task shapes.
  - Core flow: load `spots.jsonl` -> call `build_prompt`, `build_itinerary_prompt`, `build_bilingual_prompt` -> write JSONL rows with `messages` + `meta`.
  - Key inputs: `--spots`, `--out`, optional `--limit`.
  - Key outputs: one combined prompt file (`data/prompts.jsonl`) used by teacher generation.
  - Operational note: existing output is deleted before regeneration (safe reset behavior).


- `prompts.py`
  - Purpose: define all prompt templates and response contracts.
  - Core functions:
    - `build_prompt`: rich structured JSON request (history, culture, logistics, routes, geo, myths, photo spots).
    - `build_itinerary_prompt`: step-by-step navigation style output with route steps and optional coordinates.
    - `build_bilingual_prompt`: Chinese + English narration style output.
  - Why it matters: this file controls label style/format; changes here directly affect teacher output distribution and SFT behavior.
  - Best use: update this file when you want different model behavior without touching training code.

- `teacher_generate.py`
  - Purpose: generate teacher labels from prompts using either Ollama or Transformers backend.
  - Core flow: read prompts -> optional resume alignment check -> generate assistant answer per row -> append to output JSONL.
  - Resume safety:
    - auto-resume when output exists (unless `--overwrite`)
    - `--resume-verify` checks message alignment between prompt and existing output rows.
  - Key runtime features: per-row timing/token logs, backend selection, timeout control, and row limit.
  - Key outputs: `data/teacher.jsonl` and `log/teacher-<timestamp>.log`.

- `sft_train.py`
  - Purpose: train LoRA adapters for the student model on teacher data.
  - Core flow:
    - normalize base model id
    - load tokenizer/model (optional 4-bit QLoRA path)
    - create train/eval split
    - run Trainer loop with periodic eval/checkpoints
    - save adapter checkpoint.
  - Quality controls: early stopping, eval loss/perplexity tracking, deterministic seed, split ratio.
  - Performance controls: `--load-in-4bit`, `--gradient-checkpointing`, `--device-mode`, `--bnb-compute-dtype`.
  - Key outputs:
    - checkpoint dir under `data/checkpoints/...`
    - `log/train-<timestamp>.log`
    - `log/train-<timestamp>.metrics.jsonl` (step metrics for chart/report pipeline).

- `train_report.py`
  - Purpose: convert training logs/metrics into actionable quality reports and diagrams.
  - Core flow: parse train log + metrics -> compute trends (loss reduction, eval best/final, grad norms, step-time drift) -> write JSON/MD/charts.
  - Main artifacts:
    - summary JSON/Markdown in `data/report`
    - loss and optimizer charts in `data/diagrams`.
  - Practical value: quickly decide if a checkpoint is healthy enough for eval/package without manual log inspection.

- `eval.py`
  - Purpose: run inference on test prompts and collect row-level model performance signals.
  - Core flow:
    - load checkpoint (PEFT adapter-aware)
    - resume from existing `eval_outputs.jsonl`
    - generate output row-by-row
    - compute per-row runtime and similarity metrics
    - write outputs + detailed logs.
  - Per-row metrics include: prompt/output tokens, latency, tokens/sec, char-F1, sequence ratio, exact-match flag.
  - Safety checks:
    - device mode handling (`cuda0|auto|cpu`)
    - 4-bit load options
    - LFS pointer detection for broken safetensors files.
  - Key outputs:
    - `data/eval_outputs.jsonl`
    - `log/eval-<timestamp>.log`
    - `log/eval-<timestamp>.metrics.jsonl`.

- `eval_report.py`
  - Purpose: summarize eval metrics into quality/performance reports and diagrams.
  - Core flow: load latest or specified eval metrics file -> aggregate latency/throughput/similarity -> generate JSON/MD + charts.
  - Typical indicators: avg/p50/p95 latency, avg tokens/sec, output length trend, char-F1/sequence similarity means.
  - Main artifacts:
    - `data/report/eval-report-<timestamp>.json`
    - `data/report/eval-report-<timestamp>.md`
    - runtime/quality charts in `data/diagrams`.

- `package.py`
  - Purpose: package the trained model artifact for downstream deployment/export steps.
  - Current status: placeholder implementation writing an export marker file (`README.txt`) in output dir.
  - Intended next step: replace with real exporter (for example GGUF conversion or merged checkpoint export).

## Troubleshooting

## `SafetensorError: header too large`
Cause: checkpoint file is a Git LFS pointer text file, not real safetensors data.

Fix (run in `mtg-data-model` root):
```powershell
git lfs pull
git lfs checkout
```

Quick validation:
- pointer file size is usually ~100-200 bytes
- real adapter safetensors is typically MBs

## `Expected all tensors to be on the same device`
Fix:
- try `--device-mode auto`
- keep `--load-in-4bit` enabled for large checkpoints
- avoid mixing CPU/GPU offload settings manually

## Charts not generated
Cause: matplotlib missing.

Fix:
```powershell
python -m pip install -r requirements.txt
```

## `ModuleNotFoundError: mtg_data_model`
Fix:
```powershell
python -m pip install -e .
```
