# mtg-data-model

Distillation pipeline for a local tourism model (Qwen teacher -> smaller student).

## Quick Start
```powershell
cd mtg-data-model
.\venv\Scripts\activate
```

## Step 3: Generate prompts
```powershell
python -m mtg_data_model.build_prompts --spots ..\maptogo-data-collector\data\curated\spots.jsonl --out data\prompts.jsonl
```

## Step 4: Teacher generation
```powershell
python -m mtg_data_model.teacher_generate --in data\prompts.jsonl --out data\teacher.jsonl --model Qwen2.5-32B-Instruct
```
Ollama backend:
```powershell
python -m mtg_data_model.teacher_generate --backend ollama --model qwen2.5:32b-instruct-q4_K_M --in data\prompts.jsonl --out data\teacher.jsonl
```
With custom URL:
```powershell
python -m mtg_data_model.teacher_generate --backend ollama --ollama-url http://localhost:11434 --ollama-num-gpu 999 --model qwen2.5:32b-instruct-q4_K_M --in data\prompts.jsonl --out data\teacher.jsonl --resume
```
Resume with alignment check:
```powershell
python -m mtg_data_model.teacher_generate --backend ollama --ollama-url http://localhost:11434 --ollama-num-gpu 999 --model qwen2.5:32b-instruct-q4_K_M --in data\prompts.jsonl --out data\teacher.jsonl --resume --resume-verify
```
Force restart:
```powershell
python -m mtg_data_model.teacher_generate --backend ollama --ollama-url http://localhost:11434 --ollama-num-gpu 999 --model qwen2.5:32b-instruct-q4_K_M --in data\prompts.jsonl --out data\teacher.jsonl --overwrite
```

## Step 5: SFT training (LoRA)
```powershell
python -m mtg_data_model.sft_train --train data\teacher.jsonl --base Qwen/Qwen2.5-32B-Instruct --out data\checkpoints\qwen2.5-32b-lora --device-mode cuda0 --load-in-4bit --gradient-checkpointing --bnb-compute-dtype bfloat16
python -m mtg_data_model.sft_train --train data\teacher.jsonl --base Qwen/Qwen2.5-32B-Instruct --out data\checkpoints\qwen2.5-32b-lora --device-mode cuda0 --load-in-4bit --gradient-checkpointing --bnb-compute-dtype bfloat16 --early-stop --early-stop-patience 3 --early-stop-threshold 0.0001
python -m mtg_data_model.sft_train --train data\teacher.jsonl --base Qwen/Qwen2.5-32B-Instruct --out data\checkpoints\qwen2.5-32b-lora --device-mode cuda0 --load-in-4bit --gradient-checkpointing --bnb-compute-dtype bfloat16 --save-steps 100
python -m mtg_data_model.sft_train --train data\teacher.jsonl --base Qwen/Qwen2.5-32B-Instruct --out data\checkpoints\qwen2.5-32b-lora --device-mode cuda0 --load-in-4bit --gradient-checkpointing --bnb-compute-dtype bfloat16 --logging-steps 1
```
Note: training now fails fast if CUDA is unavailable. Use `--allow-cpu` only for debugging.
Note: for large models, `--device-mode auto` may offload to CPU. Use `--device-mode cuda0` to force GPU or fail fast on OOM.
32B recommended command (QLoRA):
```powershell
python run_tasks.py train --student-base Qwen/Qwen2.5-32B-Instruct --ckpt-out data\checkpoints\qwen2.5-32b-lora --device-mode cuda0 --load-in-4bit --gradient-checkpointing --bnb-compute-dtype bfloat16 --early-stop --early-stop-patience 3 --early-stop-threshold 0.0001
```
Training logs:
- `log/train-<timestamp>.log`
- `log/train-<timestamp>.metrics.jsonl` (step-level metrics, including eval loss/perplexity)

## Step 6: Training report (quality + diagrams)
```powershell
python -m pip install -e .
```

```powershell
python -m mtg_data_model.train_report --out-dir data\report --latest
python -m mtg_data_model.train_report --log log\train-20260216165502.log --out-dir data\report
python -m mtg_data_model.train_report --out-dir data\report --diagrams-dir data\diagrams --latest
```
Outputs:
- `data/report/train-report-<timestamp>.json` (structured summary)
- `data/report/train-report-<timestamp>.md` (human-readable evaluation)
- `data/diagrams/train-report-<timestamp>.loss.png` / `.optim.png` (training charts)

## Step 7: Eval
```powershell
python -m mtg_data_model.eval --model data\checkpoints\qwen2.5-32b-lora --test data\teacher.jsonl
python -m mtg_data_model.eval --model data\checkpoints\qwen2.5-32b-lora --test data\teacher.jsonl --limit 100
python -m mtg_data_model.eval --model data\checkpoints\qwen2.5-32b-lora --test data\teacher.jsonl --overwrite
python -m mtg_data_model.eval --model data\checkpoints\qwen2.5-32b-lora --test data\teacher.jsonl --device-mode auto
python -m mtg_data_model.eval --model data\checkpoints\qwen2.5-32b-lora --test data\teacher.jsonl --log-every 1
```
By default, eval has no limit (`--limit 0`) and resumes from `data\eval_outputs.jsonl` if it exists.
By default, eval uses `--device-mode cuda0 --load-in-4bit` to avoid CPU/CUDA tensor mismatch on large LoRA checkpoints.
Eval logs:
- `log/eval-<timestamp>.log` (row-level + summary)
- `log/eval-<timestamp>.metrics.jsonl` (latency/tokens/similarity metrics per row)

## Step 8: Eval report (quality + diagrams)
```powershell
python -m mtg_data_model.eval_report --out-dir data\report --latest
python -m mtg_data_model.eval_report --metrics log\eval-20260216220000.metrics.jsonl --out-dir data\report
python -m mtg_data_model.eval_report --out-dir data\report --diagrams-dir data\diagrams --latest
```
Outputs:
- `data/report/eval-report-<timestamp>.json` (structured summary)
- `data/report/eval-report-<timestamp>.md` (human-readable evaluation)
- `data/diagrams/eval-report-<timestamp>.runtime.png` / `.quality.png` (evaluation charts)
Note: run Step 7 first to generate `log/eval-<timestamp>.metrics.jsonl`.

## Step 9: Package
```powershell
python -m mtg_data_model.package --model data\checkpoints\qwen2.5-32b-lora --out data\export
```

## Run Tasks Separately
Use `run_tasks.py` to run one step at a time or all:
```powershell
python run_tasks.py prompts
python run_tasks.py teacher
python run_tasks.py teacher --resume-verify
python run_tasks.py teacher --overwrite-teacher
python run_tasks.py train
python run_tasks.py report
python run_tasks.py train --student-base Qwen/Qwen2.5-32B-Instruct --device-mode cuda0
python run_tasks.py eval
python run_tasks.py eval-report
python run_tasks.py package
python run_tasks.py all
python run_tasks.py prompts --prompts-limit 20
python run_tasks.py teacher --teacher-limit 50
python run_tasks.py all --prompts-limit 20 --teacher-limit 50
python run_tasks.py train --save-steps 100
python run_tasks.py train --logging-steps 1
```
Notes:
- If `--ckpt-out` is omitted, `run_tasks.py` auto-derives it from `--student-base`.
- `run_tasks.py report` reads latest train log by default, writes reports to `data\report`, and charts to `data\diagrams`.
- `run_tasks.py eval` defaults to no limit (`--eval-limit 0`) and resumes from existing eval output.
- `run_tasks.py eval-report` reads latest eval metrics by default, writes reports to `data\report`, and charts to `data\diagrams`.

## Notes
- Edit `src/mtg_data_model/prompts.py` to tune style and tasks.
- The scripts assume JSONL with `{"messages":[...], "meta": {...}}`.
- Use `--limit` to iterate quickly.
- `teacher_generate` auto-resumes when output file already exists (unless `--overwrite` is set).
- Install dependencies from `requirements.txt` to enable report chart generation (`matplotlib`).
- `eval` auto-resumes from `data\eval_outputs.jsonl` unless `--overwrite` is set.

