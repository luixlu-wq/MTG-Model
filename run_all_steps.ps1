# End-to-end commands for mtg-data-model
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\run_all_steps.ps1

$ErrorActionPreference = "Stop"

Set-Location "$PSScriptRoot"

# Activate venv
& .\venv\Scripts\Activate.ps1

# Ensure local src is importable
$env:PYTHONPATH = "src"

# ---------------------------
# Config (edit as needed)
# ---------------------------
$SPOTS_PATH = "..\maptogo-data-collector\data\curated\spots.jsonl"
$PROMPTS_PATH = "data\prompts.jsonl"
$TEACHER_PATH = "data\teacher.jsonl"

# Teacher model (Ollama)
$OLLAMA_URL = "http://localhost:11434"
$TEACHER_MODEL = "qwen2.5:32b-instruct-q4_K_M"

# Student train target
$STUDENT_BASE = "Qwen2.5-4B-Instruct"
$CKPT_OUT = "data\checkpoints\qwen2.5-4b-lora"
$EXPORT_OUT = "data\export"

# ---------------------------
# Step 3: Build prompts
# ---------------------------
python -m mtg_data_model.build_prompts `
  --spots $SPOTS_PATH `
  --out $PROMPTS_PATH

# ---------------------------
# Step 4: Teacher generation
# ---------------------------
python -m mtg_data_model.teacher_generate `
  --backend ollama `
  --ollama-url $OLLAMA_URL `
  --model $TEACHER_MODEL `
  --in $PROMPTS_PATH `
  --out $TEACHER_PATH

# ---------------------------
# Step 5: SFT training (LoRA)
# ---------------------------
python -m mtg_data_model.sft_train `
  --train $TEACHER_PATH `
  --base $STUDENT_BASE `
  --out $CKPT_OUT `
  --epochs 2 `
  --lr 1e-4 `
  --batch 1 `
  --grad-accum 8 `
  --max-len 2048

# ---------------------------
# Step 6: Eval
# ---------------------------
python -m mtg_data_model.eval `
  --model $CKPT_OUT `
  --test $TEACHER_PATH `
  --limit 50

# ---------------------------
# Step 7: Package
# ---------------------------
python -m mtg_data_model.package `
  --model $CKPT_OUT `
  --out $EXPORT_OUT

Write-Host "Pipeline finished."
