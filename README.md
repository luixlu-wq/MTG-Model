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
python -m mtg_data_model.teacher_generate --backend ollama --ollama-url http://localhost:11434 --model qwen2.5:32b-instruct-q4_K_M --in data\prompts.jsonl --out data\teacher.jsonl
```

## Step 5: SFT training (LoRA)
```powershell
python -m mtg_data_model.sft_train --train data\teacher.jsonl --base Qwen2.5-4B-Instruct --out data\checkpoints\qwen2.5-4b-lora
```

## Step 6: Eval
```powershell
python -m mtg_data_model.eval --model data\checkpoints\qwen2.5-4b-lora --test data\teacher.jsonl
```

## Step 7: Package
```powershell
python -m mtg_data_model.package --model data\checkpoints\qwen2.5-4b-lora --out data\export
```

## Notes
- Edit `src/mtg_data_model/prompts.py` to tune style and tasks.
- The scripts assume JSONL with `{"messages":[...], "meta": {...}}`.
- Use `--limit` to iterate quickly.
