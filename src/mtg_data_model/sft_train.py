from __future__ import annotations

import argparse
import os
import logging
import json
import math
from datetime import datetime

from .config import settings


def _normalize_model_id(model_id: str) -> str:
    mid = (model_id or "").strip()
    if "/" in mid:
        # Legacy non-existent alias in this project: map 4B -> 3B.
        if mid.lower() == "qwen/qwen2.5-4b-instruct":
            return "Qwen/Qwen2.5-3B-Instruct"
        return mid
    # Backward-compatible shorthand support:
    # Qwen2.5-4B-Instruct -> Qwen/Qwen2.5-4B-Instruct
    if mid.startswith("Qwen"):
        if mid.lower() == "qwen2.5-4b-instruct":
            return "Qwen/Qwen2.5-3B-Instruct"
        return f"Qwen/{mid}"
    return mid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="teacher jsonl")
    ap.add_argument("--base", required=True, help="base model id/path")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--save-steps", type=int, default=500, help="Checkpoint save interval (steps).")
    ap.add_argument("--logging-steps", type=int, default=1, help="Log interval in steps.")
    ap.add_argument("--val-ratio", type=float, default=0.05, help="Validation split ratio for quality tracking.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--early-stop", action="store_true", help="Enable early stopping based on eval_loss.")
    ap.add_argument("--early-stop-patience", type=int, default=3, help="Early stop patience in eval rounds.")
    ap.add_argument("--early-stop-threshold", type=float, default=0.0, help="Minimum eval_loss improvement to reset patience.")
    ap.add_argument("--allow-cpu", action="store_true", help="Allow CPU training (very slow).")
    ap.add_argument("--load-in-4bit", action="store_true", help="Enable QLoRA 4-bit loading (recommended for 32B).")
    ap.add_argument("--bnb-compute-dtype", choices=["float16", "bfloat16"], default="bfloat16")
    ap.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing to save VRAM.")
    ap.add_argument(
        "--device-mode",
        choices=["cuda0", "auto"],
        default="cuda0",
        help="Device placement: cuda0 forces single GPU; auto may offload to CPU.",
    )
    args = ap.parse_args()

    os.makedirs(settings.log_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    log_path = os.path.join(settings.log_dir, f"train-{stamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )
    log = logging.getLogger(__name__)
    metrics_path = os.path.join(settings.log_dir, f"train-{stamp}.metrics.jsonl")
    log.info(
        "train_start base=%s out=%s epochs=%s lr=%s batch=%s grad_accum=%s max_len=%s save_steps=%s logging_steps=%s early_stop=%s patience=%s threshold=%s load_in_4bit=%s bnb_compute_dtype=%s grad_ckpt=%s",
        args.base,
        args.out,
        args.epochs,
        args.lr,
        args.batch,
        args.grad_accum,
        args.max_len,
        args.save_steps,
        args.logging_steps,
        args.early_stop,
        args.early_stop_patience,
        args.early_stop_threshold,
        args.load_in_4bit,
        args.bnb_compute_dtype,
        args.gradient_checkpointing,
    )
    log.info("train_metrics_file=%s", metrics_path)

    os.makedirs(args.out, exist_ok=True)

    # Lazy import: keep environment light until needed
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        TrainingArguments,
        TrainerCallback,
        EarlyStoppingCallback,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import torch

    base_model = _normalize_model_id(args.base)
    if base_model != args.base:
        log.info("normalized_base_model input=%s normalized=%s", args.base, base_model)
    try:
        tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    except OSError as e:
        raise RuntimeError(
            "Failed to load base model. Use a valid HF model id or local path. "
            "Recommended Qwen students: "
            "Qwen/Qwen2.5-1.5B-Instruct, "
            "Qwen/Qwen2.5-3B-Instruct, "
            "Qwen/Qwen2.5-7B-Instruct. "
            f"Resolved base model: {base_model}"
        ) from e
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cuda_ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
    if not cuda_ok and not args.allow_cpu:
        raise RuntimeError(
            "CUDA is not available in this venv. "
            "Install a CUDA-enabled PyTorch build, or rerun with --allow-cpu."
        )
    if cuda_ok:
        devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        log.info("cuda_available=True cuda_version=%s devices=%s", torch.version.cuda, devices)
    else:
        log.warning("cuda_available=False; running on CPU because --allow-cpu was set")

    compute_dtype = torch.bfloat16 if args.bnb_compute_dtype == "bfloat16" else torch.float16
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": ({"": 0} if (cuda_ok and args.device_mode == "cuda0") else ("auto" if cuda_ok else None)),
    }
    if args.load_in_4bit and cuda_ok:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        model_kwargs["torch_dtype"] = torch.float16 if cuda_ok else torch.float32

    try:
        model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and cuda_ok and args.device_mode == "cuda0":
            raise RuntimeError(
                "GPU OOM while forcing cuda0. "
                "Use a smaller model, enable --load-in-4bit, or retry with --device-mode auto."
            ) from e
        raise

    if cuda_ok and hasattr(model, "hf_device_map"):
        dmap = getattr(model, "hf_device_map") or {}
        log.info("hf_device_map=%s", dmap)
        if args.device_mode == "auto":
            devs = {str(v) for v in dmap.values()}
            if devs and all(("cpu" in d.lower() or "disk" in d.lower()) for d in devs):
                raise RuntimeError(
                    "Model was auto-placed on CPU/disk only; GPU training would be ineffective. "
                    "Retry with --device-mode cuda0, or use smaller/quantized model."
                )

    if args.load_in_4bit and cuda_ok:
        model = prepare_model_for_kbit_training(model)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)

    ds = load_dataset("json", data_files=args.train, split="train")

    def _format(example):
        text = tok.apply_chat_template(example["messages"], tokenize=False)
        # Keep sequence length bounded in preprocessing for TRL/transformers compatibility.
        ids = tok(
            text,
            truncation=True,
            max_length=args.max_len,
            add_special_tokens=False,
        )["input_ids"]
        text_cut = tok.decode(ids, skip_special_tokens=False)
        return {"text": text_cut}

    ds = ds.map(_format, remove_columns=ds.column_names)

    def _tokenize(example):
        toks = tok(
            example["text"],
            truncation=True,
            max_length=args.max_len,
        )
        return toks

    ds = ds.map(_tokenize, remove_columns=ds.column_names)
    split = ds.train_test_split(test_size=args.val_ratio, seed=args.seed) if 0 < args.val_ratio < 1 else {"train": ds, "test": ds.select([])}
    train_ds = split["train"]
    eval_ds = split["test"] if len(split["test"]) > 0 else None
    log.info("dataset_rows total=%s train=%s eval=%s", len(ds), len(train_ds), len(eval_ds) if eval_ds is not None else 0)

    class JsonlMetricsCallback(TrainerCallback):
        def __init__(self, path: str, logger: logging.Logger):
            self.path = path
            self.log = logger
            if os.path.exists(self.path):
                os.remove(self.path)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            row = {"step": int(state.global_step), "epoch": float(state.epoch or 0.0)}
            row.update({k: float(v) if isinstance(v, (int, float)) else v for k, v in logs.items()})
            if "eval_loss" in row:
                try:
                    row["eval_ppl"] = float(math.exp(row["eval_loss"]))
                except OverflowError:
                    row["eval_ppl"] = float("inf")
            with open(self.path, "a", encoding="utf-8") as mf:
                mf.write(json.dumps(row, ensure_ascii=False) + "\n")
            # Single-line console log for quick quality tracking.
            loss = row.get("loss")
            eval_loss = row.get("eval_loss")
            eval_ppl = row.get("eval_ppl")
            lr = row.get("learning_rate")
            gn = row.get("grad_norm")
            self.log.info(
                "train_log step=%s epoch=%.3f loss=%s eval_loss=%s eval_ppl=%s lr=%s grad_norm=%s",
                row["step"],
                row["epoch"],
                f"{loss:.4f}" if isinstance(loss, float) else loss,
                f"{eval_loss:.4f}" if isinstance(eval_loss, float) else eval_loss,
                f"{eval_ppl:.2f}" if isinstance(eval_ppl, float) and math.isfinite(eval_ppl) else eval_ppl,
                f"{lr:.2e}" if isinstance(lr, float) else lr,
                f"{gn:.4f}" if isinstance(gn, float) else gn,
            )

    args_train = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=100 if eval_ds is not None else None,
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True if eval_ds is not None else False,
        metric_for_best_model="eval_loss" if eval_ds is not None else None,
        greater_is_better=False if eval_ds is not None else None,
        fp16=(cuda_ok and args.bnb_compute_dtype == "float16"),
        bf16=(cuda_ok and args.bnb_compute_dtype == "bfloat16"),
        gradient_checkpointing=args.gradient_checkpointing,
        optim="paged_adamw_8bit" if (args.load_in_4bit and cuda_ok) else "adamw_torch",
        report_to="none",
    )

    callbacks = [JsonlMetricsCallback(metrics_path, log)]
    if args.early_stop and eval_ds is not None and len(eval_ds) > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stop_patience,
                early_stopping_threshold=args.early_stop_threshold,
            )
        )

    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=args_train,
        data_collator=collator,
        callbacks=callbacks,
    )
    train_result = trainer.train()
    if eval_ds is not None and len(eval_ds) > 0:
        eval_metrics = trainer.evaluate()
        eval_loss = float(eval_metrics.get("eval_loss", float("nan")))
        try:
            eval_ppl = math.exp(eval_loss)
        except OverflowError:
            eval_ppl = float("inf")
        log.info("final_eval eval_loss=%.6f eval_ppl=%s", eval_loss, f"{eval_ppl:.2f}" if math.isfinite(eval_ppl) else "inf")
    log.info("train_runtime_seconds=%s train_samples_per_second=%s train_steps_per_second=%s",
             train_result.metrics.get("train_runtime"),
             train_result.metrics.get("train_samples_per_second"),
             train_result.metrics.get("train_steps_per_second"))
    trainer.save_model(args.out)
    log.info("train_done out=%s log=%s metrics=%s", args.out, log_path, metrics_path)


if __name__ == "__main__":
    main()
