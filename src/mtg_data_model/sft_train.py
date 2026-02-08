from __future__ import annotations

import argparse
import os
import logging
from datetime import datetime

from .config import settings


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
    log.info("train_start base=%s out=%s epochs=%s lr=%s batch=%s grad_accum=%s max_len=%s",
             args.base, args.out, args.epochs, args.lr, args.batch, args.grad_accum, args.max_len)

    os.makedirs(args.out, exist_ok=True)

    # Lazy import: keep environment light until needed
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer
    import torch

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )

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
        return {"text": tok.apply_chat_template(example["messages"], tokenize=False)}

    ds = ds.map(_format, remove_columns=ds.column_names)

    args_train = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=20,
        save_steps=500,
        fp16=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        args=args_train,
        max_seq_length=args.max_len,
        tokenizer=tok,
        dataset_text_field="text",
    )
    trainer.train()
    trainer.save_model(args.out)
    log.info("train_done out=%s", args.out)


if __name__ == "__main__":
    main()
