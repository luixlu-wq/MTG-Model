from __future__ import annotations

import argparse
import json
import os


def _load(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model id/path")
    ap.add_argument("--test", required=True, help="teacher jsonl or prompts jsonl")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )

    out_path = os.path.join(os.path.dirname(args.test), "eval_outputs.jsonl")
    if os.path.exists(out_path):
        os.remove(out_path)

    count = 0
    for row in _load(args.test):
        messages = row["messages"][:-1] if row["messages"] and row["messages"][-1]["role"] == "assistant" else row["messages"]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok([text], return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        gen = tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        row["model_output"] = gen
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        count += 1
        if count >= args.limit:
            break

    print(f"eval_written={count} out={out_path}")


if __name__ == "__main__":
    main()
