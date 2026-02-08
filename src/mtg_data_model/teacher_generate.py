from __future__ import annotations

import argparse
import json
import os


def _load_prompts(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def _gen_with_transformers(model_id: str, prompts, max_new_tokens: int, temperature: float):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )

    for row in prompts:
        messages = row["messages"]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok([text], return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
        gen = tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        row["messages"].append({"role": "assistant", "content": gen})
        yield row


def _gen_with_ollama(model_id: str, prompts, max_new_tokens: int, temperature: float, base_url: str):
    import httpx

    base = base_url.rstrip("/")
    url = f"{base}/api/chat"
    for row in prompts:
        messages = row["messages"]
        payload = {
            "model": model_id,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_new_tokens,
            },
        }
        r = httpx.post(url, json=payload, timeout=600.0)
        r.raise_for_status()
        data = r.json()
        content = (data.get("message") or {}).get("content") or ""
        row["messages"].append({"role": "assistant", "content": content})
        yield row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="prompts jsonl")
    ap.add_argument("--out", required=True, help="teacher jsonl")
    ap.add_argument("--model", required=True, help="teacher model id/path")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--backend", choices=["transformers", "ollama"], default="transformers")
    ap.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if os.path.exists(args.out):
        os.remove(args.out)

    count = 0
    prompts = _load_prompts(args.inp)
    if args.backend == "ollama":
        gen = _gen_with_ollama(args.model, prompts, args.max_new_tokens, args.temperature, args.ollama_url)
    else:
        gen = _gen_with_transformers(args.model, prompts, args.max_new_tokens, args.temperature)
    for row in gen:
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        count += 1
        if args.limit and count >= args.limit:
            break

    print(f"teacher_written={count}")


if __name__ == "__main__":
    main()
