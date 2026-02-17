from __future__ import annotations

import argparse
import json
import os
import logging
import time
import itertools
from datetime import datetime

from .config import settings


def _load_prompts(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def _count_valid_jsonl_rows(path: str) -> int:
    if not os.path.exists(path):
        return 0
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                json.loads(line)
                count += 1
            except Exception:
                # Ignore malformed trailing lines and resume from last valid row.
                break
    return count


def _messages_without_assistant_tail(row: dict) -> list[dict]:
    msgs = row.get("messages") or []
    if msgs and (msgs[-1].get("role") == "assistant"):
        return msgs[:-1]
    return msgs


def _verify_resume_alignment(prompts_path: str, out_path: str, rows_to_check: int, log: logging.Logger) -> None:
    if rows_to_check <= 0:
        return
    with open(prompts_path, "r", encoding="utf-8") as pf, open(out_path, "r", encoding="utf-8") as of:
        checked = 0
        for idx in range(1, rows_to_check + 1):
            p_line = pf.readline()
            o_line = of.readline()
            if not p_line or not o_line:
                raise ValueError(f"resume_verify failed at row {idx}: missing line in prompts or output")
            if not p_line.strip() or not o_line.strip():
                raise ValueError(f"resume_verify failed at row {idx}: blank line encountered")
            p_obj = json.loads(p_line)
            o_obj = json.loads(o_line)
            p_msgs = p_obj.get("messages") or []
            o_msgs = _messages_without_assistant_tail(o_obj)
            if p_msgs != o_msgs:
                p_meta = p_obj.get("meta") or {}
                o_meta = o_obj.get("meta") or {}
                raise ValueError(
                    f"resume_verify mismatch at row {idx}: "
                    f"prompt_meta={p_meta} output_meta={o_meta}"
                )
            checked += 1
    log.info("resume_verify_ok checked_rows=%s", checked)


def _gen_with_transformers(model_id: str, prompts, max_new_tokens: int, temperature: float, log: logging.Logger):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )

    for idx, row in enumerate(prompts, start=1):
        meta = row.get("meta") or {}
        spot = meta.get("spot", "?")
        ptype = meta.get("type", "general")
        messages = row["messages"]
        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        log.info("[%s] START spot=%s type=%s prompt_chars=%s", idx, spot, ptype, prompt_chars)

        t0 = time.perf_counter()
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok([text], return_tensors="pt").to(model.device)
        input_tokens = inputs["input_ids"].shape[-1]
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
        elapsed = time.perf_counter() - t0
        gen = tok.decode(out[0][input_tokens:], skip_special_tokens=True)
        output_tokens = out[0].shape[-1] - input_tokens
        tps = output_tokens / elapsed if elapsed > 0 else 0

        row["messages"].append({"role": "assistant", "content": gen})
        log.info("[%s] DONE  spot=%s type=%s elapsed=%.1fs in_tok=%s out_tok=%s tok/s=%.1f resp_chars=%s",
                 idx, spot, ptype, elapsed, input_tokens, output_tokens, tps, len(gen))
        yield row


def _gen_with_ollama(
    model_id: str,
    prompts,
    max_new_tokens: int,
    temperature: float,
    base_url: str,
    timeout: float,
    ollama_num_gpu: int,
    log: logging.Logger,
):
    import httpx

    base = base_url.rstrip("/")
    url = f"{base}/api/chat"
    for idx, row in enumerate(prompts, start=1):
        meta = row.get("meta") or {}
        spot = meta.get("spot", "?")
        ptype = meta.get("type", "general")
        messages = row["messages"]
        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        log.info("[%s] START spot=%s type=%s prompt_chars=%s", idx, spot, ptype, prompt_chars)

        t0 = time.perf_counter()
        payload = {
            "model": model_id,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_new_tokens,
                "num_gpu": ollama_num_gpu,
            },
        }
        r = httpx.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        elapsed = time.perf_counter() - t0
        data = r.json()
        content = (data.get("message") or {}).get("content") or ""

        eval_count = data.get("eval_count", 0)
        prompt_eval_count = data.get("prompt_eval_count", 0)
        eval_duration_ns = data.get("eval_duration", 0)
        tps = (eval_count / (eval_duration_ns / 1e9)) if eval_duration_ns > 0 else 0

        row["messages"].append({"role": "assistant", "content": content})
        log.info("[%s] DONE  spot=%s type=%s elapsed=%.1fs in_tok=%s out_tok=%s tok/s=%.1f resp_chars=%s",
                 idx, spot, ptype, elapsed, prompt_eval_count, eval_count, tps, len(content))
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
    ap.add_argument("--ollama-num-gpu", type=int, default=999, help="Ollama num_gpu option; larger values force more GPU layers.")
    ap.add_argument("--timeout", type=float, default=1800, help="HTTP timeout in seconds (default 1800)")
    ap.add_argument("--resume", action="store_true", help="Resume from existing output file.")
    ap.add_argument("--resume-verify", action="store_true", help="Verify prompts/output alignment before resume.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output file and start from scratch.")
    args = ap.parse_args()

    os.makedirs(settings.log_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    log_path = os.path.join(settings.log_dir, f"teacher-{stamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )
    log = logging.getLogger(__name__)
    log.info("teacher_start backend=%s model=%s max_new_tokens=%s temperature=%s in=%s out=%s",
             args.backend, args.model, args.max_new_tokens, args.temperature, args.inp, args.out)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.overwrite and os.path.exists(args.out):
        os.remove(args.out)

    resume_rows = 0
    resume_enabled = (not args.overwrite) and (args.resume or os.path.exists(args.out))
    if resume_enabled:
        resume_rows = _count_valid_jsonl_rows(args.out)
        if os.path.exists(args.out) and not args.resume:
            log.info("teacher_resume auto-enabled because output file exists and --overwrite was not set")
    if resume_rows:
        log.info("teacher_resume out=%s resume_rows=%s", args.out, resume_rows)
        if args.resume_verify:
            _verify_resume_alignment(args.inp, args.out, resume_rows, log)

    count = 0
    errors = 0
    t_start = time.perf_counter()
    prompts = _load_prompts(args.inp)
    if resume_rows:
        prompts = itertools.islice(prompts, resume_rows, None)
    if args.backend == "ollama":
        gen = _gen_with_ollama(
            args.model,
            prompts,
            args.max_new_tokens,
            args.temperature,
            args.ollama_url,
            args.timeout,
            args.ollama_num_gpu,
            log,
        )
    else:
        gen = _gen_with_transformers(args.model, prompts, args.max_new_tokens, args.temperature, log)
    for row in gen:
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        resp = row["messages"][-1].get("content", "")
        if not resp.strip():
            errors += 1
            log.warning("[%s] EMPTY response", count + 1)
        count += 1
        if args.limit and count >= args.limit:
            break

    total_elapsed = time.perf_counter() - t_start
    avg = total_elapsed / count if count else 0
    total_rows = resume_rows + count
    log.info("teacher_done written_now=%s total_rows=%s empty=%s total_time=%.1fs avg=%.1fs/row log=%s out=%s",
             count, total_rows, errors, total_elapsed, avg, log_path, args.out)
    print(f"teacher_written_now={count} teacher_total_rows={total_rows} empty={errors} total_time={total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
