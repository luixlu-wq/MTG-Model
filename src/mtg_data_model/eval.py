from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
import time
from collections import Counter
from datetime import datetime
from statistics import mean

from .config import settings


def _load(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def _count_rows(path: str) -> int:
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _is_lfs_pointer(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with open(path, "rb") as f:
            head = f.read(256)
    except OSError:
        return False
    return head.startswith(b"version https://git-lfs.github.com/spec/v1")


def _char_f1(pred: str, ref: str) -> float:
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    pred_counter = Counter(pred)
    ref_counter = Counter(ref)
    overlap = 0
    for ch, n in pred_counter.items():
        overlap += min(n, ref_counter.get(ch, 0))
    precision = overlap / max(len(pred), 1)
    recall = overlap / max(len(ref), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _get_reference(row: dict) -> str:
    messages = row.get("messages") or []
    if messages and isinstance(messages[-1], dict) and messages[-1].get("role") == "assistant":
        content = messages[-1].get("content") or ""
        return content if isinstance(content, str) else str(content)
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model id/path")
    ap.add_argument("--test", required=True, help="teacher jsonl or prompts jsonl")
    ap.add_argument("--limit", type=int, default=0, help="max rows to generate this run (0=all remaining)")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--overwrite", action="store_true", help="overwrite eval output instead of resume")
    ap.add_argument("--device-mode", choices=["cuda0", "auto", "cpu"], default="cuda0")
    ap.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--bnb-compute-dtype", choices=["float16", "bfloat16"], default="bfloat16")
    ap.add_argument("--log-every", type=int, default=1, help="Write one detail log line every N evaluated rows.")
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import torch

    os.makedirs(settings.log_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    eval_log_path = os.path.join(settings.log_dir, f"eval-{stamp}.log")
    eval_metrics_path = os.path.join(settings.log_dir, f"eval-{stamp}.metrics.jsonl")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(eval_log_path, encoding="utf-8"), logging.StreamHandler()],
    )
    log = logging.getLogger(__name__)
    if os.path.exists(eval_metrics_path):
        os.remove(eval_metrics_path)
    log.info(
        "eval_start model=%s test=%s limit=%s max_new_tokens=%s overwrite=%s device_mode=%s load_in_4bit=%s bnb_compute_dtype=%s",
        args.model,
        args.test,
        args.limit,
        args.max_new_tokens,
        args.overwrite,
        args.device_mode,
        args.load_in_4bit,
        args.bnb_compute_dtype,
    )
    log.info("eval_metrics_file=%s", eval_metrics_path)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    cuda_ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
    if args.device_mode == "cuda0" and not cuda_ok:
        raise RuntimeError("CUDA not available for --device-mode cuda0. Use --device-mode cpu or auto.")

    model_kwargs = {"trust_remote_code": True}
    if args.device_mode == "cuda0":
        model_kwargs["device_map"] = {"": 0}
    elif args.device_mode == "auto" and cuda_ok:
        model_kwargs["device_map"] = "auto"
    elif args.device_mode == "cpu":
        model_kwargs["device_map"] = {"": "cpu"}

    compute_dtype = torch.bfloat16 if args.bnb_compute_dtype == "bfloat16" else torch.float16
    if args.load_in_4bit and args.device_mode != "cpu":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        model_kwargs["torch_dtype"] = torch.float16 if (cuda_ok and args.device_mode != "cpu") else torch.float32

    is_adapter_dir = os.path.isdir(args.model) and os.path.exists(os.path.join(args.model, "adapter_config.json"))
    if is_adapter_dir:
        adapter_sf = os.path.join(args.model, "adapter_model.safetensors")
        adapter_bin = os.path.join(args.model, "adapter_model.bin")
        if _is_lfs_pointer(adapter_sf):
            raise RuntimeError(
                "adapter_model.safetensors is a Git LFS pointer, not real weights. "
                "Recover weights with `git lfs pull` in mtg-data-model, or retrain to regenerate checkpoint files. "
                f"Pointer file: {adapter_sf}"
            )
        if not os.path.exists(adapter_sf) and not os.path.exists(adapter_bin):
            raise RuntimeError(
                "Adapter checkpoint is missing weight files (adapter_model.safetensors/.bin). "
                f"Checkpoint path: {args.model}"
            )
    if is_adapter_dir:
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()
    log.info("cuda_available=%s cuda_version=%s adapter_dir=%s", cuda_ok, torch.version.cuda, is_adapter_dir)
    if hasattr(model, "hf_device_map"):
        log.info("hf_device_map=%s", getattr(model, "hf_device_map"))

    if hasattr(model, "get_input_embeddings") and model.get_input_embeddings() is not None:
        input_device = model.get_input_embeddings().weight.device
    else:
        input_device = next(model.parameters()).device
    log.info("input_device=%s", str(input_device))

    out_path = os.path.join(os.path.dirname(args.test), "eval_outputs.jsonl")
    resumed_from = 0
    if args.overwrite and os.path.exists(out_path):
        os.remove(out_path)
    elif os.path.exists(out_path):
        resumed_from = _count_rows(out_path)
    total_rows = _count_rows(args.test)
    remaining_rows = max(total_rows - resumed_from, 0)
    planned_rows = min(args.limit, remaining_rows) if args.limit > 0 else remaining_rows
    log.info(
        "eval_output_file=%s resumed_from=%s total_rows=%s remaining_rows=%s planned_rows=%s",
        out_path,
        resumed_from,
        total_rows,
        remaining_rows,
        planned_rows,
    )

    count = 0
    latencies_ms: list[float] = []
    output_tokens_list: list[int] = []
    token_per_sec_list: list[float] = []
    char_f1_list: list[float] = []
    seq_ratio_list: list[float] = []
    exact_match_count = 0
    for idx, row in enumerate(_load(args.test)):
        if idx < resumed_from:
            continue
        in_run_idx = count + 1
        global_progress = (idx + 1) / total_rows * 100.0 if total_rows > 0 else 0.0
        run_progress = (in_run_idx / planned_rows * 100.0) if planned_rows > 0 else 0.0
        spot_name = (((row.get("meta") or {}).get("name")) or "").strip()
        log.info(
            "eval_row_start row_idx=%s written_idx=%s run_idx=%s global_progress=%.2f%% run_progress=%.2f%% name=%s",
            idx,
            resumed_from + count,
            in_run_idx,
            global_progress,
            run_progress,
            spot_name if spot_name else "-",
        )
        messages = row["messages"][:-1] if row["messages"] and row["messages"][-1]["role"] == "assistant" else row["messages"]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok([text], return_tensors="pt")
        inputs = {k: v.to(input_device) for k, v in inputs.items()}
        ref_text = _get_reference(row)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        gen_ids = out[0][inputs["input_ids"].shape[-1]:]
        gen = tok.decode(gen_ids, skip_special_tokens=True)
        row["model_output"] = gen
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        output_tokens = int(gen_ids.shape[-1])
        tokens_per_sec = 0.0 if elapsed_ms <= 0 else output_tokens / (elapsed_ms / 1000.0)
        char_f1 = _char_f1(gen, ref_text) if ref_text else None
        seq_ratio = difflib.SequenceMatcher(None, gen, ref_text).ratio() if ref_text else None
        exact_match = bool(ref_text and gen.strip() == ref_text.strip())
        if exact_match:
            exact_match_count += 1
        latencies_ms.append(elapsed_ms)
        output_tokens_list.append(output_tokens)
        token_per_sec_list.append(tokens_per_sec)
        if char_f1 is not None:
            char_f1_list.append(char_f1)
        if seq_ratio is not None:
            seq_ratio_list.append(seq_ratio)
        metric_row = {
            "row_idx": idx,
            "written_idx": resumed_from + count,
            "prompt_tokens": int(inputs["input_ids"].shape[-1]),
            "output_tokens": output_tokens,
            "latency_ms": round(elapsed_ms, 3),
            "tokens_per_sec": round(tokens_per_sec, 3),
            "reference_chars": len(ref_text),
            "output_chars": len(gen),
            "exact_match": exact_match,
            "char_f1": round(char_f1, 6) if char_f1 is not None else None,
            "seq_ratio": round(seq_ratio, 6) if seq_ratio is not None else None,
        }
        with open(eval_metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metric_row, ensure_ascii=False) + "\n")
        count += 1
        if args.log_every > 0 and (count % args.log_every == 0):
            log.info(
                "eval_row_done row_idx=%s run_idx=%s/%s global_progress=%.2f%% run_progress=%.2f%% name=%s prompt_tok=%s out_tok=%s latency_ms=%.1f tok_s=%.2f char_f1=%s seq_ratio=%s exact=%s",
                idx,
                count,
                planned_rows if planned_rows > 0 else remaining_rows,
                global_progress,
                run_progress,
                spot_name if spot_name else "-",
                metric_row["prompt_tokens"],
                metric_row["output_tokens"],
                elapsed_ms,
                tokens_per_sec,
                f"{char_f1:.4f}" if char_f1 is not None else "NA",
                f"{seq_ratio:.4f}" if seq_ratio is not None else "NA",
                exact_match,
            )
        if args.limit > 0 and count >= args.limit:
            break

    avg_latency_ms = mean(latencies_ms) if latencies_ms else 0.0
    avg_tps = mean(token_per_sec_list) if token_per_sec_list else 0.0
    avg_out_tokens = mean(output_tokens_list) if output_tokens_list else 0.0
    avg_char_f1 = mean(char_f1_list) if char_f1_list else None
    avg_seq_ratio = mean(seq_ratio_list) if seq_ratio_list else None
    exact_match_rate = (exact_match_count / count) if count > 0 else 0.0
    log.info(
        "eval_done written=%s resumed_from=%s total_written=%s avg_latency_ms=%.2f avg_tok_s=%.2f avg_out_tokens=%.2f char_f1=%s seq_ratio=%s exact_rate=%.4f out=%s",
        count,
        resumed_from,
        resumed_from + count,
        avg_latency_ms,
        avg_tps,
        avg_out_tokens,
        f"{avg_char_f1:.4f}" if avg_char_f1 is not None else "NA",
        f"{avg_seq_ratio:.4f}" if avg_seq_ratio is not None else "NA",
        exact_match_rate,
        out_path,
    )
    print(
        f"eval_written={count} resumed_from={resumed_from} total_written={resumed_from + count} out={out_path} "
        f"log={eval_log_path} metrics={eval_metrics_path}"
    )


if __name__ == "__main__":
    main()
