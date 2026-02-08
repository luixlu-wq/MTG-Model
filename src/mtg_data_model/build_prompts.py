from __future__ import annotations

import argparse
import json
import os
from typing import Iterable

from .prompts import build_prompt, build_itinerary_prompt, build_bilingual_prompt


def _iter_spots(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spots", required=True, help="spots.jsonl path")
    ap.add_argument("--out", required=True, help="output jsonl")
    ap.add_argument("--limit", type=int, default=0, help="limit spots")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if os.path.exists(args.out):
        if os.path.isfile(args.out):
            os.remove(args.out)
        else:
            raise ValueError(f"Output path is not a file: {args.out}")

    count = 0
    with open(args.out, "a", encoding="utf-8") as f:
        for spot in _iter_spots(args.spots):
            msgs = build_prompt(spot)
            f.write(json.dumps({"messages": msgs, "meta": {"spot": spot.get("name")}}, ensure_ascii=False) + "\n")
            msgs2 = build_itinerary_prompt(spot)
            f.write(json.dumps({"messages": msgs2, "meta": {"spot": spot.get("name"), "type": "itinerary"}}, ensure_ascii=False) + "\n")
            msgs3 = build_bilingual_prompt(spot)
            f.write(json.dumps({"messages": msgs3, "meta": {"spot": spot.get("name"), "type": "bilingual"}}, ensure_ascii=False) + "\n")
            count += 1
            if args.limit and count >= args.limit:
                break

    print(f"prompts_written={count*2}")


if __name__ == "__main__":
    main()
