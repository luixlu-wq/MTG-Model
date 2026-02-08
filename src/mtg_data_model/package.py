from __future__ import annotations

import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model id/path")
    ap.add_argument("--out", required=True, help="output dir")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Placeholder: model packaging can be GGUF export or safetensors copy
    # For now, just write a marker file.
    marker = os.path.join(args.out, "README.txt")
    with open(marker, "w", encoding="utf-8") as f:
        f.write("Package export placeholder. Use llama.cpp or your preferred tool to export GGUF.\n")
        f.write(f"Source model: {args.model}\n")

    print(f"package_written={marker}")


if __name__ == "__main__":
    main()
