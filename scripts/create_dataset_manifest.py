#!/usr/bin/env python3
"""Create a deterministic, checksummed manifest of preprocessed NPY/NPZ data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42, help="recorded for reproducibility; ordering remains lexical")
    args = parser.parse_args()
    root = args.dataset.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"dataset directory not found: {root}")
    files = sorted(path for path in root.rglob("*") if path.suffix.lower() in {".npy", ".npz"})
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit must be positive")
        files = files[: args.limit]
    if not files:
        parser.error("no NPY/NPZ files found")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(path) for path in files]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    metadata = {
        "dataset_root": str(root),
        "manifest": str(args.output.resolve()),
        "sample_count": len(files),
        "ordering": "lexical path order",
        "seed": args.seed,
        "manifest_sha256": digest,
        "preprocessing_contract": "files must already match model dtype, color order, normalization, and layout",
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest: {args.output} ({len(files)} samples)")
    print(f"Metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())