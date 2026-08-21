#!/usr/bin/env python3
"""Compare FP16 and explicit-Q/DQ TensorRT profile summaries against P2 gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp16", required=True, type=Path, help="FP16 *_summary.json")
    parser.add_argument("--qdq", required=True, type=Path, help="Q/DQ INT8-FP16 *_summary.json")
    parser.add_argument("--output", type=Path, default=Path("results/p2_profile_acceptance.json"))
    parser.add_argument("--min-int8-share", type=float, default=20.0)
    parser.add_argument("--max-fp32-share", type=float, default=10.0)
    parser.add_argument("--max-matmul-ms", type=float, default=5.0)
    parser.add_argument("--max-pre-conv8-ms", type=float, help="optional vertical+horizontal pre_conv8 family limit")
    parser.add_argument("--max-reformat-ratio", type=float, default=2.0)
    parser.add_argument("--max-total-ratio", type=float, default=1.0)
    args = parser.parse_args()

    fp16, qdq = load(args.fp16), load(args.qdq)
    if "p2_acceptance_metrics" not in fp16 or "p2_acceptance_metrics" not in qdq:
        parser.error("regenerate both summaries with the current summarize_profile.py")
    fp16_p2 = fp16["p2_acceptance_metrics"]
    qdq_p2 = qdq["p2_acceptance_metrics"]
    fp16_total = float(fp16["summed_average_layer_time_ms"])
    qdq_total = float(qdq["summed_average_layer_time_ms"])
    fp16_reformat = float(fp16_p2["reformat_copy"]["average_ms"])
    qdq_reformat = float(qdq_p2["reformat_copy"]["average_ms"])
    reformat_ratio = qdq_reformat / fp16_reformat if fp16_reformat > 0 else (0.0 if qdq_reformat == 0 else float("inf"))
    checks = {
        "int8_share": {
            "value": qdq_p2["int8"]["share"], "limit": args.min_int8_share,
            "passed": qdq_p2["int8"]["share"] >= args.min_int8_share,
        },
        "fp32_share": {
            "value": qdq_p2["fp32"]["share"], "limit": args.max_fp32_share,
            "passed": qdq_p2["fp32"]["share"] <= args.max_fp32_share,
        },
        "matmul_ms": {
            "value": qdq_p2["matmul"]["average_ms"], "limit": args.max_matmul_ms,
            "passed": qdq_p2["matmul"]["average_ms"] <= args.max_matmul_ms,
        },
        "reformat_ratio_vs_fp16": {
            "value": reformat_ratio, "limit": args.max_reformat_ratio,
            "passed": reformat_ratio <= args.max_reformat_ratio,
        },
        "total_profile_ratio_vs_fp16": {
            "value": qdq_total / fp16_total, "limit": args.max_total_ratio,
            "passed": qdq_total <= fp16_total * args.max_total_ratio,
        },
        "fusion_signature_preserved": {
            "value": qdq_p2["conv_add_activation_fusion_signatures"]["layer_count"],
            "limit": fp16_p2["conv_add_activation_fusion_signatures"]["layer_count"],
            "passed": qdq_p2["conv_add_activation_fusion_signatures"]["layer_count"] >= fp16_p2["conv_add_activation_fusion_signatures"]["layer_count"],
        },
    }
    if args.max_pre_conv8_ms is not None:
        checks["separable_pre_conv8_ms"] = {
            "value": qdq_p2["separable_pre_conv8"]["average_ms"],
            "limit": args.max_pre_conv8_ms,
            "passed": qdq_p2["separable_pre_conv8"]["average_ms"] <= args.max_pre_conv8_ms,
        }
    passed = all(item["passed"] for item in checks.values())
    report = {
        "passed": passed,
        "fp16_summary": str(args.fp16.resolve()),
        "qdq_summary": str(args.qdq.resolve()),
        "checks": checks,
        "note": "Profile summed layer time is diagnostic; confirm GPU Compute in benchmark logs and task accuracy separately.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, check in checks.items():
        print(f"{'PASS' if check['passed'] else 'FAIL'} {name}: value={check['value']}, limit={check['limit']}")
    print(f"Report: {args.output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())