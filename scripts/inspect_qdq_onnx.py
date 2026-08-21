#!/usr/bin/env python3
"""Audit explicit Q/DQ placement and likely fusion boundaries in an ONNX graph."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-qdq", action="store_true")
    args = parser.parse_args()
    try:
        import onnx
    except ImportError as error:
        print("ERROR: inspect_qdq_onnx.py requires the onnx package", file=sys.stderr)
        return 2
    if not args.model.is_file():
        print(f"ERROR: model not found: {args.model}", file=sys.stderr)
        return 2

    model = onnx.load(str(args.model))
    onnx.checker.check_model(model)
    nodes = list(model.graph.node)
    counts = Counter(node.op_type for node in nodes)
    producers = {output: node for node in nodes for output in node.output}
    consumers: dict[str, list[Any]] = defaultdict(list)
    for node in nodes:
        for value in node.input:
            consumers[value].append(node)

    qdq_nodes = [node for node in nodes if node.op_type in {"QuantizeLinear", "DequantizeLinear"}]
    quantized_targets: Counter[str] = Counter()
    boundaries: list[dict[str, str]] = []
    for node in qdq_nodes:
        for output in node.output:
            for consumer in consumers.get(output, []):
                quantized_targets[consumer.op_type] += 1
                boundaries.append({
                    "from": node.name or node.op_type,
                    "from_type": node.op_type,
                    "to": consumer.name or consumer.op_type,
                    "to_type": consumer.op_type,
                })

    fusion_patterns: list[dict[str, Any]] = []
    for conv in (node for node in nodes if node.op_type == "Conv"):
        successors: list[Any] = []
        frontier = list(conv.output)
        visited: set[str] = set()
        while frontier and len(successors) < 4:
            tensor = frontier.pop(0)
            for consumer in consumers.get(tensor, []):
                key = consumer.name or "|".join(consumer.output)
                if key in visited:
                    continue
                visited.add(key)
                if consumer.op_type in {"QuantizeLinear", "DequantizeLinear", "Identity"}:
                    frontier.extend(consumer.output)
                else:
                    successors.append(consumer)
        kinds = [node.op_type for node in successors]
        fusion_patterns.append({
            "conv": conv.name or "<unnamed>",
            "successor_types_across_qdq": kinds,
            "add_or_leakyrelu_visible": any(kind in {"Add", "LeakyRelu"} for kind in kinds),
        })

    float_policy = {
        "pre_conv8_nodes": [node.name for node in nodes if re.search(r"(?:^|[/.])pre_conv8(?:[/.]|$)", node.name, re.I)],
        "matmul_nodes": [node.name for node in nodes if node.op_type in {"MatMul", "Gemm"}],
    }
    report = {
        "model": str(args.model.resolve()),
        "operator_counts": dict(sorted(counts.items())),
        "quantize_linear_count": counts["QuantizeLinear"],
        "dequantize_linear_count": counts["DequantizeLinear"],
        "qdq_boundary_count": len(boundaries),
        "qdq_targets": dict(sorted(quantized_targets.items())),
        "float_policy_candidates": float_policy,
        "conv_fusion_neighborhoods": fusion_patterns,
        "boundaries": boundaries,
    }
    output = args.output or args.model.with_suffix(".qdq-audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Model: {args.model}")
    print(f"QuantizeLinear: {counts['QuantizeLinear']}")
    print(f"DequantizeLinear: {counts['DequantizeLinear']}")
    print("Q/DQ targets: " + (", ".join(f"{key}={value}" for key, value in sorted(quantized_targets.items())) or "none"))
    print(f"pre_conv8 candidates: {len(float_policy['pre_conv8_nodes'])}")
    print(f"remaining MatMul/Gemm: {len(float_policy['matmul_nodes'])}")
    print(f"Audit: {output}")
    if args.require_qdq and (counts["QuantizeLinear"] == 0 or counts["DequantizeLinear"] == 0):
        print("ERROR: explicit Q/DQ nodes are required but missing", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())