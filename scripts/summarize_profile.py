#!/usr/bin/env python3
"""Summarize TensorRT trtexec layer profile JSON for optimization work."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys
from collections import defaultdict
from typing import Any


CATEGORY_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("Plugin", re.compile(r"plugin|efficientnms|nms|grid.?sample|roi.?align", re.I)),
    ("Convolution", re.compile(r"conv|convolution|deconv|deconvolution", re.I)),
    ("MatrixMultiply", re.compile(r"matmul|matrix.?multiply|gemm|fully.?connected|\bfc\b", re.I)),
    ("Normalization", re.compile(r"normalization|layernorm|instancenorm|batchnorm|scale", re.I)),
    ("Attention", re.compile(r"attention|softmax|qkv", re.I)),
    ("Resize", re.compile(r"resize|upsample|interpol", re.I)),
    ("Activation", re.compile(r"relu|gelu|sigmoid|tanh|activation|leaky", re.I)),
    ("Reduction", re.compile(r"reduce|pool|mean|sum", re.I)),
    ("ElementWise", re.compile(r"elementwise|\badd\b|\bmul\b|\bdiv\b|\bsub\b|\bpow\b", re.I)),
    ("Reformat/Copy", re.compile(r"reformat|copy|shuffle|transpose|reshape|squeeze|unsqueeze", re.I)),
    ("Concatenation", re.compile(r"concat|concatenation", re.I)),
    ("Slice/Gather", re.compile(r"slice|gather|scatter|split", re.I)),
]

CATEGORY_ADVICE = {
    "Plugin": "Inspect plugin precision, launch configuration, workspace use, and opportunities to replace it with a native TensorRT layer.",
    "Convolution": "Inspect tactic, tensor-core precision/layout, channel alignment, kernel size, and whether adjacent bias/activation layers are fused.",
    "MatrixMultiply": "Check tensor-core eligibility, aligned dimensions, transpose/layout costs, and FP16/INT8 accuracy options.",
    "Normalization": "Check epsilon-sensitive precision and whether normalization can fuse with adjacent operations or use a native TensorRT implementation.",
    "Attention": "Consider a fused attention implementation and validate softmax/reduction precision.",
    "Resize": "Verify interpolation semantics first, then reduce repeated resize/reformat traffic or fuse surrounding operations.",
    "Activation": "Standalone activation hotspots may indicate missed fusion with convolution or matrix multiplication.",
    "Reduction": "Inspect reduction axis/layout, accumulation precision, and whether upstream/downstream operations can fuse.",
    "ElementWise": "Look for chains that can be constant-folded, algebraically simplified, or fused into a plugin/native pointwise kernel.",
    "Reformat/Copy": "Reduce layout/precision conversions by keeping neighboring layers in compatible tensor formats and precisions.",
    "Concatenation": "Reduce branch materialization and memory traffic; check whether downstream layers can consume branches directly.",
    "Slice/Gather": "Check data movement, index dtype, contiguous layouts, and opportunities to make indexing static.",
    "Other": "Inspect the layer type/tactic in layer-info JSON and optimize only after confirming repeatable impact.",
}


def load_json(path: pathlib.Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_records(value: Any, required_keys: set[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(value, dict):
        lowered = {str(key).lower() for key in value}
        if lowered & required_keys:
            records.append(value)
        for child in value.values():
            records.extend(find_records(child, required_keys))
    elif isinstance(value, list):
        for child in value:
            records.extend(find_records(child, required_keys))
    return records


def value_for(record: dict[str, Any], *names: str) -> Any:
    mapping = {str(key).lower(): value for key, value in record.items()}
    for name in names:
        if name.lower() in mapping:
            return mapping[name.lower()]
    return None


def number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def category_for(name: str, layer_type: str) -> str:
    text = f"{layer_type} {name}"
    for category, pattern in CATEGORY_RULES:
        if pattern.search(text):
            return category
    return "Other"


def layer_metadata_map(path: pathlib.Path | None) -> dict[str, tuple[str, str]]:
    if path is None or not path.is_file():
        return {}
    result: dict[str, tuple[str, str]] = {}
    for record in find_records(load_json(path), {"layertype", "type"}):
        name = value_for(record, "Name", "name")
        layer_type = value_for(record, "LayerType", "type", "layer_type")
        if name is not None and layer_type is not None:
            tensor_records = value_for(record, "Outputs", "outputs") or value_for(record, "Inputs", "inputs") or []
            precisions: set[str] = set()
            if isinstance(tensor_records, list):
                for tensor in tensor_records:
                    if not isinstance(tensor, dict):
                        continue
                    tensor_format = str(value_for(tensor, "Format/Datatype", "format", "datatype") or "")
                    precisions.update(re.findall(r"\b(?:FP32|FP16|BF16|FP8|INT8|INT32|INT64|UINT8|BOOL)\b", tensor_format.upper()))
            result[str(name)] = (str(layer_type), "+".join(sorted(precisions)) or "unknown")
    return result


def recommendation(category: str, share: float, name: str) -> str:
    prefix = "Critical-path candidate. " if share >= 10.0 else "Optimization candidate. "
    note = CATEGORY_ADVICE.get(category, CATEGORY_ADVICE["Other"])
    if "||" in name or " + " in name:
        note += " The name suggests an already-fused layer; preserve fusion when changing the graph."
    return prefix + note


def pareto_count(rows: list[dict[str, Any]], threshold: float) -> int:
    cumulative = 0.0
    for index, row in enumerate(rows, start=1):
        cumulative += row["share"]
        if cumulative >= threshold:
            return index
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=pathlib.Path)
    parser.add_argument("--layer-info", type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path, help="Markdown report path")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    try:
        raw = load_json(args.profile)
    except (OSError, json.JSONDecodeError) as error:
        print(f"ERROR: cannot read TensorRT profile JSON: {error}", file=sys.stderr)
        return 2

    records = find_records(raw, {"averagems", "timems", "percentage"})
    metadata = layer_metadata_map(args.layer_info)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        name = str(value_for(record, "name", "layerName") or f"layer_{index}")
        count = int(number(value_for(record, "count", "iterations")) or 0)
        average_ms = number(value_for(record, "averageMs", "latencyMs"))
        if average_ms is None:
            total_record_ms = number(value_for(record, "timeMs"))
            average_ms = total_record_ms / count if total_record_ms is not None and count > 0 else total_record_ms
        if average_ms is None or average_ms < 0:
            continue
        layer_type, execution_precision = metadata.get(
            name,
            (str(value_for(record, "layerType", "type") or "unknown"), "unknown"),
        )
        rows.append({
            "name": name,
            "average_ms": average_ms,
            "count": count,
            "layer_type": layer_type,
            "execution_precision": execution_precision,
        })

    if not rows:
        print("ERROR: no layer timing records found in profile JSON", file=sys.stderr)
        return 2

    total_ms = sum(row["average_ms"] for row in rows)
    if total_ms <= 0:
        print("ERROR: total layer time is zero", file=sys.stderr)
        return 2

    for row in rows:
        row["share"] = row["average_ms"] / total_ms * 100.0
        row["category"] = category_for(row["name"], row["layer_type"])
    rows.sort(key=lambda row: row["average_ms"], reverse=True)

    cumulative = 0.0
    for rank, row in enumerate(rows, start=1):
        cumulative += row["share"]
        row["rank"] = rank
        row["cumulative"] = cumulative

    categories: dict[str, dict[str, float]] = defaultdict(lambda: {"time": 0.0, "count": 0.0})
    for row in rows:
        categories[row["category"]]["time"] += row["average_ms"]
        categories[row["category"]]["count"] += 1
    category_rows = sorted(categories.items(), key=lambda item: item[1]["time"], reverse=True)

    precisions: dict[str, dict[str, float]] = defaultdict(lambda: {"time": 0.0, "count": 0.0})
    for row in rows:
        precisions[row["execution_precision"]]["time"] += row["average_ms"]
        precisions[row["execution_precision"]]["count"] += 1
    precision_rows = sorted(precisions.items(), key=lambda item: item[1]["time"], reverse=True)

    def aggregate_rows(predicate: Any) -> dict[str, float | int]:
        selected = [row for row in rows if predicate(row)]
        time_ms = sum(row["average_ms"] for row in selected)
        return {
            "layer_count": len(selected),
            "average_ms": time_ms,
            "share": time_ms / total_ms * 100.0,
        }

    p2_metrics = {
        "matmul": aggregate_rows(lambda row: row["category"] == "MatrixMultiply"),
        "separable_pre_conv8": aggregate_rows(
            lambda row: re.search(r"/pre_conv8/(?:vertical|horizontal)/Conv", row["name"], re.I) is not None
        ),
        "reformat_copy": aggregate_rows(lambda row: row["category"] == "Reformat/Copy"),
        "fp32": aggregate_rows(lambda row: "FP32" in row["execution_precision"]),
        "int8": aggregate_rows(lambda row: "INT8" in row["execution_precision"]),
        "conv_add_activation_fusion_signatures": aggregate_rows(
            lambda row: "conv" in row["name"].lower()
            and ("add" in row["name"].lower() or "relu" in row["name"].lower())
        ),
    }

    candidate_rows = [row for row in rows if row["share"] >= 3.0 or row["cumulative"] - row["share"] < 80.0]
    tiny_rows = [row for row in rows if row["share"] < 1.0]
    top_limit = max(1, min(args.top, len(rows)))
    top1 = rows[0]["share"]
    top5 = sum(row["share"] for row in rows[:5])
    top10 = sum(row["share"] for row in rows[:10])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".csv")
    json_path = args.output.with_suffix(".json")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "name", "layer_type", "execution_precision", "category", "average_ms", "share", "cumulative", "count"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "profile": str(args.profile),
        "layer_count": len(rows),
        "summed_average_layer_time_ms": total_ms,
        "concentration_percent": {"top1": top1, "top5": top5, "top10": top10},
        "pareto_layer_count": {str(level): pareto_count(rows, level) for level in (50, 80, 90, 95)},
        "categories": [
            {"category": category, "layer_count": int(values["count"]), "average_ms": values["time"], "share": values["time"] / total_ms * 100.0}
            for category, values in category_rows
        ],
        "execution_precisions": [
            {"precision": precision, "layer_count": int(values["count"]), "average_ms": values["time"], "share": values["time"] / total_ms * 100.0}
            for precision, values in precision_rows
        ],
        "p2_acceptance_metrics": p2_metrics,
        "optimization_candidates": [row["name"] for row in candidate_rows],
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with args.output.open("w", encoding="utf-8") as report:
        report.write("# TensorRT layer profile optimization summary\n\n")
        report.write("> `averageMs` is TensorRT per-layer GPU execution timing, not GPU occupancy/utilization. ")
        report.write("Use Nsight Systems/Compute when SM occupancy, memory bandwidth, kernel launch gaps, or CPU/GPU overlap are required.\n\n")
        report.write("## Executive summary\n\n")
        report.write(f"- Profiled layers: **{len(rows)}**\n")
        report.write(f"- Sum of average layer times: **{total_ms:.4f} ms/inference**\n")
        report.write(f"- Time concentration: top 1 **{top1:.2f}%**, top 5 **{top5:.2f}%**, top 10 **{top10:.2f}%**\n")
        report.write(f"- Layers required for 50/80/90/95% of time: **{pareto_count(rows, 50)}/{pareto_count(rows, 80)}/{pareto_count(rows, 90)}/{pareto_count(rows, 95)}**\n")
        report.write(f"- Long tail below 1% each: **{len(tiny_rows)} layers**, together **{sum(row['share'] for row in tiny_rows):.2f}%**\n")
        report.write(f"- Optimization candidates (>=3% individually or within cumulative 80%): **{len(candidate_rows)}**\n\n")

        report.write("## P2 explicit Q/DQ acceptance signals\n\n")
        report.write("| Signal | Layers | Avg ms | Share |\n| --- | ---: | ---: | ---: |\n")
        for signal, values in p2_metrics.items():
            report.write(f"| {signal} | {values['layer_count']} | {values['average_ms']:.5f} | {values['share']:.2f}% |\n")
        report.write("\nFusion signatures are layer-name heuristics; confirm actual fusion in TensorRT LayerInfo.\n\n")

        report.write("## Hot layers by average GPU execution time\n\n")
        report.write("| Rank | Layer | Type | Precision | Category | Avg ms | Share | Cumulative |\n| ---: | --- | --- | --- | --- | ---: | ---: | ---: |\n")
        for row in rows[:top_limit]:
            safe_name = row["name"].replace("|", "\\|")
            report.write(f"| {row['rank']} | `{safe_name}` | {row['layer_type']} | {row['execution_precision']} | {row['category']} | {row['average_ms']:.5f} | {row['share']:.2f}% | {row['cumulative']:.2f}% |\n")

        report.write("\n## Time grouped by actual TensorRT tensor precision\n\n")
        report.write("This is derived from layer-info output tensor formats. Mixed or unknown entries must be inspected directly.\n\n")
        report.write("| Rank | Precision | Layers | Avg ms | Share |\n| ---: | --- | ---: | ---: | ---: |\n")
        for rank, (precision, values) in enumerate(precision_rows, start=1):
            report.write(f"| {rank} | {precision} | {int(values['count'])} | {values['time']:.5f} | {values['time'] / total_ms * 100.0:.2f}% |\n")

        report.write("\n## Time grouped by operator category\n\n")
        report.write("Categories prefer TensorRT layer metadata when available and otherwise use layer-name heuristics; verify ambiguous `Other` entries in the layer-info JSON.\n\n")
        report.write("| Rank | Category | Layers | Avg ms | Share |\n| ---: | --- | ---: | ---: | ---: |\n")
        for rank, (category, values) in enumerate(category_rows, start=1):
            report.write(f"| {rank} | {category} | {int(values['count'])} | {values['time']:.5f} | {values['time'] / total_ms * 100.0:.2f}% |\n")

        report.write("\n## Prioritized optimization candidates\n\n")
        if candidate_rows:
            for row in candidate_rows:
                report.write(f"### {row['rank']}. `{row['name']}` — {row['average_ms']:.5f} ms ({row['share']:.2f}%)\n\n")
                report.write(recommendation(row["category"], row["share"], row["name"]) + "\n\n")
        else:
            report.write("No dominant candidate was detected; investigate the long tail, launch overhead, and reformat layers with Nsight Systems.\n")

        report.write("## Recommended optimization order\n\n")
        report.write("1. Re-run at least three times under fixed power mode/clocks and optimize only stable hotspots.\n")
        report.write("2. Start with layers/categories covering the first 80% of summed layer time; estimate maximum benefit with Amdahl's law.\n")
        report.write("3. Check `Reformat/Copy` and many tiny standalone layers for precision/layout churn or missed fusion.\n")
        report.write("4. Compare FP32 and FP16 profiles and task accuracy; use INT8 only with representative calibration/Q-DQ.\n")
        report.write("5. Use Nsight Systems for pipeline/launch/transfer bottlenecks and Nsight Compute for the final few expensive kernels.\n\n")
        report.write("The summed layer time can differ from end-to-end latency because enqueue, transfers, synchronization, CUDA Graph behavior, and profiling overhead are outside or alter layer measurements.\n")

    print("TensorRT profile summary")
    print(f"  layers: {len(rows)}")
    print(f"  summed average layer time: {total_ms:.4f} ms")
    print(f"  top1/top5/top10: {top1:.2f}%/{top5:.2f}%/{top10:.2f}%")
    print(f"  80% Pareto layers: {pareto_count(rows, 80)}")
    print(f"  report: {args.output}")
    print(f"  layer CSV: {csv_path}")
    print(f"  machine summary: {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())