#!/usr/bin/env python3
"""Create an explicit Q/DQ ONNX model using representative tensor data."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from quantization_dataset import iter_samples


def imports() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import onnx
        import onnxruntime as ort
        from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod
        from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
    except ImportError as error:
        raise SystemExit(
            "缺少 P2 依赖。请安装与平台兼容的 onnx 和 onnxruntime："
            "python3 -m pip install -r requirements-p2.txt"
        ) from error
    return onnx, ort, CalibrationDataReader, CalibrationMethod, QuantFormat, (QuantType, quantize_static)


def graph_input_names(model: Any) -> list[str]:
    initializers = {item.name for item in model.graph.initializer}
    return [item.name for item in model.graph.input if item.name not in initializers]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="FP32/FP16 source ONNX")
    parser.add_argument("--dataset", type=Path, required=True, help="preprocessed NPY/NPZ directory or manifest")
    parser.add_argument("--output", type=Path, required=True, help="explicit Q/DQ ONNX output")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--op-types", default="Conv", help="comma-separated operator types to quantize")
    parser.add_argument("--pre-conv8", choices=("fp16", "int8"), default="fp16")
    parser.add_argument("--matmul", choices=("fp16", "int8"), default="fp16")
    parser.add_argument("--exclude-regex", action="append", default=[])
    parser.add_argument("--include-regex", action="append", default=[])
    parser.add_argument("--activation", choices=("uint8", "int8"), default="int8")
    parser.add_argument("--weight", choices=("uint8", "int8"), default="int8")
    parser.add_argument("--per-channel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reduce-range", action="store_true")
    parser.add_argument("--calibration", choices=("minmax", "entropy", "percentile"), default="entropy")
    parser.add_argument("--report", type=Path, help="JSON quantization selection report")
    args = parser.parse_args()

    onnx, ort, CalibrationDataReader, CalibrationMethod, QuantFormat, quantization = imports()
    QuantType, quantize_static = quantization
    if not args.model.is_file():
        parser.error(f"model not found: {args.model}")
    if args.samples <= 0:
        parser.error("--samples must be positive")

    model = onnx.load(str(args.model))
    onnx.checker.check_model(model)
    input_names = graph_input_names(model)
    op_types = [value.strip() for value in args.op_types.split(",") if value.strip()]
    if args.matmul == "int8":
        op_types.extend(value for value in ("MatMul", "Gemm") if value not in op_types)
    exclude_patterns = list(args.exclude_regex)
    if args.pre_conv8 == "fp16":
        exclude_patterns.append(r"(?:^|[/.])pre_conv8(?:[/.]|$)")
    if args.matmul == "fp16":
        exclude_patterns.append(r"(?:^|/)MatMul(?:_\d+)?$")

    exclude_re = [re.compile(pattern, re.IGNORECASE) for pattern in exclude_patterns]
    include_re = [re.compile(pattern, re.IGNORECASE) for pattern in args.include_regex]
    candidates = [node for node in model.graph.node if node.op_type in op_types]
    selected: list[str] = []
    excluded: list[str] = []
    unnamed = [node for node in candidates if not node.name]
    if unnamed:
        raise SystemExit(
            f"ERROR: {len(unnamed)} candidate nodes have no name; assign stable node names before selective PTQ"
        )
    for node in candidates:
        included = not include_re or any(pattern.search(node.name) for pattern in include_re)
        blocked = any(pattern.search(node.name) for pattern in exclude_re)
        (selected if included and not blocked else excluded).append(node.name)
    if not selected:
        raise SystemExit("ERROR: no nodes selected for quantization; check op types and regex filters")

    class DatasetReader(CalibrationDataReader):
        def __init__(self) -> None:
            self.count = 0
            self.rewind()

        def rewind(self) -> None:
            self.count = 0
            self._iterator = iter_samples(args.dataset, input_names, limit=args.samples)

        def get_next(self) -> dict[str, Any] | None:
            try:
                _, sample = next(self._iterator)
            except StopIteration:
                return None
            self.count += 1
            return sample

    methods = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
    }
    types = {"uint8": QuantType.QUInt8, "int8": QuantType.QInt8}
    reader = DatasetReader()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        model_input=str(args.model),
        model_output=str(args.output),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=types[args.activation],
        weight_type=types[args.weight],
        per_channel=args.per_channel,
        reduce_range=args.reduce_range,
        op_types_to_quantize=op_types,
        nodes_to_quantize=selected,
        calibrate_method=methods[args.calibration],
        extra_options={
            "ActivationSymmetric": args.activation == "int8",
            "WeightSymmetric": args.weight == "int8",
            "ForceQuantizeNoInputCheck": False,
        },
    )
    if reader.count == 0:
        raise SystemExit("ERROR: calibration reader produced no samples")

    result = onnx.load(str(args.output))
    onnx.checker.check_model(result)
    counts: dict[str, int] = {}
    for node in result.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    if not counts.get("QuantizeLinear") or not counts.get("DequantizeLinear"):
        raise SystemExit("ERROR: output model contains no explicit QuantizeLinear/DequantizeLinear pairs")

    report = {
        "source_model": str(args.model.resolve()),
        "output_model": str(args.output.resolve()),
        "dataset": str(args.dataset.resolve()),
        "calibration_samples": reader.count,
        "calibration_method": args.calibration,
        "activation_type": args.activation,
        "weight_type": args.weight,
        "per_channel": args.per_channel,
        "pre_conv8_policy": args.pre_conv8,
        "matmul_policy": args.matmul,
        "op_types": op_types,
        "selected_node_count": len(selected),
        "excluded_node_count": len(excluded),
        "selected_nodes": selected,
        "excluded_nodes": excluded,
        "onnx_operator_counts": counts,
        "onnxruntime_version": ort.__version__,
    }
    report_path = args.report or args.output.with_suffix(".quantization.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Explicit Q/DQ model: {args.output}")
    print(f"Calibration samples: {reader.count}")
    print(f"Selected/excluded nodes: {len(selected)}/{len(excluded)}")
    print(f"QuantizeLinear/DequantizeLinear: {counts['QuantizeLinear']}/{counts['DequantizeLinear']}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())