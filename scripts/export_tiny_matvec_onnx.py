#!/usr/bin/env python3
"""Export TensorRT-friendly tiny 3x3 matvec implementations to ONNX.

Run from the project root, for example:

    python3 scripts/export_tiny_matvec_onnx.py --implementation both
    python3 scripts/export_tiny_matvec_onnx.py --implementation reduce --two-matvecs
    python3 scripts/export_tiny_matvec_onnx.py --implementation expanded --flat-vector

Each ONNX file gets a sibling ``.graph.txt`` report containing the ordered
node list and operator counts, making the graph structure easy to inspect.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import Tensor, nn

from training_modules import TinyMatVec3x3, TwoTinyMatVec3x3


class SingleTinyMatVec(nn.Module):
    def __init__(self, implementation: str) -> None:
        super().__init__()
        self.matvec = TinyMatVec3x3(implementation)  # type: ignore[arg-type]

    def forward(self, matrix: Tensor, vector: Tensor) -> Tensor:
        return self.matvec(matrix, vector)


def _shape_text(value_info: object) -> str:
    tensor_type = value_info.type.tensor_type  # type: ignore[attr-defined]
    dimensions: list[str] = []
    for dimension in tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            dimensions.append(str(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            dimensions.append(dimension.dim_param)
        else:
            dimensions.append("?")
    return "[" + ", ".join(dimensions) + "]"


def _names(values: Iterable[object]) -> str:
    return ", ".join(value.name for value in values)  # type: ignore[attr-defined]


def graph_report(model: object) -> str:
    graph = model.graph  # type: ignore[attr-defined]
    counts = Counter(node.op_type for node in graph.node)
    lines = [
        f"graph: {graph.name or '<unnamed>'}",
        f"inputs: {_names(graph.input)}",
        f"outputs: {_names(graph.output)}",
        "",
        "input shapes:",
    ]
    lines.extend(f"  {value.name}: {_shape_text(value)}" for value in graph.input)
    lines.extend(("", "operator counts:"))
    lines.extend(f"  {name}: {count}" for name, count in sorted(counts.items()))
    lines.extend(("", "ordered nodes:"))

    for index, node in enumerate(graph.node):
        name = node.name or f"node_{index}"
        inputs = ", ".join(node.input)
        outputs = ", ".join(node.output)
        lines.append(f"  {index:03d}  {node.op_type:<12} {name}")
        lines.append(f"       in:  {inputs}")
        lines.append(f"       out: {outputs}")

    forbidden = [node.name or "<unnamed>" for node in graph.node if node.op_type in {"MatMul", "Gemm"}]
    lines.extend(("", f"MatMul/Gemm check: {'FAIL ' + str(forbidden) if forbidden else 'PASS'}"))
    return "\n".join(lines) + "\n"


def export_one(
    implementation: str,
    output_path: Path,
    *,
    batch: int,
    height: int,
    width: int,
    column_vector: bool,
    two_matvecs: bool,
    opset: int,
) -> None:
    try:
        import onnx
    except ImportError as error:
        raise SystemExit("缺少 onnx，请先安装：python3 -m pip install onnx") from error

    prefix = (batch, height, width)
    matrix_0 = torch.randn(*prefix, 3, 3)
    vector_shape = (*prefix, 3, 1) if column_vector else (*prefix, 3)
    vector_0 = torch.randn(*vector_shape)

    if two_matvecs:
        module: nn.Module = TwoTinyMatVec3x3(implementation)  # type: ignore[arg-type]
        inputs = (matrix_0, vector_0, torch.randn_like(matrix_0), torch.randn_like(vector_0))
        input_names = ("matrix_0", "vector_0", "matrix_1", "vector_1")
        output_names = ("output_0", "output_1")
    else:
        module = SingleTinyMatVec(implementation)
        inputs = (matrix_0, vector_0)
        input_names = ("matrix", "vector")
        output_names = ("output",)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    module.eval()
    with torch.no_grad():
        torch.onnx.export(
            module,
            inputs,
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset,
            do_constant_folding=True,
        )

    exported = onnx.load(str(output_path))
    onnx.checker.check_model(exported)
    inferred = onnx.shape_inference.infer_shapes(exported)
    report = graph_report(inferred)
    report_path = output_path.with_suffix(".graph.txt")
    report_path.write_text(report, encoding="utf-8")

    print(f"\n=== {implementation} ===")
    print(f"ONNX:   {output_path}")
    print(f"结构报告: {report_path}")
    print(report, end="")

    forbidden = [node for node in exported.graph.node if node.op_type in {"MatMul", "Gemm"}]
    if forbidden:
        raise SystemExit(f"ERROR: {output_path} 中仍存在 MatMul/Gemm")


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 tiny 3x3 matvec ONNX 并显示算子结构")
    parser.add_argument("--implementation", choices=("reduce", "expanded", "both"), default="both")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tiny_matvec"))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--flat-vector", action="store_true", help="使用 [...,3]，默认使用 [...,3,1]")
    parser.add_argument("--two-matvecs", action="store_true", help="在一个图中导出两份算子")
    parser.add_argument("--opset", type=int, default=13)
    args = parser.parse_args()

    if min(args.batch, args.height, args.width) <= 0:
        parser.error("batch、height、width 必须为正整数")

    implementations = ("reduce", "expanded") if args.implementation == "both" else (args.implementation,)
    vector_kind = "flat" if args.flat_vector else "column"
    graph_kind = "two" if args.two_matvecs else "single"
    for implementation in implementations:
        filename = f"tiny_matvec_3x3_{implementation}_{graph_kind}_{vector_kind}.onnx"
        export_one(
            implementation,
            args.output_dir / filename,
            batch=args.batch,
            height=args.height,
            width=args.width,
            column_vector=not args.flat_vector,
            two_matvecs=args.two_matvecs,
            opset=args.opset,
        )


if __name__ == "__main__":
    main()