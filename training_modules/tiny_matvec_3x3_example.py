"""Validate and export the two tiny batched MatMul replacements.

Examples from the project root:

    python3 -m training_modules.tiny_matvec_3x3_example --implementation reduce
    python3 -m training_modules.tiny_matvec_3x3_example --implementation expanded
    python3 -m training_modules.tiny_matvec_3x3_example --height 512 --width 416

The generated ONNX model has two matrix-vector operations, matching the usual
``/MatMul`` and ``/MatMul_1`` use case, but contains no ONNX ``MatMul`` node.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import Tensor

from .tiny_matvec_3x3 import Implementation, TwoTinyMatVec3x3


def reference(
    matrix_0: Tensor,
    vector_0: Tensor,
    matrix_1: Tensor,
    vector_1: Tensor,
) -> tuple[Tensor, Tensor]:
    """Original implementation used only as the numerical reference."""
    return torch.matmul(matrix_0, vector_0), torch.matmul(matrix_1, vector_1)


def _maximum_errors(actual: Tensor, expected: Tensor) -> tuple[float, float]:
    difference = (actual.float() - expected.float()).abs()
    absolute = float(difference.max())
    relative = float((difference / expected.float().abs().clamp_min(1e-6)).max())
    return absolute, relative


def validate(
    model: TwoTinyMatVec3x3,
    inputs: tuple[Tensor, Tensor, Tensor, Tensor],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> None:
    """Check both outputs and all four input gradients against ``matmul``."""
    optimized_inputs = tuple(value.detach().clone().requires_grad_(True) for value in inputs)
    reference_inputs = tuple(value.detach().clone().requires_grad_(True) for value in inputs)

    actual_outputs = model(*optimized_inputs)
    expected_outputs = reference(*reference_inputs)

    for index, (actual, expected) in enumerate(zip(actual_outputs, expected_outputs)):
        torch.testing.assert_close(
            actual,
            expected,
            atol=absolute_tolerance,
            rtol=relative_tolerance,
        )
        absolute, relative = _maximum_errors(actual, expected)
        print(f"output_{index}: max_abs={absolute:.8g}, max_rel={relative:.8g}")

    sum(output.float().square().mean() for output in actual_outputs).backward()
    sum(output.float().square().mean() for output in expected_outputs).backward()
    for index, (actual, expected) in enumerate(zip(optimized_inputs, reference_inputs)):
        torch.testing.assert_close(
            actual.grad,
            expected.grad,
            atol=absolute_tolerance,
            rtol=relative_tolerance,
        )
        absolute, relative = _maximum_errors(actual.grad, expected.grad)
        print(f"gradient_{index}: max_abs={absolute:.8g}, max_rel={relative:.8g}")


def export_and_check(
    model: TwoTinyMatVec3x3,
    inputs: tuple[Tensor, Tensor, Tensor, Tensor],
    output_path: Path,
) -> None:
    """Export ONNX and fail if any ``MatMul`` or ``Gemm`` remains."""
    try:
        import onnx
    except ImportError as error:
        raise RuntimeError("ONNX export/check requires: python3 -m pip install onnx") from error

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.onnx.export(
        model,
        inputs,
        str(output_path),
        input_names=("matrix_0", "vector_0", "matrix_1", "vector_1"),
        output_names=("output_0", "output_1"),
        opset_version=13,
        do_constant_folding=True,
    )

    graph = onnx.load(str(output_path))
    onnx.checker.check_model(graph)
    forbidden = [node.name or "<unnamed>" for node in graph.graph.node if node.op_type in {"MatMul", "Gemm"}]
    operator_counts: dict[str, int] = {}
    for node in graph.graph.node:
        operator_counts[node.op_type] = operator_counts.get(node.op_type, 0) + 1
    if forbidden:
        raise RuntimeError(f"export still contains MatMul/Gemm nodes: {forbidden}")

    print("ONNX:", output_path)
    print("operators:", ", ".join(f"{name}={count}" for name, count in sorted(operator_counts.items())))
    print("MatMul/Gemm check: PASS")


def make_inputs(
    batch: int,
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    shape = (batch, height, width)
    return (
        torch.randn(*shape, 3, 3, device=device, dtype=dtype),
        torch.randn(*shape, 3, 1, device=device, dtype=dtype),
        torch.randn(*shape, 3, 3, device=device, dtype=dtype),
        torch.randn(*shape, 3, 1, device=device, dtype=dtype),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("reduce", "expanded"), default="reduce")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fp16", action="store_true", help="validate in FP16; requires CUDA")
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--onnx", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16 if args.fp16 else torch.float32
    if dtype == torch.float16 and device.type != "cuda":
        parser.error("--fp16 requires a CUDA device")
    if min(args.batch, args.height, args.width) <= 0:
        parser.error("batch, height, and width must be positive")

    absolute_tolerance = args.atol if args.atol is not None else (2e-3 if args.fp16 else 1e-6)
    relative_tolerance = args.rtol if args.rtol is not None else (2e-3 if args.fp16 else 1e-5)
    implementation: Implementation = args.implementation
    model = TwoTinyMatVec3x3(implementation).to(device)
    inputs = make_inputs(args.batch, args.height, args.width, device=device, dtype=dtype)

    print(f"implementation={implementation}, device={device}, dtype={dtype}")
    validate(
        model,
        inputs,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    print("PyTorch numerical and gradient checks: PASS")

    output_path = args.onnx or Path("artifacts") / f"two_tiny_matvec_3x3_{implementation}.onnx"
    cpu_inputs = tuple(value.detach().float().cpu() for value in inputs)
    export_and_check(model.float().cpu(), cpu_inputs, output_path)


if __name__ == "__main__":
    main()