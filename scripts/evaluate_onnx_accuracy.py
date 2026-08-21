#!/usr/bin/env python3
"""Compare FP16/PTQ/QAT ONNX outputs against an FP32 ONNX reference."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from quantization_dataset import iter_samples


def parse_candidate(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("candidate must use NAME=/path/model.onnx")
    name, path = text.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("candidate must use NAME=/path/model.onnx")
    return name, Path(path)


def load_business_metric(spec: str | None) -> Callable[[dict[str, np.ndarray], dict[str, np.ndarray]], dict[str, float]] | None:
    if not spec:
        return None
    if ":" not in spec:
        raise ValueError("business metric must use module:function")
    module_name, function_name = spec.split(":", 1)
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"business metric is not callable: {spec}")
    return function


def global_ssim(reference: np.ndarray, candidate: np.ndarray, data_range: float) -> float:
    """Compute global SSIM over one output tensor without external packages."""
    x = reference.astype(np.float64, copy=False).ravel()
    y = candidate.astype(np.float64, copy=False).ravel()
    mean_x, mean_y = float(x.mean()), float(y.mean())
    variance_x, variance_y = float(x.var()), float(y.var())
    covariance = float(((x - mean_x) * (y - mean_y)).mean())
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    denominator = (mean_x * mean_x + mean_y * mean_y + c1) * (variance_x + variance_y + c2)
    return 1.0 if denominator == 0 and np.array_equal(x, y) else (
        ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / denominator if denominator else 0.0
    )


@dataclass
class OutputMetrics:
    element_count: int = 0
    sample_count: int = 0
    absolute_sum: float = 0.0
    squared_sum: float = 0.0
    maximum_absolute: float = 0.0
    maximum_relative: float = 0.0
    ssim_sum: float = 0.0
    data_range_sum: float = 0.0

    def update(self, reference: np.ndarray, candidate: np.ndarray, configured_range: float | None) -> None:
        if reference.shape != candidate.shape:
            raise ValueError(f"output shape mismatch: reference={reference.shape}, candidate={candidate.shape}")
        ref = reference.astype(np.float64, copy=False)
        actual = candidate.astype(np.float64, copy=False)
        difference = np.abs(ref - actual)
        data_range = configured_range or max(float(ref.max() - ref.min()), 1e-12)
        self.element_count += difference.size
        self.sample_count += 1
        self.absolute_sum += float(difference.sum())
        self.squared_sum += float(np.square(difference).sum())
        self.maximum_absolute = max(self.maximum_absolute, float(difference.max(initial=0.0)))
        relative = difference / np.maximum(np.abs(ref), 1e-6)
        self.maximum_relative = max(self.maximum_relative, float(relative.max(initial=0.0)))
        self.ssim_sum += global_ssim(ref, actual, data_range)
        self.data_range_sum += data_range

    def result(self) -> dict[str, float | int]:
        mae = self.absolute_sum / self.element_count
        mse = self.squared_sum / self.element_count
        data_range = self.data_range_sum / self.sample_count
        psnr = math.inf if mse == 0 else 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)
        return {
            "samples": self.sample_count,
            "elements": self.element_count,
            "max_absolute_error": self.maximum_absolute,
            "mean_absolute_error": mae,
            "max_relative_error": self.maximum_relative,
            "mse": mse,
            "psnr_db": psnr,
            "ssim_global": self.ssim_sum / self.sample_count,
        }


def session_input(sample: dict[str, np.ndarray], session: Any) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for metadata in session.get_inputs():
        if metadata.name not in sample:
            raise ValueError(f"dataset missing input {metadata.name}")
        value = sample[metadata.name]
        if metadata.type == "tensor(float16)":
            value = value.astype(np.float16)
        elif metadata.type == "tensor(float)":
            value = value.astype(np.float32)
        expected = metadata.shape
        if len(expected) != value.ndim:
            raise ValueError(f"{metadata.name}: expected rank {len(expected)}, got {value.shape}")
        for expected_dim, actual_dim in zip(expected, value.shape):
            if isinstance(expected_dim, int) and expected_dim > 0 and expected_dim != actual_dim:
                raise ValueError(f"{metadata.name}: expected shape {expected}, got {value.shape}")
        result[metadata.name] = np.ascontiguousarray(value)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path, help="FP32 reference ONNX")
    parser.add_argument("--candidate", action="append", required=True, type=parse_candidate, help="NAME=model.onnx; repeat for fp16/PTQ/QAT")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--providers", default="CUDAExecutionProvider,CPUExecutionProvider")
    parser.add_argument("--data-range", type=float, help="fixed output range used by PSNR/SSIM")
    parser.add_argument("--business-metric", help="Python callback module:function(ref_outputs, candidate_outputs)")
    parser.add_argument("--output", type=Path, default=Path("results/accuracy/onnx_accuracy.json"))
    parser.add_argument("--max-abs", type=float)
    parser.add_argument("--max-mae", type=float)
    parser.add_argument("--min-psnr", type=float)
    parser.add_argument("--min-ssim", type=float)
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    if args.data_range is not None and args.data_range <= 0:
        parser.error("--data-range must be positive")

    try:
        import onnxruntime as ort
    except ImportError as error:
        raise SystemExit("缺少 onnxruntime：python3 -m pip install -r requirements-p2.txt") from error
    requested = [name.strip() for name in args.providers.split(",") if name.strip()]
    available = set(ort.get_available_providers())
    providers = [name for name in requested if name in available]
    if not providers:
        raise SystemExit(f"ERROR: none of the requested providers are available; installed={sorted(available)}")
    if not args.reference.is_file():
        parser.error(f"reference not found: {args.reference}")

    reference_session = ort.InferenceSession(str(args.reference), providers=providers)
    output_names = [output.name for output in reference_session.get_outputs()]
    input_names = [input_.name for input_ in reference_session.get_inputs()]
    candidates: dict[str, Any] = {}
    for name, path in args.candidate:
        if name in candidates:
            parser.error(f"duplicate candidate name: {name}")
        if not path.is_file():
            parser.error(f"candidate not found: {path}")
        session = ort.InferenceSession(str(path), providers=providers)
        if [output.name for output in session.get_outputs()] != output_names:
            parser.error(f"{name}: output names/order differ from reference")
        candidates[name] = session

    business_metric = load_business_metric(args.business_metric)
    metrics = {name: {output: OutputMetrics() for output in output_names} for name in candidates}
    business_values: dict[str, dict[str, list[float]]] = {name: {} for name in candidates}
    sample_labels: list[str] = []
    for label, sample in iter_samples(args.dataset, input_names, limit=args.samples):
        sample_labels.append(label)
        ref_values = reference_session.run(output_names, session_input(sample, reference_session))
        ref_outputs = dict(zip(output_names, ref_values))
        for name, session in candidates.items():
            values = session.run(output_names, session_input(sample, session))
            outputs = dict(zip(output_names, values))
            for output_name in output_names:
                metrics[name][output_name].update(ref_outputs[output_name], outputs[output_name], args.data_range)
            if business_metric:
                for key, value in business_metric(ref_outputs, outputs).items():
                    business_values[name].setdefault(key, []).append(float(value))

    results: dict[str, Any] = {}
    failed: list[str] = []
    for candidate, outputs in metrics.items():
        output_results = {name: metric.result() for name, metric in outputs.items()}
        for output_name, values in output_results.items():
            reasons = []
            if args.max_abs is not None and values["max_absolute_error"] > args.max_abs:
                reasons.append(f"max_abs>{args.max_abs}")
            if args.max_mae is not None and values["mean_absolute_error"] > args.max_mae:
                reasons.append(f"mae>{args.max_mae}")
            if args.min_psnr is not None and values["psnr_db"] < args.min_psnr:
                reasons.append(f"psnr<{args.min_psnr}")
            if args.min_ssim is not None and values["ssim_global"] < args.min_ssim:
                reasons.append(f"ssim<{args.min_ssim}")
            if reasons:
                failed.append(f"{candidate}/{output_name}: {', '.join(reasons)}")
        business_summary = {
            key: {"mean": float(np.mean(values)), "min": min(values), "max": max(values)}
            for key, values in business_values[candidate].items()
        }
        results[candidate] = {"outputs": output_results, "business_metrics": business_summary}

    report = {
        "reference": str(args.reference.resolve()),
        "candidates": {name: str(path.resolve()) for name, path in args.candidate},
        "dataset": str(args.dataset.resolve()),
        "samples": len(sample_labels),
        "providers": providers,
        "outputs": output_names,
        "ssim_definition": "global SSIM over each complete output tensor, averaged over samples",
        "thresholds": {"max_abs": args.max_abs, "max_mae": args.max_mae, "min_psnr": args.min_psnr, "min_ssim": args.min_ssim},
        "passed": not failed,
        "failures": failed,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["candidate", "output", "samples", "elements", "max_absolute_error", "mean_absolute_error", "max_relative_error", "mse", "psnr_db", "ssim_global"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate, candidate_result in results.items():
            for output_name, values in candidate_result["outputs"].items():
                writer.writerow({"candidate": candidate, "output": output_name, **values})

    print(f"Samples: {len(sample_labels)}; outputs: {', '.join(output_names)}; providers: {providers}")
    for candidate, candidate_result in results.items():
        print(f"[{candidate}]")
        for output_name, values in candidate_result["outputs"].items():
            print(f"  {output_name}: max_abs={values['max_absolute_error']:.7g}, MAE={values['mean_absolute_error']:.7g}, PSNR={values['psnr_db']:.4g} dB, SSIM={values['ssim_global']:.7g}")
    print(f"JSON: {args.output}")
    print(f"CSV: {csv_path}")
    if failed:
        print("FAILED: " + "; ".join(failed), file=sys.stderr)
        return 1
    print("Accuracy thresholds: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())