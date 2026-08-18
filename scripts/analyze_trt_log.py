#!/usr/bin/env python3
"""Summarize TensorRT ONNX parser/build diagnostics without third-party packages."""

from __future__ import annotations

import argparse
import collections
import pathlib
import re
import sys


OPERATOR_RULES: dict[str, tuple[str, str]] = {
    "ArgMax": ("shape", "Verify output index dtype and axis semantics."),
    "ArgMin": ("shape", "Verify output index dtype and axis semantics."),
    "Cast": ("dtype", "Check casts involving INT64, BOOL, or low-precision tensors."),
    "CumSum": ("support", "Confirm native implementation; consider an equivalent prefix-sum plugin if rejected."),
    "DeformConv": ("support", "Usually requires a TensorRT plugin or graph rewrite."),
    "Div": ("precision", "Check near-zero denominators and compare FP16 output with FP32."),
    "Einsum": ("support", "Equation support varies; rewrite as MatMul/Transpose/Reduce when necessary."),
    "Erf": ("precision", "Validate FP16 error, especially when used in GELU."),
    "Exp": ("precision", "FP16 may overflow/underflow; validate the input range."),
    "GridSample": ("support", "Attributes/modes vary by TensorRT version; a plugin or rewrite may be required."),
    "If": ("support", "Control-flow support is constrained; prefer a static graph when possible."),
    "InstanceNormalization": ("precision", "Validate epsilon-sensitive FP16 output and plugin/native layer selection."),
    "LayerNormalization": ("precision", "Validate reduction and epsilon behavior in FP16."),
    "Log": ("precision", "Check non-positive inputs and FP16 underflow."),
    "Loop": ("support", "Control-flow support is constrained; unroll or export a static graph when possible."),
    "Mod": ("support", "Integer/floating mode support varies; rewrite with primitive arithmetic if rejected."),
    "NonMaxSuppression": ("support", "Often better replaced with TensorRT EfficientNMS and validated carefully."),
    "NonZero": ("shape", "Produces data-dependent shapes, which may prevent static TensorRT execution."),
    "OneHot": ("support", "Check index/depth dtypes; replace with Gather from an identity matrix if needed."),
    "Pow": ("precision", "Fractional powers and FP16 ranges require numerical validation."),
    "Reciprocal": ("precision", "Check near-zero inputs and FP16 numerical error."),
    "Resize": ("semantic", "Verify coordinate transformation, nearest mode, and output alignment against ONNX Runtime."),
    "RoiAlign": ("support", "Mode and coordinate semantics vary; a plugin may be required."),
    "Scan": ("support", "Control-flow support is constrained; export an unrolled/static graph when possible."),
    "ScatterND": ("support", "Check reduction/duplicate-index semantics and TensorRT version support."),
    "Softmax": ("precision", "Validate FP16 behavior for large logits; consider FP32 layer precision if needed."),
    "Sqrt": ("precision", "Check negative/small inputs and FP16 numerical error."),
    "TopK": ("shape", "K usually must be build-time constant and index dtype should be checked."),
}

REDUCTION_OPS = {"ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp", "ReduceMean", "ReduceProd", "ReduceSum", "ReduceSumSquare"}

ERROR_PATTERN = re.compile(
    r"\[(?:E|F)\]|\b(?:ERROR|UNSUPPORTED)\b|No importer registered|Could not find any implementation|"
    r"failed to parse|Parsing model failed|Engine could not be created|&&&& FAILED",
    re.IGNORECASE,
)
WARNING_PATTERN = re.compile(r"\[W\]|\bWARNING\b|Attempting to cast|clamp", re.IGNORECASE)
IGNORE_ERROR_PATTERN = re.compile(r"===.*Options|Error Recorder", re.IGNORECASE)
NODE_PATTERN = re.compile(r"Parsing node:\s+(.+?)\s+\[([^\]]+)\]")


def compact_diagnostics(lines: list[str], pattern: re.Pattern[str]) -> list[str]:
    results: list[str] = []
    for index, line in enumerate(lines):
        if not pattern.search(line) or IGNORE_ERROR_PATTERN.search(line):
            continue
        message = line.strip()
        next_index = index + 1
        while next_index < len(lines) and not re.search(r"^\[|^&&&&", lines[next_index]):
            message += " " + lines[next_index].strip()
            next_index += 1
        message = re.sub(r"\s+", " ", message)
        if message not in results:
            results.append(message)
    return results


def diagnostic_recommendation(message: str) -> str:
    lowered = message.lower()
    if "does not match detected package major" in lowered:
        return "Set TRTEXEC to the trtexec shipped with the intended TensorRT package and rerun the scan."
    if "int64" in lowered and "int32" in lowered:
        return "Re-export shape/index constants as INT32 where their range permits, then compare outputs; values outside INT32 are invalid for this cast."
    if "clamp" in lowered:
        return "Inspect the named weights/constants for out-of-range values and avoid relying on TensorRT's silent clamp."
    if "no importer registered" in lowered:
        return "Rewrite the operator into supported ONNX primitives or provide and explicitly load a matching TensorRT plugin."
    if "could not find any implementation" in lowered:
        return "Try FP32, increase workspace, inspect tensor formats/shapes, or replace the failing subgraph/plugin."
    if "failed to parse" in lowered or "parsing model failed" in lowered:
        return "Inspect the first preceding parser error; simplify/re-export that node or use a compatible plugin/opset."
    return "Review the complete log and verify representative outputs against the reference implementation."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=pathlib.Path)
    parser.add_argument("--report", required=True, type=pathlib.Path)
    parser.add_argument("--trt-version", default="unknown")
    parser.add_argument("--package-version", default="unknown")
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--command-status", type=int, default=0)
    args = parser.parse_args()

    text = args.log.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    nodes: list[tuple[str, str]] = []
    for line in lines:
        match = NODE_PATTERN.search(line)
        if match:
            nodes.append((match.group(1), match.group(2)))

    operator_counts = collections.Counter(operator for _, operator in nodes)
    errors = compact_diagnostics(lines, ERROR_PATTERN)
    warnings = compact_diagnostics(lines, WARNING_PATTERN)
    parsed = "Finished parsing network model" in text or "Successfully parsed ONNX file" in text
    build_passed = args.command_status == 0 and ("&&&& PASSED" in text or "Serialized engine" in text or "Engine built" in text)
    if args.command_status == 0 and not build_passed:
        warnings.append("TensorRT build completion marker was not found; the log may contain parser output only or be truncated.")
    package_major = args.package_version.split(".", 1)[0]
    binary_digits = "".join(character for character in args.trt_version if character.isdigit())
    if package_major.isdigit() and binary_digits and not binary_digits.startswith(package_major):
        warnings.append(
            f"TensorRT executable ID {args.trt_version} does not match detected package major {package_major}; "
            "TRTEXEC may point to another TensorRT installation."
        )

    findings: list[tuple[str, str, int, str]] = []
    for operator, count in sorted(operator_counts.items()):
        rule = OPERATOR_RULES.get(operator)
        if rule:
            findings.append((rule[0], operator, count, rule[1]))
        elif operator in REDUCTION_OPS:
            findings.append(("precision", operator, count, "Reduction accumulation can differ in FP16; compare with FP32/ONNX Runtime."))

    status = "ERROR" if errors or args.command_status != 0 else "WARNING" if warnings or findings else "PASS"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as report:
        report.write("# TensorRT ONNX operator compatibility report\n\n")
        report.write(f"- Result: **{status}**\n")
        report.write(f"- TensorRT binary ID: `{args.trt_version}`\n")
        report.write(f"- TensorRT package: `{args.package_version}`\n")
        report.write(f"- Requested precision: `{args.precision}`\n")
        report.write(f"- Parser completed: `{str(parsed).lower()}`\n")
        report.write(f"- Build command passed: `{str(build_passed).lower()}`\n")
        report.write(f"- ONNX nodes observed: `{len(nodes)}`\n")
        report.write(f"- Unique operator types: `{len(operator_counts)}`\n\n")
        if build_passed:
            report.write(f"**Verified build support:** all `{len(nodes)}` observed nodes (`{len(operator_counts)}` operator types) were accepted for this exact tested configuration.\n\n")
        else:
            report.write("**Verified build support:** not established because a successful complete build was not observed.\n\n")

        report.write("## Definitive TensorRT errors\n\n")
        if errors:
            report.writelines(f"- **ERROR:** {message}\n" for message in errors)
        else:
            report.write("- None detected.\n")

        report.write("\n## TensorRT warnings\n\n")
        if warnings:
            report.writelines(f"- **WARNING:** {message}\n" for message in warnings)
        else:
            report.write("- None detected.\n")

        if errors or warnings:
            report.write("\n### Recommended actions\n\n")
            for message in errors + warnings:
                report.write(f"- {diagnostic_recommendation(message)}\n")

        report.write("\n## Potential compatibility or precision review\n\n")
        report.write("These are conservative review items, not proof that an operator is unsupported.\n\n")
        if findings:
            report.write("| Category | Operator | Count | Recommendation |\n| --- | --- | ---: | --- |\n")
            for category, operator, count, recommendation in findings:
                report.write(f"| {category} | `{operator}` | {count} | {recommendation} |\n")
        else:
            report.write("- No rule-based review items detected.\n")

        report.write("\n## Operator inventory\n\n")
        report.write("| Operator | Count |\n| --- | ---: |\n")
        for operator, count in sorted(operator_counts.items()):
            report.write(f"| `{operator}` | {count} |\n")

        report.write("\n## Interpretation\n\n")
        report.write("A successful build proves that this exact TensorRT/CUDA/GPU stack can select implementations for the tested static shapes. ")
        report.write("It does not prove numerical equivalence. Compare representative FP32 and optimized outputs and task metrics before deployment.\n")

    print(f"TensorRT ONNX support result: {status}")
    print(f"  nodes/operators: {len(nodes)}/{len(operator_counts)}")
    print(f"  TensorRT errors: {len(errors)}")
    print(f"  TensorRT warnings: {len(warnings)}")
    print(f"  review findings: {len(findings)}")
    for message in errors[:5]:
        print(f"  ERROR: {message}")
    for message in warnings[:5]:
        print(f"  WARNING: {message}")
    if findings:
        print("  REVIEW: " + ", ".join(f"{operator}({category}, {count})" for category, operator, count, _ in findings))
    print(f"  report: {args.report}")
    if errors or args.command_status != 0:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
