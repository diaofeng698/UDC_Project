#!/usr/bin/env python3
"""Deterministic preprocessed tensor dataset for PTQ and accuracy evaluation.

Supported sources:

* A directory containing ``.npy`` or ``.npz`` files (sorted recursively).
* A text manifest with one ``.npy``/``.npz`` path per line.
* A JSONL manifest whose rows map model input names to tensor paths.

Images are deliberately not decoded here. Every sample must already use the
model's exact color order, normalization, dtype, and NCHW/NHWC layout. This
prevents a hidden preprocessing mismatch from invalidating calibration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np


def _resolve(path: str, base: Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else (base / value).resolve()


def discover_samples(source: Path) -> list[Path | dict[str, Path]]:
    """Return samples in deterministic order from a directory or manifest."""
    source = source.expanduser().resolve()
    if source.is_dir():
        samples = sorted(path for path in source.rglob("*") if path.suffix.lower() in {".npy", ".npz"})
    elif source.is_file() and source.suffix.lower() == ".jsonl":
        samples = []
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not record:
                raise ValueError(f"{source}:{line_number}: expected a non-empty JSON object")
            samples.append({str(name): _resolve(str(path), source.parent) for name, path in record.items()})
    elif source.is_file():
        samples = [
            _resolve(line.strip(), source.parent)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        raise FileNotFoundError(f"dataset source does not exist: {source}")
    if not samples:
        raise ValueError(f"no .npy/.npz samples found in {source}")
    return samples


def _load_array(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"sample tensor not found: {path}")
    value = np.load(path, allow_pickle=False)
    if isinstance(value, np.lib.npyio.NpzFile):
        try:
            if len(value.files) != 1:
                raise ValueError(f"{path} has {len(value.files)} arrays; use keys matching model inputs")
            return np.asarray(value[value.files[0]])
        finally:
            value.close()
    return np.asarray(value)


def load_sample(sample: Path | Mapping[str, Path], input_names: Sequence[str]) -> dict[str, np.ndarray]:
    """Load one sample and validate that every model input is present."""
    if isinstance(sample, Mapping):
        unknown = sorted(set(sample) - set(input_names))
        missing = sorted(set(input_names) - set(sample))
        if unknown or missing:
            raise ValueError(f"input mapping mismatch; missing={missing}, unknown={unknown}")
        values = {name: _load_array(sample[name]) for name in input_names}
    elif sample.suffix.lower() == ".npz":
        archive = np.load(sample, allow_pickle=False)
        try:
            if set(input_names).issubset(archive.files):
                values = {name: np.asarray(archive[name]) for name in input_names}
            elif len(input_names) == 1 and len(archive.files) == 1:
                values = {input_names[0]: np.asarray(archive[archive.files[0]])}
            else:
                raise ValueError(f"{sample}: expected NPZ keys {list(input_names)}, got {archive.files}")
        finally:
            archive.close()
    else:
        if len(input_names) != 1:
            raise ValueError("multi-input models require NPZ files or a JSONL input mapping")
        values = {input_names[0]: _load_array(sample)}

    result: dict[str, np.ndarray] = {}
    for name, value in values.items():
        if value.dtype != np.float32:
            value = value.astype(np.float32)
        result[name] = np.ascontiguousarray(value)
    return result


def iter_samples(
    source: Path,
    input_names: Sequence[str],
    *,
    limit: int | None = None,
) -> Iterator[tuple[str, dict[str, np.ndarray]]]:
    samples = discover_samples(source)
    if limit is not None:
        if limit <= 0:
            raise ValueError("sample limit must be positive")
        samples = samples[:limit]
    for index, sample in enumerate(samples):
        label = str(sample) if isinstance(sample, Path) else f"jsonl-row-{index + 1}"
        yield label, load_sample(sample, input_names)