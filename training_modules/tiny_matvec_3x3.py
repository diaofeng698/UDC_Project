"""TensorRT-friendly replacements for tiny batched 3x3 matrix-vector products.

The functions in this module replace ``torch.matmul(matrix, vector)`` where
``matrix`` ends in ``(3, 3)`` and ``vector`` ends in either ``(3,)`` or
``(3, 1)``.  They intentionally use no matrix-multiplication operator, so an
ONNX export contains elementwise operations and either ``ReduceSum`` or an
explicit three-term addition instead of tiny GEMV layers.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

Implementation = Literal["reduce", "expanded"]


def _validate_inputs(matrix: Tensor, vector: Tensor) -> bool:
    if matrix.ndim < 2 or matrix.shape[-2:] != (3, 3):
        raise ValueError(
            f"matrix must have trailing shape (3, 3), got {tuple(matrix.shape)}"
        )

    column_vector = vector.ndim >= 2 and vector.shape[-2:] == (3, 1)
    flat_vector = vector.ndim >= 1 and vector.shape[-1] == 3
    if not (column_vector or flat_vector):
        raise ValueError(
            f"vector must have trailing shape (3,) or (3, 1), got {tuple(vector.shape)}"
        )
    return column_vector


def tiny_matvec_reduce(matrix: Tensor, vector: Tensor) -> Tensor:
    """Compute a batched 3x3 matrix-vector product with ``Mul + ReduceSum``.

    Leading dimensions follow normal PyTorch broadcasting rules.  The output
    ends in ``(3,)`` for a flat input vector and ``(3, 1)`` for a column vector.
    This is the preferred first implementation to benchmark in TensorRT.
    """
    column_vector = _validate_inputs(matrix, vector)
    flat = vector.squeeze(-1) if column_vector else vector
    result = torch.sum(matrix * flat.unsqueeze(-2), dim=-1)
    return result.unsqueeze(-1) if column_vector else result


def tiny_matvec_expanded(matrix: Tensor, vector: Tensor) -> Tensor:
    """Compute a batched 3x3 matrix-vector product as three explicit products."""
    column_vector = _validate_inputs(matrix, vector)
    flat = vector.squeeze(-1) if column_vector else vector
    result = (
        matrix[..., 0] * flat[..., 0:1]
        + matrix[..., 1] * flat[..., 1:2]
        + matrix[..., 2] * flat[..., 2:3]
    )
    return result.unsqueeze(-1) if column_vector else result


class TinyMatVec3x3(nn.Module):
    """Drop-in module for a TensorRT-friendly tiny matrix-vector product.

    Replace code such as ``y = torch.matmul(a, x)`` with
    ``y = self.tiny_matvec(a, x)`` after adding this module to ``__init__``.
    The module has no parameters and works in training, inference, and ONNX
    export.  Use ``implementation="expanded"`` to avoid a reduction operator.
    """

    def __init__(self, implementation: Implementation = "reduce") -> None:
        super().__init__()
        if implementation not in ("reduce", "expanded"):
            raise ValueError(f"unsupported implementation: {implementation}")
        self.implementation = implementation

    def forward(self, matrix: Tensor, vector: Tensor) -> Tensor:
        if self.implementation == "reduce":
            return tiny_matvec_reduce(matrix, vector)
        return tiny_matvec_expanded(matrix, vector)


class TwoTinyMatVec3x3(nn.Module):
    """Convenience replacement for models containing two independent MatMuls."""

    def __init__(self, implementation: Implementation = "reduce") -> None:
        super().__init__()
        self.first = TinyMatVec3x3(implementation)
        self.second = TinyMatVec3x3(implementation)

    def forward(
        self,
        matrix_0: Tensor,
        vector_0: Tensor,
        matrix_1: Tensor,
        vector_1: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return self.first(matrix_0, vector_0), self.second(matrix_1, vector_1)