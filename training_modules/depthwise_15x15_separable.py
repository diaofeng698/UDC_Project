"""Rank-1 replacement for a 15x15 depthwise convolution.

The 15x15 kernel is replaced by 15x1 followed by 1x15 depthwise
convolutions.  This reduces weights/MACs per channel from 225 to 30.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Union

import torch
from torch import Tensor, nn

PairLike = Union[int, tuple[int, int]]


def _pair(value: PairLike) -> tuple[int, int]:
    return (value, value) if isinstance(value, int) else value


def _validate_source(conv: nn.Conv2d) -> None:
    if conv.kernel_size != (15, 15):
        raise ValueError(f"expected kernel_size=(15, 15), got {conv.kernel_size}")
    if not (conv.groups == conv.in_channels == conv.out_channels):
        raise ValueError(
            "expected a depthwise Conv2d with groups == in_channels == out_channels"
        )


class SeparableDepthwise15x15(nn.Module):
    """15x15 depthwise convolution approximated by 15x1 and 1x15.

    The API intentionally mirrors the relevant ``nn.Conv2d`` arguments.
    Bias is applied only by the second convolution. No activation is inserted,
    so an existing activation after the original convolution can be retained.

    Args:
        channels: Number of input/output channels.
        stride: Spatial stride of the original 15x15 convolution.
        dilation: Spatial dilation of the original convolution.
        bias: Add one output bias per channel.
        padding_mode: Same meaning as in ``nn.Conv2d``.
    """

    def __init__(
        self,
        channels: int,
        stride: PairLike = 1,
        dilation: PairLike = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")

        stride_h, stride_w = _pair(stride)
        dilation_h, dilation_w = _pair(dilation)

        self.vertical = nn.Conv2d(
            channels,
            channels,
            kernel_size=(15, 1),
            stride=(stride_h, 1),
            padding=(7 * dilation_h, 0),
            dilation=(dilation_h, 1),
            groups=channels,
            bias=False,
            padding_mode=padding_mode,
        )
        self.horizontal = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, 15),
            stride=(1, stride_w),
            padding=(0, 7 * dilation_w),
            dilation=(1, dilation_w),
            groups=channels,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.horizontal(self.vertical(x))

    @classmethod
    def from_conv2d(cls, conv: nn.Conv2d) -> "SeparableDepthwise15x15":
        """Build from a 15x15 depthwise ``Conv2d`` using per-channel SVD.

        Each source kernel is initialized with its best rank-1 approximation.
        This is generally not numerically identical to the source convolution;
        fine-tuning is required after conversion.
        """
        _validate_source(conv)
        replacement = cls(
            channels=conv.in_channels,
            stride=conv.stride,
            dilation=conv.dilation,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
        ).to(device=conv.weight.device, dtype=conv.weight.dtype)

        with torch.no_grad():
            # W ~= (sqrt(S) * U) outer (sqrt(S) * Vh), independently per channel.
            kernels = conv.weight[:, 0].float()
            u, singular_values, vh = torch.linalg.svd(kernels, full_matrices=False)
            scale = singular_values[:, 0].clamp_min(0).sqrt()
            vertical = u[:, :, 0] * scale[:, None]
            horizontal = vh[:, 0, :] * scale[:, None]
            replacement.vertical.weight.copy_(
                vertical[:, None, :, None].to(conv.weight.dtype)
            )
            replacement.horizontal.weight.copy_(
                horizontal[:, None, None, :].to(conv.weight.dtype)
            )
            if conv.bias is not None:
                replacement.horizontal.bias.copy_(conv.bias)

        replacement.train(conv.training)
        return replacement


def replace_depthwise_15x15(module: nn.Module) -> nn.Module:
    """Recursively replace all eligible 15x15 depthwise convolutions.

    A deep copy is returned; the input model is not modified. Source weights
    are converted with per-channel rank-1 SVD initialization.
    """
    converted = deepcopy(module)

    def visit(parent: nn.Module) -> None:
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Conv2d):
                try:
                    replacement = SeparableDepthwise15x15.from_conv2d(child)
                except ValueError:
                    continue
                setattr(parent, name, replacement)
            else:
                visit(child)

    visit(converted)
    return converted
