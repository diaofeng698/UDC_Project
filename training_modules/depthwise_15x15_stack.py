"""Seven-layer 3x3 replacement for a 15x15 depthwise convolution.

Seven stride-1 3x3 layers have the same 15x15 receptive-field size while
reducing weights/MACs per channel from 225 to 63.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Callable, Optional, Union
import warnings

import torch
from torch import Tensor, nn

PairLike = Union[int, tuple[int, int]]
ActivationFactory = Optional[Callable[[], nn.Module]]


def _pair(value: PairLike) -> tuple[int, int]:
    return (value, value) if isinstance(value, int) else value


def _validate_source(conv: nn.Conv2d) -> None:
    if conv.kernel_size != (15, 15):
        raise ValueError(f"expected kernel_size=(15, 15), got {conv.kernel_size}")
    if not (conv.groups == conv.in_channels == conv.out_channels):
        raise ValueError(
            "expected a depthwise Conv2d with groups == in_channels == out_channels"
        )


class StackedDepthwise15x15(nn.Module):
    """Replace one 15x15 depthwise convolution with seven 3x3 layers.

    By default no intermediate activations are inserted, making this a linear
    replacement with the same receptive-field size. Pass an activation factory,
    for example ``lambda: nn.LeakyReLU(0.1, inplace=True)``, to increase model
    capacity when training a redesigned architecture.

    Args:
        channels: Number of input/output channels.
        stride: Spatial stride. It is applied by the first 3x3 layer.
        dilation: Dilation used by every 3x3 layer.
        bias: Add bias only to the final layer.
        padding_mode: Same meaning as in ``nn.Conv2d``.
        activation: Optional factory creating activations between convolutions.
    """

    depth = 7

    def __init__(
        self,
        channels: int,
        stride: PairLike = 1,
        dilation: PairLike = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        activation: ActivationFactory = None,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")

        stride_h, stride_w = _pair(stride)
        dilation_h, dilation_w = _pair(dilation)
        padding = (dilation_h, dilation_w)
        layers: list[nn.Module] = []

        for index in range(self.depth):
            layers.append(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=3,
                    stride=(stride_h, stride_w) if index == 0 else 1,
                    padding=padding,
                    dilation=(dilation_h, dilation_w),
                    groups=channels,
                    bias=bias and index == self.depth - 1,
                    padding_mode=padding_mode,
                )
            )
            if activation is not None and index != self.depth - 1:
                layers.append(activation())

        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

    @property
    def convolutions(self) -> list[nn.Conv2d]:
        """Return the seven depthwise convolution layers."""
        return [layer for layer in self.layers if isinstance(layer, nn.Conv2d)]

    def reset_identity(self) -> None:
        """Initialize the linear stack as an identity mapping.

        This is useful for residual architectures. It does not approximate the
        source 15x15 kernel.
        """
        with torch.no_grad():
            for conv in self.convolutions:
                conv.weight.zero_()
                conv.weight[:, 0, 1, 1] = 1
                if conv.bias is not None:
                    conv.bias.zero_()

    @classmethod
    def from_conv2d(
        cls,
        conv: nn.Conv2d,
        *,
        activation: ActivationFactory = None,
    ) -> "StackedDepthwise15x15":
        """Create a trainable replacement for a 15x15 depthwise ``Conv2d``.

        A general 15x15 kernel cannot be copied exactly into seven 3x3 kernels.
        The replacement therefore uses Kaiming initialization and copies only
        the source bias. Fine-tune or retrain after conversion.
        """
        _validate_source(conv)
        replacement = cls(
            channels=conv.in_channels,
            stride=conv.stride,
            dilation=conv.dilation,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
            activation=activation,
        ).to(device=conv.weight.device, dtype=conv.weight.dtype)

        with torch.no_grad():
            for layer in replacement.convolutions:
                nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="linear")
            if conv.bias is not None:
                replacement.convolutions[-1].bias.copy_(conv.bias)

        replacement.train(conv.training)
        warnings.warn(
            "15x15 weights cannot be copied exactly into seven 3x3 layers; "
            "the replacement requires fine-tuning or retraining.",
            UserWarning,
            stacklevel=2,
        )
        return replacement


def replace_depthwise_15x15(
    module: nn.Module,
    *,
    activation: ActivationFactory = None,
) -> nn.Module:
    """Recursively replace eligible 15x15 depthwise convolutions.

    A deep copy is returned. The new 3x3 stacks require fine-tuning/retraining.
    """
    converted = deepcopy(module)

    def visit(parent: nn.Module) -> None:
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Conv2d):
                try:
                    replacement = StackedDepthwise15x15.from_conv2d(
                        child, activation=activation
                    )
                except ValueError:
                    continue
                setattr(parent, name, replacement)
            else:
                visit(child)

    visit(converted)
    return converted
