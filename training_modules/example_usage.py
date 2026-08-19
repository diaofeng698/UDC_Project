"""Usage examples for both 15x15 depthwise replacement modules.

Run from the project root after installing PyTorch:

    python3 -m training_modules.example_usage --method separable
    python3 -m training_modules.example_usage --method stacked
    python3 -m training_modules.example_usage --method stacked-activation
"""

from __future__ import annotations

import argparse

import torch
from torch import Tensor, nn

from . import (
    SeparableDepthwise15x15,
    StackedDepthwise15x15,
    replace_with_separable_depthwise,
    replace_with_stacked_depthwise,
)


class ExampleBlock(nn.Module):
    """Example block containing the expensive convolution to replace."""

    def __init__(self, channels: int = 72) -> None:
        super().__init__()
        self.pre_conv8 = nn.Conv2d(
            channels,
            channels,
            kernel_size=15,
            stride=1,
            padding=7,
            groups=channels,
            bias=False,
        )
        self.conv8 = nn.Conv2d(
            channels,
            channels // 2,
            kernel_size=3,
            padding=8,
            dilation=8,
            bias=False,
        )
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.conv8(self.pre_conv8(x)))


class ExampleModel(nn.Module):
    def __init__(self, channels: int = 72) -> None:
        super().__init__()
        self.body = nn.Sequential(
            ExampleBlock(channels),
            nn.Conv2d(channels // 2, channels, kernel_size=1),
            ExampleBlock(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


def direct_layer_examples() -> None:
    """Example 1: use either replacement directly in a model definition."""
    x = torch.randn(1, 72, 64, 64)

    separable = SeparableDepthwise15x15(channels=72, bias=False)
    stacked = StackedDepthwise15x15(channels=72, bias=False)

    assert separable(x).shape == x.shape
    assert stacked(x).shape == x.shape
    print("Direct modules: output shape", tuple(x.shape))


def count_target_layers(model: nn.Module) -> int:
    return sum(
        isinstance(layer, nn.Conv2d)
        and layer.kernel_size == (15, 15)
        and layer.groups == layer.in_channels == layer.out_channels
        for layer in model.modules()
    )


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def build_replacement(model: nn.Module, method: str) -> nn.Module:
    """Example 2: recursively replace all eligible layers in a model."""
    if method == "separable":
        # Best choice when converting a pretrained model. Existing 15x15
        # weights initialize the two convolutions through per-channel SVD.
        return replace_with_separable_depthwise(model)

    if method == "stacked":
        # No intermediate activation: retains a linear replacement structure.
        # The seven convolutions are newly initialized and require training.
        return replace_with_stacked_depthwise(model)

    if method == "stacked-activation":
        # Redesigned nonlinear block; use for retraining rather than attempting
        # to preserve the exact behavior of the pretrained 15x15 convolution.
        return replace_with_stacked_depthwise(
            model,
            activation=lambda: nn.LeakyReLU(0.1, inplace=True),
        )

    raise ValueError(f"unsupported replacement method: {method}")


def one_training_step(model: nn.Module, device: torch.device) -> float:
    """Example 3: replacement modules train like normal PyTorch modules."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.L1Loss()

    inputs = torch.randn(2, 72, 64, 64, device=device)

    optimizer.zero_grad(set_to_none=True)
    predictions = model(inputs)
    targets = torch.randn_like(predictions)
    loss = criterion(predictions, targets)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=("separable", "stacked", "stacked-activation"),
        default="separable",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    direct_layer_examples()

    original = ExampleModel()
    optimized = build_replacement(original, args.method)
    device = torch.device(args.device)
    optimized.to(device)

    print("Method:", args.method)
    print("Original target layers:", count_target_layers(original))
    print("Optimized target layers:", count_target_layers(optimized))
    print("Original parameters:", count_parameters(original))
    print("Optimized parameters:", count_parameters(optimized))

    with torch.no_grad():
        sample = torch.randn(1, 72, 64, 64, device=device)
        output = optimized(sample)
        print("Model output shape:", tuple(output.shape))

    loss = one_training_step(optimized, device)
    print("Training step loss:", loss)


if __name__ == "__main__":
    main()
