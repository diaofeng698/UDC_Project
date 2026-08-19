"""Training-time alternatives to expensive 15x15 depthwise convolutions."""

from .depthwise_15x15_separable import (
    SeparableDepthwise15x15,
    replace_depthwise_15x15 as replace_with_separable_depthwise,
)
from .depthwise_15x15_stack import (
    StackedDepthwise15x15,
    replace_depthwise_15x15 as replace_with_stacked_depthwise,
)

__all__ = [
    "SeparableDepthwise15x15",
    "StackedDepthwise15x15",
    "replace_with_separable_depthwise",
    "replace_with_stacked_depthwise",
]
