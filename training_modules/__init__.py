"""Training-time alternatives to expensive 15x15 depthwise convolutions."""

from .depthwise_15x15_separable import (
    SeparableDepthwise15x15,
    replace_depthwise_15x15 as replace_with_separable_depthwise,
)
from .depthwise_15x15_stack import (
    StackedDepthwise15x15,
    replace_depthwise_15x15 as replace_with_stacked_depthwise,
)
from .tiny_matvec_3x3 import (
    TinyMatVec3x3,
    TwoTinyMatVec3x3,
    tiny_matvec_expanded,
    tiny_matvec_reduce,
)

__all__ = [
    "SeparableDepthwise15x15",
    "StackedDepthwise15x15",
    "TinyMatVec3x3",
    "TwoTinyMatVec3x3",
    "replace_with_separable_depthwise",
    "replace_with_stacked_depthwise",
    "tiny_matvec_expanded",
    "tiny_matvec_reduce",
]
