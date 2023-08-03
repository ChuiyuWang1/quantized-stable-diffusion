from functools import partial
from math import ceil, log2
from typing import Union, Dict

import torch
from ldm.chop.passes.transforms.quantize.quantizers import (
    block_fp_quantizer,
    block_log_quantizer,
    block_minifloat_quantizer,
    integer_quantizer,
    log_quantizer,
    minifloat_denorm_quantizer,
    minifloat_ieee_quantizer,
    binary_quantizer,
)
from torch import Tensor
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t


class _GroupNormBase(torch.nn.GroupNorm):
    def __init__(
        self,
        num_groups: int, 
        num_channels: int, 
        eps: float = 1e-5, 
        affine: bool = True,
        device=None, 
        dtype=None) -> None:
        super().__init__(
            num_groups, 
            num_channels, 
            eps, 
            affine,
            device,
            dtype,
        )
        self.bypass = False
        self.x_quantizer = None
        self.w_quantizer = None
        self.b_quantizer = None

    def forward(self, x: Tensor) -> Tensor:
        if self.bypass:
            return super().forward(x)
        x = self.x_quantizer(x)
        w = self.w_quantizer(self.weight) if self.affine else None
        bias = self.b_quantizer(self.bias) if self.affine else None
        return F.group_norm(x, self.num_groups, w, bias, self.eps)

    def get_quantized_weight(self) -> Tensor:
        return self.w_quantizer(self.weight)

    def get_quantized_weights_with_inputs(self, x: Tensor) -> Dict:
        x = self.x_quantizer(x)
        w = self.w_quantizer(self.weight)
        bias = self.b_quantizer(self.bias) if self.bias is not None else None
        y = F.group_norm(x, self.num_groups, w, bias, self.eps)
        return {
            "x": x,
            "w": w,
            "bias": bias,
            "y": y,
        }
    
class GroupNormInteger(_GroupNormBase):
    def __init__(self,
        num_groups: int, 
        num_channels: int, 
        eps: float = 1e-5, 
        affine: bool = True,
        device=None, 
        dtype=None,
        config=None,
    ) -> None:
        super().__init__(
            num_groups, 
            num_channels, 
            eps, 
            affine,
            device,
            dtype,
        )
        assert config is not None, "config is None!"
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return

        # establish quantizers
        w_width, w_frac_width = config["weight_width"], config["weight_frac_width"]
        x_width, x_frac_width = config["data_in_width"], config["data_in_frac_width"]
        # check bias quantizer, if not, use weight quantizer
        b_width, b_frac_width = config["bias_width"], config["bias_frac_width"]
        self.w_quantizer = partial(
            integer_quantizer, width=w_width, frac_width=w_frac_width
        )
        self.x_quantizer = partial(
            integer_quantizer, width=x_width, frac_width=x_frac_width
        )
        self.b_quantizer = partial(
            integer_quantizer, width=b_width, frac_width=b_frac_width
        )