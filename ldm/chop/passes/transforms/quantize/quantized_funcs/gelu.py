from functools import partial

import torch
import torch.nn.functional as F

# from ....graph.mase_tracer import mark_as_leaf_func
from ..quantizers import (
    block_fp_quantizer,
    block_log_quantizer,
    block_minifloat_quantizer,
    integer_quantizer,
    log_quantizer,
    minifloat_denorm_quantizer,
    minifloat_ieee_quantizer,
    binary_quantizer,
)

def gelu_integer(x, config=None):
    bypass = config.get("bypass", False)
    if bypass:
        return F.gelu(x)
    else:
        x_width, x_frac_width = config["data_in_width"], config["data_in_frac_width"]
        x_quantizer = partial(
            integer_quantizer, width=x_width, frac_width=x_frac_width, is_signed=True
        )

        return F.gelu(x_quantizer(x))

def gelu_block_fp(x, config=None):
    bypass = config.get("bypass", False)
    if bypass:
        return F.gelu(x)
    else:
        x_width, x_exponent_width, x_exponent_bias, x_block_size = (
            config["data_in_width"],
            config["data_in_exponent_width"],
            config["data_in_exponent_bias"],
            config["data_in_block_size"],
        )

        x_more_than_2_dims = x.ndim > 2
        x_quantizer = partial(
            block_fp_quantizer,
            width=x_width,
            exponent_width=x_exponent_width,
            exponent_bias=x_exponent_bias,
            block_size=x_block_size,
            skip_first_dim=x_more_than_2_dims,
        )

        x_shape = [i for i in x.shape]
        if x_more_than_2_dims:
            x = torch.flatten(x, start_dim=0, end_dim=-3)
        x = x_quantizer(x)
        x = torch.reshape(x, x_shape)
        return F.gelu(x)