import logging
import math
from typing import Dict

import numpy as np
import torch

logger = logging.getLogger(__name__)


def compute_tensor_bits_fp(tensor_shape: np.ndarray, width: int):
    return np.prod(tensor_shape) * width


def compute_tensor_bits_integer(tensor_shape: np.ndarray, width: int):
    return np.prod(tensor_shape) * width


def compute_tensor_bits_block_fp(
    tensor_shape: np.ndarray, width: int, exponent_width: int, block_size: np.ndarray
):
    if tensor_shape.size > block_size.size:
        block_size = np.append([1] * (tensor_shape.size - block_size.size), block_size)
    elif tensor_shape.size < block_size.size:
        block_size = block_size[-tensor_shape.ndim :]

    num_blocks = np.prod(np.ceil(tensor_shape / block_size))
    return num_blocks * np.prod(block_size) * width + num_blocks * exponent_width


def profile_linear_layer(
    quant_config: Dict, in_features: int, out_features: int, bias: bool, batch_size: int
):
    """
    {
        "num_params": 0,
        "num_acts": 0,
        "param_bits": 0,
        "act_bits": 0,
        "flops": 0,
        "flops_bitwidth": 0,
    }
    """
    # logger.debug(
    #     f"quant_config = {quant_config},\nin_features = {in_features}, out_features = {out_features}, bias = {bias}, batch_size = {batch_size}"
    # )
    w_shape = np.array((in_features, out_features))
    b_shape = np.array((out_features,))
    x_shape = np.array((batch_size, in_features))

    w_width = quant_config["weight_width"]
    x_width = quant_config["data_in_width"]
    if bias:
        b_width = quant_config["bias_width"]

    # compute num of params, bias and activations
    num_params = in_features * out_features
    if bias:
        num_params += out_features
    num_xs = batch_size * in_features

    # compute param, bias and activation bits
    quant_arith = quant_config["name"]
    if quant_config.get("bypass", False):
        w_width = 32
        b_width = 32
        x_width = 32

        p_bits = compute_tensor_bits_fp(w_shape, w_width)
        if bias:
            p_bits += compute_tensor_bits_fp(b_shape, b_width)
        x_bits = compute_tensor_bits_fp(x_shape, x_width)
    else:
        if quant_arith == "integer":
            p_bits = compute_tensor_bits_integer(w_shape, w_width)
            if bias:
                p_bits += compute_tensor_bits_integer(b_shape, b_width)
            x_bits = compute_tensor_bits_integer(x_shape, x_width)
        elif quant_arith == "block_fp":
            w_block_size = np.array(quant_config["weight_block_size"])
            if bias:
                b_block_size = np.array(quant_config["bias_block_size"])
            x_block_size = np.array(quant_config["data_in_block_size"])

            p_bits = compute_tensor_bits_block_fp(
                w_shape,
                w_width,
                quant_config["weight_exponent_width"],
                w_block_size,
            )
            if bias:
                p_bits += compute_tensor_bits_block_fp(
                    b_shape,
                    b_width,
                    quant_config["bias_exponent_width"],
                    b_block_size,
                )
            x_bits = compute_tensor_bits_block_fp(
                x_shape,
                x_width,
                quant_config["data_in_exponent_width"],
                x_block_size,
            )
        else:
            raise ValueError(f"Unknown quant_arith: {quant_arith}")
    # logger.debug(
    #     f"num_params = {num_params}, num_xs = {num_xs}, p_bits = {p_bits}, x_bits = {x_bits}"
    # )
    # x [batch_size, in_features], w [in_features, out_features], b [out_features]
    # flops = batch_size * out_features * (2 * in_features - 1) + in_features * out_features
    flops = batch_size * out_features * (2 * in_features - 1)
    flops_bitwidth = flops * x_width * w_width
    if bias:
        flops_b = batch_size * out_features
        flops += flops_b
        flops_bitwidth += flops_b * b_width * x_width
    return {
        "num_params": np.rint(num_params).astype(np.int64),
        "num_acts": np.rint(num_xs).astype(np.int64),
        "param_bits": np.rint(p_bits).astype(np.int64),
        "act_bits": np.rint(x_bits).astype(np.int64),
        "flops": np.rint(flops).astype(np.int64),
        "flops_bitwidth": np.rint(flops_bitwidth).astype(np.int64),
    }


def profile_matmul_layer(quant_config: Dict, data_in_0_size, data_in_1_size):
    """
    {
        "num_params": 0,
        "num_acts": 0,
        "param_bits": 0,
        "act_bits": 0,
        "flops": 0,
        "flops_bitwidth": 0,
    """

    x0_shape = np.array((data_in_0_size,))
    x1_shape = np.array((data_in_1_size,))
    num_xs = np.prod(x0_shape) + np.prod(x1_shape)

    quant_arith = quant_config["name"]
    x0_width = quant_config["data_in_width"]
    x1_width = quant_config["data_in_width"]
    num_params = 0

    param_bits = 0
    if quant_config.get("bypass", False):
        x0_width = x1_width = 32
        x_bits = compute_tensor_bits_fp(x0_shape, x0_width) + compute_tensor_bits_fp(
            x1_shape, x1_width
        )
    else:
        if quant_arith == "integer":
            x_bits = compute_tensor_bits_integer(x0_shape, x0_width) + compute_tensor_bits_integer(x1_shape, x1_width)
        elif quant_arith == "block_fp":
            x0_block_size = np.array(quant_config["data_in_block_size"])
            x1_block_size = np.array(quant_config["weight_block_size"])
            x_bits = compute_tensor_bits_block_fp(
                x0_shape,
                x0_width,
                quant_config["data_in_exponent_width"],
                x0_block_size,
            ) + compute_tensor_bits_block_fp(
                x1_shape,
                x1_width,
                quant_config["weight_exponent_width"],
                x1_block_size,
            )
        else:
            raise ValueError(f"Unknown quant_arith: {quant_arith}")

    flops = data_in_0_size[0] * data_in_1_size[1] * (2 * data_in_0_size[1] - 1)
    flops_bitwidth = flops * x0_width * x1_width
    return {
        "num_params": np.rint(num_params).astype(np.int64),
        "num_acts": np.rint(num_xs).astype(np.int64),
        "param_bits": np.rint(param_bits).astype(np.int64),
        "act_bits": np.rint(x_bits).astype(np.int64),
        "flops": np.rint(flops).astype(np.int64),
        "flops_bitwidth": np.rint(flops_bitwidth).astype(np.int64),
    }


def profile_conv1d_layer(
    quant_config: Dict, in_channels: int, out_channels: int, kernel_size: int, stride: int, bias: bool, batch_size: int, sequence_length: int
):
    """
    {
        "num_params": 0,
        "num_acts": 0,
        "param_bits": 0,
        "act_bits": 0,
        "flops": 0,
        "flops_bitwidth": 0,
    """
    w_shape = np.array((out_channels, kernel_size, in_channels))
    b_shape = np.array((out_channels,))
    x_shape = np.array((batch_size, sequence_length, in_channels))

    w_width = quant_config["weight_width"]
    x_width = quant_config["data_in_width"]
    if bias:
        b_width = quant_config["bias_width"]

    num_params = out_channels * in_channels * kernel_size
    if bias:
        num_params += out_channels
    num_xs = batch_size * in_channels * sequence_length

    quant_arith = quant_config["name"]
    if quant_config.get("bypass", False):
        w_width = 32
        b_width = 32
        x_width = 32

        p_bits = compute_tensor_bits_fp(w_shape, w_width)
        if bias:
            p_bits += compute_tensor_bits_fp(b_shape, b_width)
        x_bits = compute_tensor_bits_fp(x_shape, x_width)
    else:
        if quant_arith == "integer":
            p_bits = compute_tensor_bits_integer(w_shape, w_width)
            if bias:
                p_bits += compute_tensor_bits_integer(b_shape, b_width)
            x_bits = compute_tensor_bits_integer(x_shape, x_width)
        elif quant_arith == "block_fp":
            w_block_size = np.array(quant_config["weight_block_size"])
            if bias:
                b_block_size = np.array(quant_config["bias_block_size"])
            x_block_size = np.array(quant_config["data_in_block_size"])

            p_bits = compute_tensor_bits_block_fp(
                w_shape,
                w_width,
                quant_config["weight_exponent_width"],
                w_block_size,
            )
            if bias:
                p_bits += compute_tensor_bits_block_fp(
                    b_shape,
                    b_width,
                    quant_config["bias_exponent_width"],
                    b_block_size,
                )
            x_bits = compute_tensor_bits_block_fp(
                x_shape,
                x_width,
                quant_config["data_in_exponent_width"],
                x_block_size,
            )
        else:
            raise ValueError(f"Unknown quant_arith: {quant_arith}")

    flops = batch_size * out_channels * (2 * in_channels * kernel_size - 1) * (sequence_length // stride)
    flops_bitwidth = flops * x_width * w_width
    if bias:
        flops_b = batch_size * out_channels * (sequence_length // stride)
        flops += flops_b
        flops_bitwidth += flops_b * b_width * x_width
    return {
        "num_params": np.rint(num_params).astype(np.int64),
        "num_acts": np.rint(num_xs).astype(np.int64),
        "param_bits": np.rint(p_bits).astype(np.int64),
        "act_bits": np.rint(x_bits).astype(np.int64),
        "flops": np.rint(flops).astype(np.int64),
        "flops_bitwidth": np.rint(flops_bitwidth).astype(np.int64),
    }

def profile_conv2d_layer(
    quant_config: Dict, in_channels: int, out_channels: int, kernel_size: int, stride: int, bias: bool, batch_size: int, height: int, width: int
):
    """
    {
        "num_params": 0,
        "num_acts": 0,
        "param_bits": 0,
        "act_bits": 0,
        "flops": 0,
        "flops_bitwidth": 0,
    """
    w_shape = np.array((out_channels, kernel_size * kernel_size, in_channels))
    b_shape = np.array((out_channels,))
    x_shape = np.array((batch_size, height * width, in_channels))

    w_width = quant_config["weight_width"]
    x_width = quant_config["data_in_width"]
    if bias:
        b_width = quant_config["bias_width"]

    num_params = out_channels * in_channels * kernel_size * kernel_size
    if bias:
        num_params += out_channels
    num_xs = batch_size * in_channels * height * width

    quant_arith = quant_config["name"]
    if quant_config.get("bypass", False):
        w_width = 32
        b_width = 32
        x_width = 32

        p_bits = compute_tensor_bits_fp(w_shape, w_width)
        if bias:
            p_bits += compute_tensor_bits_fp(b_shape, b_width)
        x_bits = compute_tensor_bits_fp(x_shape, x_width)
    else:
        if quant_arith == "integer":
            p_bits = compute_tensor_bits_integer(w_shape, w_width)
            if bias:
                p_bits += compute_tensor_bits_integer(b_shape, b_width)
            x_bits = compute_tensor_bits_integer(x_shape, x_width)
        elif quant_arith == "block_fp":
            w_block_size = np.array(quant_config["weight_block_size"])
            if bias:
                b_block_size = np.array(quant_config["bias_block_size"])
            x_block_size = np.array(quant_config["data_in_block_size"])

            p_bits = compute_tensor_bits_block_fp(
                w_shape,
                w_width,
                quant_config["weight_exponent_width"],
                w_block_size,
            )
            if bias:
                p_bits += compute_tensor_bits_block_fp(
                    b_shape,
                    b_width,
                    quant_config["bias_exponent_width"],
                    b_block_size,
                )
            x_bits = compute_tensor_bits_block_fp(
                x_shape,
                x_width,
                quant_config["data_in_exponent_width"],
                x_block_size,
            )
        else:
            raise ValueError(f"Unknown quant_arith: {quant_arith}")

    flops = batch_size * out_channels * (2 * in_channels * kernel_size * kernel_size - 1) * (height // stride) * (width // stride)
    flops_bitwidth = flops * x_width * w_width
    if bias:
        flops_b = batch_size * out_channels * (height // stride) * (width // stride)
        flops += flops_b
        flops_bitwidth += flops_b * b_width * x_width
    return {
        "num_params": np.rint(num_params).astype(np.int64),
        "num_acts": np.rint(num_xs).astype(np.int64),
        "param_bits": np.rint(p_bits).astype(np.int64),
        "act_bits": np.rint(x_bits).astype(np.int64),
        "flops": np.rint(flops).astype(np.int64),
        "flops_bitwidth": np.rint(flops_bitwidth).astype(np.int64),
    }




def update_profile(profile, delta):
    profile["num_params"] += delta["num_params"]
    profile["num_acts"] += delta["num_acts"]
    profile["param_bits"] += delta["param_bits"]
    profile["act_bits"] += delta["act_bits"]
    profile["flops"] += delta["flops"]
    profile["flops_bitwidth"] += delta["flops_bitwidth"]
    return profile


def register_a_stat_hook(stat_manager, name: str, module: torch.nn.Module, entry: str):
    if entry == "data_in":
        module.register_forward_pre_hook(
            stat_manager.get_pre_forward_act_hook(name)
        )
    elif entry == "weight":
        module.register_forward_pre_hook(
            stat_manager.get_pre_forward_weight_hook(name, weight_name="weight")
        )
    elif entry == "bias":
        module.register_forward_pre_hook(
            stat_manager.get_pre_forward_weight_hook(name, weight_name="bias")
        )
    elif entry == "data_out":
        module.register_forward_hook(stat_manager.get_post_forward_act_hook(name))
    else:
        raise ValueError(f"Unknown entry: {entry}")
