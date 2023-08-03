import os
import re
from copy import deepcopy
from typing import List, Dict, Tuple
from dataclasses import dataclass

import toml
from ldm.chop.tools.config_load import convert_str_na_to_none

from ..quant_utils import parse_node_config

"""
An example of quant_config for llama

{
    "model_layer": {
        "self_attn": {
            "q_proj": {},
            "k_proj": {},
            "v_proj": {},
            "o_proj": {},
            "matmul_0": {},
            "matmul_1": {},
        },
        "mlp": {
            "gate_proj": {},
            "down_proj": {},
            "up_proj": {},
        },
    }
    "linear_default": {},
    "matmul_default": {},

}
"""


def cp_multi_values(src: Dict, dst: Dict, src_keys: Tuple, dst_keys: Tuple = None):
    """Copy multiple values from src dict to dst dict."""
    if dst_keys is None:
        for key in src_keys:
            dst[key] = deepcopy(src[key])
    else:
        for src_key, dst_key in zip(src_keys, dst_keys):
            dst[dst_key] = deepcopy(src[src_key])


def has_multi_keys(src: Dict, keys: Tuple):
    """Check if src dict has multiple keys."""
    for key in keys:
        if key not in src:
            return False
    return True


def match_a_pattern(name: str, patterns: List[str]) -> str | None:
    for pattern in patterns:
        match = re.fullmatch(pattern, name)
        if match:
            return pattern
    return None


def create_a_layer_config(
    linear_qc: Dict = None, matmul_qc: Dict = None, layer_qc=None
) -> Dict:
    if (layer_qc is None and matmul_qc is None) and layer_qc is None:
        raise ValueError("Must provide either (linear_qc & matmul_qc) or layer_qc")
    if layer_qc is None:
        layer_qc = {}
    # fmt: off
    qc = {
        "self_attn": {
            "q_proj": parse_node_config(layer_qc.get("self_attn", {}).get("q_proj", linear_qc), "linear"),
            "k_proj": parse_node_config(layer_qc.get("self_attn", {}).get("k_proj", linear_qc), "linear"),
            "v_proj": parse_node_config(layer_qc.get("self_attn", {}).get("v_proj", linear_qc), "linear"),
            "o_proj": parse_node_config(layer_qc.get("self_attn", {}).get("o_proj", linear_qc), "linear"),
            "matmul_0": parse_node_config(layer_qc.get("self_attn", {}).get("matmul_0", matmul_qc), "matmul"),
            "matmul_1": parse_node_config(layer_qc.get("self_attn", {}).get("matmul_1", matmul_qc), "matmul"),
        },
        "mlp": {
            "gate_proj": parse_node_config(layer_qc.get("mlp", {}).get("gate_proj", linear_qc), "linear"),
            "down_proj": parse_node_config(layer_qc.get("mlp", {}).get("down_proj", linear_qc), "linear"),
            "up_proj": parse_node_config(layer_qc.get("mlp", {}).get("up_proj", linear_qc), "linear")
        },
    }
    # fmt: on
    return qc


def by_type_parser(config: Dict, num_hidden_layers: int) -> Dict:
    assert "default" in config, "Must provide default config for by_class_parser"
    default_qc: Dict = config["default"]
    linear_qc: Dict = parse_node_config(
        config.get("linear", default_qc), mase_op="linear"
    )
    matmul_qc: Dict = parse_node_config(
        config.get("matmul", default_qc), mase_op="matmul"
    )
    layer_qc: Dict = config.get("model_layer", None)

    # parsed config
    p_config = {}
    for i in range(num_hidden_layers):
        layer_entry = f"model_layer_{i}"
        p_config[layer_entry] = create_a_layer_config(linear_qc, matmul_qc, layer_qc)
    p_config["default"] = default_qc
    return p_config


def by_name_parser(config: Dict, num_hidden_layers: int) -> Dict:
    assert "default" in config, "Must provide default config for by_name_parser"
    default_qc: Dict = config["default"]
    linear_qc: Dict = parse_node_config(
        config.get("linear", default_qc), mase_op="linear"
    )
    matmul_qc: Dict = parse_node_config(
        config.get("matmul", default_qc), mase_op="matmul"
    )

    # parsed config
    p_config = {}
    for i in range(num_hidden_layers):
        layer_entry = f"model_layer_{i}"
        layer_qc = config.get(layer_entry, None)
        p_config[layer_entry] = create_a_layer_config(linear_qc, matmul_qc, layer_qc)
    p_config["default"] = default_qc
    return p_config


def parse_llama_quantized_config(config: str | Dict, num_hidden_layers: int) -> Dict:
    assert isinstance(
        config, (str, dict)
    ), "config must be a str path to config toml or dict"
    if isinstance(config, str):
        config = toml.load(config)
    config = convert_str_na_to_none(config)
    by = config.pop("by", "type")
    # match by:
    #     case "type":
    if by == "type": 
        #     return by_type_parser(config, num_hidden_layers)
        return by_type_parser(config, num_hidden_layers)
    #     case "name":
    elif by == "name": 
        #     return by_name_parser(config, num_hidden_layers)
        return by_name_parser(config, num_hidden_layers)
    #     case _:
    else:
        #     raise ValueError(f"Unknown by: {by}")
        raise ValueError(f"Unknown by: {by}")
