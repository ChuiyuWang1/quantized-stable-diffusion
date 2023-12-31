import os
import re
from typing import List, Dict, Tuple, Union
from copy import deepcopy

import toml
from ldm.chop.tools.config_load import convert_str_na_to_none

from ..quant_utils import parse_node_config

"""
An example of quant_config for bert

{
    "default": {}
    "model_layer": {
        "attention": {
            "query": {},
            "key": {},
            "value": {},
            "output": {
                "dense": {},
            },
            "matmul_0": {},
            "matmul_1": {},
        },
        "crossattntion": { # if config.add_cross_attention is True
            "query": {},
            "key": {},
            "value": {},
            "output": {
                "dense": {},
            },
            "matmul_0": {},
            "matmul_1": {},
        }
        "intermediate": {
            "dense": {},
        },
        "output": {
            "dense": {},
        },
    }
    "linear_default": {},
    "matmul_default": {},
    "model_layer_0": {
        "attention": {
            ...
        },
        ...
    }
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


def match_a_pattern(name: str, patterns: List[str]) -> Union[str, None]:
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
        "attention": {
            "query": parse_node_config(layer_qc.get("attention", {}).get("query", linear_qc), "linear"),
            "key": parse_node_config(layer_qc.get("attention", {}).get("key", linear_qc), "linear"),
            "value": parse_node_config(layer_qc.get("attention", {}).get("value", linear_qc), "linear"),
            "matmul_0": parse_node_config(layer_qc.get("attention", {}).get("matmul_0", matmul_qc), "matmul"),
            "matmul_1": parse_node_config(layer_qc.get("attention", {}).get("matmul_1", matmul_qc), "matmul"),
            "output": {
                "dense": parse_node_config(layer_qc.get("attention", {}).get("output", {}).get("dense", linear_qc), "linear"),
            },
        },
        "intermediate": {
            "dense": parse_node_config(layer_qc.get("intermediate", {}).get("dense", linear_qc), "linear"),
        },
        "output": {
            "dense": parse_node_config(layer_qc.get("output", {}).get("dense", linear_qc), "linear"),
        },
    }
    # fmt: on
    return qc


def by_type_parser(config: Dict, num_hidden_layers: int) -> Dict:
    assert "default" in config, "Must provide a default config"
    default_qc: Dict = config["default"]
    linear_qc: Dict = parse_node_config(
        config.get("linear", default_qc), mase_op="linear"
    )
    matmul_qc: Dict = parse_node_config(
        config.get("matmul", default_qc), mase_op="matmul"
    )
    layer_qc: Dict = config.get("model_layer", None)

    p_config = {}
    for i in range(num_hidden_layers):
        layer_entry = f"model_layer_{i}"
        p_config[layer_entry] = create_a_layer_config(linear_qc, matmul_qc, layer_qc)
    p_config["default"] = default_qc
    return p_config


def by_name_parser(config: Dict, num_hidden_layers: int) -> Dict:
    assert "default" in config, "Must provide a default config"
    default_qc: Dict = config["default"]
    linear_qc: Dict = parse_node_config(
        config.get("linear", default_qc), mase_op="linear"
    )
    matmul_qc: Dict = parse_node_config(
        config.get("matmul", default_qc), mase_op="matmul"
    )

    p_config = {}
    for i in range(num_hidden_layers):
        layer_entry = f"model_layer_{i}"
        layer_qc = config.get(layer_entry, None)
        p_config[layer_entry] = create_a_layer_config(linear_qc, matmul_qc, layer_qc)
    p_config["default"] = default_qc
    return p_config


def parse_bert_quantized_config(config: Union[str, Dict], num_hidden_layers: int) -> Dict:
    assert isinstance(config, (str, dict)), "Must provide either a path or a dict"
    if isinstance(config, str):
        config = toml.load(config)
    config = convert_str_na_to_none(config)
    by = config.pop("by", "type")
    # match by:
    if by == "type": 
    #     case "type":
        #     return by_type_parser(config, num_hidden_layers)
        return by_type_parser(config, num_hidden_layers)
    #     case "name":
    elif by == "name": 
        #     return by_name_parser(config, num_hidden_layers)
        return by_name_parser(config, num_hidden_layers)
    #     case _:
    else:
        #     raise ValueError(f"Unknown quantized config type: {by}")
        raise ValueError(f"Unknown quantized config type: {by}")
