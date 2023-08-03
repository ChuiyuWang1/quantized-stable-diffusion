from typing import Callable, Dict

from ldm.chop.passes import parse_node_config
from ldm.chop.passes import quantized_func_map
from ldm.chop.passes import quantized_module_map


def get_quantized_cls(mase_op: str, quant_config: Dict) -> type:
    quant_arith = quant_config["name"]
    return quantized_module_map[f"{mase_op}_{quant_arith}"]


def get_quantized_func(mase_op: str, quant_config: Dict) -> Callable:
    quant_arith = quant_config["name"]
    return quantized_func_map[f"{mase_op}_{quant_arith}"]


def parse_op_quant_config(mase_op: str, config: Dict) -> Dict:
    return parse_node_config(config=config, mase_op=mase_op)
