from ldm.chop.passes.transforms.quantize.quantized_modules import *
from ldm.chop.passes.transforms.quantize.quantized_layer_profiler import *
from ldm.modules.diffusionmodules_quant.openaimodel import QKVAttentionLegacy
import torch.nn as nn
import torch
import numpy as np

def forward_hook_fn(module, input, output):
    # Profile the layer and update statistics
    if isinstance(module, LinearInteger):
        profile_result = profile_linear_layer(module.config, module.in_features, module.out_features, module.bias is not None, input[0].shape[0])
    elif isinstance(module, Conv1dInteger):
        profile_result = profile_conv1d_layer(module.config, module.in_channels, module.out_channels, module.kernel_size[0], module.stride[0], module.bias is not None, input[0].shape[0], input[0].shape[-1])
    elif isinstance(module, Conv2dInteger):
        profile_result = profile_conv2d_layer(module.config, module.in_channels, module.out_channels, module.kernel_size[0], module.stride[0], module.bias is not None, input[0].shape[0], input[0].shape[-2], input[0].shape[-1])
    # ... handle other layer types ...
    elif isinstance(module, QKVAttentionLegacy):
        profile_result = profile_matmul_layer(module.config_0, torch.permute(module.q, (0,2,1)).shape, module.k.shape)
        profile_result_2 = profile_matmul_layer(module.config_1, torch.permute(module.weight, (0,2,1)).shape, module.v.shape)
        for key in profile_result.keys():
            profile_result[key] += profile_result_2[key]
    elif isinstance(module, SiLUInteger):
        profile_result = {}
        profile_result["num_acts"] = input[0].numel()
        profile_result["act_bits"] = module.config["data_in_width"] * input[0].numel()


# Create dictionaries to store layer statistics
layer_stats = {}

for name, layer in model.named_children():
    layer_stats[name] = {"num_params": 0, "num_acts": 0, "param_bits": 0, "act_bits": 0, "flops": 0}
    layer.register_forward_hook(forward_hook_fn)

# Perform a forward pass to collect statistics with hooks

# Calculate average bitwidths and total FLOPs
total_param_bits = sum(stats["param_bits"] for stats in layer_stats.values())
total_activation_bits = sum(stats["act_bits"] for stats in layer_stats.values())
total_flops = sum(stats["flops"] for stats in layer_stats.values())
total_layer_count = len(layer_stats)

average_param_bitwidth = total_param_bits / total_layer_count
average_activation_bitwidth = total_activation_bits / total_layer_count

# Print or store the results
print(f"Average Parameter Bitwidth: {average_param_bitwidth}")
print(f"Average Activation Bitwidth: {average_activation_bitwidth}")
print(f"Total FLOPs: {total_flops}")

# Detach the hooks from the layers
for name, layer in model.named_children():
    layer.register_forward_hook(None)