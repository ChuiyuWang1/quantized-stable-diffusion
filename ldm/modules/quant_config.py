from typing import Dict

class UNetQuantConfig():
    def __init__(self) -> None:
        self.quant_config = {}
        self.config_keys = [
            "attpool_qkf", 
            "attpool_c",
            "upsample",
            "downsample",
            "avg_pool2d",
            "silu_res_in",
            "resblock_in",
            "silu_res_emb",
            "resblock_emb",
            "silu_res_out",
            "resblock_out",
            "resblock_skip",
            "attblock_qkv",
            "attblock_out",
            "unet_temb",
            "silu_unet_temb",
            "unet_in",
            "silu_unet_out",
            "unet_out",
            "unet_codebook",
            ]
        self.quant_config["attpool_qkf"] = {"name": "integer", "weight_width": 8, "weight_frac_width": 4,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 4}
        self.quant_config["attpool_c"] = {"name": "integer", "weight_width": 8, "weight_frac_width": 4,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 4}
        self.quant_config["upsample"] = {"name": "integer", "weight_width": 8, "weight_frac_width": 4,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 4}
        self.quant_config["downsample"] = {"name": "integer", "weight_width": 8, "weight_frac_width": 4,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 4}
        self.quant_config["avg_pool2d"] = {"name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 4}
        self.quant_config["silu_res_in"] = {"name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 4}
        self.quant_config["resblock_in"] = {"name": "integer", "weight_width": 8, "weight_frac_width": 4,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 4}
        self.quant_config["silu_res_emb"] = {"name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 4}
        self.quant_config["resblock_emb"] = {"name": "integer", "weight_width": 8, "weight_frac_width": 4,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 4}
        self.quant_config["silu_res_out"] = {"name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 4}
        self.quant_config["resblock_out"] = {"name": "integer", "weight_width": 8, "weight_frac_width": 4,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 4}
        self.quant_config["resblock_skip"] = {"name": "integer", "weight_width": 8, "weight_frac_width": 4,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 4}
        self.quant_config["attblock_qkv"] = {"name": "integer", "weight_width": 8, "weight_frac_width": 4,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 4}
        self.quant_config["attblock_out"] = {"name": "integer", "weight_width": 8, "weight_frac_width": 4,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 4}
        self.quant_config["unet_temb"] = {"name": "integer", "weight_width": 8, "weight_frac_width": 4,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 4}
        self.quant_config["silu_unet_temb"] = {"name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 4}
        self.quant_config["unet_in"] = {"name": "integer", "weight_width": 8, "weight_frac_width": 4,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 4}
        self.quant_config["silu_unet_out"] = {"name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 4}
        self.quant_config["unet_out"] = {"name": "integer", "weight_width": 8, "weight_frac_width": 4,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 4}
        self.quant_config["unet_codebook"] = {"name": "integer", "weight_width": 8, "weight_frac_width": 4,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 4}
        print("Loaded quantization config.")
    
    def get(self, item: str) -> Dict:
        if item in self.config_keys:
            return self.quant_config[item]
        else:
            return None