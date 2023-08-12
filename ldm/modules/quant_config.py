from typing import Dict

def add_int_recursive(item, add_value):
    if isinstance(item, int):
        return item + add_value
    elif isinstance(item, list):
        return [add_int_recursive(element, add_value) for element in item]
    elif isinstance(item, dict):
        return {key: add_int_recursive(value, add_value) if not isinstance(value, bool) else value for key, value in item.items()}
    else:
        return item

class CelebA256UNetQuantConfig():
    def __init__(self, bitwidth=8) -> None:
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
            "matmul_0",
            "matmul_1",
            "attblock_out",
            "temb_1",
            "silu_temb",
            "temb_2",
            "unet_in",
            "silu_unet_out",
            "unet_out",
            "unet_codebook",
            ]
        
        self.quant_config["attpool_qkf"] = {"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 5,
                                            "data_in_width": 8, "data_in_frac_width": 5, 
                                            "bias_width": 8, "bias_frac_width": 5}
        self.quant_config["attpool_c"] = {"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 5,
                                            "data_in_width": 8, "data_in_frac_width": 5, 
                                            "bias_width": 8, "bias_frac_width": 5}
        self.quant_config["unet_codebook"] = {"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 5,
                                            "data_in_width": 8, "data_in_frac_width": 5, 
                                            "bias_width": 8, "bias_frac_width": 5}
        self.quant_config["avg_pool2d"] = {"bypass": True, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 5}
        
        """
        self.quant_config["upsample"] = {"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 5,
                                            "data_in_width": 8, "data_in_frac_width": 5, 
                                            "bias_width": 8, "bias_frac_width": 5}
        self.quant_config["downsample"] = {"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 5,
                                            "data_in_width": 8, "data_in_frac_width": 5, 
                                            "bias_width": 8, "bias_frac_width": 5}
        self.quant_config["avg_pool2d"] = {"bypass": True, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 5}
        self.quant_config["silu_res_in"] = {"bypass": True, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 5}
        self.quant_config["resblock_in"] = {"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 5,
                                            "data_in_width": 8, "data_in_frac_width": 5, 
                                            "bias_width": 8, "bias_frac_width": 5}
        self.quant_config["silu_res_emb"] = {"bypass": True, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 5}
        self.quant_config["resblock_emb"] = {"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 5,
                                            "data_in_width": 8, "data_in_frac_width": 5, 
                                            "bias_width": 8, "bias_frac_width": 5}
        self.quant_config["silu_res_out"] = {"bypass": True, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 5}
        self.quant_config["resblock_out"] = {"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 5,
                                            "data_in_width": 8, "data_in_frac_width": 5, 
                                            "bias_width": 8, "bias_frac_width": 5}
        self.quant_config["resblock_skip"] = {"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 5,
                                            "data_in_width": 8, "data_in_frac_width": 5, 
                                            "bias_width": 8, "bias_frac_width": 5}
        self.quant_config["attblock_qkv"] = {"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 5,
                                            "data_in_width": 8, "data_in_frac_width": 5, 
                                            "bias_width": 8, "bias_frac_width": 5}
        self.quant_config["attblock_out"] = {"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 5,
                                            "data_in_width": 8, "data_in_frac_width": 5, 
                                            "bias_width": 8, "bias_frac_width": 5}
        self.quant_config["unet_temb"] = {"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 5,
                                            "data_in_width": 8, "data_in_frac_width": 5, 
                                            "bias_width": 8, "bias_frac_width": 5}
        self.quant_config["silu_unet_temb"] = {"bypass": True, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 5}
        self.quant_config["unet_in"] = {"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 5,
                                            "data_in_width": 8, "data_in_frac_width": 5, 
                                            "bias_width": 8, "bias_frac_width": 5}
        self.quant_config["silu_unet_out"] = {"bypass": True, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 5}
        self.quant_config["unet_out"] = {"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 5,
                                            "data_in_width": 8, "data_in_frac_width": 5, 
                                            "bias_width": 8, "bias_frac_width": 5}

        """


        self.quant_config["temb_1"] = {"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 9,
                                            "data_in_width": 8, "data_in_frac_width": 7, 
                                            "bias_width": 8, "bias_frac_width": 11}
        self.quant_config["silu_temb"] = {"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 4}
        self.quant_config["temb_2"] = {"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 9}
        self.quant_config["unet_in"] = {"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 9}
        # Begin ResBlock[0]
        self.quant_config["silu_res_in"] = [{"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 4}]
        self.quant_config["resblock_in"] = [{"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 7,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 9}]
        self.quant_config["silu_res_emb"] = [{"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 2}]
        self.quant_config["resblock_emb"] = [{"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 7,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 9}]
        self.quant_config["silu_res_out"] = [{"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 2}]
        self.quant_config["resblock_out"] = [{"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 10}]
        self.quant_config["resblock_skip"] = [None]

        # Begin Resblock[1]
        self.quant_config["silu_res_in"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_in"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 7,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["silu_res_emb"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 2})
        self.quant_config["resblock_emb"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["silu_res_out"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 4})
        self.quant_config["resblock_out"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 7,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["resblock_skip"].append(None)

        # Begin DownSample[0]
        self.quant_config["downsample"] = [{"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 10}]
        
        # Begin ResBlock[2]
        self.quant_config["silu_res_in"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_in"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 7,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 8})
        self.quant_config["silu_res_emb"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 2})
        self.quant_config["resblock_emb"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 6,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 8})
        self.quant_config["silu_res_out"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_out"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["resblock_skip"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 9,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 9})
        
        # Begin AttentionBlock[0]
        self.quant_config["attblock_qkv"] = [{"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 7}]
        self.quant_config["matmul_0"] = [{"bypass": False, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3,
                                            "weight_width": 8, "weight_frac_width": 3}]
        self.quant_config["matmul_1"] = [{"bypass": False, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4,
                                            "weight_width": 8, "weight_frac_width": 7}]
        self.quant_config["attblock_out"] = [{"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 9}]
        
        # Begin ResBlock[3]
        self.quant_config["silu_res_in"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_in"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 6,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["silu_res_emb"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 2})
        self.quant_config["resblock_emb"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 7,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["silu_res_out"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_out"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["resblock_skip"].append(None)

        # Begin AttentionBlock[1]
        self.quant_config["attblock_qkv"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 7})
        self.quant_config["matmul_0"].append({"bypass": False, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3,
                                            "weight_width": 8, "weight_frac_width": 3})
        self.quant_config["matmul_1"].append({"bypass": False, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4,
                                            "weight_width": 8, "weight_frac_width": 7})
        self.quant_config["attblock_out"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 9,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 8})
        
        # Begin DownSample[1]
        self.quant_config["downsample"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 9,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 9})
        
        # Begin ResBlock[4]
        self.quant_config["silu_res_in"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 2})
        self.quant_config["resblock_in"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 2, 
                                            "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["silu_res_emb"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 2})
        self.quant_config["resblock_emb"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 7,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["silu_res_out"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_out"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["resblock_skip"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 10,
                                            "data_in_width": 8, "data_in_frac_width": 2, 
                                            "bias_width": 8, "bias_frac_width": 9})
        
        # Begin AttentionBlock[2]
        self.quant_config["attblock_qkv"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 8})
        self.quant_config["matmul_0"].append({"bypass": False, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4,
                                            "weight_width": 8, "weight_frac_width": 4})
        self.quant_config["matmul_1"].append({"bypass": False, "name": "integer", "data_in_width": 8, "data_in_frac_width": 5,
                                            "weight_width": 8, "weight_frac_width": 7})
        self.quant_config["attblock_out"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 9,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 10})
        
        # Begin ResBlock[5]
        self.quant_config["silu_res_in"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_in"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["silu_res_emb"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 2})
        self.quant_config["resblock_emb"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["silu_res_out"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_out"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 9,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["resblock_skip"].append(None)
        
        # Begin AttentionBlock[3]
        self.quant_config["attblock_qkv"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 8})
        self.quant_config["matmul_0"].append({"bypass": False, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4,
                                            "weight_width": 8, "weight_frac_width": 3})
        self.quant_config["matmul_1"].append({"bypass": False, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4,
                                            "weight_width": 8, "weight_frac_width": 7})
        self.quant_config["attblock_out"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 9,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 10})
        
        # Begin DownSample[2]
        self.quant_config["downsample"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 9,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 9})
        
        # Begin ResBlock[6]
        self.quant_config["silu_res_in"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 4})
        self.quant_config["resblock_in"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_emb"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 2})
        self.quant_config["resblock_emb"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 9,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_out"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_out"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["resblock_skip"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 10,
                                            "data_in_width": 8, "data_in_frac_width": 2, 
                                            "bias_width": 8, "bias_frac_width": 9})
        
        # Begin AttentionBlock[4]
        self.quant_config["attblock_qkv"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 9,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["matmul_0"].append({"bypass": False, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4,
                                            "weight_width": 8, "weight_frac_width": 4})
        self.quant_config["matmul_1"].append({"bypass": False, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4,
                                            "weight_width": 8, "weight_frac_width": 7})
        self.quant_config["attblock_out"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 9,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 9})
        
        # Begin ResBlock[7]
        self.quant_config["silu_res_in"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_in"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 8})
        self.quant_config["silu_res_emb"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 2})
        self.quant_config["resblock_emb"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 8})
        self.quant_config["silu_res_out"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_out"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 11})
        self.quant_config["resblock_skip"].append(None)
        
        # Begin AttentionBlock[5]
        self.quant_config["attblock_qkv"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 9,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["matmul_0"].append({"bypass": False, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4,
                                            "weight_width": 8, "weight_frac_width": 4})
        self.quant_config["matmul_1"].append({"bypass": False, "name": "integer", "data_in_width": 8, "data_in_frac_width": 5,
                                            "weight_width": 8, "weight_frac_width": 7})
        self.quant_config["attblock_out"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 9,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 11})
        
        # Begin MiddleBlock
        # Begin ResBlock[8]
        self.quant_config["silu_res_in"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_in"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["silu_res_emb"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 2})
        self.quant_config["resblock_emb"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 8})
        self.quant_config["silu_res_out"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_out"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 9,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["resblock_skip"].append(None)

        # Begin AttentionBlock[6]
        self.quant_config["attblock_qkv"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 9,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["matmul_0"].append({"bypass": False, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4,
                                            "weight_width": 8, "weight_frac_width": 4})
        self.quant_config["matmul_1"].append({"bypass": False, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4,
                                            "weight_width": 8, "weight_frac_width": 7})
        self.quant_config["attblock_out"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 9,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 10})
        
        # Begin ResBlock[9]
        self.quant_config["silu_res_in"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_in"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 8})
        self.quant_config["silu_res_emb"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 2})
        self.quant_config["resblock_emb"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 9,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 8})
        self.quant_config["silu_res_out"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_out"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["resblock_skip"].append(None)

        # Begin OutBlock
        # Begin ResBlock[10]
        self.quant_config["silu_res_in"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_in"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 9,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_emb"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 2})
        self.quant_config["resblock_emb"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 9,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["silu_res_out"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_out"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["resblock_skip"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 9,
                                            "data_in_width": 8, "data_in_frac_width": 2, 
                                            "bias_width": 8, "bias_frac_width": 10})
        
        # Begin AttentionBlock[7]
        self.quant_config["attblock_qkv"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 9,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["matmul_0"].append({"bypass": False, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4,
                                            "weight_width": 8, "weight_frac_width": 4})
        self.quant_config["matmul_1"].append({"bypass": False, "name": "integer", "data_in_width": 8, "data_in_frac_width": 5,
                                            "weight_width": 8, "weight_frac_width": 7})
        self.quant_config["attblock_out"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 9,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 10})
        
        # Begin ResBlock[11]
        self.quant_config["silu_res_in"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 1})
        self.quant_config["resblock_in"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 1, 
                                            "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["silu_res_emb"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 2})
        self.quant_config["resblock_emb"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["silu_res_out"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_out"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["resblock_skip"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 9,
                                            "data_in_width": 8, "data_in_frac_width": 0, 
                                            "bias_width": 8, "bias_frac_width": 10})
        
        # Begin AttentionBlock[8]
        self.quant_config["attblock_qkv"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["matmul_0"].append({"bypass": False, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4,
                                            "weight_width": 8, "weight_frac_width": 4})
        self.quant_config["matmul_1"].append({"bypass": False, "name": "integer", "data_in_width": 8, "data_in_frac_width": 5,
                                            "weight_width": 8, "weight_frac_width": 7})
        self.quant_config["attblock_out"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 9,
                                            "data_in_width": 8, "data_in_frac_width": 5, 
                                            "bias_width": 8, "bias_frac_width": 10})
        
        # Begin ResBlock[12]
        self.quant_config["silu_res_in"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_in"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 7,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 7})
        self.quant_config["silu_res_emb"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 2})
        self.quant_config["resblock_emb"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 7,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 7})
        self.quant_config["silu_res_out"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_out"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["resblock_skip"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 9,
                                            "data_in_width": 8, "data_in_frac_width": 2, 
                                            "bias_width": 8, "bias_frac_width": 10})
        
        # Begin AttentionBlock[8]
        self.quant_config["attblock_qkv"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["matmul_0"].append({"bypass": False, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4,
                                            "weight_width": 8, "weight_frac_width": 4})
        self.quant_config["matmul_1"].append({"bypass": False, "name": "integer", "data_in_width": 8, "data_in_frac_width": 5,
                                            "weight_width": 8, "weight_frac_width": 7})
        self.quant_config["attblock_out"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 9,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 10})
        
        # Begin Upsample[0]
        self.quant_config["upsample"] = [{"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 2, 
                                            "bias_width": 8, "bias_frac_width": 8}]
        
        # Begin ResBlock[13]
        self.quant_config["silu_res_in"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_in"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_emb"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 2})
        self.quant_config["resblock_emb"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 9,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_out"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_out"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["resblock_skip"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 9,
                                            "data_in_width": 8, "data_in_frac_width": 1, 
                                            "bias_width": 8, "bias_frac_width": 9})
        
        # Begin AttentionBlock[9]
        self.quant_config["attblock_qkv"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 8})
        self.quant_config["matmul_0"].append({"bypass": False, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3,
                                            "weight_width": 8, "weight_frac_width": 3})
        self.quant_config["matmul_1"].append({"bypass": False, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4,
                                            "weight_width": 8, "weight_frac_width": 7})
        self.quant_config["attblock_out"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 10})
        
        # Begin ResBlock[14]
        self.quant_config["silu_res_in"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_in"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 7,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["silu_res_emb"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 2})
        self.quant_config["resblock_emb"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 9,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["silu_res_out"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_out"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["resblock_skip"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 10,
                                            "data_in_width": 8, "data_in_frac_width": 2, 
                                            "bias_width": 8, "bias_frac_width": 9})
        
        # Begin AttentionBlock[10]
        self.quant_config["attblock_qkv"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 8})
        self.quant_config["matmul_0"].append({"bypass": False, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3,
                                            "weight_width": 8, "weight_frac_width": 3})
        self.quant_config["matmul_1"].append({"bypass": False, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4,
                                            "weight_width": 8, "weight_frac_width": 7})
        self.quant_config["attblock_out"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 9})

        # Begin ResBlock[15]
        self.quant_config["silu_res_in"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_in"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["silu_res_emb"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 2})
        self.quant_config["resblock_emb"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["silu_res_out"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_out"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["resblock_skip"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 10,
                                            "data_in_width": 8, "data_in_frac_width": 2, 
                                            "bias_width": 8, "bias_frac_width": 10})
        
        # Begin AttentionBlock[11]
        self.quant_config["attblock_qkv"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 8})
        self.quant_config["matmul_0"].append({"bypass": False, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4,
                                            "weight_width": 8, "weight_frac_width": 3})
        self.quant_config["matmul_1"].append({"bypass": False, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4,
                                            "weight_width": 8, "weight_frac_width": 7})
        self.quant_config["attblock_out"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 10})
        
        # Begin Upsample[1]
        self.quant_config["upsample"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 9,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 8})
        
        # Begin ResBlock[16]
        self.quant_config["silu_res_in"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_in"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 7,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["silu_res_emb"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 2})
        self.quant_config["resblock_emb"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["silu_res_out"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_out"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 7,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["resblock_skip"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 7,
                                            "data_in_width": 8, "data_in_frac_width": 1, 
                                            "bias_width": 8, "bias_frac_width": 9})
        
        # Begin AttentionBlock[12]
        self.quant_config["attblock_qkv"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 8})
        self.quant_config["matmul_0"].append({"bypass": False, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3,
                                            "weight_width": 8, "weight_frac_width": 3})
        self.quant_config["matmul_1"].append({"bypass": False, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3,
                                            "weight_width": 8, "weight_frac_width": 7})
        self.quant_config["attblock_out"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 10})
        
        # Begin ResBlock[17]
        self.quant_config["silu_res_in"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_in"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 7,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["silu_res_emb"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 2})
        self.quant_config["resblock_emb"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 8})
        self.quant_config["silu_res_out"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_out"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 7,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["resblock_skip"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 2, 
                                            "bias_width": 8, "bias_frac_width": 10})
        
        # Begin AttentionBlock[13]
        self.quant_config["attblock_qkv"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 8})
        self.quant_config["matmul_0"].append({"bypass": False, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3,
                                            "weight_width": 8, "weight_frac_width": 3})
        self.quant_config["matmul_1"].append({"bypass": False, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3,
                                            "weight_width": 8, "weight_frac_width": 7})
        self.quant_config["attblock_out"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 10})
        
        # Begin ResBlock[18]
        self.quant_config["silu_res_in"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_in"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 6,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["silu_res_emb"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 2})
        self.quant_config["resblock_emb"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["silu_res_out"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_out"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 7,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["resblock_skip"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 9,
                                            "data_in_width": 8, "data_in_frac_width": 2, 
                                            "bias_width": 8, "bias_frac_width": 9})
        
        # Begin AttentionBlock[14]
        self.quant_config["attblock_qkv"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 8})
        self.quant_config["matmul_0"].append({"bypass": False, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3,
                                            "weight_width": 8, "weight_frac_width": 3})
        self.quant_config["matmul_1"].append({"bypass": False, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3,
                                            "weight_width": 8, "weight_frac_width": 7})
        self.quant_config["attblock_out"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 9})
        
        # Begin Upsample[2]
        self.quant_config["upsample"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 2, 
                                            "bias_width": 8, "bias_frac_width": 8})
        
        # Begin ResBlock[19]
        self.quant_config["silu_res_in"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_in"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 7,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["silu_res_emb"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 2})
        self.quant_config["resblock_emb"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 9,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["silu_res_out"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_out"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 7,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["resblock_skip"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 7,
                                            "data_in_width": 8, "data_in_frac_width": 0, 
                                            "bias_width": 8, "bias_frac_width": 9})
        
        # Begin ResBlock[20]
        self.quant_config["silu_res_in"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_in"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 6,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_emb"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 2})
        self.quant_config["resblock_emb"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_out"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_out"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 6,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["resblock_skip"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 1, 
                                            "bias_width": 8, "bias_frac_width": 10})
        
        # Begin ResBlock[21]
        self.quant_config["silu_res_in"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_in"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 6,
                                            "data_in_width": 8, "data_in_frac_width": 3,
                                            "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["silu_res_emb"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 2})
        self.quant_config["resblock_emb"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 7,
                                            "data_in_width": 8, "data_in_frac_width": 3, 
                                            "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["silu_res_out"].append({"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_out"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 6,
                                            "data_in_width": 8, "data_in_frac_width": 4, 
                                            "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["resblock_skip"].append({"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 8,
                                            "data_in_width": 8, "data_in_frac_width": 2, 
                                            "bias_width": 8, "bias_frac_width": 10})
        
        # Begin unet.out
        self.quant_config["silu_unet_out"] = {"bypass": False, "name": "integer",
                                           "data_in_width": 8, "data_in_frac_width": 4}
        self.quant_config["unet_out"] = {"bypass": False, "name": "integer", "weight_width": 8, "weight_frac_width": 9,
                                            "data_in_width": 8, "data_in_frac_width": 5, 
                                            "bias_width": 8, "bias_frac_width": 13}
        
        self.quant_config = add_int_recursive(self.quant_config, bitwidth - 8)
        # print(self.quant_config)

        print("Loaded quantization config.")
    
    def get(self, item: str) -> Dict:
        if item in self.config_keys:
            return self.quant_config[item]
        else:
            return None


class ImageNet256UNetQuantConfig():
    def __init__(self, bitwidth=8):
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
            "proj_in",
            "matmul_0_0",
            "matmul_1_0",
            "to_q_0",
            "to_k_0",
            "to_v_0",
            "to_out_0",
            "proj_out_0",
            "ff_gelu",
            "ff_proj",
            "ff_out",
            "matmul_0_1",
            "matmul_1_1",
            "to_q_1",
            "to_k_1",
            "to_v_1",
            "to_out_1",
            "proj_out",
            "temb_1",
            "silu_temb",
            "temb_2",
            "unet_in",
            "silu_unet_out",
            "unet_out",
            "unet_codebook",
            ]


        self.quant_config["temb_1"] = {"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 7, "bias_width": 8, "bias_frac_width": 10}
        self.quant_config["silu_temb"] = {"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 5}
        self.quant_config["temb_2"] = {"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 5, "bias_width": 8, "bias_frac_width": 11}
        self.quant_config["unet_in"] = {"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 4, "bias_width": 8, "bias_frac_width": 9}
        self.quant_config["silu_res_in"] = [{"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4}]
        self.quant_config["resblock_in"] = [{"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 4, "bias_width": 8, "bias_frac_width": 10}]
        self.quant_config["silu_res_emb"] = [{"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3}]
        self.quant_config["resblock_emb"] = [{"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10}]
        self.quant_config["silu_res_out"] = [{"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 2}]
        self.quant_config["resblock_out"] = [{"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10}]
        self.quant_config["resblock_skip"] = [None]
        self.quant_config["silu_res_in"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 2})
        self.quant_config["resblock_in"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 4, "bias_width": 8, "bias_frac_width": 11})
        self.quant_config["silu_res_emb"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_emb"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 11})
        self.quant_config["silu_res_out"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 4, "bias_width": 8, "bias_frac_width": 12})
        self.quant_config["resblock_skip"].append(None)
        self.quant_config["downsample"] = [{"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 4, "bias_width": 8, "bias_frac_width": 11}]
        self.quant_config["silu_res_in"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_in"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 11})
        self.quant_config["silu_res_emb"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_emb"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 11})
        self.quant_config["silu_res_out"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 4, "bias_width": 8, "bias_frac_width": 11})
        self.quant_config["resblock_skip"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["proj_in"] = [{"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 4, "bias_width": 8, "bias_frac_width": 9}]
        self.quant_config["matmul_0_0"] = [{"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4, "weight_width": 8, "weight_frac_width": 4}]
        self.quant_config["matmul_1_0"] = [{"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 7, "weight_width": 8, "weight_frac_width": 4}]
        self.quant_config["to_q_0"] = [{"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3}]
        self.quant_config["to_k_0"] = [{"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3}]
        self.quant_config["to_v_0"] = [{"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3}]
        self.quant_config["to_out_0"] = [{"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 5, "bias_width": 8, "bias_frac_width": 9}]
        self.quant_config["ff_gelu"] = [{"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3}]
        self.quant_config["ff_proj"] = [{"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 9}]
        self.quant_config["ff_out"] = [{"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 2, "bias_width": 8, "bias_frac_width": 9}]
        self.quant_config["matmul_0_1"] = [{"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 5, "weight_width": 8, "weight_frac_width": 6}]
        self.quant_config["matmul_1_1"] = [{"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 7, "weight_width": 8, "weight_frac_width": 6}]
        self.quant_config["to_q_1"] = [{"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 11, "data_in_width": 8, "data_in_frac_width": 3}]
        self.quant_config["to_k_1"] = [{"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 11, "data_in_width": 8, "data_in_frac_width": 5}]
        self.quant_config["to_v_1"] = [{"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 5}]
        self.quant_config["to_out_1"] = [{"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 6, "bias_width": 8, "bias_frac_width": 8}]
        self.quant_config["proj_out"] = [{"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 4, "bias_width": 8, "bias_frac_width": 11}]
        self.quant_config["silu_res_in"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_in"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 11})
        self.quant_config["silu_res_emb"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_emb"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_out"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4})
        self.quant_config["resblock_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 4, "bias_width": 8, "bias_frac_width": 11})
        self.quant_config["resblock_skip"].append(None)
        self.quant_config["proj_in"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 4, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["matmul_0_0"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4, "weight_width": 8, "weight_frac_width": 4})
        self.quant_config["matmul_1_0"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 7, "weight_width": 8, "weight_frac_width": 5})
        self.quant_config["to_q_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["to_k_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["to_v_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["to_out_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 6, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["ff_gelu"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4})
        self.quant_config["ff_proj"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["ff_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 2, "bias_width": 8, "bias_frac_width": 11})
        self.quant_config["matmul_0_1"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 5, "weight_width": 8, "weight_frac_width": 6})
        self.quant_config["matmul_1_1"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 7, "weight_width": 8, "weight_frac_width": 8})
        self.quant_config["to_q_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 11, "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["to_k_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 11, "data_in_width": 8, "data_in_frac_width": 5})
        self.quant_config["to_v_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 11, "data_in_width": 8, "data_in_frac_width": 5})
        self.quant_config["to_out_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 11, "data_in_width": 8, "data_in_frac_width": 8, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["proj_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["downsample"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 11})
        self.quant_config["silu_res_in"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_in"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 7, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_emb"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_emb"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_out"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["resblock_skip"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 2, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["proj_in"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["matmul_0_0"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4, "weight_width": 8, "weight_frac_width": 4})
        self.quant_config["matmul_1_0"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 7, "weight_width": 8, "weight_frac_width": 4})
        self.quant_config["to_q_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 4})
        self.quant_config["to_k_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 4})
        self.quant_config["to_v_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 4})
        self.quant_config["to_out_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 5, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["ff_gelu"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["ff_proj"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 4, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["ff_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 1, "bias_width": 8, "bias_frac_width": 11})
        self.quant_config["matmul_0_1"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 5, "weight_width": 8, "weight_frac_width": 6})
        self.quant_config["matmul_1_1"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 7, "weight_width": 8, "weight_frac_width": 6})
        self.quant_config["to_q_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 11, "data_in_width": 8, "data_in_frac_width": 4})
        self.quant_config["to_k_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 11, "data_in_width": 8, "data_in_frac_width": 5})
        self.quant_config["to_v_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 5})
        self.quant_config["to_out_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 6, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["proj_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_in"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_in"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 7, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_emb"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_emb"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_out"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 11})
        self.quant_config["resblock_skip"].append(None)
        self.quant_config["proj_in"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 4, "bias_width": 8, "bias_frac_width": 8})
        self.quant_config["matmul_0_0"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4, "weight_width": 8, "weight_frac_width": 4})
        self.quant_config["matmul_1_0"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 7, "weight_width": 8, "weight_frac_width": 4})
        self.quant_config["to_q_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 4})
        self.quant_config["to_k_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 4})
        self.quant_config["to_v_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 4})
        self.quant_config["to_out_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 6, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["ff_gelu"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["ff_proj"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 4, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["ff_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 1, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["matmul_0_1"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 5, "weight_width": 8, "weight_frac_width": 6})
        self.quant_config["matmul_1_1"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 7, "weight_width": 8, "weight_frac_width": 6})
        self.quant_config["to_q_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 11, "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["to_k_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 11, "data_in_width": 8, "data_in_frac_width": 5})
        self.quant_config["to_v_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 5})
        self.quant_config["to_out_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 6, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["proj_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["downsample"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_in"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_in"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_emb"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_emb"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_out"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["resblock_skip"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 1, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["proj_in"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 4, "bias_width": 8, "bias_frac_width": 8})
        self.quant_config["matmul_0_0"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4, "weight_width": 8, "weight_frac_width": 4})
        self.quant_config["matmul_1_0"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 7, "weight_width": 8, "weight_frac_width": 5})
        self.quant_config["to_q_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["to_k_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["to_v_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["to_out_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 5, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["ff_gelu"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["ff_proj"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["ff_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 1, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["matmul_0_1"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 5, "weight_width": 8, "weight_frac_width": 6})
        self.quant_config["matmul_1_1"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 7, "weight_width": 8, "weight_frac_width": 5})
        self.quant_config["to_q_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 12, "data_in_width": 8, "data_in_frac_width": 2})
        self.quant_config["to_k_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 11, "data_in_width": 8, "data_in_frac_width": 5})
        self.quant_config["to_v_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 5})
        self.quant_config["to_out_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 5, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["proj_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 2, "bias_width": 8, "bias_frac_width": 11})
        self.quant_config["silu_res_in"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_in"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_emb"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_emb"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_out"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["resblock_skip"].append(None)
        self.quant_config["proj_in"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 4, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["matmul_0_0"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4, "weight_width": 8, "weight_frac_width": 4})
        self.quant_config["matmul_1_0"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 7, "weight_width": 8, "weight_frac_width": 4})
        self.quant_config["to_q_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 4})
        self.quant_config["to_k_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 4})
        self.quant_config["to_v_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 4})
        self.quant_config["to_out_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 5, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["ff_gelu"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["ff_proj"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 4, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["ff_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 1, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["matmul_0_1"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 5, "weight_width": 8, "weight_frac_width": 6})
        self.quant_config["matmul_1_1"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 7, "weight_width": 8, "weight_frac_width": 6})
        self.quant_config["to_q_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 12, "data_in_width": 8, "data_in_frac_width": 4})
        self.quant_config["to_k_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 11, "data_in_width": 8, "data_in_frac_width": 5})
        self.quant_config["to_v_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 5})
        self.quant_config["to_out_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 6, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["proj_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_in"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_in"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 11})
        self.quant_config["silu_res_emb"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_emb"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_out"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["resblock_skip"].append(None)
        self.quant_config["proj_in"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 4, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["matmul_0_0"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4, "weight_width": 8, "weight_frac_width": 4})
        self.quant_config["matmul_1_0"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 7, "weight_width": 8, "weight_frac_width": 5})
        self.quant_config["to_q_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 5})
        self.quant_config["to_k_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 5})
        self.quant_config["to_v_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 5})
        self.quant_config["to_out_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 5, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["ff_gelu"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["ff_proj"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 4, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["ff_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": -1, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["matmul_0_1"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 5, "weight_width": 8, "weight_frac_width": 6})
        self.quant_config["matmul_1_1"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 7, "weight_width": 8, "weight_frac_width": 6})
        self.quant_config["to_q_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 12, "data_in_width": 8, "data_in_frac_width": 4})
        self.quant_config["to_k_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 11, "data_in_width": 8, "data_in_frac_width": 5})
        self.quant_config["to_v_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 5})
        self.quant_config["to_out_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 6, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["proj_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_in"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_in"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_emb"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_emb"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_out"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["resblock_skip"].append(None)
        self.quant_config["silu_res_in"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_in"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 11})
        self.quant_config["silu_res_emb"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_emb"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 11})
        self.quant_config["silu_res_out"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["resblock_skip"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 0, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["proj_in"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 8})
        self.quant_config["matmul_0_0"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4, "weight_width": 8, "weight_frac_width": 3})
        self.quant_config["matmul_1_0"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 7, "weight_width": 8, "weight_frac_width": 4})
        self.quant_config["to_q_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 5})
        self.quant_config["to_k_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 5})
        self.quant_config["to_v_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 5})
        self.quant_config["to_out_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 5, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["ff_gelu"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["ff_proj"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 4, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["ff_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 0, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["matmul_0_1"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 5, "weight_width": 8, "weight_frac_width": 6})
        self.quant_config["matmul_1_1"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 7, "weight_width": 8, "weight_frac_width": 6})
        self.quant_config["to_q_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 12, "data_in_width": 8, "data_in_frac_width": 4})
        self.quant_config["to_k_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 11, "data_in_width": 8, "data_in_frac_width": 5})
        self.quant_config["to_v_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 5})
        self.quant_config["to_out_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 6, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["proj_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_in"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_in"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 11})
        self.quant_config["silu_res_emb"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_emb"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_out"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 11})
        self.quant_config["resblock_skip"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 1, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["proj_in"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["matmul_0_0"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4, "weight_width": 8, "weight_frac_width": 4})
        self.quant_config["matmul_1_0"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 7, "weight_width": 8, "weight_frac_width": 4})
        self.quant_config["to_q_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 5})
        self.quant_config["to_k_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 5})
        self.quant_config["to_v_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 5})
        self.quant_config["to_out_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 5, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["ff_gelu"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4})
        self.quant_config["ff_proj"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 4, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["ff_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 1, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["matmul_0_1"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 5, "weight_width": 8, "weight_frac_width": 6})
        self.quant_config["matmul_1_1"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 7, "weight_width": 8, "weight_frac_width": 5})
        self.quant_config["to_q_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 12, "data_in_width": 8, "data_in_frac_width": 4})
        self.quant_config["to_k_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 11, "data_in_width": 8, "data_in_frac_width": 5})
        self.quant_config["to_v_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 5})
        self.quant_config["to_out_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 5, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["proj_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_in"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_in"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 7, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_emb"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_emb"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_out"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 11})
        self.quant_config["resblock_skip"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 1, "bias_width": 8, "bias_frac_width": 11})
        self.quant_config["proj_in"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["matmul_0_0"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4, "weight_width": 8, "weight_frac_width": 4})
        self.quant_config["matmul_1_0"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 7, "weight_width": 8, "weight_frac_width": 4})
        self.quant_config["to_q_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 5})
        self.quant_config["to_k_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 5})
        self.quant_config["to_v_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 5})
        self.quant_config["to_out_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 5, "bias_width": 8, "bias_frac_width": 8})
        self.quant_config["ff_gelu"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4})
        self.quant_config["ff_proj"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 4, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["ff_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 1, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["matmul_0_1"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 5, "weight_width": 8, "weight_frac_width": 6})
        self.quant_config["matmul_1_1"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 7, "weight_width": 8, "weight_frac_width": 6})
        self.quant_config["to_q_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 12, "data_in_width": 8, "data_in_frac_width": 4})
        self.quant_config["to_k_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 11, "data_in_width": 8, "data_in_frac_width": 5})
        self.quant_config["to_v_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 5})
        self.quant_config["to_out_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 6, "bias_width": 8, "bias_frac_width": 8})
        self.quant_config["proj_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["upsample"] = [{"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 2, "bias_width": 8, "bias_frac_width": 9}]
        self.quant_config["silu_res_in"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_in"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 6, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_emb"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_emb"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_out"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["resblock_skip"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 0, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["proj_in"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 8})
        self.quant_config["matmul_0_0"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4, "weight_width": 8, "weight_frac_width": 4})
        self.quant_config["matmul_1_0"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 7, "weight_width": 8, "weight_frac_width": 4})
        self.quant_config["to_q_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["to_k_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["to_v_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["to_out_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 4, "bias_width": 8, "bias_frac_width": 7})
        self.quant_config["ff_gelu"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["ff_proj"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 4, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["ff_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 1, "bias_width": 8, "bias_frac_width": 8})
        self.quant_config["matmul_0_1"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 5, "weight_width": 8, "weight_frac_width": 6})
        self.quant_config["matmul_1_1"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 7, "weight_width": 8, "weight_frac_width": 6})
        self.quant_config["to_q_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 11, "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["to_k_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 11, "data_in_width": 8, "data_in_frac_width": 5})
        self.quant_config["to_v_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 5})
        self.quant_config["to_out_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 6, "bias_width": 8, "bias_frac_width": 7})
        self.quant_config["proj_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 2, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_in"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_in"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 7, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_emb"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_emb"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_out"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["resblock_skip"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 1, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["proj_in"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 4, "bias_width": 8, "bias_frac_width": 8})
        self.quant_config["matmul_0_0"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4, "weight_width": 8, "weight_frac_width": 4})
        self.quant_config["matmul_1_0"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 7, "weight_width": 8, "weight_frac_width": 4})
        self.quant_config["to_q_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["to_k_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["to_v_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["to_out_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 5, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["ff_gelu"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["ff_proj"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 8})
        self.quant_config["ff_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 1, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["matmul_0_1"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 5, "weight_width": 8, "weight_frac_width": 6})
        self.quant_config["matmul_1_1"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 7, "weight_width": 8, "weight_frac_width": 6})
        self.quant_config["to_q_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 11, "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["to_k_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 11, "data_in_width": 8, "data_in_frac_width": 5})
        self.quant_config["to_v_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 5})
        self.quant_config["to_out_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 6, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["proj_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 2, "bias_width": 8, "bias_frac_width": 11})
        self.quant_config["silu_res_in"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_in"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 6, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_emb"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_emb"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_out"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 11})
        self.quant_config["resblock_skip"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 2, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["proj_in"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 4, "bias_width": 8, "bias_frac_width": 7})
        self.quant_config["matmul_0_0"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4, "weight_width": 8, "weight_frac_width": 4})
        self.quant_config["matmul_1_0"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 7, "weight_width": 8, "weight_frac_width": 4})
        self.quant_config["to_q_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["to_k_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["to_v_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["to_out_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 5, "bias_width": 8, "bias_frac_width": 7})
        self.quant_config["ff_gelu"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4})
        self.quant_config["ff_proj"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 4, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["ff_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 1, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["matmul_0_1"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 5, "weight_width": 8, "weight_frac_width": 6})
        self.quant_config["matmul_1_1"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 7, "weight_width": 8, "weight_frac_width": 6})
        self.quant_config["to_q_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 11, "data_in_width": 8, "data_in_frac_width": 2})
        self.quant_config["to_k_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 11, "data_in_width": 8, "data_in_frac_width": 5})
        self.quant_config["to_v_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 5})
        self.quant_config["to_out_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 6, "bias_width": 8, "bias_frac_width": 7})
        self.quant_config["proj_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 2, "bias_width": 8, "bias_frac_width": 11})
        self.quant_config["upsample"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["silu_res_in"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_in"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 7, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_emb"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_emb"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_out"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["resblock_skip"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 1, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["proj_in"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["matmul_0_0"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4, "weight_width": 8, "weight_frac_width": 4})
        self.quant_config["matmul_1_0"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 7, "weight_width": 8, "weight_frac_width": 4})
        self.quant_config["to_q_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["to_k_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["to_v_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["to_out_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 5, "bias_width": 8, "bias_frac_width": 8})
        self.quant_config["ff_gelu"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4})
        self.quant_config["ff_proj"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 4, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["ff_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 2, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["matmul_0_1"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 5, "weight_width": 8, "weight_frac_width": 6})
        self.quant_config["matmul_1_1"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 7, "weight_width": 8, "weight_frac_width": 7})
        self.quant_config["to_q_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 11, "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["to_k_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 11, "data_in_width": 8, "data_in_frac_width": 5})
        self.quant_config["to_v_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 11, "data_in_width": 8, "data_in_frac_width": 5})
        self.quant_config["to_out_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 7, "bias_width": 8, "bias_frac_width": 8})
        self.quant_config["proj_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 11})
        self.quant_config["silu_res_in"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4})
        self.quant_config["resblock_in"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 4, "bias_width": 8, "bias_frac_width": 11})
        self.quant_config["silu_res_emb"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_emb"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_out"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 11})
        self.quant_config["resblock_skip"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 2, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["proj_in"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 4, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["matmul_0_0"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4, "weight_width": 8, "weight_frac_width": 4})
        self.quant_config["matmul_1_0"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 7, "weight_width": 8, "weight_frac_width": 4})
        self.quant_config["to_q_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 4})
        self.quant_config["to_k_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 4})
        self.quant_config["to_v_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 4})
        self.quant_config["to_out_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 5, "bias_width": 8, "bias_frac_width": 8})
        self.quant_config["ff_gelu"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4})
        self.quant_config["ff_proj"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["ff_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 2, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["matmul_0_1"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 5, "weight_width": 8, "weight_frac_width": 6})
        self.quant_config["matmul_1_1"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 7, "weight_width": 8, "weight_frac_width": 7})
        self.quant_config["to_q_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 11, "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["to_k_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 11, "data_in_width": 8, "data_in_frac_width": 5})
        self.quant_config["to_v_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 11, "data_in_width": 8, "data_in_frac_width": 5})
        self.quant_config["to_out_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 11, "data_in_width": 8, "data_in_frac_width": 7, "bias_width": 8, "bias_frac_width": 8})
        self.quant_config["proj_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 11})
        self.quant_config["silu_res_in"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4})
        self.quant_config["resblock_in"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 4, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_emb"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_emb"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 7, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_out"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 4, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["resblock_skip"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["proj_in"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 4, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["matmul_0_0"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4, "weight_width": 8, "weight_frac_width": 4})
        self.quant_config["matmul_1_0"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 7, "weight_width": 8, "weight_frac_width": 4})
        self.quant_config["to_q_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["to_k_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["to_v_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["to_out_0"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 5, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["ff_gelu"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["ff_proj"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["ff_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 2, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["matmul_0_1"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 5, "weight_width": 8, "weight_frac_width": 6})
        self.quant_config["matmul_1_1"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 7, "weight_width": 8, "weight_frac_width": 7})
        self.quant_config["to_q_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 11, "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["to_k_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 11, "data_in_width": 8, "data_in_frac_width": 5})
        self.quant_config["to_v_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 10, "data_in_width": 8, "data_in_frac_width": 5})
        self.quant_config["to_out_1"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 11, "data_in_width": 8, "data_in_frac_width": 7, "bias_width": 8, "bias_frac_width": 9})
        self.quant_config["proj_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 11})
        self.quant_config["upsample"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_in"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_in"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 7, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 11})
        self.quant_config["silu_res_emb"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_emb"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_out"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4})
        self.quant_config["resblock_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 4, "bias_width": 8, "bias_frac_width": 11})
        self.quant_config["resblock_skip"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 2, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_in"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_in"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 4, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_emb"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_emb"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_out"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4})
        self.quant_config["resblock_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 7, "data_in_width": 8, "data_in_frac_width": 4, "bias_width": 8, "bias_frac_width": 11})
        self.quant_config["resblock_skip"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_in"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_in"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 7, "data_in_width": 8, "data_in_frac_width": 4, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_emb"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_emb"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 3, "bias_width": 8, "bias_frac_width": 10})
        self.quant_config["silu_res_out"].append({"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 3})
        self.quant_config["resblock_out"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 8, "data_in_width": 8, "data_in_frac_width": 4, "bias_width": 8, "bias_frac_width": 12})
        self.quant_config["resblock_skip"].append({"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 4, "bias_width": 8, "bias_frac_width": 11})
        self.quant_config["silu_unet_out"] = {"bypass": True, "name": "integer", "data_in_width": 8, "data_in_frac_width": 4}
        self.quant_config["unet_out"] = {"bypass": True, "name": "integer", "weight_width": 8, "weight_frac_width": 9, "data_in_width": 8, "data_in_frac_width": 4, "bias_width": 8, "bias_frac_width": 14}

        self.quant_config = add_int_recursive(self.quant_config, bitwidth - 8)
        # print(self.quant_config)

        print("Loaded quantization config.")
    
    def get(self, item: str) -> Dict:
        if item in self.config_keys:
            return self.quant_config[item]
        else:
            return None