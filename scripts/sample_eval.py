import argparse, os, sys, glob, datetime, yaml
import torch
import time
import numpy as np
import typing
import shutil
from tqdm import trange
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch_fidelity import calculate_metrics

from omegaconf import OmegaConf
from PIL import Image, ImageFile

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from ldm.data.imagenet import ImageNetTrain, ImageNetValidation
from taming.data.faceshq import CelebAHQTrain, CelebAHQValidation

from ldm.chop.passes.transforms.quantize.quantized_modules import *
from ldm.chop.passes.transforms.quantize.quantized_layer_profiler import *
from ldm.modules.diffusionmodules_quant.openaimodel import QKVAttentionLegacy
from ldm.modules.diffusionmodules_quant.attention import GEGLU, CrossAttention

import random
import json
ImageFile.LOAD_TRUNCATED_IMAGES = True

# torch.manual_seed(23)
# random.seed(3080)

rescale = lambda x: (x + 1.) / 2.

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

def compute_tensor_bits_block_fp(
    tensor_shape: np.ndarray, width: int, exponent_width: int, block_size: np.ndarray
):
    if tensor_shape.size > block_size.size:
        block_size = np.append([1] * (tensor_shape.size - block_size.size), block_size)
    elif tensor_shape.size < block_size.size:
        block_size = block_size[-tensor_shape.ndim :]

    num_blocks = np.prod(np.ceil(tensor_shape / block_size))
    return num_blocks * np.prod(block_size) * width + num_blocks * exponent_width

def forward_hook_fn(module, input, output, layer_name, layer_stats):
    # Profile the layer and update statistics
    if isinstance(module, LinearInteger) or isinstance(module, LinearBlockFP):
        profile_result = profile_linear_layer(module.config, module.in_features, module.out_features, module.bias is not None, input[0].shape[0])
        layer_stats[layer_name] = profile_result
    elif isinstance(module, Conv1dInteger) or isinstance(module, Conv1dBlockFP):
        profile_result = profile_conv1d_layer(module.config, module.in_channels, module.out_channels, module.kernel_size[0], module.stride[0], module.bias is not None, input[0].shape[0], input[0].shape[-1])
        layer_stats[layer_name] = profile_result
    elif isinstance(module, Conv2dInteger) or isinstance(module, Conv2dBlockFP):
        profile_result = profile_conv2d_layer(module.config, module.in_channels, module.out_channels, module.kernel_size[0], module.stride[0], module.bias is not None, input[0].shape[0], input[0].shape[-2], input[0].shape[-1])
        layer_stats[layer_name] = profile_result
    elif isinstance(module, QKVAttentionLegacy):
        profile_result = profile_matmul_layer(module.config_0, module.q.shape, module.k.shape)
        profile_result_2 = profile_matmul_layer(module.config_1, module.weight.shape, module.v.shape)
        update_profile(profile_result, profile_result_2)
        layer_stats[layer_name] = profile_result
    elif isinstance(module, SiLUInteger) or isinstance(module, SiLUBlockFP):
        profile_result = {"num_params": 0, "num_acts": 0, "param_bits": 0, "act_bits": 0, "flops": 0, "flops_bitwidth": 0}
        profile_result["num_acts"] = input[0].numel()
        if module.config.get("bypass", False):
            profile_result["act_bits"] = 32 * profile_result["num_acts"]
        elif module.config["name"] == "integer":
            profile_result["act_bits"] = module.config["data_in_width"] * profile_result["num_acts"]
        else:
            profile_result["act_bits"] = compute_tensor_bits_block_fp(np.array(input[0].size()), module.config["data_in_width"], 
                                                                      module.config["data_in_exponent_width"], 
                                                                      np.array(module.config["data_in_block_size"]))
        layer_stats[layer_name] = profile_result
    elif isinstance(module, GEGLU):
        profile_result = {"num_params": 0, "num_acts": 0, "param_bits": 0, "act_bits": 0, "flops": 0, "flops_bitwidth": 0}
        profile_result["num_acts"] = module.gate.numel()
        if module.gelu_config.get("bypass", False):
            profile_result["act_bits"] = 32 * profile_result["num_acts"]
        elif module.gelu_config["name"] == "integer":
            profile_result["act_bits"] = module.gelu_config["data_in_width"] * profile_result["num_acts"]
        else:
            profile_result["act_bits"] = compute_tensor_bits_block_fp(np.array(module.gate.size()), module.gelu_config["data_in_width"], 
                                                                      module.gelu_config["data_in_exponent_width"], 
                                                                      np.array(module.gelu_config["data_in_block_size"]))
        layer_stats[layer_name] = profile_result
    elif isinstance(module, CrossAttention):
        profile_result = profile_matmul_layer(module.config_0, module.q.shape, module.k.shape)
        profile_result_2 = profile_matmul_layer(module.config_1, module.attn.shape, module.v.shape)
        update_profile(profile_result, profile_result_2)
        layer_stats[layer_name] = profile_result


class GeneratedImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = os.listdir(root)
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.root, self.images[index])
        image = Image.open(image_path).convert("RGB")
        # image = np.array(image).astype(np.uint8)
        if self.transform is not None:
            image = self.transform(image)
        image = self.to_tensor(image)
        image = (image * 255).to(torch.uint8)  # Scale to [0, 255] and convert to uint8
        # print("Image data type:", image.dtype)
        #print("Minimum pixel value:", torch.min(image))
        #print("Maximum pixel value:", torch.max(image))
        return image

class ModifiedImageNetTrain(ImageNetTrain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.to_tensor = ToTensor()  # Modify the transform to convert to Tensor

    def __getitem__(self, index):
        img_dict = super().__getitem__(index)
        img = self.to_tensor(img_dict["image"])  # Convert the image to Tensor
        img = ((img + 1) * 127.5).to(torch.uint8)
        # label = img_dict["class_label"]
        # print("Image data type:", img.dtype)
        # print("Minimum pixel value:", torch.min(img))
        # print("Maximum pixel value:", torch.max(img))
        return img
    
class ModifiedImageNetValidation(ImageNetValidation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.to_tensor = ToTensor()  # Modify the transform to convert to Tensor

    def __getitem__(self, index):
        img_dict = super().__getitem__(index)
        img = self.to_tensor(img_dict["image"])  # Convert the image to Tensor
        img = ((img + 1) * 127.5).to(torch.uint8)
        # label = img_dict["class_label"]
        # print("Image data type:", img.dtype)
        # print("Minimum pixel value:", torch.min(img))
        # print("Maximum pixel value:", torch.max(img))
        return img
    
class ModifiedCelebAHQTrain(CelebAHQTrain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.to_tensor = ToTensor()  # Modify the transform to convert to Tensor

    def __getitem__(self, index):
        img_dict = super().__getitem__(index)
        img = self.to_tensor(img_dict["image"])  # Convert the image to Tensor
        img = ((img + 1) * 127.5).to(torch.uint8)
        # label = img_dict["class_label"]
        # print("Image data type:", img.dtype)
        # print("Minimum pixel value:", torch.min(img))
        # print("Maximum pixel value:", torch.max(img))
        return img
    
class ModifiedCelebAHQValidation(CelebAHQValidation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.to_tensor = ToTensor()  # Modify the transform to convert to Tensor

    def __getitem__(self, index):
        img_dict = super().__getitem__(index)
        img = self.to_tensor(img_dict["image"])  # Convert the image to Tensor
        img = ((img + 1) * 127.5).to(torch.uint8)
        # label = img_dict["class_label"]
        # print("Image data type:", img.dtype)
        # print("Minimum pixel value:", torch.min(img))
        # print("Maximum pixel value:", torch.max(img))
        return img


def custom_to_np(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample


def logs2pil(logs, keys=["sample"]):
    imgs = dict()
    for k in logs:
        try:
            if len(logs[k].shape) == 4:
                img = custom_to_pil(logs[k][0, ...])
            elif len(logs[k].shape) == 3:
                img = custom_to_pil(logs[k])
            else:
                print(f"Unknown format for key {k}. ")
                img = None
        except:
            img = None
        imgs[k] = img
    return imgs


@torch.no_grad()
def convsample(model, shape, return_intermediates=True,
               verbose=True,
               make_prog_row=False):


    if not make_prog_row:
        return model.p_sample_loop(None, shape,
                                   return_intermediates=return_intermediates, verbose=verbose)
    else:
        return model.progressive_denoising(
            None, shape, verbose=True
        )


@torch.no_grad()
def convsample_ddim(model, steps, shape, eta=1.0
                    ):
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False,)
    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(model, batch_size, vanilla=False, custom_steps=None, eta=1.0,):


    log = dict()

    shape = [batch_size,
             model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]

    with model.ema_scope("Plotting"):
        t0 = time.time()
        if vanilla:
            sample, progrow = convsample(model, shape,
                                         make_prog_row=True)
        else:
            sample, intermediates = convsample_ddim(model,  steps=custom_steps, shape=shape,
                                                    eta=eta)

        t1 = time.time()

    x_sample = model.decode_first_stage(sample)

    log["sample"] = x_sample
    log["time"] = t1 - t0
    log['throughput'] = sample.shape[0] / (t1 - t0)
    print(f'Throughput for this batch: {log["throughput"]}')
    return log

@torch.no_grad()
def make_classcond_convolutional_sample(model, batch_size, cond_input=None, vanilla=False, custom_steps=None, eta=1.0,):


    log = dict()

    shape = [batch_size,
             model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]

    with model.ema_scope("Plotting"):
        t0 = time.time()
        if vanilla:
            sample, progrow = convsample(model, shape,
                                         make_prog_row=True)
        else:
            #sample, intermediates = convsample_ddim(model, shape=shape,
            #                                        eta=eta)
            
            uc = model.get_learned_conditioning(
                    {model.cond_stage_key: torch.tensor(batch_size*[1000]).to(model.device)}
                    )
            xc = torch.tensor(cond_input)
            c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
            ddim = DDIMSampler(model)
            img_shape = shape[1:]
            sample, intermediates = ddim.sample(custom_steps, conditioning=c, batch_size=batch_size, shape=img_shape, 
                                                unconditional_guidance_scale=1.5, unconditional_conditioning=uc, eta=eta, verbose=False,)
        t1 = time.time()

    x_sample = model.decode_first_stage(sample)

    log["sample"] = x_sample
    log["time"] = t1 - t0
    log['throughput'] = sample.shape[0] / (t1 - t0)
    print(f'Throughput for this batch: {log["throughput"]}')
    return log

def run(model, logdir, batch_size=50, vanilla=False, custom_steps=None, eta=None, n_samples=50000, nplog=None):
    if vanilla:
        print(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
    else:
        print(f'Using DDIM sampling with {custom_steps} sampling steps and eta={eta}')


    tstart = time.time()
    n_saved = len(glob.glob(os.path.join(logdir,'*.png')))-1
    # path = logdir
    if model.cond_stage_model is None:
        all_images = []

        print(f"Running unconditional sampling for {n_samples} samples")
        for _ in trange(n_samples // batch_size, desc="Sampling Batches (unconditional)"):
            logs = make_convolutional_sample(model, batch_size=batch_size,
                                             vanilla=vanilla, custom_steps=custom_steps,
                                             eta=eta)
            n_saved = save_logs(logs, logdir, n_saved=n_saved, key="sample")
            all_images.extend([custom_to_np(logs["sample"])])
            if n_saved >= n_samples:
                print(f'Finish after generating {n_saved} samples')
                break
        all_img = np.concatenate(all_images, axis=0)
        all_img = all_img[:n_samples]
        shape_str = "x".join([str(x) for x in all_img.shape])
        nppath = os.path.join(nplog, f"{shape_str}-samples.npz")
        np.savez(nppath, all_img)

    elif model.cond_stage_key == 'class_label':
        # Your dictionary with keys and their occurrence counts (weights)
        # Read the distribution from the text file and extract a dictionary
        output_file = 'distribution.txt'
        with open(output_file, 'r') as f:   
            distribution_dict = json.load(f)
        
        # Extract keys and weights (occurrence counts) from the dictionary
        keys = [int(i) for i in distribution_dict.keys()]
        weights = list(distribution_dict.values())

        
        all_images = []

        print(f"Running class conditional sampling for {n_samples} samples")
        for _ in trange(n_samples // batch_size, desc="Sampling Batches (class conditional)"):
            random_class_sample = random.choices(keys, weights=weights, k=batch_size)

            #print(random_class_sample)
            logs = make_classcond_convolutional_sample(model, batch_size=batch_size, cond_input = random_class_sample,
                                             vanilla=vanilla, custom_steps=custom_steps,
                                             eta=eta)
            n_saved = save_logs(logs, logdir, n_saved=n_saved, key="sample")
            all_images.extend([custom_to_np(logs["sample"])])
            if n_saved >= n_samples:
                print(f'Finish after generating {n_saved} samples')
                break
        all_img = np.concatenate(all_images, axis=0)
        all_img = all_img[:n_samples]
        shape_str = "x".join([str(x) for x in all_img.shape])
        nppath = os.path.join(nplog, f"{shape_str}-samples.npz")
        np.savez(nppath, all_img)
    
    else:
        raise NotImplementedError('Currently only sampling for unconditional models supported.')
        

    print(f"sampling of {n_saved} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")


def save_logs(logs, path, n_saved=0, key="sample", np_path=None):
    for k in logs:
        if k == key:
            batch = logs[key]
            if np_path is None:
                for x in batch:
                    img = custom_to_pil(x)
                    imgpath = os.path.join(path, f"{key}_{n_saved:06}.png")
                    img.save(imgpath)
                    n_saved += 1
            else:
                npbatch = custom_to_np(batch)
                shape_str = "x".join([str(x) for x in npbatch.shape])
                nppath = os.path.join(np_path, f"{n_saved}-{shape_str}-samples.npz")
                np.savez(nppath, npbatch)
                n_saved += npbatch.shape[0]
    return n_saved

"""
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        nargs="?",
        help="number of samples to draw",
        default=50000
    )
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=1.0
    )
    parser.add_argument(
        "-v",
        "--vanilla_sample",
        default=False,
        action='store_true',
        help="vanilla sampling (default option is DDIM sampling)?",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default="none"
    )
    parser.add_argument(
        "-c",
        "--custom_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fastdpm sampling",
        default=50
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        help="the bs",
        default=10
    )
    return parser
"""

def load_model_from_config(config, sd, trial=None):
    model = instantiate_from_config(config, trial)
    model.load_state_dict(sd,strict=False)
    model.cuda()
    model.eval()
    return model


def load_model(config, ckpt, gpu, eval_mode, trial=None):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            global_step = pl_sd["global_step"]
        else:
            global_step = None
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"], trial=trial)

    return model, global_step


def sampling_main(
        trial,
        quant_mode,
        resume: str,
        opt_logdir: str,
        eta: float = 1.0,
        vanilla_sample: bool = False,
        n_samples: int = 50000,
        custom_steps: int = 50,
        batch_size: int = 10,
):
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sys.path.append(os.getcwd())
    # command = " ".join(sys.argv)

    # parser = get_parser()
    # opt, unknown = parser.parse_known_args()
    ckpt = None

    if not os.path.exists(resume):
        raise ValueError("Cannot find {}".format(resume))
    if os.path.isfile(resume):
        # paths = resume.split("/")
        try:
            logdir = '/'.join(resume.split('/')[:-1])
            # idx = len(paths)-paths[::-1].index("logs")+1
            print(f'Logdir is {logdir}')
        except ValueError:
            paths = resume.split("/")
            idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
        ckpt = resume
    else:
        assert os.path.isdir(resume), f"{resume} is not a directory"
        logdir = resume.rstrip("/")
        ckpt = os.path.join(logdir, "model.ckpt")

    if quant_mode == "integer":
        base_configs = sorted(glob.glob(os.path.join(logdir, "config.yaml")))
    elif quant_mode == "block_fp":
        base_configs = sorted(glob.glob(os.path.join(logdir, "config_bfp.yaml")))
    else:
        raise NotImplementedError('Currently only integer and block_fp supported.')

    opt_base = base_configs

    configs = [OmegaConf.load(cfg) for cfg in opt_base]
    # cli = OmegaConf.from_dotlist(unknown)
    # config = OmegaConf.merge(*configs, cli)
    config = OmegaConf.merge(*configs)

    gpu = True
    # gpu = False
    eval_mode = True

    if opt_logdir != "none":
        locallog = logdir.split(os.sep)[-1]
        if locallog == "": locallog = logdir.split(os.sep)[-2]
        print(f"Switching logdir from '{logdir}' to '{os.path.join(opt_logdir, locallog)}'")
        logdir = os.path.join(opt_logdir, locallog)

    print(config)

    model, global_step = load_model(config, ckpt, gpu, eval_mode, trial=trial)
    if not global_step:
        global_step = 0
    print(f"global step: {global_step}")
    print(75 * "=")

    # Calculate FLOPs and total bits
    layer_stats = {}
    hooks = []

    def attach_hooks_recursive(module, cur_name, layer_stats):
        # Attach hooks to the current module
        hook = module.register_forward_hook(lambda module, input, output, name=cur_name: forward_hook_fn(module, input, output, name, layer_stats))
        hooks.append(hook)
        
        # Recursively attach hooks to child modules
        for chd_name, child in module.named_children():
            nested_name = cur_name + '.' + chd_name
            attach_hooks_recursive(child, nested_name, layer_stats)


    """
    for name, layer in model.model.diffusion_model.named_children():
        layer_stats[name] = {"num_params": 0, "num_acts": 0, "param_bits": 0, "act_bits": 0, "flops": 0, "flops_bitwidth": 0}
        hook = layer.register_forward_hook(lambda module, input, output, name=name: forward_hook_fn(module, input, output, name, layer_stats))
        hooks.append(hook)
    """
    attach_hooks_recursive(model.model.diffusion_model, "diffusion_model", layer_stats)

    # Perform a forward pass to collect statistics with hooks
    print("calculating FLOPs and bits")
    print("logging to:")
    logdir_test = os.path.join(logdir, "flops", f"{global_step:08}", now)
    imglogdir = os.path.join(logdir_test, "img")
    numpylogdir = os.path.join(logdir_test, "numpy")
    os.makedirs(imglogdir)
    os.makedirs(numpylogdir)
    print(logdir_test)
    print(75 * "=")

    run(model, imglogdir, eta=eta,
        vanilla=vanilla_sample,  n_samples=batch_size, custom_steps=custom_steps,
        batch_size=batch_size, nplog=numpylogdir)

    # Calculate average bitwidths and total FLOPs
    profile_overall = {"num_params": 0, "num_acts": 0, "param_bits": 0, "act_bits": 0, "flops": 0, "flops_bitwidth": 0}
    for layer_name in layer_stats.keys():
        update_profile(profile_overall, layer_stats[layer_name])

    total_num_params = profile_overall["num_params"]
    total_num_acts = profile_overall["num_acts"]
    total_param_bits = profile_overall["param_bits"]
    total_activation_bits = profile_overall["act_bits"]
    total_flops = profile_overall["flops"]
    total_flops_bitwidth = profile_overall["flops_bitwidth"]

    total_bits = total_param_bits + total_activation_bits
    
    # original models uses fp32
    compare = 32 
    compare_total_bitwidth = compare * (total_num_params + total_num_acts)

    mem_density = compare_total_bitwidth / total_bits

    # Print or store the results
    average_bitwidth = total_bits / (total_num_params + total_num_acts)
    print(f"Total params: {total_num_params}")
    print(f"Average Bitwidth: {average_bitwidth}")
    print(f"Total FLOPs: {total_flops}")
    print(f"Total FLOPs bitwidth: {total_flops_bitwidth}")
    print(f"Memory density: {mem_density}")
    avg_flops_bitwidth = total_flops_bitwidth / total_flops
    
    file_name = os.path.join(logdir_test, "layer_stats.csv")
    with open(file_name, "w") as fp:
        fp.write("Name, num_params, num_acts, param_bits, act_bits, flops, flop_bitwidth\n")
        for name in layer_stats.keys():
            fp.write(f"{name},{layer_stats[name]['num_params']}, {layer_stats[name]['num_acts']}, {layer_stats[name]['param_bits']}, {layer_stats[name]['act_bits']}, {layer_stats[name]['flops']}, {layer_stats[name]['flops_bitwidth']}\n")
    
    # shutil.rmtree(imglogdir)
    shutil.rmtree(numpylogdir)

    # Detach the hooks from the layers
    for hook in hooks:
        hook.remove()

    print("logging to:")
    logdir = os.path.join(logdir, "samples", f"{global_step:08}", now)
    imglogdir = os.path.join(logdir, "img")
    numpylogdir = os.path.join(logdir, "numpy")

    os.makedirs(imglogdir)
    os.makedirs(numpylogdir)
    print(logdir)
    print(75 * "=")

    # write config out
    # sampling_file = os.path.join(logdir, "sampling_config.yaml")
    # sampling_conf = vars(opt)

    # with open(sampling_file, 'w') as f:
    #     yaml.dump(sampling_conf, f, default_flow_style=False)
    # print(sampling_conf)


    run(model, imglogdir, eta=eta,
        vanilla=vanilla_sample,  n_samples=n_samples, custom_steps=custom_steps,
        batch_size=batch_size, nplog=numpylogdir)
    shutil.rmtree(numpylogdir)

    print("done.")
    return imglogdir, mem_density, avg_flops_bitwidth



def evaluation(
        imagedir: str,
        dataset: str,
        dataset_part: str,
):
    assert dataset in ["imagenet", "celebahq"]
    assert dataset_part in ["train", "validation"]
    if dataset == "imagenet":
        config = {"size": 256}
        if dataset_part == "validation":
            real_dataset = ModifiedImageNetValidation(config)
        elif dataset_part == "train": 
            real_dataset = ModifiedImageNetTrain(config)
    elif dataset == "celebahq":
        if dataset_part == "train":
            real_dataset = ModifiedCelebAHQTrain(size=256)
        elif dataset_part == "validation":
            real_dataset = ModifiedCelebAHQValidation(size=256)
    else:
        raise NotImplementedError

    generated_dataset = GeneratedImageDataset(root=imagedir)
    metrics = calculate_metrics(input1=generated_dataset, input2=real_dataset, cuda=True, fid=True, isc=True, prc=True)
    print(metrics)
    shutil.rmtree(imagedir)
    return metrics