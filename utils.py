import torch
import importlib
from rich import print
from typing import Union
import numpy as np
import os

def print0(*args, **kwargs):
    print(*args, **kwargs)  # python -m rich.color

def tensor_to_uint8(tensor):
    tensor = torch.clamp(tensor, -1.0, 1.0)
    tensor = (tensor + 1.0) / 2.0
    tensor = (tensor.cpu().numpy() * 255).astype(np.uint8)
    return tensor

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def load_model_from_config(config, sd, gpu=True, eval_mode=True):
    model = instantiate_from_config(config)
    if sd is not None:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if len(missing) != 0:
            raise ValueError(f"Missing keys: {missing}")
    if gpu:
        model.cuda()
    if eval_mode:
        model.eval()
    return {"model": model}

def get_valid_dirs(dir1: str, dir2: Union[None, str] = None, dir3: Union[None, str] = None) -> Union[None, str]:
    if (dir1 is not None) and os.path.isdir(dir1): return dir1
    elif (dir2 is not None) and os.path.isdir(dir2): return dir2
    elif (dir3 is not None) and os.path.isdir(dir3): return dir3
    else: return None

def get_valid_paths(path1: str, path2: Union[None, str] = None, path3: Union[None, str] = None) -> Union[None, str]:
    if (path1 is not None) and os.path.isfile(path1): return path1
    elif (path2 is not None) and os.path.isfile(path2): return path2
    elif (path3 is not None) and os.path.isfile(path3): return path3
    else: return None

def load_model(config, ckpt, gpu, eval_mode):
    if str(ckpt).endswith(".bin"):
        weight = torch.load(ckpt)
    elif ckpt:
        weight = torch.load(ckpt, map_location="cpu")["state_dict"]
    model = load_model_from_config(config.model, weight, gpu=gpu, eval_mode=eval_mode)["model"]
    model.load_state_dict(weight, strict=False)
    model.to(torch.float16)
    
    return model

