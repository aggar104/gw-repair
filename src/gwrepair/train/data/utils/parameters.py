import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
import importlib

from torch.distributions import Uniform
from ml4gw.distributions import Cosine
from typing import Optional

EXTRINSIC_PRIORS= {
    "dec": Cosine(),
    "psi": Uniform(0, 3.14),
    "phi": Uniform(0, 6.28),
}

def instantiate_from_string(class_path: str):
    """
    Convert 'package.module.ClassName' into an instance: ClassName(*args, **kwargs)
    """
    module_name, cls_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name)
    return cls

def dists(prior: dict):
    new = {}
    priors = prior['init_args']['priors']
    for k in priors.keys():
        p = priors[k]
        low, high = p['init_args']['low'], p['init_args']['high']
        dist = instantiate_from_string(p['class_path'])(low, high)
        new[k] = dist
    return new

def get_prior(file):
    cfg = OmegaConf.load(file)
    cfg = instantiate(cfg)
    prior_args = dists(cfg)
    return prior_args

class ParameterSampler:
    def __init__(
        self, 
        prior_file: str,
        extrinsic_priors: Optional[dict] = None,
        conversion_fn: Optional[dict] = None,
    ):
        self.priors = get_prior(prior_file)
        self.extrinsic_priors = extrinsic_priors or EXTRINSIC_PRIORS
        self.conversion_fn = conversion_fn or (lambda x: x)

    def __call__(self, X, ext=True):
        N = X.shape[0]
        device = X.device
        params = {
            k: v.sample((N,)).to(device)
            for k, v in self.priors.items()
        }
        params = self.conversion_fn(params)

        if not ext:
            return params
            
        extrinsic_params = {
            k: v.sample((N,)).to(device)
            for k, v in self.extrinsic_priors.items()
        }
        return extrinsic_params, params

class FixedParameterSampler:
    def __init__(
        self,
        param_dict: dict,
        conversion_fn: Optional[dict] = None,
        device: Optional[str] = 'cpu',
    ):
        self.param_dict = param_dict
        self.conversion_fn = conversion_fn or (lambda x: x)
        self.device = device

    def __call__(self, _):
        keys = self.param_dict.keys()
        device = self.device

        extrinsic_keys = EXTRINSIC_PRIORS.keys()

        params = {
            k: torch.tensor(v, device=device)
            for k, v in self.param_dict.items()
            if k not in extrinsic_keys
        }
        extrinsic_params = {
            k: torch.tensor(v, device=device)
            for k, v in self.param_dict.items()
            if k in extrinsic_keys
        }
        
        params = self.conversion_fn(params)
        return extrinsic_params, params
        
            


        