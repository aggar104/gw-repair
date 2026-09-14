import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
import importlib

from torch.distributions import Uniform
from ml4gw.distributions import Cosine

from pathlib import Path
from typing import Callable, List, Optional

import ml4gw
from ml4gw.waveforms.generator import TimeDomainCBCWaveformGenerator
from ml4gw.waveforms import SineGaussian


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

class GlitchParameterSampler:
    def __init__(
        self, 
        prior_file: str,
        fmin: float|None = 0,
    ):
        self.level1 = ["N", "fc", "fw", "tw"]
        self.level2 = ["quality", "phase", "hrss", "eccentricity"]
        self.priors = get_prior(prior_file)
        self.f_min = fmin

    def __call__(self, X, level=1):

        if level == 1:
            
            B = X.shape[0]
            params = {
                k: v.sample((B,))
                for k, v in self.priors.items()
                if k in self.level1
            }
            return params

        else:
            N = int(X["N"])

            fmin = X["fc"] - X["fw"]/2 
            fmax = X["fc"] + X["fw"]/2
            fmin = fmin if fmin > self.f_min else self.f_min
            fmax = fmax if fmax < 500 else 500

            if fmin > fmax:
                fmax = fmin + fmax

            frequency_prior = torch.distributions.Uniform(fmin, fmax)

            params = {
                k: v.sample((N,))
                for k, v in self.priors.items()
                if k in self.level2
            }
            params["frequency"] = frequency_prior.sample((N,))

            return params
def time_shift(x, shifts):
    """
    c: [N, T]
    p: [N, T]
    returns: shifted tensor of same shape
    """
    N, T = x.shape
    x_shifted = torch.zeros_like(x)

    for n in range(N):
        s = shifts[n]
        if s > 0:
            x_shifted[n, s:] = x[n, :-s]
        elif s < 0:
            x_shifted[n, :s] = x[n, -s:]
        else:
            x_shifted[n] = x[n]

    return x_shifted

def time_shift(x, shifts):
    """
    c: [N, T]
    p: [N, T]
    returns: shifted tensor of same shape
    """
    N, T = x.shape
    x_shifted = torch.zeros_like(x)

    for n in range(N):
        s = shifts[n]
        if s > 0:
            x_shifted[n, s:] = x[n, :-s]
        elif s < 0:
            x_shifted[n, :s] = x[n, -s:]
        else:
            x_shifted[n] = x[n]

    return x_shifted

class ClusterGlitchSampler(torch.nn.Module):
    def __init__(
        self,
        prior: Path,
        sample_rate: float = 2048,
        duration: float = 5,
        fduration: float = 2,
        right_pad: float = 2,
        avoid_time: float = 0.2,
        include_time: float = 1,
        f_min: float = None,
        device: str = "cpu",
        
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.duration = duration
        self.fduration = fduration
        self.right_pad = right_pad
        self.avoid_time = avoid_time
        self.include_time = include_time
        self.device = device

        self.set_time()

        self.length = int(sample_rate * duration)   
        self.parameter_sampler = GlitchParameterSampler(
            prior, f_min
        )
        self.generator = SineGaussian(
            sample_rate = sample_rate,
            duration = duration,
        )
        

    def wavelets(self, X, l1_params):

        B = X.shape[0]
        glitch = torch.zeros_like(X)

        t0 = self.sample_time_shifts(B)*self.sample_rate

        tw = l1_params["tw"]
        jitters = ((10**tw)*self.sample_rate).int()

        

        for i in range(glitch.shape[0]):

            l1 = {k: v[i] for k, v in l1_params.items()}
            
            j = jitters[i].item()
            N = int(l1["N"].item())
            jitter = torch.randint(-j, j, size=(N, ), device=self.device)
            jitter = jitter + t0[i].int()

            params = self.parameter_sampler(l1, 2)
            c, _ = self.generator(**params)
            c = time_shift(c, jitter)
            c_ = c.sum(dim=0)
            glitch[i, 0] += c_

        
        return glitch
        
    @torch.no_grad()
    def forward(self, X):

        l1_params = self.parameter_sampler(X)

        return self.wavelets(X, l1_params)
        

            

