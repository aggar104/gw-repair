import torch
from pathlib import Path
from typing import Callable, List, Optional

import ml4gw
from ml4gw.waveforms.generator import TimeDomainCBCWaveformGenerator
from ml4gw.waveforms import SineGaussian

from gwrepair.train.data.utils import ParameterSampler, FixedParameterSampler
from gwrepair.train.data.utils import WaveformProjector
from gwrepair.train.data.utils.conversions import precessing_to_lalsimulation_parameters

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

def trigger_shift(x: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    """
    x:      [B, C, T]
    shifts: [B]
        positive = shift right
        negative = shift left
    """
    B, C, T = x.shape
    shifts = shifts.to(device=x.device, dtype=torch.long)

    out = torch.empty_like(x)

    for i in range(B):
        out[i] = torch.roll(x[i], shifts=int(shifts[i].item()), dims=-1)

    return out

class WaveformSampler(torch.nn.Module):
    def __init__(
        self,
        prior: Path,
        sample_rate: float = 2048,
        duration: float = 30,
        ifos: List[str] = ["H1", "L1", "V1"],
        f_min: float = 20,
        f_ref: float = 25,
        right_pad: float = 2,
        jitter: float = 0,
        approximant: Optional[Callable] = None,
        extrinsic_priors: Optional[dict] = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.duration = duration
        self.ifos = ifos
        self.f_min = f_min
        self.f_ref = f_ref
        self.right_pad = right_pad
        self.jitter_index = int(jitter * sample_rate)
        self.approximant = approximant or ml4gw.waveforms.IMRPhenomPv2()
        self.extrinsic_priors = extrinsic_priors

        self.generator = TimeDomainCBCWaveformGenerator(
            approximant = self.approximant,
            sample_rate=self.sample_rate,
            duration=self.duration,
            f_min=self.f_min,
            f_ref=self.f_ref,
            right_pad=self.right_pad,
        )
        self.parameter_sampler = ParameterSampler(
            prior,
            conversion_fn = precessing_to_lalsimulation_parameters,
        )
        
        self.projector = WaveformProjector(self.ifos, self.sample_rate)
        
    @torch.no_grad()
    def forward(self, X):
        extrinsic_params, params = self.parameter_sampler(X)
        cross, plus = self.generator(**params)
        params.update(extrinsic_params)
        waveforms = self.projector(params['dec'], params['psi'], params['phi'], cross=cross, plus=plus)

        ## jitters ##
        if self.jitter_index != 0:
            jitters = torch.randint(-self.jitter_index, self.jitter_index+1, size=(X.shape[0],), device=X.device)
            waveforms = trigger_shift(waveforms, jitters)
        else:
            jitters = torch.zeros(X.shape[0], device=X.device)
        
        return waveforms, params, jitters/self.sample_rate

class GlitchSampler(torch.nn.Module):
    def __init__(
        self,
        prior: Path,
        sample_rate: float = 2048,
        duration: float = 30,
        glitch_ifos: List[str] = ["H1"],
        extrinsic_priors: Optional[dict] = None,
        max_wavelets: int = 50,
        wavelet_jitter: float = 0,
        jitter: float = 0,
        right_gpad: float = 1,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.duration = duration
        self.glitch_ifos = glitch_ifos
        self.extrinsic_priors = extrinsic_priors 
        self.max_wavelets = max_wavelets
        self.wavelet_jitter = int(wavelet_jitter * sample_rate)
        self.jitter = int(jitter * sample_rate)
        self.right_gpad = (right_gpad * sample_rate)
        self.length = int(sample_rate * duration)   

        self.parameter_sampler = ParameterSampler(
            prior,
        )
        self.generator = SineGaussian(
            sample_rate = sample_rate,
            duration = duration,
        )
        self.projector = WaveformProjector(self.glitch_ifos, sample_rate)

    def wavelets(self, X):

        cross = torch.zeros_like(X[:, 0, :])
        plus = torch.zeros_like(X[:, 0, :])

        for i in range(self.max_wavelets):
            jitters = torch.randint(-self.wavelet_jitter, self.wavelet_jitter+1, size=(X.shape[0],), device=X.device)
            extrinsic_params, params = self.parameter_sampler(X)
            c, p = self.generator(**params)
            c = time_shift(c, jitters)
            p = time_shift(p, jitters)
            cross += c
            plus += c

        return extrinsic_params, cross, plus
        
    @torch.no_grad()
    def forward(self, X):

        t0 = torch.randint(-self.jitter, self.jitter+1, size=(X.shape[0],), device=X.device)
        jitters = t0 + int(self.length / 2 - self.right_gpad)
        params, cross, plus = self.wavelets(X)
        cross = time_shift(cross, jitters)
        plus = time_shift(plus, jitters)

        glitch_ = self.projector(params['dec'], params['psi'], params['phi'], cross=cross, plus=plus).squeeze(1)
        glitch = torch.zeros_like(X)
        glitch[:, 0, :] = glitch_
        return glitch      #([B, num_params], [B, C, T]

