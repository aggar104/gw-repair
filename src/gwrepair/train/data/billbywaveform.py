import torch
import numpy as np
import bilby
from ml4gw import gw


class BilbyWaveform:
    def __init__(
        self, 
        duration: float = 5,
        sample_rate: float = 2048,
        fduration: float = 2,
        right_pad: float = 1,
        f_min: float = 20,
        ifos = ["H1", "L1"],
        snr_range = [20, 80]
    ):
        self.sample_rate = sample_rate
        self.duration = duration
        self.ifos = ifos
        self.snr_range = snr_range
        self.f_min = f_min



        self.priors = bilby.gw.prior.BBHPriorDict()
        waveform_arguments = dict(
            waveform_approximant="IMRPhenomPv2",
            reference_frequency=40,
            minimum_frequency=f_min,
        )
        self.waveform_generator = bilby.gw.WaveformGenerator(
            duration=duration,
            sampling_frequency=sample_rate,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=waveform_arguments,
        )
        self.ifos_ = bilby.gw.detector.InterferometerList(ifos)
        self.ifos_.set_strain_data_from_zero_noise(
            sampling_frequency=sample_rate,
            duration=duration,
            start_time=0,
        )
        self.geo_time = self.duration - right_pad - fduration//2

    def waveform_gen(self, params):

        params = {k: v[0] for k, v in params.items()}
        params["geocent_time"] = self.geo_time
        
        signals = []
        for i in range(len(self.ifos)):
            signal = self.ifos_[i].get_detector_response(
                waveform_polarizations=self.waveform_generator.frequency_domain_strain(params),
                parameters=params,
            )
            signal_td = np.fft.irfft(signal) * self.sample_rate
            signals.append(signal_td)
        signals = np.vstack(signals)
        return torch.from_numpy(signals).unsqueeze(0), params

    def compute_snr(self, S, psds, net=True, ifos=None):

        ifos = ifos or len(self.ifos)

        snrs = gw.compute_ifo_snr(
            S, psds, 
            sample_rate = self.sample_rate,
            highpass = self.f_min
        )

        if net:
            snrs = snrs.sum(axis=-1) ** 0.5
        else:
            snrs = torch.min(snrs[:, 0:ifos], dim=-1).values
        return snrs
        
    def rescale_snr(self, S, psds, ifos=None):

        snr = self.compute_snr(S, psds, net=False, ifos=ifos)

        if snr[0] > self.snr_range[0] and snr[0] < self.snr_range[1]:
            return S, 1
        
        target_snr = torch.empty_like(snr).uniform_(self.snr_range[0], self.snr_range[1])
        scale = target_snr / (snr + 1e-8)
        scale = scale[0].item()
        S = S * scale   

        return S, scale
        

    def forward(self, params=None, psds=None):

        if params is None:
            params = self.priors.sample(3)
            
        waveform, params = self.waveform_gen(params)
        scale = 1

        if psds is not None:
            waveform, scale = self.rescale_snr(waveform, psds, 1)

            params["luminosity_distance"] = params["luminosity_distance"] / scale
        return waveform, params