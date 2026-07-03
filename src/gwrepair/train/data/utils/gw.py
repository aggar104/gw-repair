from typing import Optional, Tuple
import torch
from ml4gw import gw

Tensor = torch.Tensor


class WaveformProjector(torch.nn.Module):
    def __init__(
        self,
        ifos: list[str],
        sample_rate: float,
    ) -> None:
        super().__init__()
        tensors, vertices = gw.get_ifo_geometry(*ifos)
        self.register_buffer("tensors", tensors)
        self.register_buffer("vertices", vertices)

        self.sample_rate = sample_rate

    def forward(
        self,
        dec: torch.Tensor,
        psi: torch.Tensor,
        phi: torch.Tensor,
        **polarizations: torch.Tensor,
    ) -> torch.Tensor:
        responses = gw.compute_observed_strain(
            dec,
            psi,
            phi,
            detector_tensors=self.tensors,
            detector_vertices=self.vertices,
            sample_rate=self.sample_rate,
            **polarizations,
        )
        return responses

        
