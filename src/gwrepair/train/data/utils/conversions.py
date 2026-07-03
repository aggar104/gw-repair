import torch
from ml4gw.waveforms.conversion import (
    bilby_spins_to_lalsim,
    chirp_mass_and_mass_ratio_to_components,
)
from typing import Optional


def precessing_to_lalsimulation_parameters(
    parameters: dict[str, torch.Tensor],
    f_ref: Optional[float] = None,
) -> dict[str, torch.Tensor]:
    """
    Convert precessing spin parameters to lalsimulation parameters
    """
    f_ref = f_ref or 40.0
    mass_1, mass_2 = chirp_mass_and_mass_ratio_to_components(
        parameters["chirp_mass"], parameters["mass_ratio"]
    )

    parameters["mass_1"] = mass_1
    parameters["mass_2"] = mass_2

    # TODO: hard coding f_ref = 40 here b/c not sure best way to link this
    # to the f_ref specified in the config file
    incl, s1x, s1y, s1z, s2x, s2y, s2z = bilby_spins_to_lalsim(
        parameters["inclination"],
        parameters["phi_jl"],
        parameters["tilt_1"],
        parameters["tilt_2"],
        parameters["phi_12"],
        parameters["a_1"],
        parameters["a_2"],
        parameters["mass_1"],
        parameters["mass_2"],
        f_ref,
        torch.zeros(len(mass_1), device=mass_1.device),
    )

    parameters["s1x"] = s1x
    parameters["s1y"] = s1y
    parameters["s1z"] = s1z
    parameters["s2x"] = s2x
    parameters["s2y"] = s2y
    parameters["s2z"] = s2z
    parameters["inclination"] = incl
    return parameters
