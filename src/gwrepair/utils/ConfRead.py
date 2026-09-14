import os
from omegaconf import OmegaConf
from pathlib import Path

from gwrepair.train.data.dataset import RepairDataset


def ReadConfigs(HOME_DIR="/Users/shreyaggarwal/Desktop/work/git", DATA_DIR=None, CONFIG_DIR=None, base_name="base.yaml"):

    os.environ["HOME_DIR"] = HOME_DIR

    PIPELINE_DIR = HOME_DIR + "/gw-repair"

    
    os.environ["DATA_DIR"] =  DATA_DIR or f"{HOME_DIR}/data/background"
    os.environ["CONFIG_DIR"] = CONFIG_DIR or PIPELINE_DIR + "/src/gwrepair/train/config"

    config = dict(OmegaConf.load(
        Path(os.environ["CONFIG_DIR"]) / base_name
    ))
    
    data_cfg = dict(config["data"])
    waveform_args = config["waveform"]
    glitch_args = config["glitch"]
    
    data_cfg["ifos"] = list(data_cfg["ifos"])
    data_cfg["waveform_args"] = waveform_args
    data_cfg["glitch_args"] = glitch_args

    return RepairDataset(**data_cfg)
