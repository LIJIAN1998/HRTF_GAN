import os
import pickle

import scipy
import torch
import torch.nn.functional as F
import numpy as np
from model.ae import AutoEncoder
from model.util import load_hrtf
from config import Config

from pathlib import Path
import importlib
from hrtfdata.transforms.hrirs import SphericalHarmonicsTransform
import matplotlib.pyplot as plt

def plot_lsd():
    config = Config("ari-upscale-4", using_hpc=True)
    data_dir = config.raw_hrtf_dir / config.dataset
    imp = importlib.import_module('hrtfdata.full')
    load_function = getattr(imp, config.dataset)
    domain = config.domain
    ds = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate,
                                                         'side': 'left', 'domain': domain}}, subject_ids='first')
    num_row_angles = len(ds.row_angles)
    num_col_angles = len(ds.column_angles)
    num_radii = len(ds.radii)
    max_order = config.max_order
    upscale_factor = config.upscale_factor
    degree = int(np.sqrt(num_row_angles*num_col_angles*num_radii/upscale_factor) - 1)

    ngpu = config.ngpu
    valid_dir = config.valid_path
    valid_gt_dir = config.valid_gt_path

    nbins = config.nbins_hrtf
    if config.merge_flag:
        nbins = config.nbins_hrtf * 2

    device = torch.device(config.device_name if (
        torch.cuda.is_available() and ngpu > 0) else "cpu")
    model = AutoEncoder(nbins=nbins, in_order=degree, latent_dim=config.latent_dim, base_channels=256, num_features=512, out_oder=max_order)
    print("Build VAE model successfully.")
    model.load_state_dict(torch.load(f"{config.model_path}/Gen.pt", map_location=torch.device('cpu')))
    print(f"Load VAE model weights `{os.path.abspath(config.model_path)}` successfully.")

    _, test_prefetcher = load_hrtf(config)
    test_prefetcher.reset()
    lr_coefficient = batch_data["lr_coefficient"].to(device=device, memory_format=torch.contiguous_format,
                                                         non_blocking=True, dtype=torch.float)