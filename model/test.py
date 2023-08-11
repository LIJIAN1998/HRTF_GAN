import os
import pickle

import scipy
import torch
import torch.nn.functional as F
import numpy as np

from model.model import VAE
import shutil
from pathlib import Path

import importlib

from hrtfdata.transforms.hrirs import SphericalHarmonicsTransform

from plot import plot_hrtf
import matplotlib.pyplot as plt

def test(config, val_prefetcher):
    # source: https://github.com/Lornatang/SRGAN-PyTorch/blob/main/test.py
    # Initialize super-resolution model

    # load the dataset to get the row, column angles info
    data_dir = config.raw_hrtf_dir / config.dataset
    imp = importlib.import_module('hrtfdata.full')
    load_function = getattr(imp, config.dataset)
    ds = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate,
                                                         'side': 'left', 'domain': 'magnitude'}}, subject_ids='first')
    num_row_angles = len(ds.row_angles)
    num_col_angles = len(ds.column_angles)
    num_radii = len(ds.radii)
    degree = int(np.sqrt(num_row_angles*num_col_angles*num_radii/config.upscale_factor) - 1)

    ngpu = config.ngpu
    valid_dir = config.valid_path
    valid_gt_dir = config.valid_gt_path

    nbins = config.nbins_hrtf
    if config.merge_flag:
        nbins = config.nbins_hrtf * 2

    device = torch.device(config.device_name if (
            torch.cuda.is_available() and ngpu > 0) else "cpu")
    model = VAE(nbins=nbins, max_degree=degree, latent_dim=config.latent_dim).to(device)
    print("Build VAE model successfully.")

    # Load vae model weights (always uses the CPU due to HPC having long wait times)
    model.load_state_dict(torch.load(f"{config.model_path}/vae.pt", map_location=torch.device('cpu')))
    print(f"Load VAE model weights `{os.path.abspath(config.model_path)}` successfully.")

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_param_mb = param_size / 1024 ** 2
    size_buffer_mb = buffer_size / 1024 ** 2
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print('param size: {:.3f}MB'.format(size_param_mb))
    print('buffer size: {:.3f}MB'.format(size_buffer_mb))
    print('model size: {:.3f}MB'.format(size_all_mb))

    # get list of positive frequencies of HRTF for plotting magnitude spectrum
    all_freqs = scipy.fft.fftfreq(256, 1 / config.hrir_samplerate)
    pos_freqs = all_freqs[all_freqs >= 0]

    # Start the verification mode of the model.
    model.eval()

    # Initialize the data loader and load the first batch of data
    val_prefetcher.reset()
    batch_data = val_prefetcher.next()

    # Clear/Create directories
    shutil.rmtree(Path(valid_dir), ignore_errors=True)
    Path(valid_dir).mkdir(parents=True, exist_ok=True)
    shutil.rmtree(Path(valid_gt_dir), ignore_errors=True)
    Path(valid_gt_dir).mkdir(parents=True, exist_ok=True)

    if config.transform_flag:
        mean_std_dir = config.mean_std_coef_dir
        mean_std_full = mean_std_dir + "/mean_std_full.pickle"
        with open(mean_std_full, "rb") as f:
            mean, std = pickle.load(f)
        mean = mean.float().to(device)
        std = std.float().to(device)

    plot_flag = True
    while batch_data is not None:
        # Transfer in-memory data to CUDA devices to speed up validation 
        lr_coefficient = batch_data["lr_coefficient"].to(device=device, memory_format=torch.contiguous_format,
                                                         non_blocking=True, dtype=torch.float)
        hrtf = batch_data["hrtf"]
        masks = batch_data["mask"]
        sample_id = batch_data["id"].item()

        # Use the generator model to generate fake samples
        with torch.no_grad():
            _, _, recon = model(lr_coefficient)

        SHT = SphericalHarmonicsTransform(28, ds.row_angles, ds.column_angles, ds.radii, masks[0].numpy().astype(bool))
        harmonics = torch.from_numpy(SHT.get_harmonics()).float().to(device)
        if config.transform_flag:
            recon = recon * std + mean
        sr = harmonics @ recon[0].T
        sr = sr.reshape(-1, num_row_angles, num_col_angles, num_radii, nbins)
        margin = 1.8670232e-08
        if config.domain == "magnitude":
            sr = F.relu(sr) + margin
        file_name = '/' + f"{config.dataset}_{sample_id}.pickle"
        sr = sr[0].detach().cpu()
        # sr = torch.permute(sr[0], (2, 3, 1, 0)).detach().cpu() # w x h x r x nbins
        hr = torch.permute(hrtf[0], (1, 2, 3, 0)).detach().cpu() # r x w x h x nbins

        with open(valid_dir + file_name, "wb") as file:
            pickle.dump(sr, file)

        with open(valid_gt_dir + file_name, "wb") as file:
            pickle.dump(hr, file)

        if plot_flag:
            print("plot")
            generated = sr[0]
            target = hr.permute(1, 2, 0, 3)
            path = '/rds/general/user/jl2622/home/HRTF_GAN'
            filename = f"sample_{sample_id}"
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            x = generated[0, 0, 0, :]
            y = target[0, 0, 0, :]
            ax1.plot(x)
            ax1.set_title('recon')
            ax2.plot(y)
            ax2.set_title('original')
            plt.savefig(f"{path}/{filename}.png")
            plot_flag = False
        # Preload the next batch of data
        batch_data = val_prefetcher.next()
