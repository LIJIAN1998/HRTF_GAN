import os
import pickle

import scipy
import torch
import numpy as np

from model.model import VAE
import shutil
from pathlib import Path

import importlib

from hrtfdata.transforms.hrirs import SphericalHarmonicsTransform


def test(config, val_prefetcher):
    # source: https://github.com/Lornatang/SRGAN-PyTorch/blob/main/test.py
    # Initialize super-resolution model

    # load the dataset to get the row, column angles info
    data_dir = config.raw_hrtf_dir / config.dataset
    imp = importlib.import_module('hrtfdata.full')
    load_function = getattr(imp, config.dataset)
    ds = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate,
                                                         'side': 'left', 'domain': 'time'}}, subject_ids='first')
    num_row_angles = len(ds.row_angles)
    num_col_angles = len(ds.column_angles)
    num_radii = len(ds.radii)
    degree = int(np.sqrt(num_row_angles*num_col_angles*num_radii/config.upscale_factor) - 1)

    ngpu = config.ngpu
    valid_dir = config.valid_path

    nbins = config.nbins_hrtf
    if config.merge_flag:
        nbins = config.nbins_hrtf * 2

    device = torch.device(config.device_name if (
            torch.cuda.is_available() and ngpu > 0) else "cpu")
    model = VAE(nbins=nbins, max_degree=degree, latent_dim=10).to(device)
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

    sample_index = 0
    while batch_data is not None:
        # Transfer in-memory data to CUDA devices to speed up validation
        lr_coefficient = batch_data["lr_coefficient"].to(device=device, memory_format=torch.contiguous_format,
                                                         non_blocking=True, dtype=torch.float)
        hrir = batch_data["hrir"]
        masks = batch_data["mask"]
        val_sample = {}

        # Use the generator model to generate fake samples
        with torch.no_grad():
            _, _, recon = model(lr_coefficient)

        SHT = SphericalHarmonicsTransform(28, ds.row_angles, ds.column_angles, ds.radii, masks[0].numpy().astype(bool))
        harmonics = torch.from_numpy(SHT.get_harmonics()).float()
        sr = harmonics @ recon[0].T
        sr = torch.abs(sr.reshape(-1, nbins, num_radii, num_row_angles, num_col_angles))
        file_name = '/' + os.path.basename(f"val_sample_{sample_index}.pkl")
        # file_name = '/' + os.path.basename(batch_data["filename"][0])
        val_sample['sr'] = torch.permute(sr[0], (2, 3, 1, 0)).detach().cpu() # w x h x r x nbins
        val_sample['hr'] = hrir.detach().cpu()

        with open(valid_dir + file_name, "wb") as file:
            pickle.dump(val_sample, file)
        
        # Preload the next batch of data
        batch_data = val_prefetcher.next()
