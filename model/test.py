import os
import pickle

import scipy
import torch
import torch.nn.functional as F
import numpy as np

from model.DBPN import D_DBPN
# from model.model import VAE, D_DBPN
# from model.ae import AutoEncoder
import shutil
from pathlib import Path

import importlib

from hrtfdata.transforms.hrirs import SphericalHarmonicsTransform

from plot import plot_hrtf
import matplotlib.pyplot as plt

def spectral_distortion_inner(input_spectrum, target_spectrum):
    numerator = target_spectrum
    denominator = input_spectrum
    return torch.mean((20 * np.log10(numerator / denominator)) ** 2)

def plot_tf(ir_id, ori_hrtf, recon_hrtf):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        ax1.plot(ori_hrtf[ir_id])
        ax1.set_title(f'original (ID: {ir_id})')
        ax2.plot(recon_hrtf[ir_id])
        ax2.set_title(f'recon (ID {ir_id})')

        # plt.show()
        plt.savefig(f"tf_{ir_id}.png")
        plt.close()

def test(config, val_prefetcher):
    # source: https://github.com/Lornatang/SRGAN-PyTorch/blob/main/test.py
    # Initialize super-resolution model

    # load the dataset to get the row, column angles info
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
    # model = AutoEncoder(nbins=nbins, in_order=degree, latent_dim=config.latent_dim, base_channels=512, num_features=512, out_oder=max_order)
    model = D_DBPN(nbins, max_order)
    # model = D_DBPN(channels=nbins, base_channels=256, num_features=512, scale_factor=upscale_factor, max_order=max_order)
    # model = VAE(nbins=nbins, max_degree=degree, latent_dim=config.latent_dim).to(device)
    print("Build VAE model successfully.")

    # Load vae model weights (always uses the CPU due to HPC having long wait times)
    # model.load_state_dict(torch.load(f"{config.model_path}/vae.pt", map_location=torch.device('cpu')))
    model.load_state_dict(torch.load(f"{config.model_path}/Gen.pt", map_location=torch.device('cpu')))
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

    margin = 1.8670232e-08

    plot_flag = True
    count = 0
    while batch_data is not None:
        print("count: ", count+1)
        count += 1
        # Transfer in-memory data to CUDA devices to speed up validation 
        lr_coefficient = batch_data["lr_coefficient"].to(device=device, memory_format=torch.contiguous_format,
                                                         non_blocking=True, dtype=torch.float)
        hr_coefficient = batch_data["hr_coefficient"].to(device=device, memory_format=torch.contiguous_format,
                                                         non_blocking=True, dtype=torch.float)
        hrtf = batch_data["hrtf"]
        masks = batch_data["mask"]
        sample_id = batch_data["id"].item()

        # Use the generator model to generate fake samples
        with torch.no_grad():
            # _, _, recon = model(lr_coefficient)
            recon = model(lr_coefficient)

        original_mask = masks[0].numpy().astype(bool)
        SHT = SphericalHarmonicsTransform(max_order, ds.row_angles, ds.column_angles, ds.radii, original_mask)
        harmonics = torch.from_numpy(SHT.get_harmonics()).float().to(device)
        if config.transform_flag:
            recon = recon * std + mean
        recon_hrtf = harmonics @ recon[0].T
        total_positions = len(recon_hrtf)
        ori_hrtf = hrtf[0].reshape(nbins, -1).T
        total_all_positions = 0
        sr = recon_hrtf.reshape(-1, num_row_angles, num_col_angles, num_radii, nbins)
        if config.domain == "magnitude":
            sr = F.relu(sr) + margin
        
        total_sd_metric = 0

        ir_id = 0
        max_value = None
        max_id = None
        min_value = None
        min_id = None
        print("subject: ", sample_id)
        for ori, gen in zip(ori_hrtf, recon_hrtf):
            if domain == 'magnitude_db':
                ori = 10 ** (ori/20)
                gen = 10 ** (gen/20)

            if domain == 'magnitude_db' or domain == 'magnitude':
                average_over_frequencies = spectral_distortion_inner(abs(gen), abs(ori))
                total_all_positions += np.sqrt(average_over_frequencies)
            elif domain == 'time':

                nbins = 128
                ori_tf_left = abs(scipy.fft.rfft(ori[:nbins], nbins*2)[1:])
                ori_tf_right = abs(scipy.fft.rfft(ori[nbins:], nbins*2)[1:])
                gen_tf_left = abs(scipy.fft.rfft(gen[:nbins], nbins*2)[1:])
                gen_tf_right = abs(scipy.fft.rfft(gen[nbins:], nbins*2)[1:])

                ori_tf = np.ma.concatenate([ori_tf_left, ori_tf_right])
                gen_tf = np.ma.concatenate([gen_tf_left, gen_tf_right])

                average_over_frequencies = spectral_distortion_inner(gen_tf, ori_tf)
                total_all_positions += np.sqrt(average_over_frequencies)

            print('Log SD (for %s position): %s' % (ir_id, np.sqrt(average_over_frequencies)))
            if max_value is None or np.sqrt(average_over_frequencies) > max_value:
                max_value = np.sqrt(average_over_frequencies)
                max_id = ir_id
            if min_value is None or np.sqrt(average_over_frequencies) < min_value:
                min_value = np.sqrt(average_over_frequencies)
                min_id = ir_id
            ir_id += 1
        
        sd_metric = total_all_positions / total_positions
        total_sd_metric += sd_metric

        print('Min Log SD (for %s position): %s' % (min_id, min_value))
        print('Max Log SD (for %s position): %s' % (max_id, max_value))

        if plot_flag:
            plot_tf(min_id, ori_hrtf, recon_hrtf)
            plot_tf(max_id, ori_hrtf, recon_hrtf)
            plot_flag = False

        print('Log SD (across all positions): %s' % float(sd_metric))
        
        # file_name = '/' + f"{config.dataset}_{sample_id}.pickle"
        # sr = sr[0].detach().cpu()
        # # sr = torch.permute(sr[0], (2, 3, 1, 0)).detach().cpu() # w x h x r x nbins
        # hr = torch.permute(hrtf[0], (1, 2, 3, 0)).detach().cpu() # r x w x h x nbins

        # with open(valid_dir + file_name, "wb") as file:
        #     pickle.dump(sr, file)

        # with open(valid_gt_dir + file_name, "wb") as file:
        #     pickle.dump(hr, file)

        # if plot_flag:
        #     print("plot")
        #     generated = sr
        #     target = hr.permute(1, 2, 0, 3)
        #     path = '/rds/general/user/jl2622/home/HRTF_GAN'
        #     filename = f"sample_{sample_id}"
        #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        #     x = generated[0, 0, 0, :]
        #     y = target[0, 0, 0, :]
        #     ax1.plot(x)
        #     ax1.set_title('recon')
        #     ax2.plot(y)
        #     ax2.set_title('original')
        #     plt.savefig(f"{path}/{filename}.png")
        #     plot_flag = False
        # Preload the next batch of data
        batch_data = val_prefetcher.next()
