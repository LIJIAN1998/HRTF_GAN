import os
import pickle

import scipy
import torch
import torch.nn.functional as F
import numpy as np
from model.ae import AutoEncoder
from model.util import load_hrtf
from model.dataset import get_sample_ratio
from config import Config

from pathlib import Path
import importlib
from hrtfdata.transforms.hrirs import SphericalHarmonicsTransform
import matplotlib.pyplot as plt

def spectral_distortion_inner(input_spectrum, target_spectrum):
    numerator = target_spectrum
    denominator = input_spectrum
    return torch.mean((20 * np.log10(numerator / denominator)) ** 2)

def calc_lsd(ori_hrtf, recon_hrtf, domain):
    total_all_positions = 0
    total_positions = len(recon_hrtf)
    lsd_list = []
    for ori, gen in zip(ori_hrtf, recon_hrtf):
        if domain == 'magnitude_db':
            ori = 10 ** (ori/20)
            gen = 10 ** (gen/20)
        average_over_frequencies = spectral_distortion_inner(abs(gen), abs(ori))
        total_all_positions += np.sqrt(average_over_frequencies)
        lsd_list.append(np.sqrt(average_over_frequencies))
        sd_metric = total_all_positions / total_positions
        print('Log SD (across all positions): %s' % float(sd_metric))
    return np.array(lsd_list)

def replace_lsd(lsd_arr, upscale_factor):
    lsd_2d = lsd_arr.reshape(72,12)
    row_ratio, column_ratio = get_sample_ratio(upscale_factor)
    for i in range(72 // row_ratio):
        for j in range(12 // column_ratio):
            lsd_2d[row_ratio*i, column_ratio*j] = 0
    return lsd_2d

def plot_lsd(lsd_2d, row_angles, column_angles, filename):
    row_indices, col_indices = np.meshgrid(row_angles, column_angles)
    x = row_indices.flatten()
    y = col_indices.flatten()
    values = lsd_2d.flatten()

    plt.scatter(x, y, c=values, cmap='viridis', s=50, marker='o')
    plt.colorbar(label='Values')
    plt.xlabel('X Coordinates')
    plt.ylabel('Y Coordinates')
    plt.title('Scatter Plot of 2D Array')
    plt.savefig(filename)

print("start visualize.py")
config = Config("ari-upscale-4", using_hpc=True)
data_dir = config.raw_hrtf_dir / config.dataset
imp = importlib.import_module('hrtfdata.full')
load_function = getattr(imp, config.dataset)
domain = config.domain
ds = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate,
                                                        'side': 'left', 'domain': domain}}, subject_ids='first')
row_angles = list(ds.row_angles)
column_angles = list(ds.column_angles)
num_row_angles = len(ds.row_angles)
num_col_angles = len(ds.column_angles)
num_radii = len(ds.radii)
max_order = config.max_order
upscale_factor = config.upscale_factor
degree = int(np.sqrt(num_row_angles*num_col_angles*num_radii/upscale_factor) - 1)

print("domain: ", domain, "upscale factor: ", upscale_factor)

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
model.load_state_dict(torch.load(f"{config.model_path}/Gen32_1.pt", map_location=torch.device('cpu')))
print(f"Load VAE model weights `{os.path.abspath(config.model_path)}` successfully.")

_, test_prefetcher = load_hrtf(config)
test_prefetcher.reset()
batch_data = test_prefetcher.next()
lr_coefficient = batch_data["lr_coefficient"].to(device=device, memory_format=torch.contiguous_format,
                                                    non_blocking=True, dtype=torch.float)
hrtf = batch_data["hrtf"]
masks = batch_data["mask"]
sample_id = batch_data["id"].item()

with torch.no_grad():
    recon = model(lr_coefficient)

original_mask = masks[0].numpy().astype(bool)
SHT = SphericalHarmonicsTransform(max_order, ds.row_angles, ds.column_angles, ds.radii, original_mask)
harmonics = torch.from_numpy(SHT.get_harmonics()).float().to(device)
recon_hrtf = harmonics @ recon[0].T
ori_hrtf = hrtf[0].reshape(nbins, -1).T
print("subject: ", sample_id)

lsd_arr = calc_lsd(ori_hrtf, recon_hrtf, domain='magnitude_db')
lsd_2d = replace_lsd(lsd_arr, upscale_factor)
filename = "lsd.png"
plot_lsd(lsd_2d, row_angles, column_angles, filename)