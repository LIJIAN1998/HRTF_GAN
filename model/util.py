import torch
import os
import shutil
from pathlib import Path
import numpy as np

from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.utils.data import random_split

from model.dataset import CUDAPrefetcher, TrainValidHRTFDataset, CPUPrefetcher, CustomHRTFDataset, MergeHRTFDataset

import importlib
from torch.utils.data._utils.collate import default_collate

def collate_dict_dataset(batch, features_key_name='features', target_key_name='target'):
    return [default_collate(x) for x in zip(*((d[features_key_name], d[target_key_name]) for d in batch))]

def initialise_folders(config, overwrite):
    """Set up folders for given tag

    :param tag: label to use for run
    :param overwrite: whether to overwrite existing model outputs
    """
    if overwrite:
        shutil.rmtree(Path(config.path), ignore_errors=True)
        Path(config.path).mkdir(parents=True, exist_ok=True)

def split_dataset(dataset, train_ratio=0.8):
    total_len = len(dataset)
    train_len = int(total_len * train_ratio)
    test_len = total_len - train_len

    return random_split(dataset, lengths=[train_len, test_len])

def load_hrtf(config):
    data_dir = config.raw_hrtf_dir / config.dataset
    imp = importlib.import_module('hrtfdata.full')
    load_function = getattr(imp, config.dataset)

    if config.merge_flag:
        left = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate, 'side': 'left', 'domain': 'magnitude'}})
        right = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate, 'side': 'right', 'domain': 'magnitude'}})
        custom_dataset = MergeHRTFDataset(left, right, config.upscale_factor)
    else:
        ds = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate, 'side': 'both', 'domain': 'magnitude'}})
        custom_dataset = CustomHRTFDataset(ds, config.upscale_factor)

    train_dataset, test_dataset = split_dataset(custom_dataset, config.train_samples_ratio)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  drop_last=False,
                                  persistent_workers=True)
    test_dataloader = DataLoader(test_dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=1,
                                  pin_memory=True,
                                  drop_last=False,
                                  persistent_workers=True)
    
    # Place all data on the preprocessing data loader
    if torch.cuda.is_available() and config.ngpu > 0:
        device = torch.device(config.device_name)
        train_prefetcher = CUDAPrefetcher(train_dataloader, device)
        test_prefetcher = CUDAPrefetcher(test_dataloader, device)
    else:
        train_prefetcher = CPUPrefetcher(train_dataloader)
        test_prefetcher = CPUPrefetcher(test_dataloader)
    return train_prefetcher, test_prefetcher


def load_dataset(config, mean=None, std=None) -> [CUDAPrefetcher, CUDAPrefetcher, CUDAPrefetcher]:
    """Based on https://github.com/Lornatang/SRGAN-PyTorch/blob/main/train_srgan.py"""

    # define transforms
    if mean is None or std is None:
        transform = None
    else:
        transform = transforms.Normalize(mean=mean, std=std)

    # Load train, test and valid datasets
    if config.merge_flag:
        train_datasets = TrainValidHRTFDataset(config.train_hrtf_merge_dir, config.hrtf_size, config.upscale_factor, transform)
        valid_datasets = TrainValidHRTFDataset(config.valid_hrtf_merge_dir, config.hrtf_size, config.upscale_factor, transform)
    else:
        train_datasets = TrainValidHRTFDataset(config.train_hrtf_dir, config.hrtf_size, config.upscale_factor, transform)
        valid_datasets = TrainValidHRTFDataset(config.valid_hrtf_dir, config.hrtf_size, config.upscale_factor, transform)

    # Generator all dataloader
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=True)
    valid_dataloader = DataLoader(valid_datasets,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=1,
                                  pin_memory=True,
                                  drop_last=False,
                                  persistent_workers=True)

    # Place all data on the preprocessing data loader
    if torch.cuda.is_available() and config.ngpu > 0:
        device = torch.device(config.device_name)
        train_prefetcher = CUDAPrefetcher(train_dataloader, device)
        valid_prefetcher = CUDAPrefetcher(valid_dataloader, device)
    else:
        train_prefetcher = CPUPrefetcher(train_dataloader)
        valid_prefetcher = CPUPrefetcher(valid_dataloader)

    return train_prefetcher, valid_prefetcher


def progress(i, batches, n, num_epochs, timed):
    """Prints progress to console

    :param i: Batch index
    :param batches: total number of batches
    :param n: Epoch number
    :param num_epochs: Total number of epochs
    :param timed: Time per batch
    """
    message = 'batch {} of {}, epoch {} of {}'.format(i, batches, n, num_epochs)
    print(f"Progress: {message}, Time per iter: {timed}")


def spectral_distortion_inner(input_spectrum, target_spectrum):
    numerator = target_spectrum
    denominator = input_spectrum
    x = torch.log10(numerator / denominator)
    print("spectral_distortion_inner nan?", torch.isnan(x), torch.isnan(x).all(), torch.isnan(x).any())
    return torch.mean((20 * torch.log10(numerator / denominator)) ** 2)


def spectral_distortion_metric(generated, target, reduction='mean'):
    """Computes the mean spectral distortion metric for a 5 dimensional tensor (N x C x P x W x H)
    Where N is the batch size, C is the number of frequency bins, P is the number of panels (usually 5),
    H is height, and W is width.

    Computes the mean over every HRTF in the batch"""
    batch_size = generated.size(0)
    num_panels = generated.size(2)
    height = generated.size(3)
    width = generated.size(4)
    total_positions = num_panels * height * width
    print("!!!!!!!!spectral_distortion_metric!!!!!!!!!!!")
    total_sd_metric = 0
    for b in range(batch_size):
        total_all_positions = 0
        for i in range(num_panels):
            for j in range(height):
                for k in range(width):
                    print(f"{b}, {i}, {j}, {k}")
                    average_over_frequencies = spectral_distortion_inner(generated[b, :, i, j, k],
                                                                         target[b, :, i, j, k])
                    total_all_positions += torch.sqrt(average_over_frequencies)
        sd_metric = total_all_positions / total_positions
        total_sd_metric += sd_metric

    print("total_sd_metric nan? ", torch.isnan(total_sd_metric))
    print("all nan? ", torch.isnan(total_sd_metric).all())
    print("any nan? ", torch.isnan(total_sd_metric).any())
    if reduction == 'mean':
        output_loss = total_sd_metric / batch_size
    elif reduction == 'sum':
        output_loss = total_sd_metric
    else:
        raise RuntimeError("Please specify a valid method for reduction (either 'mean' or 'sum').")

    return output_loss


def spectral_distortion_metric_for_plot(generated, target):
    """Computes the mean spectral distortion metric for a 4 dimensional tensor (P x W x H x C)
    Where P is the number of panels (usually 5), H is height, W is width, and C is the number of frequency bins.

    Wrapper for spectral_distortion_metric, used for plot_magnitude_spectrums"""
    generated = torch.permute(generated, (3, 0, 1, 2))
    target = torch.permute(target, (3, 0, 1, 2))

    generated = torch.unsqueeze(generated, 0)
    target = torch.unsqueeze(target, 0)

    return spectral_distortion_metric(generated, target).item()


def ILD_metric_inner(config, input_spectrum, target_spectrum):
    input_left = input_spectrum[:config.nbins_hrtf]
    input_right = input_spectrum[config.nbins_hrtf:]
    target_left = target_spectrum[:config.nbins_hrtf]
    target_right = target_spectrum[config.nbins_hrtf:]
    input_ILD = torch.mean((20 * torch.log10(input_left / input_right)))
    target_ILD = torch.mean((20 * torch.log10(target_left / target_right)))
    return torch.abs(input_ILD - target_ILD)


def ILD_metric(config, generated, target, reduction="mean"):
    batch_size = generated.size(0)
    num_panels = generated.size(2)
    height = generated.size(3)
    width = generated.size(4)
    total_positions = num_panels * height * width

    total_ILD_metric = 0
    for b in range(batch_size):
        total_all_positions = 0
        for i in range(num_panels):
            for j in range(height):
                for k in range(width):
                    average_over_frequencies = ILD_metric_inner(config, generated[b, :, i, j, k], target[b, :, i, j, k])
                    total_all_positions += average_over_frequencies
        ILD_metric_batch = total_all_positions / total_positions
        total_ILD_metric += ILD_metric_batch
    print("total_ILD_metric nan? ", torch.isnan(total_ILD_metric))
    print("all nan? ", torch.isnan(total_ILD_metric).all())
    print("any nan? ", torch.isnan(total_ILD_metric).any())

    if reduction == 'mean':
        output_loss = total_ILD_metric / batch_size
    elif reduction == 'sum':
        output_loss = total_ILD_metric
    else:
        raise RuntimeError("Please specify a valid method for reduction (either 'mean' or 'sum').")

    return output_loss


def ILD_metric_for_plot(config, generated, target):
    """Computes the ILD metric for a 4 dimensional tensor (P x W x H x C)
    Where P is the number of panels (usually 5), H is height, W is width, and C is the number of frequency bins.

    Wrapper for ILD_metric"""
    generated = torch.permute(generated, (3, 0, 1, 2))
    target = torch.permute(target, (3, 0, 1, 2))

    generated = torch.unsqueeze(generated, 0)
    target = torch.unsqueeze(target, 0)

    return ILD_metric(config, generated, target).item()


def sd_ild_loss(config, generated, target, sd_mean, sd_std, ild_mean, ild_std):
    """Computes the mean sd/ild loss for a 5 dimensional tensor (N x C x P x W x H)
    Where N is the batch size, C is the number of frequency bins, P is the number of panels (usually 5),
    H is height, and W is width.

    Computes the mean over every HRTF in the batch"""

    # calculate SD and ILD metrics
    print("generated nan? ", torch.isnan(generated).any())
    print("target nan? ", torch.isnan(target).any())
    sd_metric = spectral_distortion_metric(generated, target)
    print("sd_metric: ", sd_metric)
    ild_metric = ILD_metric(config, generated, target)
    print("ild_metric: ", ild_metric)

    # normalize SD and ILD based on means/standard deviations passed to the function
    sd_norm = torch.div(torch.sub(sd_metric, sd_mean), sd_std)
    ild_norm = torch.div(torch.sub(ild_metric, ild_mean), ild_std)

    # add normalized metrics together
    sum_norms = torch.add(sd_norm, ild_norm)

    # un-normalize
    sum_std = (sd_std ** 2 + ild_std ** 2) ** 0.5
    sum_mean = sd_mean + ild_mean

    output = torch.add(torch.mul(sum_norms, sum_std), sum_mean)

    return output
