import argparse
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import importlib

from config import Config
from model.train import train, test_train
from model.test import test
from model.util import load_dataset, load_hrtf, get_train_val_loader, spectral_distortion_metric
from model import util
from preprocessing.cubed_sphere import CubedSphere
from preprocessing.hrtf_sphere import HRTF_Sphere
from preprocessing.utils import interpolate_fft, generate_euclidean_cube, convert_to_sofa, my_convert_to_sofa,\
     merge_files, gen_sofa_preprocess, get_hrtf_from_ds, clear_create_directories, get_sphere_coords

from baselines.barycentric_interpolation import run_barycentric_interpolation, my_barycentric_interpolation, debug_barycentric
from baselines.hrtf_selection import run_hrtf_selection
from evaluation.evaluation import run_lsd_evaluation, run_localisation_evaluation, check_sofa, run_target_localisation_evaluation

from hrtfdata.transforms.hrirs import SphericalHarmonicsTransform
from scipy.ndimage import binary_dilation

import matlab.engine

import shutil
from pathlib import Path
import matplotlib.pyplot as plt

PI_4 = np.pi / 4

# Random seed to maintain reproducible results
torch.manual_seed(0)
np.random.seed(0)

def main(config, mode):
    # Initialize Config
    data_dir = config.raw_hrtf_dir / config.dataset
    print(os.getcwd())
    print(config.dataset)

    imp = importlib.import_module('hrtfdata.full')
    load_function = getattr(imp, config.dataset)

    if mode == 'preprocess':
        ds_left = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate,
                                                                  'side': 'left', 'domain': 'magnitude'}})
        ds_right = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate,
                                                                  'side': 'left', 'domain': 'magnitude'}})
        sphere = HRTF_Sphere(mask=ds_left[0]['features'].mask, row_angles=ds_left.row_angles, column_angles=ds_left.column_angles)

        # Split data into train and test sets
        train_size = int(len(set(ds_left.subject_ids)) * config.train_samples_ratio)
        train_sample = np.random.choice(list(set(ds_left.subject_ids)), train_size, replace=False)
        val_sample = list(set(ds_left.subject_ids) - set(train_sample))
        print("num train samples: ", len(train_sample))
        print(train_sample)
        print("num validation samples: ", len(val_sample))
        print(val_sample)
        id_file_dir = config.train_val_id_dir
        if not os.path.exists(id_file_dir):
            os.makedirs(id_file_dir)
        id_filename = id_file_dir + '/train_val_id.pickle'
        with open(id_filename, "wb") as file:
            pickle.dump((train_sample, val_sample), file)

        valid_gt_dir = config.valid_gt_path
        shutil.rmtree(Path(valid_gt_dir), ignore_errors=True)
        Path(valid_gt_dir).mkdir(parents=True, exist_ok=True)

        # collect all train_hrtfs to get mean and sd
        num_rows = len(ds_left.row_angles)
        num_columns = len(ds_left.column_angles)
        j = 0
        train_hrtfs = torch.empty(size=(2 * train_size, 1, num_rows, num_columns, config.nbins_hrtf))
        for i in range(len(ds_left)):
            left = ds_left[i]['features'][:, :, :, 1:]
            right = ds_right[i]['features'][:, :, :, 1:]
            merge = np.ma.concatenate([left, right], axis=3)
            merge = torch.from_numpy(merge.data).permute(2, 0, 1, 3) # r x w x h x nbins
            if ds_left.subject_ids[i] in train_sample:
                train_hrtfs[j] = merge[:, :, :, :config.nbins_hrtf] # add left
                j += 1
                train_hrtfs[j] = merge[:, :, :, config.nbins_hrtf:] # add right
                j += 1
            else: # store test HRTFs
                subject_id = str(ds_left.subject_ids[i])
                file_name = '/' + f"{config.dataset}_{subject_id}.pickle"
                with open(valid_gt_dir + file_name, "wb") as file:
                    pickle.dump(merge, file)

        # save dataset mean and standard deviation for each channel, across all HRTFs in the training data
        mean = torch.mean(train_hrtfs, [0, 1, 2, 3])
        std = torch.std(train_hrtfs, [0, 1, 2, 3])
        min_hrtf = torch.min(train_hrtfs)
        max_hrtf = torch.max(train_hrtfs)
        mean_std_filename = config.mean_std_filename
        with open(mean_std_filename, "wb") as file:
            pickle.dump((mean, std, min_hrtf, max_hrtf), file)

    elif mode == 'train':
        bs, optmizer, lr_G, lr_D, latent_dim, critic_iters = config.get_train_params()
        with open(f"log.txt", "a") as f:
            # f.write(f"config loaded: {config_file_path}\n")
            f.write(f"batch size: {bs}\n")
            f.write(f"optimizer: {optmizer}\n")
            f.write(f"generator lr: {lr_G}\n")
            f.write(f"discriminator lr: {lr_D}\n")
            f.write(f"latent_dim: {latent_dim}\n")
            f.write(f"critic iters: {critic_iters}\n")
        train_prefetcher, _ = load_hrtf(config)
        print("train fetcher: ", len(train_prefetcher))
        # Trains the model, according to the parameters specified in Config
        util.initialise_folders(config, overwrite=True)
        train(config, train_prefetcher)

    elif mode == 'test':
        # config.upscale_factor = 216
        with open("log.txt", "a") as f:
            f.write(f"upscale factor: {config.upscale_factor}\n")
        if config.transform_flag:
            mean_std_dir = config.mean_std_coef_dir
            mean_std_full = mean_std_dir + "/mean_std_full.pickle"
            with open(mean_std_full, "rb") as f:
                mean_full, std_full = pickle.load(f)
            
            mean_std_lr = mean_std_dir + f"/mean_std_{config.upscale_factor}.pickle"
            with open(mean_std_lr, "rb") as f:
                mean_lr, std_lr = pickle.load(f)
            mean = (mean_lr, mean_full)
            std = (std_lr, std_full)
            _, test_prefetcher = load_hrtf(config, mean, std)
        else:
            _, test_prefetcher = load_hrtf(config)
        print("Loaded all datasets successfully.")

        test(config, test_prefetcher)

        run_lsd_evaluation(config, config.valid_path, config.valid_gt_path)
        run_localisation_evaluation(config, config.recon_mag_path, config.valid_mag_path)

    elif mode == 'barycentric_baseline':
        config.domain = "magnitude"
        # config.upscale_factor = 216
        # print("domain: ", config.domain)
        # print("upsacle factor: ", config.upscale_factor)
        #  store hr hrtf pickles
        _, test_prefetcher = load_hrtf(config)
        valid_mag_dir = config.valid_mag_path
        shutil.rmtree(Path(valid_mag_dir), ignore_errors=True)
        Path(valid_mag_dir).mkdir(parents=True, exist_ok=True)
        test_prefetcher.reset()
        batch_data = test_prefetcher.next()
        while batch_data is not None:
            hrtf = batch_data["hrtf"]
            sample_id = batch_data["id"].item()
            hr = torch.permute(hrtf[0], (1, 2, 3, 0)).detach().cpu()  # r x w x h x nbins
            # print(f"data {sample_id} has negative? ", (hr<0).any())
            file_name = '/' + f"{config.dataset}_{sample_id}.pickle"
            with open(valid_mag_dir + file_name, "wb") as file:
                pickle.dump(hr, file)
            batch_data = test_prefetcher.next()

        barycentric_data_folder = f'/barycentric_interpolated_data_{config.upscale_factor}'
        barycentric_output_path = config.barycentric_hrtf_dir + barycentric_data_folder
        # run_barycentric_interpolation(config, barycentric_output_path)
        # print("!!!!!!!!!!!!!!!!!!my interpolation!!!!!!!!!!!!!!!!!!!!!!!!")
        sphere_coords = my_barycentric_interpolation(config, barycentric_output_path)
        # if config.gen_sofa_flag:
        #     row_angles = list(set([x[1] for x in sphere_coords]))  # rad
        #     column_angles = list(set([x[0] for x  in sphere_coords]))  # rad
        #     my_convert_to_sofa(barycentric_output_path, config, row_angles, column_angles)
        #     print('Created barycentric baseline sofa files')

        config.path = config.barycentric_hrtf_dir
        file_ext = f'lsd_errors_barycentric_interpolated_data_{config.upscale_factor}.pickle'
        run_lsd_evaluation(config, barycentric_output_path, valid_mag_dir, file_ext)

        file_ext = f'loc_errors_barycentric_interpolated_data_{config.upscale_factor}.pickle'
        run_localisation_evaluation(config, barycentric_output_path, valid_mag_dir, file_ext)
        # run_target_localisation_evaluation(config)

    elif mode == 'hrtf_selection_baseline':
        config.domain = "magnitude"
        run_hrtf_selection(config, config.hrtf_selection_dir)

        if config.gen_sofa_flag:
            ds = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate,
                                                                 'side': 'left', 'domain': 'magnitude'}}, subject_ids='first')
            row_angles = ds.row_angles
            column_angles = ds.column_angles
            my_convert_to_sofa(config.hrtf_selection_dir, config, row_angles, column_angles)

        config.path = config.hrtf_selection_dir

        file_ext = f'lsd_errors_hrtf_selection_minimum_data.pickle'
        run_lsd_evaluation(config, config.hrtf_selection_dir, valid_mag_dir, file_ext, hrtf_selection='minimum')
        file_ext = f'loc_errors_hrtf_selection_minimum_data.pickle'
        run_localisation_evaluation(config, config.hrtf_selection_dir, valid_mag_dir, file_ext, hrtf_selection='minimum')

        file_ext = f'lsd_errors_hrtf_selection_maximum_data.pickle'
        run_lsd_evaluation(config, config.hrtf_selection_dir, valid_mag_dir, file_ext, hrtf_selection='maximum')
        file_ext = f'loc_errors_hrtf_selection_maximum_data.pickle'
        run_localisation_evaluation(config, config.hrtf_selection_dir, valid_mag_dir, file_ext, hrtf_selection='maximum')

    # ignore the debugging code
    elif mode == "debug":
        # ds = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate, 'side': 'both', 'domain': 'time'}})
        # cs = CubedSphere(mask=ds[0]['features'].mask, row_angles=ds.row_angles, column_angles=ds.column_angles)
        # projection_filename = f'{config.projection_dir}/{config.dataset}_projection_{config.hrtf_size}'
        # with open(projection_filename, "rb") as file:
        #     cube, sphere, sphere_triangles, sphere_coeffs = pickle.load(file)
        # features = ds[0]['features'].data
        # print("features: ", type(features), features.shape)
        # print(*ds[0]['features'].shape[:-2])
        # features = features.reshape(*ds[0]['features'].shape[:-2], -1)
        # clean_hrtf = interpolate_fft(config, cs, features, sphere, sphere_triangles, sphere_coeffs,
        #                              cube, fs_original=ds.hrir_samplerate, edge_len=config.hrtf_size)
        # print("clean_hrtf", clean_hrtf.shape)

        left_hrtf = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate, 
                                                             'side': 'left', 'domain': 'magnitude_db'}})
        right_hrtf = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate, 
                                                             'side': 'right', 'domain': 'magnitude_db'}})
        # min_list = []
        # all_valid = True
        with open('/vol/bitbucket/jl2622/HRTF-results/data/SONICOM/train_val_id/train_val_id.pickle', "rb") as f:
            train_ids, val_ids = pickle.load(f)

        left_train = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate, 'side': 'left', 'domain': 'magnitude_db'}},
                                   subject_ids=train_ids)
        right_train = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate, 'side': 'right', 'domain': 'magnitude_db'}},
                                   subject_ids=train_ids)

        means = []
        stds = []
        for sample_id in range(len(left_train)):
            left = left_train[sample_id]['features'][:, :, :, 1:]
            right = right_train[sample_id]['features'][:, :, :, 1:]
            merge = np.ma.concatenate([left, right], axis=3)
            mask = np.all(np.ma.getmaskarray(left), axis=3)
            SHT = SphericalHarmonicsTransform(28, left_hrtf.row_angles, left_hrtf.column_angles, left_hrtf.radii, mask)
            sh_coef = torch.from_numpy(SHT(merge)).T
            means.append(torch.mean(sh_coef, 1))
            stds.append(torch.std(sh_coef, 1))
        means = torch.stack(means, 0)
        stds = torch.stack(stds, 0)
        print("mean shape: ", means.shape)
        mean = torch.mean(means, 0)
        std = torch.std(torch.tensor(stds), 0)
        print("mean: ", mean.shape)
        # print(mean)
        print("std: ", std.shape)
        # print(std)

        sample_id = 55
        left = left_hrtf[sample_id]['features'][:, :, :, 1:]
        right = right_hrtf[sample_id]['features'][:, :, :, 1:]
        merge = np.ma.concatenate([left, right], axis=3)
        mask = np.ones((72, 12, 1), dtype=bool)
        original_mask = np.all(np.ma.getmaskarray(left), axis=3)
        # row_ratio = 8
        # col_ratio = 4
        # for i in range(72 // row_ratio):
        #     for j in range(12 // col_ratio):
        #         mask[row_ratio*i, col_ratio*j, :] = original_mask[row_ratio*i, col_ratio*j, :]
        order = 28
        SHT = SphericalHarmonicsTransform(order, left_hrtf.row_angles, left_hrtf.column_angles, left_hrtf.radii, original_mask)
        sh_coef = torch.from_numpy(SHT(merge))
        print("coef: ", sh_coef.shape, sh_coef.dtype)
        norm_coef = (sh_coef - mean[:, None]) / std[:, None]
        print("max norm: ", torch.max(norm_coef))
        print("min norm: ", torch.min(norm_coef))
        print("avg norm: ", torch.mean(norm_coef))
        # merge = torch.from_numpy(merge.data).float() # w x h x r x nbins
        # harmonics = torch.from_numpy(SHT.get_harmonics()).float()
        # print("harmonics shape: ", harmonics.shape, harmonics.dtype)
        # print("max harmonics: ", torch.max(harmonics))
        # print("min harmonics: ", torch.min(harmonics))
        # print("avg harmonics: ", torch.mean(harmonics))
        # inverse = harmonics @ sh_coef
        # print("inverse: ", inverse.shape)
        # recon = inverse.reshape(72, 12, 1, 256).detach().cpu() # w x h x r x nbins
        # print("recon: ", recon.shape)
        # margin = 1.8670232e-08
        # generated = recon[None,:].permute(0, 4, 3, 1, 2) # 1 x nbins x r x w x h
        # target = merge[None,:].permute(0,4,3,1,2)
        # error = spectral_distortion_metric(generated, target)
        # print("id: ", sample_id)
        # print("lsd error: ", error)

        # x = recon[15, 6, 0, :]
        # y = merge[15, 6, 0, :]
        # mean_recon1 = torch.mean(recon)
        # max1 = torch.max(recon)
        # min1 = torch.min(recon)
        # mean_original = torch.mean(merge)
        # max_original = torch.max(merge)
        # min_original = torch.min(merge)
        # print("order: ", order)
        # print("mean 1: ", mean_recon1)
        # print("original mean: ", mean_original)
        # print("max 1: ", max1)
        # print("max original: ", max_original)
        # print("min 1: ", min1)
        # print("min original: ", min_original)

        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        # ax1.plot(x)
        # ax1.set_title('recon')
        # ax2.plot(y)
        # ax2.set_title('original')
        # plt.savefig("output.png")


        

        
        
        # config.batch_size = 1
        # train_prefetcher, test_prefetcher = load_hrtf(config)
        # train_prefetcher.reset()
        # train_batch = train_prefetcher.next()
        # while train_batch is not None:
        #     lr_coefficient = train_batch["lr_coefficient"]
        #     if torch.isnan(lr_coefficient).any():
        #         id = train_batch["id"]
        #         print("nan coef in train sample ", id)
        #     train_batch = train_prefetcher.next()

        # test_prefetcher.reset()
        # test_batch = test_prefetcher.next()
        # while test_batch is not None:
        #     lr_coefficient = test_batch["lr_coefficient"]
        #     if torch.isnan(lr_coefficient).any():
        #         id = test_batch["id"]
        #         print("nan in test sample ", id)
        #     test_batch = test_prefetcher.next()


        # ds = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate, 'side': 'both', 'domain': 'magnitude'}})
        # subject_ids = list(ds.subject_ids)
        # hrtf = ds[0]['features'][:, :, :, 1:]
        # for i in range(len(subject_ids)):
        #     hrtf = ds[i]['features'][:, :, :, 1:]
        #     data = torch.from_numpy(hrtf.data)
        #     if torch.isnan(data).any():
        #         print("id: ", subject_ids[i])
        #         print("target: ", ds[i]['target'])
        #         print("group: ", ds[i]['group'])
        #         print()
            
        # train_prefetcher, _ = get_train_val_loader(config)
        # print("train size: ", len(train_prefetcher))
        # test_train(config, train_prefetcher)

    print("finished")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mode")
    parser.add_argument("-t", "--tag")
    parser.add_argument("-c", "--hpc")
    args = parser.parse_args()

    if args.hpc == "True":
        hpc = True
    elif args.hpc == "False":
        hpc = False
    else:
        raise RuntimeError("Please enter 'True' or 'False' for the hpc tag (-c/--hpc)")
    
    if args.tag:
        tag = args.tag
    else:
        tag = None

    config = Config(tag, using_hpc=hpc)
    main(config, args.mode)