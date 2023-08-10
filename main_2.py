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
from model.util import load_dataset, load_hrtf, get_train_val_loader, spectral_distortion_metric, sd_ild_loss
from model.dataset import get_sample_ratio
from model import util
from preprocessing.cubed_sphere import CubedSphere
from preprocessing.hrtf_sphere import HRTF_Sphere
from preprocessing.utils import interpolate_fft, generate_euclidean_cube, convert_to_sofa, my_convert_to_sofa,\
     merge_files, gen_sofa_preprocess, get_hrtf_from_ds, clear_create_directories, get_sphere_coords

from baselines.barycentric_interpolation import run_barycentric_interpolation, my_barycentric_interpolation, debug_barycentric
from baselines.hrtf_selection import run_hrtf_selection
# from evaluation.evaluation import run_lsd_evaluation, run_localisation_evaluation, check_sofa

from hrtfdata.transforms.hrirs import SphericalHarmonicsTransform
from scipy.ndimage import binary_dilation

# import matlab.engine

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

    if mode == 'generate_projection':
        # Must be run in this mode once per dataset, finds barycentric coordinates for each point in the cubed sphere
        # No need to load the entire dataset in this case
        ds = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate, 
                                                              'side': 'left', 'domain': 'time'}}, subject_ids='first')
        # need to use protected member to get this data, no getters
        print("projection dir: ", config.projection_dir)
        cs = CubedSphere(mask=ds[0]['features'].mask, row_angles=ds.row_angles, column_angles=ds.column_angles)
        generate_euclidean_cube(config, cs.get_sphere_coords(), edge_len=config.hrtf_size)

    elif mode == 'preprocess':
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
            else:
                subject_id = str(ds_left.subject_ids[i])
                file_name = '/' + f"{config.dataset}_{subject_id}.pickle"
                with open(valid_gt_dir + file_name, "wb") as file:
                    pickle.dump(merge, file)

        if config.gen_sofa_flag:
            my_convert_to_sofa(valid_gt_dir, config, ds_left.row_angles, ds_left.column_angles)

        # save dataset mean and standard deviation for each channel, across all HRTFs in the training data
        mean = torch.mean(train_hrtfs, [0, 1, 2, 3])
        std = torch.std(train_hrtfs, [0, 1, 2, 3])
        min_hrtf = torch.min(train_hrtfs)
        max_hrtf = torch.max(train_hrtfs)
        mean_std_filename = config.mean_std_filename
        with open(mean_std_filename, "wb") as file:
            pickle.dump((mean, std, min_hrtf, max_hrtf), file)

    elif mode == 'train':
        print("using cuda? ", torch.cuda.is_available())
        # config_file_path = f"{config.path}/config_files/config_150.json"
        # config.load(150)
        config.upscale_factor = 2
        bs, optmizer, lr, alpha, lambda_feature, latent_dim, critic_iters = config.get_train_params()
        with open(f"log.txt", "a") as f:
            # f.write(f"config loaded: {config_file_path}\n")
            f.write(f"batch size: {bs}\n")
            f.write(f"optimizer: {optmizer}\n")
            f.write(f"lr: {lr}\n")
            f.write(f"alpha: {alpha}\n")
            f.write(f"lambda: {lambda_feature}\n")
            f.write(f"latent_dim: {latent_dim}\n")
            f.write(f"critic iters: {critic_iters}\n")

        if config.transform_flag:
            mean_std_coef_filename = config.mean_std_coef_filename
            with open(mean_std_coef_filename, 'rb') as f:
                mean, std = pickle.load(f)
            train_prefetcher, _ = load_hrtf(config, mean, std)
        else:
            train_prefetcher, _ = load_hrtf(config)
        print("transform applied: ", config.transform_flag)
        print("train fetcher: ", len(train_prefetcher))
        # data = train_prefetcher.next()
        # print(data.keys())
        # Trains the model, according to the parameters specified in Config
        # util.initialise_folders(config, overwrite=True)
        train(config, train_prefetcher)

    elif mode == 'test':
        config.upscale_factor = 32
        _, test_prefetcher = load_hrtf(config)
        print("Loaded all datasets successfully.")

        test(config, test_prefetcher)

        # run_lsd_evaluation(config, config.valid_path)
        # run_localisation_evaluation(config, config.valid_path)

    elif mode == 'barycentric_baseline':
        barycentric_data_folder = f'/barycentric_interpolated_data_{config.upscale_factor}'
        barycentric_output_path = config.barycentric_hrtf_dir + barycentric_data_folder
        # run_barycentric_interpolation(config, barycentric_output_path)
        # print("!!!!!!!!!!!!!!!!!!my interpolation!!!!!!!!!!!!!!!!!!!!!!!!")
        # sphere_coords = debug_barycentric(config, barycentric_output_path)
        sphere_coords = my_barycentric_interpolation(config, barycentric_output_path)
        if config.gen_sofa_flag:
            row_angles = list(set([x[1] for x in sphere_coords]))
            column_angles = list(set([x[0] for x  in sphere_coords]))
            my_convert_to_sofa(barycentric_output_path, config, row_angles, column_angles)
            print('Created barycentric baseline sofa files')

        config.path = config.barycentric_hrtf_dir
        file_ext = f'lsd_errors_barycentric_interpolated_data_{config.upscale_factor}.pickle'
        # run_lsd_evaluation(config, barycentric_output_path, file_ext)

        file_ext = f'loc_errors_barycentric_interpolated_data_{config.upscale_factor}.pickle'
        # run_localisation_evaluation(config, barycentric_output_path, file_ext)

    elif mode == 'hrtf_selection_baseline':
        run_hrtf_selection(config, config.hrtf_selection_dir)

        if config.gen_sofa_flag:
            ds = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate,
                                                                 'side': 'left', 'domain': 'magnitude'}}, subject_ids='first')
            row_angles = ds.row_angles
            column_angles = ds.column_angles
            my_convert_to_sofa(config.hrtf_selection_dir, config, row_angles, column_angles)

        config.path = config.hrtf_selection_dir

        file_ext = f'lsd_errors_hrtf_selection_minimum_data.pickle'
        # run_lsd_evaluation(config, config.hrtf_selection_dir, file_ext, hrtf_selection='minimum')
        file_ext = f'loc_errors_hrtf_selection_minimum_data.pickle'
        # run_localisation_evaluation(config, config.hrtf_selection_dir, file_ext, hrtf_selection='minimum')

        file_ext = f'lsd_errors_hrtf_selection_maximum_data.pickle'
        # run_lsd_evaluation(config, config.hrtf_selection_dir, file_ext, hrtf_selection='maximum')
        file_ext = f'loc_errors_hrtf_selection_maximum_data.pickle'
        # run_localisation_evaluation(config, config.hrtf_selection_dir, file_ext, hrtf_selection='maximum')

    elif mode == "debug":
        left_hrtf = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate, 
                                                             'side': 'left', 'domain': 'magnitude_db'}})
        right_hrtf = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate, 
                                                             'side': 'right', 'domain': 'magnitude_db'}})

        with open('/vol/bitbucket/jl2622/HRTF-results/data/SONICOM/train_val_id/train_val_id.pickle', "rb") as f:
            train_ids, val_ids = pickle.load(f)

        left_train = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate, 
                                                                     'side': 'left', 'domain': 'magnitude_db'}},
                                   subject_ids=train_ids)
        right_train = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate, 
                                                                      'side': 'right', 'domain': 'magnitude_db'}},
                                   subject_ids=train_ids)

        means = []
        stds = []
        orders = [19, 13, 9, 6, 4, 3, 2, 1]
        upscale_factors = [2, 4, 8, 16, 32, 48, 72, 108, 216]
        # row_ratio, column_ratio = get_sample_ratio()
        for order in orders:
            print("order: ", order)
            coefs = []
            for sample_id in range(len(left_train)):
                left = left_train[sample_id]['features'][:, :, :, 1:]
                right = right_train[sample_id]['features'][:, :, :, 1:]
                merge = np.ma.concatenate([left, right], axis=3)
                mask = np.all(np.ma.getmaskarray(left), axis=3)
                SHT = SphericalHarmonicsTransform(order, left_hrtf.row_angles, left_hrtf.column_angles, left_hrtf.radii, mask)
                sh_coef = torch.from_numpy(SHT(merge)).T
                coefs.append(sh_coef)
            coefs = torch.stack(coefs)
            print("all train coefs: ", coefs.shape)
            mean = torch.mean(coefs, 0)
            std = torch.std(coefs, 0)
            print("mean: ", mean.shape)
            print(mean[0][:20])
            print("std: ", std.shape)
            print(std[0][:20])
            print("max: ", torch.max(coefs))
            print("min: ", torch.min(coefs))
            print()
            # means.append(torch.mean(sh_coef, 0))
            # stds.append(torch.std(sh_coef, 0))
        # means = torch.stack(means, 0)
        # stds = torch.stack(stds, 0)
        # print("mean shape: ", means.shape)
        # mean = torch.mean(means, 0)
        # std = torch.std(stds, 0)
        # print("mean: ", mean.shape)
        # print("std: ", std.shape)
        
        # mean_std_coef_filename = config.mean_std_coef_filename
        # with open(mean_std_coef_filename, 'rb') as f:
        #     # pickle.dump((mean, std), f) 
        #     mean, std = pickle.load(f)

        # sample_id = 34
        # left = left_hrtf[sample_id]['features'][:, :, :, 1:]
        # right = right_hrtf[sample_id]['features'][:, :, :, 1:]
        # merge = np.ma.concatenate([left, right], axis=3)
        # mask = np.ones((72, 12, 1), dtype=bool)
        # original_mask = np.all(np.ma.getmaskarray(left), axis=3)
        # row_ratio = 1
        # col_ratio = 2
        # for i in range(72 // row_ratio):
        #     for j in range(12 // col_ratio):
        #         mask[row_ratio*i, col_ratio*j, :] = original_mask[row_ratio*i, col_ratio*j, :]
        # order = 15
        # SHT = SphericalHarmonicsTransform(order, left_hrtf.row_angles, left_hrtf.column_angles, left_hrtf.radii, original_mask)
        # sh_coef = torch.from_numpy(SHT(merge))
        # print("coef: ", sh_coef.shape, sh_coef.dtype)
        # norm_coef = (sh_coef.T - mean[:, None]) / std[:, None]
        # print("max coef: ", torch.max(sh_coef))
        # print("min coef: ", torch.min(sh_coef))
        # print("avg coef: ", torch.mean(sh_coef))
        # print("max norm: ", torch.max(norm_coef))
        # print("min norm: ", torch.min(norm_coef))
        # print("avg norm: ", torch.mean(norm_coef))
        # un_norm = norm_coef * std[:, None] + mean[:, None]
        # merge = torch.from_numpy(merge.data).float() # w x h x r x nbins
        # SHT = SphericalHarmonicsTransform(order, left_hrtf.row_angles, left_hrtf.column_angles, left_hrtf.radii, original_mask)
        # harmonics = torch.from_numpy(SHT.get_harmonics())
        # inverse = harmonics @ un_norm.T
        # print("harmonics shape: ", harmonics.shape, harmonics.dtype)
        # print("max harmonics: ", torch.max(harmonics))
        # print("min harmonics: ", torch.min(harmonics))
        # print("avg harmonics: ", torch.mean(harmonics))
        # inverse = harmonics.float() @ sh_coef.float()
        # print("inverse: ", inverse.shape)
        # recon = inverse.reshape(72, 12, 1, 256).detach().cpu() # w x h x r x nbins
        # print("recon: ", recon.shape)
        # margin = 1.8670232e-08
        # generated = recon[None,:].permute(0, 4, 3, 1, 2) # 1 x nbins x r x w x h
        # generated = F.relu(generated) + margin
        # target = merge[None,:].permute(0,4,3,1,2)
        # error = spectral_distortion_metric(generated, target)
        # print("id: ", sample_id)
        # print("lsd error: ", error)

        # sd_mean = 7.387559253346883
        # sd_std = 0.577364154400081
        # ild_mean = 3.6508303231127868
        # ild_std = 0.5261339271318863
        # content_loss = sd_ild_loss(config, generated, target, sd_mean, sd_std, ild_mean, ild_std)
        # print("content loss: ", content_loss)

        # x = recon[70, 1, 0, :]
        # y = merge[70, 1, 0, :]
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