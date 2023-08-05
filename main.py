import argparse
import os
import pickle
import torch
import torch.nn.functional as F
import numpy as np
import importlib

from config import Config
from model.train import train, test_train
from model.test import test
from model.util import load_dataset, load_hrtf, get_train_val_loader
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
        ds = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate,
                                                             'side': 'left', 'domain': 'magnitude'}})
        sphere = HRTF_Sphere(mask=ds[0]['features'].mask, row_angles=ds.row_angles, column_angles=ds.column_angles)

        # Split data into train and test sets
        train_size = int(len(set(ds.subject_ids)) * config.train_samples_ratio)
        train_sample = np.random.choice(list(set(ds.subject_ids)), train_size, replace=False)
        val_sample = list(set(ds.subject_ids) - set(train_sample))
        id_file_dir = config.train_val_id_dir
        if not os.path.exists(id_file_dir):
            os.makedirs(id_file_dir)
        id_filename = id_file_dir + '/train_val_id.pickle'
        with open(id_filename, "wb") as file:
            pickle.dump((train_sample, val_sample), file)



        # Interpolates data to find HRIRs on cubed sphere, then FFT to obtain HRTF, finally splits data into train and
        # val sets and saves processed data
        ds = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate, 'side': 'both', 'domain': 'time'}})
        cs = CubedSphere(mask=ds[0]['features'].mask, row_angles=ds.row_angles, column_angles=ds.column_angles)

        # need to use protected member to get this data, no getters
        projection_filename = f'{config.projection_dir}/{config.dataset}_projection_{config.hrtf_size}'
        with open(projection_filename, "rb") as file:
            cube, sphere, sphere_triangles, sphere_coeffs = pickle.load(file)

        # Clear/Create directories
        clear_create_directories(config)

        # Split data into train and test sets
        train_size = int(len(set(ds.subject_ids)) * config.train_samples_ratio)
        train_sample = np.random.choice(list(set(ds.subject_ids)), train_size, replace=False)
        val_sample = list(set(ds.subject_ids) - set(train_sample))
        id_file_dir = config.train_val_id_dir
        if not os.path.exists(id_file_dir):
            os.makedirs(id_file_dir)
        id_filename = id_file_dir + '/train_val_id.pickle'
        with open(id_filename, "wb") as file:
            pickle.dump((train_sample, val_sample), file)

        # collect all train_hrtfs to get mean and sd
        train_hrtfs = torch.empty(size=(2 * train_size, 5, config.hrtf_size, config.hrtf_size, config.nbins_hrtf))
        j = 0
        for i in range(len(ds)):
            if i % 10 == 0:
                print(f"HRTF {i} out of {len(ds)} ({round(100 * i / len(ds))}%)")

            # Verification that HRTF is valid
            if np.isnan(ds[i]['features']).any():
                print(f'HRTF (Subject ID: {i}) contains nan values')
                continue

            features = ds[i]['features'].data.reshape(*ds[i]['features'].shape[:-2], -1)
            clean_hrtf = interpolate_fft(config, cs, features, sphere, sphere_triangles, sphere_coeffs,
                                             cube, fs_original=ds.hrir_samplerate, edge_len=config.hrtf_size)
            hrtf_original, phase_original, sphere_original = get_hrtf_from_ds(config, ds, i)

            # save cleaned hrtfdata
            if ds.subject_ids[i] in train_sample:
                projected_dir = config.train_hrtf_dir
                projected_dir_original = config.train_original_hrtf_dir
                train_hrtfs[j] = clean_hrtf
                j += 1
            else:
                projected_dir = config.valid_hrtf_dir
                projected_dir_original = config.valid_original_hrtf_dir

            subject_id = str(ds.subject_ids[i])
            side = ds.sides[i]
            with open('%s/%s_mag_%s%s.pickle' % (projected_dir, config.dataset, subject_id, side), "wb") as file:
                pickle.dump(clean_hrtf, file)

            with open('%s/%s_mag_%s%s.pickle' % (projected_dir_original, config.dataset, subject_id, side), "wb") as file:
                pickle.dump(hrtf_original, file)

            with open('%s/%s_phase_%s%s.pickle' % (projected_dir_original, config.dataset, subject_id, side), "wb") as file:
                pickle.dump(phase_original, file)

        if config.merge_flag:
            merge_files(config)

        if config.gen_sofa_flag:
            gen_sofa_preprocess(config, cube, sphere, sphere_original)

        # save dataset mean and standard deviation for each channel, across all HRTFs in the training data
        mean = torch.mean(train_hrtfs, [0, 1, 2, 3])
        std = torch.std(train_hrtfs, [0, 1, 2, 3])
        min_hrtf = torch.min(train_hrtfs)
        max_hrtf = torch.max(train_hrtfs)
        mean_std_filename = config.mean_std_filename
        with open(mean_std_filename, "wb") as file:
            pickle.dump((mean, std, min_hrtf, max_hrtf), file)

    elif mode == 'train':
        # print("using cuda? ", torch.cuda.is_available())
        config_file_path = f"{config.path}/config_files/config_150.json"
        config.load(150)
        config.upscale_factor = 32
        bs, optmizer, lr, alpha, lambda_feature, latent_dim, critic_iters = config.get_train_params()
        with open(f"log.txt", "a") as f:
            f.write(f"config loaded: {config_file_path}\n")
            f.write(f"batch size: {bs}\n")
            f.write(f"optimizer: {optmizer}\n")
            f.write(f"lr: {lr}\n")
            f.write(f"alpha: {alpha}\n")
            f.write(f"lambda: {lambda_feature}\n")
            f.write(f"latent_dim: {latent_dim}\n")
            f.write(f"critic iters: {critic_iters}\n")
        train_prefetcher, _ = load_hrtf(config)
        print("train fetcher: ", len(train_prefetcher))
        # Trains the model, according to the parameters specified in Config
        # util.initialise_folders(config, overwrite=True)
        train(config, train_prefetcher)

    elif mode == 'test':
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
                                                             'side': 'left', 'domain': 'magnitude'}})
        right_hrtf = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate, 
                                                             'side': 'right', 'domain': 'magnitude'}})
        left_ids = left_hrtf.subject_ids
        right_ids = right_hrtf.subject_ids
        print(left_ids)
        print(right_ids)

        # row_angles = ds.row_angles
        # column_angles = ds.column_angles
        # print("num row: ", len(row_angles))
        # with open('log.txt', 'a') as f:
        #     f.write('dataset loaded')


        valid_dir = config.valid_path
        valid_gt_dir = config.valid_gt_path
        shutil.rmtree(Path(valid_dir), ignore_errors=True)
        Path(valid_dir).mkdir(parents=True, exist_ok=True)
        shutil.rmtree(Path(valid_gt_dir), ignore_errors=True)
        Path(valid_gt_dir).mkdir(parents=True, exist_ok=True)
        sample_id = 108
        left = left_hrtf[sample_id]['features'][:, :, :, 1:]
        right = right_hrtf[sample_id]['features'][:, :, :, 1:]
        merge = np.ma.concatenate([left, right], axis=3)
        original_mask = np.all(np.ma.getmaskarray(merge), axis=3)
        SHT = SphericalHarmonicsTransform(28, left_hrtf.row_angles, left_hrtf.column_angles, left_hrtf.radii, original_mask.astype(bool))
        sh_coef = torch.from_numpy(SHT(merge))
        print("coef: ", sh_coef.shape)
        merge = torch.from_numpy(merge.data) # w x h x r x nbins
        harmonics = torch.from_numpy(SHT.get_harmonics())
        print("harmonics shape: ", harmonics.shape)
        inverse = harmonics @ sh_coef
        print("inverse: ", inverse.shape)
        inverse2 = torch.from_numpy(SHT.inverse(sh_coef.numpy()))
        print("inverse2: ", inverse2.shape) 
        recon = inverse.reshape(72, 12, 1, 256).detach().cpu() # w x h x r x nbins
        recon2 = inverse2.reshape(72, 12, 1, 256).detach().cpu()
        # recon = torch.permute(recon[0], (2, 3, 1, 0)).detach().cpu() 
        # recon2 = torch.permute(recon2[0], (2, 3, 1, 0)).detach().cpu()
        print("recon: ", recon.shape)
        # file_name = '/' + f"{config.dataset}_{0}.pickle"
        # with open(valid_dir + file_name, "wb") as file:
        #     pickle.dump(recon, file)
        # hr = torch.permute(merge, (2, 0, 1, 3)).detach().cpu()   # r x w x h x nbins
        # print("gt: ", hr.shape)
        # with open(valid_gt_dir + file_name, "wb") as file:
        #     pickle.dump(hr, file)

        x = recon[24, 8, 0, :]
        y = merge[24, 8, 0, :]
        mean_recon1 = torch.mean(recon)
        max1 = torch.max(recon)
        min1 = torch.min(recon)
        mean_recon2 = torch.mean(recon2)
        max2 = torch.max(recon2)
        min2 = torch.min(recon2)
        mean_original = torch.mean(merge)
        max_original = torch.max(merge)
        min_original = torch.min(merge)
        # print("x: ", x)
        print("mean 1: ", mean_recon1)
        print("mean 2: ", mean_recon2)
        print("original mean: ", mean_original)
        print("max 1: ", max1)
        print("max 2: ", max2)
        print("max original: ", max_original)
        print("min 1: ", min1)
        print("min 2: ", min2)
        print("min original: ", min_original)

        # print("y: ", y)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.plot(x)
        ax1.set_title('recon')
        ax2.plot(y)
        ax2.set_title('original')
        # plt.plot(x)
        plt.savefig("output.png")


        

        
        
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