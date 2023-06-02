import argparse
import os
import pickle
import torch
import numpy as np
import importlib

from config import Config
from model.train import train
from model.test import test
from model.util import load_dataset, load_hrtf
from model import util
from preprocessing.cubed_sphere import CubedSphere
from preprocessing.utils import interpolate_fft, generate_euclidean_cube, convert_to_sofa, \
     merge_files, gen_sofa_preprocess, get_hrtf_from_ds, clear_create_directories

from torch.utils.data import ConcatDataset

from hrtfdata.transforms.hrirs import SphericalHarmonicsTransform
from scipy.ndimage import binary_dilation

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
        x = torch.randn(1,3).to(device=config.device_name)
        print("device: ", x.device.type)
        print("merge?: ", config.merge_flag)
        train_prefetcher, valid_prefetcher, _ = load_hrtf(config)
        print("Loaded all datasets successfully.")
        print("train fetcher: ", len(train_prefetcher))
        print("val: ", len(valid_prefetcher))
        data = train_prefetcher.next()
        coef = data['sh_coefficient']
        print("coef: ", coef.shape)
        hrir = data['original_hrir']
        print("hrir:", hrir.shape)

        # Trains the model, according to the parameters specified in Config
        # train_prefetcher, _ = load_dataset(config, mean=None, std=None)
        # print("Loaded all datasets successfully.")

        # train_prefetcher.reset()
        # batch_data = train_prefetcher.next()
        # lr = batch_data["lr"]
        # hr = batch_data["hr"]
        # print("lr: ", type(lr), lr.shape)
        # print("hr: ", type(hr), hr.shape)

        # ds = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate, 
        #                                                       'side': 'left', 'domain': 'magnitude'}}, subject_ids='first')
        # # remove the dc component
        # p = ds[0]['features'][:, :, :, 1:]
        # maskarray = np.ma.getmaskarray(ds[0]['features'])
        # print('feature shape: ', ds[0]['features'].shape)
        # print("mask array: ", maskarray.shape)
        # print("all false: ", np.all(maskarray == False).shape)
        # # print(maskarray)
        # print("np.all: ", np.all(np.ma.getmaskarray(ds[0]['features']), axis=3).shape)
        # SHT = SphericalHarmonicsTransform(10, ds.row_angles, ds.column_angles, ds.radii, 
        #                                   np.all(np.ma.getmaskarray(ds[0]['features']), axis=3))
        
        # original_mask = np.all(np.ma.getmaskarray(ds[0]['features']), axis=3)
        # mask = np.ones((72, 12, 1), dtype=bool)
        # for i in range(36):
        #     for j in range(6):
        #         mask[2*i, 2*j, :] = original_mask[2*i, 2*j, :]
        # SHT = SphericalHarmonicsTransform(10, ds.row_angles, ds.column_angles, ds.radii, mask)
        # x = SHT(ds[0]['features'])
        # print('feature: ', ds[0]['features'].shape)
        # print('coef: ', x.shape)
        # print('inverse: ', SHT.inverse(x).shape)

        

        # perform SHT on each low resolution data, and stack them back into a batch
        # sh_coeffs_list = []
        # for i in range(lr.size(0)):
        #     sh_coeffs_list.append(SHT(lr[i]))
        # sh_coeffs = torch.stack(sh_coeffs_list, dim=0)
        # print("sh coef: ", sh_coeffs.shape)
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