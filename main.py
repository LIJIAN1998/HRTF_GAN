import argparse
import os
import pickle
import torch
import numpy as np
import importlib

from config import Config
from model.train import train, test_train
from model.test import test
from model.util import load_dataset, load_hrtf
from model import util
from preprocessing.cubed_sphere import CubedSphere
from preprocessing.utils import interpolate_fft, generate_euclidean_cube, convert_to_sofa, \
     merge_files, gen_sofa_preprocess, get_hrtf_from_ds, clear_create_directories, get_sphere_coords

from evaluation.evaluation import run_lsd_evaluation, run_localisation_evaluation

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
        train_prefetcher, _ = load_hrtf(config)
        print("train fetcher: ", len(train_prefetcher))
        # Trains the model, according to the parameters specified in Config
        util.initialise_folders(config, overwrite=True)
        train(config, train_prefetcher)

    elif mode == 'test':
        _, test_prefetcher = load_hrtf(config)
        print("Loaded all datasets successfully.")

        test(config, test_prefetcher)

        run_lsd_evaluation(config, config.valid_path)
        run_localisation_evaluation(config, config.valid_path)

        with open('/rds/general/user/jl2622/home/HRTF-projection/runs-hpc/ari-upscale-4/valid/nodes_replaced/SONICOM_100.pickle', "rb") as file:
            valid = pickle.load(file)
        print("valid shape: ", valid.shape)
        with open('/rds/general/user/jl2622/home/HRTF-projection/runs-hpc/ari-upscale-4/valid_gt/SONICOM_100.pickle', "rb") as file:
            gt = pickle.load(file)
        print("gt: ", gt.shape)



    elif mode == 'barycentric_baseline':
        barycentric_data_folder = f'/barycentric_interpolated_data_{config.upscale_factor}'
        barycentric_output_path = config.barycentric_hrtf_dir + barycentric_data_folder
        ds = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate,
                                                             'side': 'left', 'domain': 'magnitude'}}, subject_ids='first')
        row_angles = ds.row_angles
        column_angles = ds.column_angles
        mask = ds[0]['features'].mask
        if type(mask) is np.bool_:
            print("bool mask")

        def elevation_validate(a, b): return None if b else a
        num_elevation_measurements = len(column_angles)
        elevation_indices = list(range(num_elevation_measurements))
        elevation = column_angles * np.pi / 180

        for azimuth_index, azimuth in enumerate(row_angles):
            azimuth = azimuth * np.pi / 180
            a = [x.flatten().any() for x in mask[azimuth_index]]
            print("length of []", len(a))
            print(a)
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