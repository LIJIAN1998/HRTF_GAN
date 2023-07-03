import pickle
import os
import glob
import numpy as np
import torch
import shutil
from pathlib import Path
import importlib

from model.dataset import downsample_hrtf, get_sample_ratio
from preprocessing.cubed_sphere import CubedSphere
from preprocessing.utils import interpolate_fft, get_sphere_coords
from preprocessing.convert_coordinates import convert_cube_to_sphere
from preprocessing.barycentric_calcs import get_triangle_vertices, calc_barycentric_coordinates

PI_4 = np.pi / 4

def my_barycentric_interpolation(config, barycentric_output_path):
    valid_gt_path = glob.glob('%s/%s_*' % (config.valid_gt_path, config.dataset))
    valid_gt_file_names = ['/' + os.path.basename(x) for x in valid_gt_path]

    # Clear/Create directory
    shutil.rmtree(Path(barycentric_output_path), ignore_errors=True)
    Path(barycentric_output_path).mkdir(parents=True, exist_ok=True)

    imp = importlib.import_module('hrtfdata.full')
    load_function = getattr(imp, config.dataset)
    data_dir = config.raw_hrtf_dir / config.dataset
    ds = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate, 
                                                         'side': 'left', 'domain': 'magnitude'}}, subject_ids='first')
    row_angles = ds.row_angles
    column_angles = ds.column_angles
    mask = ds[0]['features'].mask
    sphere_coords, indices = get_sphere_coords(row_angles, column_angles, mask)
    radii = ds.radii
    row_ratio, col_ratio = get_sample_ratio(config.upscale_factor)

    for file_name in valid_gt_file_names:
        with open(config.valid_gt_path + file_name, "rb") as f:
            hr_hrtf = pickle.load(f)

        sphere_coords_lr = []
        sphere_coords_lr_index = []

        for i in range(hr_hrtf.size(0) // row_ratio):
            for j in range(hr_hrtf.size(1) // col_ratio):
                sphere_coords_lr.append(column_angles[col_ratio*j], row_angles[row_ratio * i])

        euclidean_sphere_triangles = []
        euclidean_sphere_coeffs = []





def run_barycentric_interpolation(config, barycentric_output_path, subject_file=None):

    if subject_file is None:
        valid_data_paths = glob.glob('%s/%s_*' % (config.valid_hrtf_merge_dir, config.dataset))
        valid_data_file_names = ['/' + os.path.basename(x) for x in valid_data_paths]
    else:
        valid_data_file_names = ['/' + subject_file]

    # Clear/Create directory
    shutil.rmtree(Path(barycentric_output_path), ignore_errors=True)
    Path(barycentric_output_path).mkdir(parents=True, exist_ok=True)

    projection_filename = f'{config.projection_dir}/{config.dataset}_projection_{config.hrtf_size}'
    with open(projection_filename, "rb") as f:
        (cube_coords, sphere_coords, euclidean_sphere_triangles, euclidean_sphere_coeffs) = pickle.load(f)

    for file_name in valid_data_file_names:
        with open(config.valid_hrtf_merge_dir + file_name, "rb") as f:
            hr_hrtf = pickle.load(f)

        lr_hrtf = torch.permute(downsample_hrtf(torch.permute(hr_hrtf, (3, 0, 1, 2)), config.hrtf_size, config.upscale_factor), (1, 2, 3, 0))

        sphere_coords_lr = []
        sphere_coords_lr_index = []
        for panel, x, y in cube_coords:
            # based on cube coordinates, get indices for magnitudes list of lists
            i = panel - 1
            j = round(config.hrtf_size * (x - (PI_4 / config.hrtf_size) + PI_4) / (np.pi / 2))
            k = round(config.hrtf_size * (y - (PI_4 / config.hrtf_size) + PI_4) / (np.pi / 2))
            if hr_hrtf[i, j, k] in lr_hrtf:
                sphere_coords_lr.append(convert_cube_to_sphere(panel, x, y))
                sphere_coords_lr_index.append([int(i), int(j / config.upscale_factor), int(k / config.upscale_factor)])

        euclidean_sphere_triangles = []
        euclidean_sphere_coeffs = []
        for sphere_coord in sphere_coords:
            # based on cube coordinates, get indices for magnitudes list of lists
            triangle_vertices = get_triangle_vertices(elevation=sphere_coord[0], azimuth=sphere_coord[1],
                                                      sphere_coords=sphere_coords_lr)
            coeffs = calc_barycentric_coordinates(elevation=sphere_coord[0], azimuth=sphere_coord[1],
                                                  closest_points=triangle_vertices)
            euclidean_sphere_triangles.append(triangle_vertices)
            euclidean_sphere_coeffs.append(coeffs)

        cs = CubedSphere(sphere_coords=sphere_coords_lr, indices=sphere_coords_lr_index)

        lr_hrtf_left = lr_hrtf[:, :, :, :config.nbins_hrtf]
        lr_hrtf_right = lr_hrtf[:, :, :, config.nbins_hrtf:]

        barycentric_hr_left = interpolate_fft(config, cs, lr_hrtf_left, sphere_coords, euclidean_sphere_triangles,
                                         euclidean_sphere_coeffs, cube_coords, fs_original=config.hrir_samplerate,
                                         edge_len=config.hrtf_size)
        barycentric_hr_right = interpolate_fft(config, cs, lr_hrtf_right, sphere_coords, euclidean_sphere_triangles,
                                              euclidean_sphere_coeffs, cube_coords, fs_original=config.hrir_samplerate,
                                              edge_len=config.hrtf_size)

        barycentric_hr_merged = torch.tensor(np.concatenate((barycentric_hr_left, barycentric_hr_right), axis=3))

        with open(barycentric_output_path + file_name, "wb") as file:
            pickle.dump(barycentric_hr_merged, file)

        print('Created barycentric baseline %s' % file_name.replace('/', ''))

    return cube_coords, sphere_coords
