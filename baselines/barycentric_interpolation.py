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
from preprocessing.utils import interpolate_fft, get_sphere_coords, my_interpolate_fft
from preprocessing.convert_coordinates import convert_cube_to_sphere
from preprocessing.barycentric_calcs import get_triangle_vertices, calc_barycentric_coordinates
from preprocessing.hrtf_sphere import HRTF_Sphere

from config import Config
from pprint import pprint
import time

PI_4 = np.pi / 4

def debug_barycentric(config, barycentric_output_path):
    with open("log.txt", 'a') as f:
        f.write("start debug barycentric\n")
    # with open('/rds/general/user/jl2622/home/HRTF-projection/data/SONICOM/hr_merge/valid/SONICOM_mag_100.pickle', 'rb') as f:
    #     hrtf1 = pickle.load(f)
    # print("hrtf1: ", hrtf1.shape)
    # with open("log.txt", 'a') as f:
    #     f.write("hrtf loaded\n")

    # projection_filename = f'{config.projection_dir}/{config.dataset}_projection_{config.hrtf_size}'
    # with open(projection_filename, "rb") as f:
    #     (cube_coords, sphere_coords, euclidean_sphere_triangles, euclidean_sphere_coeffs) = pickle.load(f)

    # lr1 = torch.permute(downsample_hrtf(torch.permute(hrtf1, (3, 0, 1, 2)), config.hrtf_size, config.upscale_factor), (1, 2, 3, 0))
    # with open('log.txt', 'a') as f:
    #     f.write("lr initialized\n")
    # sphere_coords_lr = []
    # sphere_coords_lr_index = []
    # for panel, x, y in cube_coords:
    #     # based on cube coordinates, get indices for magnitudes list of lists
    #     i = panel - 1
    #     j = round(config.hrtf_size * (x - (PI_4 / config.hrtf_size) + PI_4) / (np.pi / 2))
    #     k = round(config.hrtf_size * (y - (PI_4 / config.hrtf_size) + PI_4) / (np.pi / 2))
    #     if hrtf1[i, j, k] in lr1:
    #         sphere_coords_lr.append(convert_cube_to_sphere(panel, x, y))
    #         sphere_coords_lr_index.append([int(i), int(j / config.upscale_factor), int(k / config.upscale_factor)])

    # euclidean_sphere_triangles = []
    # euclidean_sphere_coeffs = []
    # n = 0
    # with open("log.txt", 'a') as f:
    #     f.write(f"total num coords: {len(sphere_coords)}\n")
    # start = time.time()
    # for sphere_coord in sphere_coords:
    #     n += 1
    #     # based on cube coordinates, get indices for magnitudes list of lists
    #     triangle_vertices = get_triangle_vertices(elevation=sphere_coord[0], azimuth=sphere_coord[1],
    #                                                 sphere_coords=sphere_coords_lr)
    #     coeffs = calc_barycentric_coordinates(elevation=sphere_coord[0], azimuth=sphere_coord[1],
    #                                             closest_points=triangle_vertices)
    #     euclidean_sphere_triangles.append(triangle_vertices)
    #     euclidean_sphere_coeffs.append(coeffs)
    #     with open("log.txt", 'a') as f:
    #         f.write(f"{n}\n")
    # end = time.time()
    # time_elapsed = end - start
    # with open('log.txt', 'a') as f:
    #     f.write(f"time used: {time_elapsed}\n")
    #     f.write("triangles calculated\n")
    
    # cs = CubedSphere(sphere_coords=sphere_coords_lr, indices=sphere_coords_lr_index)
    # lr1_left = lr1[:, :, :, :config.nbins_hrtf]
    # lr1_right = lr1[:, :, :, config.nbins_hrtf:]
    # print("lr1_left: ", lr1_left.shape)

    # start = time.time()
    # barycentric_hr_left = interpolate_fft(config, cs, lr1_left, sphere_coords, euclidean_sphere_triangles,
    #                                      euclidean_sphere_coeffs, cube_coords, fs_original=config.hrir_samplerate,
    #                                      edge_len=config.hrtf_size)
    # barycentric_hr_right = interpolate_fft(config, cs, lr1_right, sphere_coords, euclidean_sphere_triangles,
    #                                      euclidean_sphere_coeffs, cube_coords, fs_original=config.hrir_samplerate,
    #                                      edge_len=config.hrtf_size)
    
    # barycentric_hr_merged = torch.tensor(np.concatenate((barycentric_hr_left, barycentric_hr_right), axis=3))
    # with open("log.txt", "a") as f:
    #     f.write(f"barycentric hr merge: {barycentric_hr_merged.shape}\n")

    # end = time.time()
    # time_elapsed = end - start
    # print("interpolation results: ", barycentric_hr_left.shape)
    # with open('log.txt', 'a') as f:
    #     f.write(f"fft time: {time_elapsed}\n")
    #     f.write("interpolation done")

    ###########################################################################
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
    full_size = (len(row_angles), len(column_angles))
    mask = ds[0]['features'].mask
    whole_sphere = HRTF_Sphere(mask=mask, row_angles=row_angles, column_angles=column_angles)

    nbins = config.nbins_hrtf
    if config.merge_flag:
        nbins = config.nbins_hrtf * 2

    sphere_coords = whole_sphere.get_sphere_coords()
    with open("log.txt", "a") as f:
        f.write(f"num coords: {len(sphere_coords)}\n")

    row_ratio, column_ratio = get_sample_ratio(config.upscale_factor)
    with open(config.valid_gt_path + '/SONICOM_100.pickle', "rb") as f:
        hr_hrtf = pickle.load(f)  # r x w x h x nbins

    print("hrtf2: ", hr_hrtf.shape)
    # initialize an empty lr_hrtf
    lr_hrtf = torch.zeros(1, hr_hrtf.size(1) // row_ratio, hr_hrtf.size(2) // column_ratio, nbins)

    sphere_coords_lr = []
    sphere_coords_lr_index = []
    
    for i in range(hr_hrtf.size(1) // row_ratio):
        for j in range(hr_hrtf.size(2) // column_ratio):
            elevation = column_angles[column_ratio*j] * np.pi / 180
            azimuth = row_angles[row_ratio * i] * np.pi / 180
            sphere_coords_lr.append((elevation, azimuth))
            sphere_coords_lr_index.append((j ,i))
            lr_hrtf[:, i, j] = hr_hrtf[:, row_ratio * i, column_ratio*j]

    print("lr: ", lr_hrtf.shape)
    print("num of my lr coords: ", len(sphere_coords_lr))
    with open("log.txt", "a") as f:
        f.write(f"my num lr coords: {len(sphere_coords_lr)}\n")

    euclidean_sphere_triangles = []
    euclidean_sphere_coeffs = []

    start_time = time.time()
    n = 0
    for sphere_coord in sphere_coords:
        n += 1
        # based on cube coordinates, get indices for magnitudes list of lists
        triangle_vertices = get_triangle_vertices(elevation=sphere_coord[0], azimuth=sphere_coord[1],  # (v0, v1, v2) v0: (elev, azi)
                                                  sphere_coords=sphere_coords_lr)
        coeffs = calc_barycentric_coordinates(elevation=sphere_coord[0], azimuth=sphere_coord[1],
                                              closest_points=triangle_vertices)
        euclidean_sphere_triangles.append(triangle_vertices)
        euclidean_sphere_coeffs.append(coeffs)
        with open("log.txt", "a") as f:
            f.write(f"{n}\n")

    lr_sphere = HRTF_Sphere(sphere_coords=sphere_coords_lr, indices=sphere_coords_lr_index)

    lr_hrtf_left = lr_hrtf[:, :, :, :config.nbins_hrtf]  
    lr_hrtf_right = lr_hrtf[:, :, :, config.nbins_hrtf:]

    barycentric_hr_left = my_interpolate_fft(config, lr_sphere, lr_hrtf_left, full_size, sphere_coords,
                                             euclidean_sphere_triangles,euclidean_sphere_coeffs)
    barycentric_hr_right = my_interpolate_fft(config, lr_sphere, lr_hrtf_right, full_size, sphere_coords,
                                              euclidean_sphere_triangles, euclidean_sphere_coeffs)
    
    barycentric_hr_merged = torch.tensor(np.concatenate((barycentric_hr_left, barycentric_hr_right), axis=3)).permute(1, 2, 0, 3)
    with open("log.txt", "a") as f:
        f.write(f"barycentric hr merge: {barycentric_hr_merged.shape}\n")

    with open(barycentric_output_path + '/SONICOM_100.pickle', "wb") as file:
        pickle.dump(barycentric_hr_merged, file)
        
    print('Created barycentric baseline %s' % '/SONICOM_100.pickle'.replace('/', ''))
    end_time = time.time()
    elapsed_time = end_time - start_time
    with open("log.txt", "a") as f:
        f.write(f"time for one file: {elapsed_time}\n")
    return sphere_coords

def my_barycentric_interpolation(config, barycentric_output_path):
    print("my barycentric interpolation")
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
    whole_sphere = HRTF_Sphere(mask=mask, row_angles=row_angles, column_angles=column_angles)
    sphere_coords = whole_sphere.get_sphere_coords()

    row_ratio, column_ratio = get_sample_ratio(config.upscale_factor)

    nbins = config.nbins_hrtf
    if config.merge_flag:
        nbins = config.nbins_hrtf * 2

    print("before loop through gt files")
    num_file = 0
    for file_name in valid_gt_file_names:
        with open(config.valid_gt_path + file_name, "rb") as f:
            hr_hrtf = pickle.load(f)  # r x w x h x nbins

        sphere_coords_lr = []
        sphere_coords_lr_index = []
        num_file += 1
        print("file opened: ", num_file)

        # initialize an empty lr_hrtf
        lr_hrtf = torch.zeros(1, hr_hrtf.size(1) // row_ratio, hr_hrtf.size(2) // column_ratio, nbins)

        for i in range(hr_hrtf.size(1) // row_ratio):
            for j in range(hr_hrtf.size(2) // column_ratio):
                elevation = column_angles[column_ratio*j] * np.pi / 180
                azimuth = row_angles[row_ratio * i] * np.pi / 180
                sphere_coords_lr.append((elevation, azimuth))
                sphere_coords_lr_index.append((j ,i))
                lr_hrtf[:, i, j] = hr_hrtf[:, row_ratio * i, column_ratio*j]

        print("my lr sphere coords:", len(sphere_coords_lr))
        with open("log.txt", "a") as f:
            f.write(f"num lr coords: {len(sphere_coords_lr)}\n")
        # pprint(sphere_coords_lr)
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
        with open("log.txt", "a") as f:
            f.write(f"{num_file}, triangle\n")

        lr_sphere = HRTF_Sphere(sphere_coords=sphere_coords_lr, indices=sphere_coords_lr_index)

        lr_hrtf_left = lr_hrtf[:, :, :, :config.nbins_hrtf]  
        lr_hrtf_right = lr_hrtf[:, :, :, config.nbins_hrtf:]

        barycentric_hr_left = my_interpolate_fft(config, lr_sphere, lr_hrtf_left, sphere_coords,
                                                 euclidean_sphere_triangles,euclidean_sphere_coeffs)
        barycentric_hr_right = my_interpolate_fft(config, lr_sphere, lr_hrtf_right, sphere_coords,
                                                  euclidean_sphere_triangles, euclidean_sphere_coeffs)
        
        barycentric_hr_merged = torch.tensor(np.concatenate((barycentric_hr_left, barycentric_hr_right), axis=3))

        with open(barycentric_output_path + file_name, "wb") as file:
            pickle.dump(barycentric_hr_merged, file)
        
        print('Created barycentric baseline %s' % file_name.replace('/', ''))
    return sphere_coords


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
    with open("log.txt", "a") as f:
        f.write(f"num sphere coords: {len(sphere_coords)}\n")

    for file_name in valid_data_file_names:
        file_name = '/SONICOM_mag_100.pickle'
        with open(config.valid_hrtf_merge_dir + file_name, "rb") as f:
            hr_hrtf = pickle.load(f)

        print("hrtf shape: ", hr_hrtf.shape)
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

        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("num lr coords: ", len(sphere_coords_lr))
        with open("log.txt", "a") as f:
            f.write(f"num lr coords: {len(sphere_coords_lr)}\n")

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
        return
        barycentric_hr_right = interpolate_fft(config, cs, lr_hrtf_right, sphere_coords, euclidean_sphere_triangles,
                                              euclidean_sphere_coeffs, cube_coords, fs_original=config.hrir_samplerate,
                                              edge_len=config.hrtf_size)

        barycentric_hr_merged = torch.tensor(np.concatenate((barycentric_hr_left, barycentric_hr_right), axis=3))

        with open(barycentric_output_path + file_name, "wb") as file:
            pickle.dump(barycentric_hr_merged, file)

        print('Created barycentric baseline %s' % file_name.replace('/', ''))

    return cube_coords, sphere_coords