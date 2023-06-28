from model.util import spectral_distortion_metric
from model.dataset import downsample_hrtf
from preprocessing.utils import convert_to_sofa, my_convert_to_sofa

import shutil
from pathlib import Path
import importlib
import subprocess

import glob
import torch
import pickle
import os
import re
import numpy as np

import matlab.engine

from model.dataset import get_sample_ratio

def replace_nodes(config, val_dir, file_name):
    # Overwrite the generated points that exist in the original data
    with open(val_dir + file_name, "rb") as f:
        data = pickle.load(f)

    sr_hrir = data['sr'] # w x h x r x nbins
    hr_hrir = data['hr']

    row_ratio, col_ratio = get_sample_ratio(config.upscale_factor)
    for i in range(sr_hrir.size(0) // row_ratio):  # sr_hrir.size(0) = num of row angles
        for j in range(sr_hrir.size(1) // col_ratio): # sr_hrir.size(1) = num of column angles
            sr_hrir[row_ratio*i, col_ratio*j, :] = hr_hrir[row_ratio*i, col_ratio*j, :]  # replace the nodes
    
    # 1 x w x h x r x nbins
    generated = torch.permute(sr_hrir[None, :], (0, 4, 3, 1, 2)) # 1 x nbins x r x w x h
    target = torch.permute(hr_hrir[None, :], (0, 4, 3, 1, 2))

    return target, generated

def run_lsd_evaluation(config, val_dir, file_ext=None, hrtf_selection=None):

    file_ext = 'lsd_errors.pickle' if file_ext is None else file_ext

    if hrtf_selection == 'minimum' or hrtf_selection == 'maximum':
        lsd_errors = []
        valid_data_paths = glob.glob('%s/%s_*' % (config.valid_hrtf_merge_dir, config.dataset))
        valid_data_file_names = ['/' + os.path.basename(x) for x in valid_data_paths]

        for file_name in valid_data_file_names:
            # Overwrite the generated points that exist in the original data
            with open(config.valid_hrtf_merge_dir + file_name, "rb") as f:
                hr_hrtf = pickle.load(f)

            with open(f'{val_dir}/{hrtf_selection}.pickle', "rb") as f:
                sr_hrtf = pickle.load(f)

            generated = torch.permute(sr_hrtf[:, None], (1, 4, 0, 2, 3))
            target = torch.permute(hr_hrtf[:, None], (1, 4, 0, 2, 3))

            error = spectral_distortion_metric(generated, target)
            subject_id = ''.join(re.findall(r'\d+', file_name))
            lsd_errors.append([subject_id,  float(error.detach())])
            print('LSD Error of subject %s: %0.4f' % (subject_id, float(error.detach())))
    else:
        val_data_paths = glob.glob(f"{val_dir}/val_sample_*")
        # sr_data_paths = glob.glob('%s/%s_*' % (val_dir, config.dataset))
        val_data_file_names = ['/' + os.path.basename(x) for x in val_data_paths]

        lsd_errors = []
        for file_name in val_data_file_names:
            target, generated = replace_nodes(config, val_dir, file_name)
            error = spectral_distortion_metric(generated, target)
            subject_id = ''.join(re.findall(r'\d+', file_name))
            lsd_errors.append([subject_id,  float(error.detach())])
            print('LSD Error of subject %s: %0.4f' % (subject_id, float(error.detach())))

    print('Mean LSD Error: %0.3f' % np.mean([error[1] for error in lsd_errors]))
    with open(f'{config.path}/{file_ext}', "wb") as file:
        pickle.dump(lsd_errors, file)

def run_localisation_evaluation(config, sr_dir, file_ext=None, hrtf_selection=None):

    imp = importlib.import_module('hrtfdata.full')
    load_function = getattr(imp, config.dataset)
    data_dir = config.raw_hrtf_dir / config.dataset
    ds = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate, 
                                                         'side': 'left', 'domain': 'magnitude'}}, subject_ids='first')
    row_angles = ds.row_angles
    column_angles = ds.column_angles

    file_ext = 'loc_errors.pickle' if file_ext is None else file_ext

    if hrtf_selection == 'minimum' or hrtf_selection == 'maximum':
        nodes_replaced_path = sr_dir
        hrtf_file_names = [hrtf_file_name for hrtf_file_name in os.listdir(config.valid_hrtf_merge_dir + '/sofa_min_phase')]
    else:
        sr_data_paths = glob.glob('%s/%s_*' % (sr_dir, config.dataset))
        sr_data_file_names = ['/' + os.path.basename(x) for x in sr_data_paths]

        # Clear/Create directories
        nodes_replaced_path = sr_dir + '/nodes_replaced'
        shutil.rmtree(Path(nodes_replaced_path), ignore_errors=True)
        Path(nodes_replaced_path).mkdir(parents=True, exist_ok=True)

        for file_name in sr_data_file_names:
            target, generated = replace_nodes(config, sr_dir, file_name)

            with open(nodes_replaced_path + file_name, "wb") as file:
                pickle.dump(torch.permute(generated[0], (1, 2, 3, 0)), file)

        # projection_filename = f'{config.projection_dir}/{config.dataset}_projection_{config.hrtf_size}'
        # with open(projection_filename, "rb") as file:
        #     cube, sphere, sphere_triangles, sphere_coeffs = pickle.load(file)

        my_convert_to_sofa(nodes_replaced_path, config, row_angles, column_angles)
        my_convert_to_sofa(config.valid_gt_path, config, row_angles, column_angles)
        # convert_to_sofa(nodes_replaced_path, config, cube, sphere)
        print('Created valid sofa files')

        hrtf_file_names = [hrtf_file_name for hrtf_file_name in os.listdir(nodes_replaced_path + '/sofa_min_phase')]

    eng = matlab.engine.start_matlab()
    s = eng.genpath(config.amt_dir)
    eng.addpath(s, nargout=0)
    s = eng.genpath(config.data_dirs_path)
    eng.addpath(s, nargout=0)

    loc_errors = []
    for file in hrtf_file_names:
        target_sofa_file = config.valid_gt_path + '/sofa_min_phase/' + file
        # target_sofa_file = config.valid_hrtf_merge_dir + '/sofa_min_phase/' + file
        if hrtf_selection == 'minimum' or hrtf_selection == 'maximum':
            generated_sofa_file = f'{nodes_replaced_path}/sofa_min_phase/{hrtf_selection}.sofa'
        else:
            generated_sofa_file = nodes_replaced_path+'/sofa_min_phase/' + file

        print(f'Target: {target_sofa_file}')
        print(f'Generated: {generated_sofa_file}')
        [pol_acc1, pol_rms1, querr1] = eng.calc_loc(generated_sofa_file, target_sofa_file, nargout=3)
        subject_id = ''.join(re.findall(r'\d+', file))
        loc_errors.append([subject_id, pol_acc1, pol_rms1, querr1])
        print('pol_acc1: %s' % pol_acc1)
        print('pol_rms1: %s' % pol_rms1)
        print('querr1: %s' % querr1)

    print('Mean ACC Error: %0.3f' % np.mean([error[1] for error in loc_errors]))
    print('Mean RMS Error: %0.3f' % np.mean([error[2] for error in loc_errors]))
    print('Mean QUERR Error: %0.3f' % np.mean([error[3] for error in loc_errors]))
    with open(f'{config.path}/{file_ext}', "wb") as file:
        pickle.dump(loc_errors, file)


def run_target_localisation_evaluation(config):

    eng = matlab.engine.start_matlab()
    s = eng.genpath(config.amt_dir)
    eng.addpath(s, nargout=0)
    s = eng.genpath(config.data_dirs_path)
    eng.addpath(s, nargout=0)

    loc_target_errors = []
    target_sofa_path = config.valid_hrtf_merge_dir + '/sofa_min_phase'
    hrtf_file_names = [hrtf_file_name for hrtf_file_name in os.listdir(target_sofa_path)]
    for file in hrtf_file_names:
        target_sofa_file = target_sofa_path + '/' + file
        generated_sofa_file = target_sofa_file
        print(f'Target: {target_sofa_file}')
        print(f'Generated: {generated_sofa_file}')
        [pol_acc1, pol_rms1, querr1] = eng.calc_loc(generated_sofa_file, target_sofa_file, nargout=3)
        subject_id = ''.join(re.findall(r'\d+', file))
        loc_target_errors.append([subject_id, pol_acc1, pol_rms1, querr1])
        print('pol_acc1: %s' % pol_acc1)
        print('pol_rms1: %s' % pol_rms1)
        print('querr1: %s' % querr1)

    print('Mean ACC Error: %0.3f' % np.mean([error[1] for error in loc_target_errors]))
    print('Mean RMS Error: %0.3f' % np.mean([error[2] for error in loc_target_errors]))
    print('Mean QUERR Error: %0.3f' % np.mean([error[3] for error in loc_target_errors]))
    with open(f'{config.data_dir}/{config.dataset}_loc_target_valid_errors.pickle', "wb") as file:
        pickle.dump(loc_target_errors, file)
