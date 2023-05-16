import argparse
import os
import pickle
import torch
import numpy as np
import importlib

from config import Config
from preprocessing.cubed_sphere import CubedSphere
from preprocessing.utils import interpolate_fft, generate_euclidean_cube, convert_to_sofa, \
     merge_files, gen_sofa_preprocess, get_hrtf_from_ds, clear_create_directories

PI_4 = np.pi / 4

# Random seed to maintain reproducible results
torch.manual_seed(0)
np.random.seed(0)

def main(config, mode):
    # Initialize Config
    data_dir = config.raw_hrtf_dir / config.dataset
    print("data dir: ", data_dir)
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
        print("ds mask:", ds[0]['features'].mask)
        # cs = CubedSphere(mask=ds[0]['features'].mask, row_angles=ds.row_angles, column_angles=ds.column_angles)
        # generate_euclidean_cube(config, cs.get_sphere_coords(), edge_len=config.hrtf_size)

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