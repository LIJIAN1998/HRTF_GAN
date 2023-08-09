import argparse
import torch
import json
import torch.nn as nn

import os
import numpy as np
import importlib
import sys
from config import Config
import matplotlib.pyplot as plt

import torch.nn.functional as F
from model.util import spectral_distortion_metric

from scipy.special import sph_harm

from hrtfdata.util import interaural2spherical, cartesian2spherical

class SphericalHarmonicsTransform:

    def __init__(self, max_degree, row_angles, column_angles, radii, selection_mask, coordinate_system='spherical'):
        self.grid = np.stack(np.meshgrid(row_angles, column_angles, radii, indexing='ij'), axis=-1)
        if coordinate_system == 'spherical':
            # elevations, azimuths, radii -> azimuths, elevations, radii
            self.grid[..., 0], self.grid[..., 1] = np.deg2rad(self.grid[..., 1]), np.deg2rad(self.grid[..., 0])
        elif coordinate_system == 'interaural':
            # lateral, vertical, radius -> azimuths, elevations, radii
            self.grid[..., 0], self.grid[..., 1], self.grid[..., 2] = interaural2spherical(self.grid[..., 0], self.grid[..., 1], self.grid[..., 2],
                                                                            out_angles_as_degrees=False)
        else:
            # X, Y, Z -> azimuths, elevations, radii
            self.grid[..., 0], self.grid[..., 1], self.grid[..., 2] = cartesian2spherical(self.grid[..., 0], self.grid[..., 1], self.grid[..., 2],
                                                                           out_angles_as_degrees=False)
        # Convert elevations to zeniths, azimuths, elevations, radii
        self.grid[..., 1] = np.pi + self.grid[..., 1]
        self.grid[..., 0] = np.pi / 2 + self.grid[..., 0]


        self.selected_angles = self.grid[~selection_mask]
        self._harmonics = np.column_stack(
            [np.real(sph_harm(order, degree, self.selected_angles[:, 1], self.selected_angles[:, 0])) for degree in
             np.arange(max_degree + 1) for order in np.arange(-degree, degree + 1)])
        self._valid_mask = ~selection_mask

    def __call__(self, hrirs):
        return np.linalg.lstsq(self._harmonics, hrirs[self._valid_mask].data, rcond=None)[0]

    def inverse(self, coefficients):
        return self._harmonics @ coefficients

    def get_harmonics(self):
        return self._harmonics

    def get_grid(self):
        return self.grid

    def get_selected_angles(self):
        return self.selected_angles

dataset = 'SONICOM'
config = Config('debug', using_hpc=False, dataset=dataset, data_dir='/data/' + dataset)


# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print("device: ", device)
