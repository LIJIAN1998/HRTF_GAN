import itertools

import numpy as np
import pandas as pd

class HRTF_Sphere(object):
    def __init__(self, mask=None, row_angles=None, column_angles=None, sphere_coords=None, indices=None) -> None:
        self.sphere_coords = []  # list of (elevation, azimuth) for every measurement point
        self.indices = []        # list of (elevation_index. azimuth_index) for every measurement point

        if indices is None:
            # assume same number of elevation measurement points at every azimuth angle
            def elevation_validate(a, b): return None if b else a
