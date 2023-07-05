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
            num_elevation_measurements = len(column_angles)
            elevation_indices = list(range(num_elevation_measurements))
            elevation = column_angles * np.pi / 180

            # loop through all azimuth positions
            for azimuth_index, azimuth in enumerate(row_angles):
                azimuth = azimuth * np.pi / 180
                if type(mask) is np.bool_:
                    if not mask:
                        elevation_valid = elevation
                else:
                    elevation_valid = list(map(elevation_validate, list(elevation), [x.flatten().any() for x in mask[azimuth_index]]))

                self.sphere_coords += list(zip(elevation_valid, [azimuth] * num_elevation_measurements))
                self.indices += list(zip(elevation_indices, [azimuth_index] * num_elevation_measurements))
            
        else:
            self.sphere_coords = sphere_coords
            self.indices = indices

        # create pandas dataframe containing sphere coordinate and indices
        self.df = pd.concat([pd.DataFrame(self.indices, columns=["elevation_index", "azimuth_index"]),
                             pd.DataFrame(self.sphere_coords, columns=["elevation", "azimuth"])], axis="columns")
        
    def get_sphere_coords(self):
        return self.sphere_coords
    
    def get_df(self):
        return self.df