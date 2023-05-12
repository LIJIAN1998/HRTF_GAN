from hrtfdata.planar import CIPICPlane, ARIPlane, ListenPlane, BiLiPlane, ITAPlane, HUTUBSPlane, SADIE2Plane, ThreeDThreeAPlane, CHEDARPlane, WidespreadPlane, SONICOMPlane
from hrtfdata import HRTFDataset
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

base_dir = Path('/rds/general/user/jl2622/projects/sonicom/live/HRTF Datasets')

plane = 'median'
domain = 'magnitude_db'
side = 'left'

ds = SONICOMPlane(base_dir/'SONICOM', plane, domain, side)

print("is a class of HRTFDataset? ", isinstance(ds, HRTFDataset))
print("length of datset: ", len(ds))

p = ds[0]
print("keys of a datapoint: ", p.keys())