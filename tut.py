from hrtfdata.planar import CIPICPlane, ARIPlane, ListenPlane, BiLiPlane, ITAPlane, HUTUBSPlane, SADIE2Plane, ThreeDThreeAPlane, CHEDARPlane, WidespreadPlane, SONICOMPlane
from hrtfdata import HRTFDataset
from hrtfdata.full import CIPIC, ARI, Listen, BiLi, ITA, HUTUBS, SADIE2, ThreeDThreeA, CHEDAR, Widespread, SONICOM
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

base_dir = Path('/rds/general/user/jl2622/projects/sonicom/live/HRTF Datasets')

plane = 'median'
domain = 'magnitude_db'
side = 'left'
subject_ids= 'first'

# ds = SONICOMPlane(base_dir/'SONICOM', plane, domain, side)

# print("is a class of HRTFDataset? ", isinstance(ds, HRTFDataset))
# print("length of datset: ", len(ds))

# p = ds[0]
# print("keys of a datapoint: ", p.keys())

domain = 'magnitude_db'
side = 'left'
ds = SONICOM(base_dir / 'SONICOM',  feature_spec={'hrirs': {'side': side, 'domain': domain}}, 
         target_spec={'side': {}}, group_spec={'subject': {}})
print("length of datset: ", len(ds))
p = ds[0]
print("keys of a datapoint: ", p.keys())
print("target: ", ds[0]['target'])
print("group: ", ds[0]['group'])

ds = SONICOM(base_dir / 'SONICOM',  feature_spec={'hrirs': {'side': side, 'domain': domain}}, target_spec={'side': {}}, group_spec={'subject': {}}, subject_ids='random')
print("number of random sample: ", len(ds))
print("id of random sample: ", ds.subject_ids)
print("available subject ids: ", ds.available_subject_ids[:10])
print("shape of a datapoint feature: ", ds[0]['features'].shape)
print("num row, column angles:", len(ds.row_angles), len(ds.column_angles))
print("row, column angles: ", ds.row_angles, ds.column_angles)
