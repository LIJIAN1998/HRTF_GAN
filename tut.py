import torch
from hrtfdata.planar import CIPICPlane, ARIPlane, ListenPlane, BiLiPlane, ITAPlane, HUTUBSPlane, SADIE2Plane, ThreeDThreeAPlane, CHEDARPlane, WidespreadPlane, SONICOMPlane
from hrtfdata import HRTFDataset
from hrtfdata.full import CIPIC, ARI, Listen, BiLi, ITA, HUTUBS, SADIE2, ThreeDThreeA, CHEDAR, Widespread, SONICOM
from hrtfdata.torch import collate_dict_dataset
from hrtfdata.planar import CIPICPlane, ARIPlane
from torch.utils.data import DataLoader, ConcatDataset, Subset
from pathlib import Path
import matplotlib.pyplot as plt
from hrtfdata.transforms.hrirs import SphericalHarmonicsTransform
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

domain = 'time'
# side_options = ['left', 'right', 'both', 'both-left', 'both-right']
# for side in side_options:
#     print("side: ", side)
#     sonicom_ds = SONICOM(base_dir / 'SONICOM',  feature_spec={'hrirs': {'side': side, 'domain': domain}}, 
#             target_spec={'side': {}}, group_spec={'subject': {}})
#     print("length of datset: ", len(sonicom_ds))
#     p = sonicom_ds[0]
#     print("target: ", sonicom_ds[0]['target'])
#     print("group: ", sonicom_ds[0]['group'])
#     print("num row, column angles:", len(sonicom_ds.row_angles), len(sonicom_ds.column_angles))
#     print("row, column angles: ", sonicom_ds.row_angles, sonicom_ds.column_angles)
#     print("radii: ", sonicom_ds.radii)
#     print("---------------------------------------------------------------")

# ds = SONICOM(base_dir / 'SONICOM',  feature_spec={'hrirs': {'side': side, 'domain': domain}}, 
#              target_spec={'side': {}}, group_spec={'subject': {}}, subject_ids='random')
# print("number of random sample: ", len(ds))
# print("id of random sample: ", ds.subject_ids)
# print("available subject ids: ", ds.available_subject_ids[:10])
# print("shape of a datapoint feature: ", ds[0]['features'].shape)
# print("num row, column angles:", len(ds.row_angles), len(ds.column_angles))
# print("row, column angles: ", ds.row_angles, ds.column_angles)
# print("radii: ", ds.radii)

sonicom_ds = SONICOM(base_dir / 'SONICOM',  feature_spec={'hrirs': {'side': 'left', 'domain': domain}}, 
            target_spec={'side': {}}, group_spec={'subject': {}})
sonicom_loader = DataLoader(sonicom_ds, collate_fn=collate_dict_dataset)
features, target = next(iter(sonicom_loader))
print("data from data loader, shape: ", features.shape)
print("target: ", target)

mask = torch.ones(len(sonicom_ds.row_angles), len(sonicom_ds.column_angles), dtype=torch.int32)
SHTransform = SphericalHarmonicsTransform(max_degree=10, row_angles=sonicom_ds.row_angles, column_angles=sonicom_ds.column_angles,
                                          radii=sonicom_ds.radii, selection_mask=mask, coordinate_system='spherical')

sphericalHarmonics = SHTransform(features[0])
print("spherical harmonics shape: ", sphericalHarmonics.shape)
print("finished")
# hrir = SHTransform.inverse(sphericalHarmonics)
# print("reverse SH transform: ", hrir.shape)