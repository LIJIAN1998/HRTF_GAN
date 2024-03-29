import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset

from hrtfdata.transforms.hrirs import SphericalHarmonicsTransform

# based on https://github.com/Lornatang/SRGAN-PyTorch/blob/7292452634137d8f5d4478e44727ec1166a89125/dataset.py
def downsample_hrtf(hr_hrtf, hrtf_size, upscale_factor):
    # downsample hrtf
    if upscale_factor == hrtf_size:
        mid_pos = int(hrtf_size / 2)
        lr_hrtf = hr_hrtf[:, :, mid_pos, mid_pos, None, None]
    else:
        lr_hrtf = torch.nn.functional.interpolate(hr_hrtf, scale_factor=1 / upscale_factor)

    return lr_hrtf

def get_sample_ratio(upscale_factor):
    if upscale_factor == 2:
        return 2, 1
    if upscale_factor == 4:
        return 2, 2
    if upscale_factor == 8:
        return 4, 2
    if upscale_factor == 16:
        return 8, 2
    if upscale_factor == 32:
        return 8, 4
    if upscale_factor == 48:
        return 12, 4
    if upscale_factor == 72:
        return 12, 6
    if upscale_factor == 108:
        return 18, 6
    if upscale_factor == 216:
        return 36, 6

class CustomHRTFDataset(Dataset):
    def __init__(self, original_hrtf_dataset, upscale_factor, max_degree=28, transform=None) -> None:
        super(CustomHRTFDataset, self).__init__()
        self.original_hrtf_dataset = original_hrtf_dataset
        self.upscale_factor = upscale_factor
        self.num_row_angles, self.num_col_angles = len(self.original_hrtf_dataset.row_angles), len(self.original_hrtf_dataset.column_angles)
        self.num_radii = len(self.original_hrtf_dataset.radii)
        # self.degree = int(np.sqrt(self.num_row_angles*self.num_col_angles*self.num_radii/upscale_factor) - 1)
        if upscale_factor in [2, 4, 8]:
            self.degree = 7
        elif upscale_factor in [16 ,32, 48]:
            self.degree = 3
        elif upscale_factor in [72, 108, 216]:
            self.degree = 1
        self.max_dgree = max_degree
        self.transform = transform

    def __getitem__(self, index: int):
        hrtf = self.original_hrtf_dataset[index]['features'][:, :, :, 1:]
        sample_id = self.original_hrtf_dataset.subject_ids[index]
        original_mask = np.all(np.ma.getmaskarray(hrtf), axis=3)
        mask = np.ones((self.num_row_angles, self.num_col_angles, self.num_radii), dtype=bool)
        row_ratio, col_ratio = get_sample_ratio(self.upscale_factor)
        for i in range(self.num_row_angles // row_ratio):
            for j in range(self.num_col_angles // col_ratio):
                mask[row_ratio*i, col_ratio*j, :] = original_mask[row_ratio*i, col_ratio*j, :]
        lr_SHT = SphericalHarmonicsTransform(self.degree, self.original_hrtf_dataset.row_angles,
                                             self.original_hrtf_dataset.column_angles,
                                             self.original_hrtf_dataset.radii,
                                             mask)
        lr_coefficient = torch.from_numpy(lr_SHT(hrtf).T)
        hr_SHT = SphericalHarmonicsTransform(self.max_dgree, self.original_hrtf_dataset.row_angles,
                                             self.original_hrtf_dataset.column_angles,
                                             self.original_hrtf_dataset.radii,
                                             original_mask)
        hr_coefficient = torch.from_numpy(hr_SHT(hrtf).T)

        hrtf = torch.from_numpy(hrtf.data).permute(3, 2, 0, 1) # nbins x r x w x h
        if self.transform is not None:
            mean_lr, mean_full = self.transform[0]
            std_lr, std_full = self.transform[1]
            lr_coefficient = (lr_coefficient - mean_lr) / std_lr
            hr_coefficient = (hr_coefficient - mean_full) / std_full

        return {"lr_coefficient": lr_coefficient, "hr_coefficient": hr_coefficient, 
                "hrtf": hrtf, "mask": original_mask, "id": sample_id}
    
    def __len__(self):
        return len(self.original_hrtf_dataset)
    
class MergeHRTFDataset(Dataset):
    def __init__(self, left_hrtf, right_hrtf, upscale_factor, max_degree=28, transform=None) -> None:
        super(MergeHRTFDataset, self).__init__()
        self.left_hrtf = left_hrtf
        self.right_hrtf = right_hrtf
        self.upscale_factor = upscale_factor
        self.num_row_angles, self.num_col_angles = len(self.left_hrtf.row_angles), len(self.left_hrtf.column_angles)
        self.num_radii = len(self.left_hrtf.radii)
        self.degree = int(np.sqrt(self.num_row_angles*self.num_col_angles*self.num_radii/upscale_factor) - 1)
        # if upscale_factor == 216:
        #     self.degree = 19
        # if upscale_factor in [2, 4, 8]:
        #     self.degree = 7
        # elif upscale_factor in [16 ,32, 48]:
        #     self.degree = 3
        # elif upscale_factor in [72, 108, 216]:
        #     self.degree = 1
        self.max_degree = max_degree
        self.transform = transform

    def __getitem__(self, index: int):
        left = self.left_hrtf[index]['features'][:, :, :, 1:]
        right = self.right_hrtf[index]['features'][:, :, :, 1:]
        sample_id = self.left_hrtf.subject_ids[index]
        merge = np.ma.concatenate([left, right], axis=3)
        original_mask = np.all(np.ma.getmaskarray(left), axis=3)
        mask = np.ones((self.num_row_angles, self.num_col_angles, self.num_radii), dtype=bool)
        row_ratio, col_ratio = get_sample_ratio(self.upscale_factor)
        for i in range(self.num_row_angles // row_ratio):
            for j in range(self.num_col_angles // col_ratio):
                mask[row_ratio*i, col_ratio*j, :] = original_mask[row_ratio*i, col_ratio*j, :]
        lr_SHT = SphericalHarmonicsTransform(self.degree, self.left_hrtf.row_angles,
                                             self.left_hrtf.column_angles,
                                             self.left_hrtf.radii,
                                             mask)
        lr_coefficient = torch.from_numpy(lr_SHT(merge).T)
        hr_SHT = SphericalHarmonicsTransform(self.max_degree, self.left_hrtf.row_angles,
                                             self.left_hrtf.column_angles,
                                             self.left_hrtf.radii,
                                             original_mask)
        hr_coefficient = torch.from_numpy(hr_SHT(merge).T)

        if self.transform is not None:
            mean_lr, mean_full = self.transform[0]
            std_lr, std_full = self.transform[1]
            lr_coefficient = (lr_coefficient - mean_lr) / std_lr
            hr_coefficient = (hr_coefficient - mean_full) / std_full

        merge = torch.from_numpy(merge.data).permute(3, 2, 0, 1)  # nbins x r x w x h
        return {"lr_coefficient": lr_coefficient, "hr_coefficient": hr_coefficient,
                "hrtf": merge, "mask": original_mask, "id": sample_id}
    
    def __len__(self):
        return len(self.left_hrtf)



class TrainValidHRTFDataset(Dataset):
    """Define training/valid dataset loading methods.
    Args:
        hrtf_dir (str): Train/Valid dataset address.
        hrtf_size (int): High resolution hrtf size.
        upscale_factor (int): hrtf up scale factor.
        transform (callable): A function/transform that takes in an HRTF and returns a transformed version.
    """

    def __init__(self, hrtf_dir: str, hrtf_size: int, upscale_factor: int, transform=None, run_validation =True) -> None:
        super(TrainValidHRTFDataset, self).__init__()
        # Get all hrtf file names in folder
        self.hrtf_file_names = [os.path.join(hrtf_dir, hrtf_file_name) for hrtf_file_name in os.listdir(hrtf_dir)
                                if os.path.isfile(os.path.join(hrtf_dir, hrtf_file_name))]

        if run_validation:
            valid_hrtf_file_names = []
            for hrtf_file_name in self.hrtf_file_names:
                file = open(hrtf_file_name, 'rb')
                hrtf = pickle.load(file)
                if not np.isnan(np.sum(hrtf.cpu().data.numpy())):
                    valid_hrtf_file_names.append(hrtf_file_name)
            self.hrtf_file_names = valid_hrtf_file_names

        # Specify the high-resolution hrtf size, with equal length and width
        self.hrtf_size = hrtf_size
        # How many times the high-resolution hrtf is the low-resolution hrtf
        self.upscale_factor = upscale_factor
        # transform to be applied to the data
        self.transform = transform

    def __getitem__(self, batch_index: int) -> [torch.Tensor, torch.Tensor]:
        # Read a batch of hrtf data
        with open(self.hrtf_file_names[batch_index], "rb") as file:
            hrtf = pickle.load(file)

        # hrtf processing operations
        if self.transform is not None:
            # If using a transform, treat panels as batch dim such that dims are (panels, channels, X, Y)
            hr_hrtf = torch.permute(hrtf, (0, 3, 1, 2))
            # Then, transform hr_hrtf to normalize and swap panel/channel dims to get channels first
            hr_hrtf = torch.permute(self.transform(hr_hrtf), (1, 0, 2, 3))
        else:
            # If no transform, go directly to (channels, panels, X, Y)
            hr_hrtf = torch.permute(hrtf, (3, 0, 1, 2))

        # downsample hrtf
        lr_hrtf = downsample_hrtf(hr_hrtf, self.hrtf_size, self.upscale_factor)

        return {"lr": lr_hrtf, "hr": hr_hrtf, "filename": self.hrtf_file_names[batch_index]}

    def __len__(self) -> int:
        return len(self.hrtf_file_names)


class CPUPrefetcher:
    """Use the CPU side to accelerate data reading.
    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader) -> None:
        self.original_dataloader = dataloader
        self.data = iter(dataloader)

    def next(self):
        try:
            return next(self.data)
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader)


class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.
    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v) and k != 'mask' and k != 'id':
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)
