import json
from pathlib import Path


class Config:
    """Config class

    Set using HPC to true in order to use appropriate paths for HPC
    """

    def __init__(self, tag, using_hpc, dataset=None, existing_model_tag=None, data_dir=None):

        # overwrite settings with arguments provided
        self.tag = tag if tag is not None else 'pub-prep-upscale-sonicom-sonicom-synthetic-tl-2'
        self.dataset = dataset if dataset is not None else 'SONICOM'
        self.data_dir = data_dir if data_dir is not None else '/data/' + self.dataset

        if existing_model_tag is not None:
            self.start_with_existing_model = True
        else:
            self.start_with_existing_model = False

        self.existing_model_tag = existing_model_tag if existing_model_tag is not None else None

        # Data processing parameters
        self.merge_flag = True
        self.gen_sofa_flag = True
        self.nbins_hrtf = 128  # make this a power of 2
        self.hrtf_size = 16
        self.upscale_factor = 2  # can only take values: 2, 4 ,8, 16
        self.train_samples_ratio = 0.8
        self.hrir_samplerate = 48000.0

        # Data dirs
        if using_hpc:
            # HPC data dirs
            self.data_dirs_path = '/rds/general/user/jl2622/home/HRTF-projection'
            self.raw_hrtf_dir = '/rds/general/project/sonicom/live/HRTF Datasets'
            self.amt_dir = '/rds/general/user/jl2622/home/amt'
        else:
            # local data dirs
            self.data_dirs_path = '/home/aos13/HRTF-GANs-27Sep22-prep-for-publication'
            self.raw_hrtf_dir = '/home/aos13/HRTF_datasets'
            self.amt_dir = '/home/aos13/AMT/amt_code'

        self.runs_folder = '/runs-hpc'
        self.path = f'{self.data_dirs_path}{self.runs_folder}/{self.tag}'
        self.existing_model_path = f'{self.data_dirs_path}{self.runs_folder}/{self.existing_model_tag}'

        self.valid_path = f'{self.data_dirs_path}{self.runs_folder}/{self.tag}/valid'
        self.valid_gt_path = f'{self.data_dirs_path}{self.runs_folder}/{self.tag}/valid_gt'
        self.model_path = f'{self.data_dirs_path}{self.runs_folder}/{self.tag}'

        self.projection_dir = f'{self.data_dirs_path}/projection_coordinates'
        self.baseline_dir = '/baseline_results/' + self.dataset

        self.train_val_id_dir = self.data_dirs_path + self.data_dir + '/train_val_id'

        self.train_hrtf_dir = self.data_dirs_path + self.data_dir + '/hr/train'
        self.valid_hrtf_dir = self.data_dirs_path + self.data_dir + '/hr/valid'
        self.train_original_hrtf_dir = self.data_dirs_path + self.data_dir + '/original/train'
        self.valid_original_hrtf_dir = self.data_dirs_path + self.data_dir + '/original/valid'

        self.train_hrtf_merge_dir = self.data_dirs_path + self.data_dir + '/hr_merge/train'
        self.valid_hrtf_merge_dir = self.data_dirs_path + self.data_dir + '/hr_merge/valid'
        self.train_original_hrtf_merge_dir = self.data_dirs_path + self.data_dir + '/merge_original/train'
        self.valid_original_hrtf_merge_dir = self.data_dirs_path + self.data_dir + '/merge_original/valid'

        self.mean_std_filename = self.data_dirs_path + self.data_dir + '/mean_std_' + self.dataset
        self.barycentric_hrtf_dir = self.data_dirs_path + self.baseline_dir + '/barycentric/valid'
        self.hrtf_selection_dir = self.data_dirs_path + self.baseline_dir + '/hrtf_selection/valid'

        # Training hyperparams
        self.batch_size = 4
        self.num_workers = 1
        self.optimizer = 'adam'
        self.num_epochs = 200  # was originally 250
        self.lr = 0.0001
        self.alpha = 0.01
        # self.lr_encoder = 0.0002
        # self.lr_decoder = 0.0002
        # self.lr_dis = 0.0000015
        self.latent_dim = 10
        # how often to train the generator
        self.critic_iters = 4

        # Loss function weight
        self.content_weight = 0.01
        self.adversarial_weight = 0.01
        self.gamma = 15
        self.beta = 5

        # betas for Adam optimizer
        self.beta1 = 0.9
        self.beta2 = 0.999

        self.ngpu = 1
        if self.ngpu > 0:
            self.device_name = "cuda:0"
        else:
            self.device_name = 'cpu'

    def save(self, n):
        j = {}
        for k, v in self.__dict__.items():
            j[k] = v
        save_path = ""
        with open(f'{self.path}/config_files/config_{n}.json', 'w') as f:
            json.dump(list(j), f)

    def load(self, n):
        with open(f'{self.path}/config_files/config_{n}.json', 'r') as f:
            j = json.load(f)
            for k, v in j.items():
                setattr(self, k, v)
            self.raw_hrtf_dir = Path(self.raw_hrtf_dir)

    def get_train_params(self):
        return self.batch_size, self.beta1, self.beta2, self.num_epochs, self.lr_encoder, self.lr_decoder, self.lr_dis, self.critic_iters
