from functools import partial
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from model.util import *
from hrtfdata.transforms.hrirs import SphericalHarmonicsTransform

import importlib
from model.model import VAE, Discriminator

from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler

print("import done!")
print("using cuda? ", torch.cuda.is_available())

def optim_hyperparameter(config):
    data_dir = config.raw_hrtf_dir / config.dataset
    imp = importlib.import_module('hrtfdata.full')
    load_function = getattr(imp, config.dataset)
    ds = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate,
                                                         'side': 'left', 'domain': 'time'}}, subject_ids='first')
    num_row_angles = len(ds.row_angles)
    num_col_angles = len(ds.column_angles)
    num_radii = len(ds.radii)

    ngpu = config.ngpu
    path = config.path

    nbins = config.nbins_hrtf
    if config.merge_flag:
        nbins = config.nbins_hrtf * 2

    device = torch.device(config.device_name if (
            torch.cuda.is_available() and ngpu > 0) else "cpu")
    print(f'Using {ngpu} GPUs')
    print(device, " will be used.\n")
    cudnn.benchmark = True

    degree = int(np.sqrt(num_row_angles*num_col_angles*num_radii/config.upscale_factor) - 1)
    vae = VAE(nbins=nbins, max_degree=degree, latent_dim=10).to(device)
    netD = Discriminator(nbins=nbins).to(device)
    if ('cuda' in str(device)) and (ngpu > 1):
        netD = (nn.DataParallel(netD, list(range(ngpu)))).to(device)
        vae = nn.DataParallel(vae, list(range(ngpu))).to(device)
     
    beta1, beta2 = config.beta1, config.beta2
    optEncoder = optim.SGD(vae.encoder.parameters(), lr=config["lr"], momentum=0.9)
    optDecoder = optim.SGD(vae.decoder.parameters(), lr=config["lr"], momentum=0.9)
    optD = optim.SGD(netD.parameters(), lr=config["lr"], betas=(beta1, beta2))

    # Define loss functions
    adversarial_criterion = nn.BCEWithLogitsLoss()
    content_criterion = sd_ild_loss

    sd_mean = 7.387559253346883
    sd_std = 0.577364154400081
    ild_mean = 3.6508303231127868
    ild_std = 0.5261339271318863

    checkpoint = session.get_checkpoint()
    if checkpoint:
        checkpoint_state = checkpoint.to_dict()
        start_epoch = checkpoint_state["epoch"]
        optEncoder.load_state_dict(checkpoint_state["optEncoder_state_dict"])
    else:
        start_epoch = 0

    train_prefetcher, test_prefetcher = load_hrtf(config)

    critic_iters = config["critic_iters"]

    for epoch in range(start_epoch, 200):
        train_loss_Dis = 0.
        train_loss_Dis_hr = 0.
        train_loss_Dis_recon = 0.
        train_loss_Dec = 0.
        train_loss_Dec_gan = 0.
        train_loss_Dec_sim = 0.
        train_loss_Dec_content = 0.
        train_loss_Enc = 0.
        train_loss_Enc_prior = 0.
        train_loss_Enc_sim = 0.

        batch_index = 0

        train_prefetcher.reset()
        batch_data = train_prefetcher.next()

        while batch_data is not None:
            lr_coefficient = batch_data["lr_coefficient"].to(device=device, memory_format=torch.contiguous_format,
                                                             non_blocking=True, dtype=torch.float)
            hr_coefficient = batch_data["hr_coefficient"].to(device=device, memory_format=torch.contiguous_format,
                                                             non_blocking=True, dtype=torch.float)
            hrtf = batch_data["hrtf"].to(device=device, memory_format=torch.contiguous_format,
                                         non_blocking=True, dtype=torch.float)
            masks = batch_data["mask"]
            
            bs = lr_coefficient.size(0)
            ones_label = Variable(torch.ones(bs,1)).to(device) # labels for real data
            zeros_label = Variable(torch.zeros(bs,1)).to(device) # labels for generated data

            # Generate fake samples using VAE
            mu, log_var, recon = vae(lr_coefficient)

            bs = lr_coefficient.size(0)
            ones_label = Variable(torch.ones(bs,1)).to(device) # labels for real data
            zeros_label = Variable(torch.zeros(bs,1)).to(device) # labels for generated data

            # Generate fake samples using VAE
            mu, log_var, recon = vae(lr_coefficient)

            # Discriminator Training
            # train on real coefficient
            pred_real = netD(hr_coefficient)[0]
            loss_D_hr = adversarial_criterion(pred_real, ones_label)
            train_loss_Dis_hr += loss_D_hr.item()
            # train on reconstructed coefficient 
            pred_recon = netD(recon.detach().clone())[0]
            loss_D_recon = adversarial_criterion(pred_recon, zeros_label)
            train_loss_Dis_recon += loss_D_recon.item()
            # Compute the discriminator loss
            gan_loss = loss_D_hr + loss_D_recon
            train_loss_Dis += gan_loss.item()
            # Update D
            netD.zero_grad()
            gan_loss.backward()
            optD.step()

            # training VAE
            if batch_index % int(critic_iters) == 0:
                # train decoder
                pred_real, feature_real = netD(hr_coefficient)
                err_dec_real = adversarial_criterion(pred_real, ones_label)
                pred_recon, feature_recon = netD(recon)
                err_dec_recon = adversarial_criterion(pred_recon, zeros_label)
                gan_loss_dec = err_dec_real + err_dec_recon
                train_loss_Dec_gan += gan_loss_dec.item() # gan / adversarial loss
                feature_sim_loss_D = config.gamma * ((feature_recon - feature_real) ** 2).mean() # feature loss
                train_loss_Dec_sim += feature_sim_loss_D.item()
                # convert reconstructed coefficient back to hrtf
                harmonics_list = []
                for i in range(masks.size(0)):
                    SHT = SphericalHarmonicsTransform(28, ds.row_angles, ds.column_angles, ds.radii, masks[i].numpy().astype(bool))
                    harmonics = torch.from_numpy(SHT.get_harmonics()).float()
                    harmonics_list.append(harmonics)
                harmonics_tensor = torch.stack(harmonics_list).to(device)
                recons = harmonics_tensor @ recon.permute(0, 2, 1)
                recons = torch.abs(recons.reshape(bs, nbins, num_radii, num_row_angles, num_col_angles))
                unweighted_content_loss = content_criterion(config, recons, hrtf, sd_mean, sd_std, ild_mean, ild_std)
                content_loss = config.content_weight * unweighted_content_loss
                train_loss_Dec_content += content_loss.item()
                err_dec = feature_sim_loss_D + content_loss - gan_loss_dec
                train_loss_Dec += err_dec.item()
                # Update decoder
                optDecoder.zero_grad()
                err_dec.backward()
                optDecoder.step()

                # train encoder
                mu, log_var, recon = vae(lr_coefficient)
                prior_loss = 1 + log_var - mu.pow(2) - log_var.exp()
                prior_loss = (-0.5 * torch.sum(prior_loss))/torch.numel(mu.data) # prior loss
                train_loss_Enc_prior += prior_loss.item()
                feature_recon = netD(recon)[1]
                feature_real = netD(hr_coefficient)[1]
                feature_sim_loss_E = config.beta * ((feature_recon - feature_real) ** 2).mean() # feature loss
                train_loss_Enc_sim += feature_sim_loss_E.item()
                err_enc = prior_loss + feature_sim_loss_E
                train_loss_Enc += err_enc.item()
                # Update encoder
                optEncoder.zero_grad()
                err_enc.backward()
                optEncoder.step()


    
