import pickle

import scipy

import importlib

from model.util import *
from model.model import *

import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import time

from plot import plot_losses, plot_magnitude_spectrums

from hrtfdata.transforms.hrirs import SphericalHarmonicsTransform

def train(config, train_prefetcher):
    """ Train the generator and discriminator models

    :param config: Config object containing model hyperparameters
    :param train_prefetcher: prefetcher for training data
    """
    data_dir = config.raw_hrtf_dir / config.dataset
    imp = importlib.import_module('hrtfdata.full')
    load_function = getattr(imp, config.dataset)
    ds = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate,
                                                         'side': 'left', 'domain': 'time'}}, subject_ids='first')
    # Calculate how many batches of data are in each Epoch
    batches = len(train_prefetcher)

    # Assign torch device
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

    # Get train params
    batch_size, beta1, beta2, num_epochs, lr_encoder, lr_decoder, lr_dis, critic_iters = config.get_train_params()

    # get list of positive frequencies of HRTF for plotting magnitude spectrum
    all_freqs = scipy.fft.fftfreq(256, 1 / config.hrir_samplerate)
    pos_freqs = all_freqs[all_freqs >= 0]

    # Define Generator network and transfer to CUDA
    degree = int(np.sqrt(72*12)/config.upscale_factor - 1)
    vae = VAE(nbins=nbins, max_degree=degree, latent_dim=10).to(device)
    # netG = Generator(upscale_factor=config.upscale_factor, nbins=nbins).to(device)
    netD = Discriminator(nbins=nbins).to(device)
    if ('cuda' in str(device)) and (ngpu > 1):
        netD = (nn.DataParallel(netD, list(range(ngpu)))).to(device)
        vae = nn.DataParallel(vae, list(range(ngpu))).to(device)
        # netG = nn.DataParallel(netG, list(range(ngpu))).to(device)

    # Define optimizers
    optD = optim.Adam(netD.parameters(), lr=lr_dis, betas=(beta1, beta2))
    optEncoder = optim.Adam(vae.encoder.parameters(), lr=lr_encoder, betas=(beta1, beta2))
    optDecoder = optim.Adam(vae.decoder.parameters(), lr=lr_decoder, betas=(beta1, beta2))
    # optG = optim.Adam(netG.parameters(), lr=lr_gen, betas=(beta1, beta2))

    # Define loss functions
    adversarial_criterion = nn.BCEWithLogitsLoss()
    content_criterion = sd_ild_loss

    # mean and std for ILD and SD, which are used for normalization
    # computed based on average ILD and SD for training data, when comparing each individual
    # to every other individual in the training data
    sd_mean = 7.387559253346883
    sd_std = 0.577364154400081
    ild_mean = 3.6508303231127868
    ild_std = 0.5261339271318863

    if config.start_with_existing_model:
        print(f'Initialized weights using an existing model - {config.existing_model_path}')
        vae.load_state_dict(torch.load(f'{config.existing_model_path}/Vae.pt'))
        netD.load_state_dict(torch.load(f'{config.existing_model_path}/Disc.pt'))

    train_losses_G = []
    train_losses_G_adversarial = []
    train_losses_G_content = []
    train_losses_D = []
    train_losses_D_hr = []
    train_losses_D_sr = []

    train_SD_metric = []

    for epoch in range(num_epochs):
        times = []
        train_loss_G = 0.
        train_loss_G_adversarial = 0.
        train_loss_G_content = 0.
        train_loss_D = 0.
        train_loss_D_hr = 0.
        train_loss_D_recon = 0.

        prior_los_list, vae_loss_list, recon_loss_list = [], [], []
        dis_real_list, dis_fake_list, dis_prior_list = [], [], []

        # Initialize the number of data batches to print logs on the terminal
        batch_index = 0

        # Initialize the data loader and load the first batch of data
        train_prefetcher.reset()
        batch_data = train_prefetcher.next()

        while batch_data is not None:
            if ('cuda' in str(device)) and (ngpu > 1):
                start_overall = torch.cuda.Event(enable_timing=True)
                end_overall = torch.cuda.Event(enable_timing=True)
                start_overall.record()
            else:
                start_overall = time.time()

            # Transfer in-memory data to CUDA devices to speed up training
            lr_coefficient = batch_data["lr_coefficient"].to(device=device, memory_format=torch.contiguous_format,
                                                             non_blocking=True, dtype=torch.float)
            hr_coefficient = batch_data["hr_coefficient"].to(device=device, memory_format=torch.contiguous_format,
                                                             non_blocking=True, dtype=torch.float)
            hrir = batch_data["hrir"].to(device=device, memory_format=torch.contiguous_format,
                                         non_blocking=True, dtype=torch.float)
            masks = batch_data["mask"].to(device=device, memory_format=torch.contiguous_format,
                                         non_blocking=True, dtype=torch.float)
            
            bs = lr_coefficient.size(0)
            ones_label = Variable(torch.ones(bs,1)).to(device) # labels for real data
            zeros_label = Variable(torch.zeros(bs,1)).to(device) # labels for generated data

            # Generate fake samples using VAE
            mu, log_var, recon = vae(lr_coefficient)

            # during every 25th epoch and last epoch, save filename for mag spectrum plot
            if epoch % 25 == 0 or epoch == (num_epochs - 1):
                filename = batch_data["filename"]

            # Discriminator Training
            # Initialize the discriminator model gradients
            
            # train on real coefficient
            pred_real = netD(hr_coefficient)[0]
            loss_D_hr = adversarial_criterion(pred_real, ones_label)
            # train on reconstructed coefficient 
            pred_recon = netD(recon.detach().clone())[0]
            loss_D_recon = adversarial_criterion(pred_recon, zeros_label)
            # Compute the discriminator loss
            gan_loss = loss_D_hr + loss_D_recon
            # Update D
            netD.zero_grad()
            gan_loss.backward(retain_graph=True)
            optD.step()
            
            loss_D = loss_D_hr + loss_D_recon
            train_loss_D += loss_D.item()
            train_loss_D_hr += loss_D_hr.item()
            train_loss_D_recon += loss_D_recon.item()

            # train decoder
            pred_real, feature_real = netD(hr_coefficient)
            errD_real = adversarial_criterion(output, ones_label)
            pred_recon, feature_recon = netD(recon)
            errD_recon = adversarial_criterion(pred_recon, zeros_label)
            gan_loss = errD_real + errD_recon
            feature_sim_loss = ((feature_recon - feature_real) ** 2).mean()
            # convert reconstructed coefficient back to hrir
            recon_coef_list = []
            for i, mask in enumerate(masks):
                SHT = SphericalHarmonicsTransform(28, ds.row_angles, ds.column_angles, ds.radii, mask[i])
                recon_coef_list.append(torch.from_numpy(SHT.inverse(recon[i].T)))
            recons = torch.stack(recon_coef_list)
            unweighted_content_loss = content_criterion(config)


            # Generator training
            if batch_index % int(critic_iters) == 0:
                # Initialize generator model gradients
                netG.zero_grad()
                sr = netG(lr)
                label.fill_(1.)
                # Calculate adversarial loss
                output = netD(sr).view(-1)

                unweighted_content_loss_G = content_criterion(config, sr, hr, sd_mean, sd_std, ild_mean, ild_std)
                content_loss_G = config.content_weight * unweighted_content_loss_G
                adversarial_loss_G = config.adversarial_weight * adversarial_criterion(output, label)

                # Calculate the generator total loss value and backprop
                loss_G = content_loss_G + adversarial_loss_G
                loss_G.backward()

                train_loss_G += loss_G.item()
                train_loss_G_adversarial += adversarial_loss_G.item()
                train_loss_G_content += content_loss_G.item()
                train_SD_metric.append(unweighted_content_loss_G.item())

                optG.step()

            if ('cuda' in str(device)) and (ngpu > 1):
                end_overall.record()
                torch.cuda.synchronize()
                times.append(start_overall.elapsed_time(end_overall))
            else:
                end_overall = time.time()
                times.append(end_overall - start_overall)

            # Every 0th batch log useful metrics
            if batch_index == 0:
                with torch.no_grad():
                    torch.save(netG.state_dict(), f'{path}/Gen.pt')
                    torch.save(netD.state_dict(), f'{path}/Disc.pt')

                    progress(batch_index, batches, epoch, num_epochs, timed=np.mean(times))
                    times = []

            # Preload the next batch of data
            batch_data = train_prefetcher.next()

            # After training a batch of data, add 1 to the number of data batches to ensure that the
            # terminal print data normally
            batch_index += 1

        train_losses_D.append(train_loss_D / len(train_prefetcher))
        train_losses_D_hr.append(train_loss_D_hr / len(train_prefetcher))
        train_losses_D_sr.append(train_loss_D_sr / len(train_prefetcher))
        train_losses_G.append(train_loss_G / len(train_prefetcher))
        train_losses_G_adversarial.append(train_loss_G_adversarial / len(train_prefetcher))
        train_losses_G_content.append(train_loss_G_content / len(train_prefetcher))
        print(f"Average epoch loss, discriminator: {train_losses_D[-1]}, generator: {train_losses_G[-1]}")
        print(f"Average epoch loss, D_real: {train_losses_D_hr[-1]}, D_fake: {train_losses_D_sr[-1]}")
        print(f"Average epoch loss, G_adv: {train_losses_G_adversarial[-1]}, train_losses_G_content: {train_losses_G_content[-1]}")

        # create magnitude spectrum plot every 25 epochs and last epoch
        if epoch % 25 == 0 or epoch == (num_epochs - 1):
            i_plot = 0
            magnitudes_real = torch.permute(hr.detach().cpu()[i_plot], (1, 2, 3, 0))
            magnitudes_interpolated = torch.permute(sr.detach().cpu()[i_plot], (1, 2, 3, 0))

            plot_label = filename[i_plot].split('/')[-1] + '_epoch' + str(epoch)
            plot_magnitude_spectrums(pos_freqs, magnitudes_real[:, :, :, :config.nbins_hrtf], magnitudes_interpolated[:, :, :, :config.nbins_hrtf],
                                     "left", "training", plot_label, path, log_scale_magnitudes=True)

    plot_losses(train_losses_D, train_losses_G,
                label_1='Discriminator loss', label_2='Generator loss',
                color_1="#5ec962", color_2="#440154",
                path=path, filename='loss_curves', title="Loss curves")
    plot_losses(train_losses_D_hr, train_losses_D_sr,
                label_1='Discriminator loss, real', label_2='Discriminator loss, fake',
                color_1="#b5de2b", color_2="#1f9e89",
                path=path, filename='loss_curves_D', title="Discriminator loss curves")
    plot_losses(train_losses_G_adversarial, train_losses_G_content,
                label_1='Generator loss, adversarial', label_2='Generator loss, content',
                color_1="#31688e", color_2="#440154",
                path=path, filename='loss_curves_G', title="Generator loss curves")

    with open(f'{path}/train_losses.pickle', "wb") as file:
        pickle.dump((train_losses_G, train_losses_G_adversarial, train_losses_G_content,
                     train_losses_D, train_losses_D_hr, train_losses_D_sr, train_SD_metric), file)

    print("TRAINING FINISHED")
