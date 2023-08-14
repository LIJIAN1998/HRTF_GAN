import pickle

import scipy

import importlib

from model.util import *
# from model.model import *
from model.ae import *

import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import time

from plot import plot_losses, plot_magnitude_spectrums, plot_hrtf

from hrtfdata.transforms.hrirs import SphericalHarmonicsTransform

def test_train(config, train_prefetcher):
    # load the dataset to get the row, column angles info
    data_dir = config.raw_hrtf_dir / config.dataset
    imp = importlib.import_module('hrtfdata.full')
    load_function = getattr(imp, config.dataset)
    ds = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate,
                                                         'side': 'left', 'domain': 'time'}}, subject_ids='first')
    num_row_angles = len(ds.row_angles)
    num_col_angles = len(ds.column_angles)
    num_radii = len(ds.radii)
    
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

    # # get list of positive frequencies of HRTF for plotting magnitude spectrum
    # all_freqs = scipy.fft.fftfreq(256, 1 / config.hrir_samplerate)
    # pos_freqs = all_freqs[all_freqs >= 0]

    # Define VAE and transfer to CUDA
    degree = int(np.sqrt(num_row_angles*num_col_angles*num_radii/config.upscale_factor) - 1)
    vae = VAE(nbins=nbins, max_degree=degree, latent_dim=10).to(device)
    netD = Discriminator(nbins=nbins).to(device)
    if ('cuda' in str(device)) and (ngpu > 1):
        netD = (nn.DataParallel(netD, list(range(ngpu)))).to(device)
        vae = nn.DataParallel(vae, list(range(ngpu))).to(device)

    # Define optimizers
    optD = optim.Adam(netD.parameters(), lr=lr_dis, betas=(beta1, beta2))
    optEncoder = optim.Adam(vae.encoder.parameters(), lr=lr_encoder, betas=(beta1, beta2))
    optDecoder = optim.Adam(vae.decoder.parameters(), lr=lr_decoder, betas=(beta1, beta2))

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

    train_loss_Dec = 0.
    train_loss_Dec_gan = 0.
    train_loss_Dec_sim = 0.
    train_loss_Dec_content = 0.

    batch_data = train_prefetcher.next()

    lr_coefficient = batch_data["lr_coefficient"].to(device=device, memory_format=torch.contiguous_format,
                                                             non_blocking=True, dtype=torch.float)
    hr_coefficient = batch_data["hr_coefficient"].to(device=device, memory_format=torch.contiguous_format,
                                                        non_blocking=True, dtype=torch.float)
    hrir = batch_data["hrtf"].to(device=device, memory_format=torch.contiguous_format,
                                    non_blocking=True, dtype=torch.float)
    masks = batch_data["mask"]
    
    bs = lr_coefficient.size(0)
    ones_label = Variable(torch.ones(bs,1)).to(device) # labels for real data
    zeros_label = Variable(torch.zeros(bs,1)).to(device) # labels for generated data

    mu, log_var, recon = vae(lr_coefficient)
    with open('train_log.txt', "a") as f:
        f.write(f"recon negative?: {(recon<0).any()}\n")

    # train decoder
    pred_real, feature_real = netD(hr_coefficient)
    err_dec_real = adversarial_criterion(pred_real, ones_label)
    pred_recon, feature_recon = netD(recon)
    err_dec_recon = adversarial_criterion(pred_recon, zeros_label)
    gan_loss_dec = err_dec_real + err_dec_recon
    train_loss_Dec_gan += gan_loss_dec.item() # gan / adversarial loss
    feature_sim_loss_D = config.gamma * ((feature_recon - feature_real) ** 2).mean() # feature loss
    with open('train_log.txt', "a") as f:
        f.write(f"sim loss D: {feature_sim_loss_D}\n")
        if torch.isnan(feature_recon).all():
            f.write("all feature recon is nan\n")
        elif torch.isnan(feature_recon).any():
            f.write("feature recon has some nan\n")
        if torch.isnan(feature_real).all():
            f.write("all feature real is nan\n")
        elif torch.isnan(feature_real).any():
            f.write("feature real has some nan\n")
    train_loss_Dec_sim += feature_sim_loss_D.item()
    # convert reconstructed coefficient back to hrtf
    harmonics_list = []
    for i in range(masks.size(0)):
        SHT = SphericalHarmonicsTransform(45, ds.row_angles, ds.column_angles, ds.radii, masks[i].numpy().astype(bool))
        harmonics = torch.from_numpy(SHT.get_harmonics()).float()
        harmonics_list.append(harmonics)
    harmonics_tensor = torch.stack(harmonics_list).to(device)
    print("any negative recon? ", (recon < 0).any())
    print("any negative harmonics? ", (harmonics < 0).any())
    recons = harmonics_tensor @ recon.permute(0, 2, 1)
    print("any negative result? ", (recons < 0).any())
    with open('train_log.txt', "a") as f:
        f.write(f"inverse transformation negative?: {(recons<0).any()}\n")
    recons = F.softplus(recons.reshape(bs, nbins, num_radii, num_row_angles, num_col_angles)) 
    unweighted_content_loss = content_criterion(config, recons, hrir, sd_mean, sd_std, ild_mean, ild_std)
    with open('train_log.txt', "a") as f:
        f.write(f"after softplus, negative? {(recons<0).any()}\n")
        f.write(f"unweighted_content_loss: {unweighted_content_loss}\n")
    content_loss = config.content_weight * unweighted_content_loss
    train_loss_Dec_content += content_loss
    err_dec = feature_sim_loss_D - gan_loss_dec
    train_loss_Dec += err_dec


def train(config, train_prefetcher):
    """ Train the generator and discriminator models

    :param config: Config object containing model hyperparameters
    :param train_prefetcher: prefetcher for training data
    """
    # load the dataset to get the row, column angles info
    domain = config.domain
    data_dir = config.raw_hrtf_dir / config.dataset
    imp = importlib.import_module('hrtfdata.full')
    load_function = getattr(imp, config.dataset)
    ds = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate,
                                                         'side': 'left', 'domain': domain}}, subject_ids='first')
    num_row_angles = len(ds.row_angles)
    num_col_angles = len(ds.column_angles)
    num_radii = len(ds.radii)
    
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
    bs, optmizer, lr, alpha, lambda_feature, latent_dim, critic_iters = config.get_train_params()
    decay_lr = config.decay_lr
    upscale_factor = config.upscale_factor
    max_order = config.max_order

    # # get list of positive frequencies of HRTF for plotting magnitude spectrum
    # all_freqs = scipy.fft.fftfreq(256, 1 / config.hrir_samplerate)
    # pos_freqs = all_freqs[all_freqs >= 0]

    # Define VAE and transfer to CUDA
    in_order = int(np.sqrt(num_row_angles*num_col_angles*num_radii/config.upscale_factor) - 1)
    netG = AutoEncoder(nbins=nbins, in_order=in_order, latent_dim=latent_dim, out_oder=max_order).to(device)
    # netG = D_DBPN(nbins, base_channels=256, num_features=512, scale_factor=upscale_factor, max_order=max_order).to(device)
    # vae = VAE(nbins=nbins, max_degree=in_order, latent_dim=latent_dim).to(device)
    netD = Discriminator(nbins=nbins).to(device)
    if ('cuda' in str(device)) and (ngpu > 1):
        netD = (nn.DataParallel(netD, list(range(ngpu)))).to(device)
        netG = nn.DataParallel(netG, list(range(ngpu))).to(device)
        # vae = nn.DataParallel(vae, list(range(ngpu))).to(device)

    # Define optimizers
    optD = optim.Adam(netD.parameters(), lr=0.0001)
    optG = optim.Adam(netG.parameters(), lr=0.001)
    scheduler_D = ExponentialLR(optD, gamma=decay_lr)
    scheduler_G = ExponentialLR(optG, gamma=decay_lr)
    # optD = optim.Adam(netD.parameters(), lr=lr*alpha)
    # optEncoder = optim.Adam(vae.encoder.parameters(), lr=lr)
    # optDecoder = optim.Adam(vae.decoder.parameters(), lr=lr)
    # optEncoder = optim.RMSprop(vae.encoder.parameters(), lr=lr, alpha=0.9, eps=1e-8, weight_decay=0, momentum=0, centered=False)
    # lr_encoder = ExponentialLR(optEncoder, gamma=decay_lr)
    # optDecoder = optim.RMSprop(vae.decoder.parameters(), lr=lr, alpha=0.9, eps=1e-8, weight_decay=0, momentum=0, centered=False)
    # lr_decoder = ExponentialLR(optDecoder, gamma=decay_lr)
    # optD = optim.RMSprop(netD.parameters(), lr=0.0000015, alpha=0.9, eps=1e-8, weight_decay=0, momentum=0, centered=False)
    # lr_discriminator = ExponentialLR(optD, gamma=decay_lr)

    # Define loss functions
    adversarial_criterion = nn.BCEWithLogitsLoss()
    cos_similarity_criterion = nn.CosineSimilarity(dim=2)
    content_criterion = sd_ild_loss

    # mean and std for ILD and SD, which are used for normalization
    # computed based on average ILD and SD for training data, when comparing each individual
    # to every other individual in the training data
    sd_mean = 7.387559253346883
    sd_std = 0.577364154400081
    ild_mean = 3.6508303231127868
    ild_std = 0.5261339271318863

    margin = 1.8670232e-08

    real_label = 1.
    fake_label = 0.

    if config.transform_flag:
        mean_std_dir = config.mean_std_coef_dir
        mean_std_full = mean_std_dir + "/mean_std_full.pickle"
        with open(mean_std_full, "rb") as f:
            mean, std = pickle.load(f)
        mean = mean.float().to(device)
        std = std.float().to(device)

    if config.start_with_existing_model:
        print(f'Initialized weights using an existing model - {config.existing_model_path}')
        vae.load_state_dict(torch.load(f'{config.existing_model_path}/Vae.pt'))
        netD.load_state_dict(torch.load(f'{config.existing_model_path}/Disc.pt'))

    train_loss_G_list = []
    train_loss_G_adversarial_list = []
    train_loss_G_content_list = []
    train_loss_G_sh_mse_list = []
    train_loss_G_sh_cos_list = []
    train_loss_D_list = []
    train_loss_D_hr_list = []
    train_loss_D_sr_list = []

    train_SD_metric = []

    num_epochs = config.num_epochs
    for epoch in range(num_epochs):
        with open("log.txt", "a") as f:
            f.write(f"\nEpoch: {epoch}\n")
        times = []
        train_loss_G = 0.
        train_loss_G_adversarial = 0.
        train_loss_G_content = 0.
        train_loss_G_sh_mse = 0.
        train_loss_G_sh_cos = 0.
        train_loss_D = 0.
        train_loss_D_hr = 0.
        train_loss_D_sr = 0.
        
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
            hrtf = batch_data["hrtf"].to(device=device, memory_format=torch.contiguous_format,
                                         non_blocking=True, dtype=torch.float)
            masks = batch_data["mask"]
            
            bs = lr_coefficient.size(0)
            # ones_label = Variable(torch.ones(bs,1)).to(device) # labels for real data
            # zeros_label = Variable(torch.zeros(bs,1)).to(device) # labels for generated data

            # Generate fake samples using D-DBPN
            sr = netG(lr_coefficient)

            # Discriminator Training
            netD.zero_grad()
            # train on real coefficient
            pred_real = netD(hr_coefficient.detach().clone()).view(-1)
            label = torch.full((bs,), real_label, dtype=hr_coefficient.dtype, device=device)
            loss_D_hr = adversarial_criterion(pred_real, label)
            loss_D_hr.backward()
            # train on reconstructed coefficient
            pred_fake = netD(sr.detach().clone()).view(-1)
            label.fill_(fake_label)
            loss_D_sr = adversarial_criterion(pred_fake, label)
            loss_D_sr.backward()

            loss_D = loss_D_hr + loss_D_sr
            train_loss_D += loss_D.item()
            train_loss_D_hr += loss_D_hr.item()
            train_loss_D_sr += loss_D_sr.item()
            # Update D
            optD.step()
            
            # training VAE
            if batch_index % int(critic_iters) == 0:
                # train decoder
                netG.zero_grad()
                pred_fake = netD(sr).view(-1)
                label.fill_(real_label)
                adversarial_loss_G = config.adversarial_weight * adversarial_criterion(pred_fake, label)
                sh_cos_loss = 1 - cos_similarity_criterion(sr, hr_coefficient).mean()
                sh_mse_loss = ((sr - hr_coefficient) ** 2).mean()  # sh coefficient loss
                sr0 = sr[0].T
                hr0 = hr_coefficient[0].T
                with open("log.txt", "a") as f:
                    f.write(f"sr: {sr0.shape}, {sr0[0, :20]}\n")
                    f.write(f"hr: {hr0.shape}, {hr0[0, :20]}\n")
                    # print("sr: ",sr0.shape, sr0[0, :20])
                    # print("hr: ",hr0.shape, hr0[0, :20])
                # convert reconstructed coefficient back to hrtf
                harmonics_list = []
                for i in range(masks.size(0)):
                    SHT = SphericalHarmonicsTransform(config.max_order, ds.row_angles, ds.column_angles, ds.radii, masks[i].numpy().astype(bool))
                    harmonics = torch.from_numpy(SHT.get_harmonics()).float()
                    harmonics_list.append(harmonics)
                harmonics_tensor = torch.stack(harmonics_list).to(device)
                if config.transform_flag:  # unormalize the coefficient
                    recon = recon * std + mean
                recons = (harmonics_tensor @ sr.permute(0, 2, 1)).reshape(bs, num_row_angles, num_col_angles, num_radii, nbins)
                recons = recons.permute(0, 4, 3, 1, 2)  # bs x nbins x r x w x h
                if domain == "magnitude":
                    recons = F.relu(recons) + margin # filter out negative values and make it non-zero
                # during every 25th epoch and last epoch, save filename for mag spectrum plot
                if epoch % 25 == 0 or epoch == (num_epochs - 1):
                    generated = recons[0].permute(2, 3, 1, 0)  # w x h x r x nbins
                    target = hrtf[0].permute(2, 3, 1, 0)
                    id = batch_data['id'][0].item()
                    filename = f"magnitude_{id}_{epoch}"
                    plot_hrtf(generated.detach().cpu(), target.detach().cpu(), path, filename)
                unweighted_content_loss_G = content_criterion(config, recons, hrtf, sd_mean, sd_std, ild_mean, ild_std)
                # with open('log.txt', "a") as f:
                #     f.write(f"unweighted_content_loss: {unweighted_content_loss}\n")
                content_loss_G = config.content_weight * unweighted_content_loss_G
                # Generator total loss
                # loss_G = content_loss_G + adversarial_loss_G + sh_loss_G
                loss_G = content_loss_G + adversarial_loss_G + sh_cos_loss
                loss_G.backward()

                train_loss_G += loss_G.item()
                train_loss_G_adversarial += adversarial_loss_G.item()
                train_loss_G_content += content_loss_G.item()
                train_loss_G_sh_mse += sh_mse_loss.item()
                train_loss_G_sh_cos += sh_cos_loss.item()
                train_SD_metric.append(unweighted_content_loss_G.item())

                optG.step()

                with open("log.txt", "a") as f:
                    f.write(f"{batch_index}/{len(train_prefetcher)}\n")
                    f.write(f"dis: {loss_D.item()}\t generator: {loss_G.item()}\n")
                    f.write(f"D_real: {loss_D_hr.item()}, D_fake: {loss_D_sr.item()}\n")
                    f.write(f"content loss: {content_loss_G.item()}, adversarial: {adversarial_loss_G.item()}\n")
                    f.write(f"sh mse: {sh_mse_loss.item()}, sh cos: {sh_cos_loss.item()}\n")

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
        scheduler_D.step()
        scheduler_G.step()

        train_loss_D_list.append(train_loss_D / len(train_prefetcher))
        train_loss_D_hr_list.append(train_loss_D_hr / len(train_prefetcher))
        train_loss_D_sr_list.append(train_loss_D_sr / len(train_prefetcher))
        train_loss_G_list.append(train_loss_G / len(train_prefetcher))
        train_loss_G_content_list.append(train_loss_G_content / len(train_prefetcher))
        train_loss_G_adversarial_list.append(train_loss_G_adversarial / len(train_prefetcher))
        train_loss_G_sh_mse_list.append(train_loss_G_sh_mse / len(train_prefetcher))
        train_loss_G_sh_cos_list.append(train_loss_G_sh_cos / len(train_prefetcher))
        print(f"Avearge epoch loss, discriminator: {train_loss_D_list[-1]}, generator: {train_loss_G_list[-1]}")
        print(f"Avearge epoch loss, D_real: {train_loss_D_hr_list[-1]}, D_fake: {train_loss_D_sr_list[-1]}")
        print(f"Avearge content loss: {train_loss_G_content_list[-1]}, adversarial loss: {train_loss_G_adversarial_list[-1]}")
        print(f"Average sh mse loss: {train_loss_G_sh_mse_list[-1]}, sh cos loss: {train_loss_G_sh_cos_list[-1]}")

        # # create magnitude spectrum plot every 25 epochs and last epoch
        # if epoch % 25 == 0 or epoch == (num_epochs - 1):
        #     i_plot = 0
        #     magnitudes_real = torch.permute(hr.detach().cpu()[i_plot], (1, 2, 3, 0))
        #     magnitudes_interpolated = torch.permute(sr.detach().cpu()[i_plot], (1, 2, 3, 0))

        #     plot_label = filename[i_plot].split('/')[-1] + '_epoch' + str(epoch)
        #     plot_magnitude_spectrums(pos_freqs, magnitudes_real[:, :, :, :config.nbins_hrtf], magnitudes_interpolated[:, :, :, :config.nbins_hrtf],
        #                              "left", "training", plot_label, path, log_scale_magnitudes=True)

    plot_losses([train_loss_D_list, train_loss_G_list], 
                ['Discriminator loss', 'Generator loss'],
                ['red', 'green'], 
                path=path, filename='loss_curves', title="Loss curves")
    plot_losses([train_loss_D_list],['Discriminator loss'],['red'], path=path, filename='Discriminator_loss', title="Dis loss")
    plot_losses([train_loss_G_list],['Generator loss'],['green'], path=path, filename='Generator_loss', title="Gen loss")
    plot_losses([train_loss_D_hr_list, train_loss_D_sr],
                ['Discriminator loss real', 'Discriminator loss fake'],
                ["#5ec962", "#440154"], 
                path=path, filename='loss_curves_Dis', title="Discriminator loss curves")
    plot_losses([train_loss_G_sh_mse_list],['SH mse loss'],['blue'], path=path, filename='SH_mse_loss', title="SH mse loss")
    plot_losses([train_loss_G_sh_cos_list],['SH cos loss'],['blue'], path=path, filename='SH_cos_loss', title="SH cos loss")
    plot_losses([train_loss_G_adversarial_list, train_loss_G_content_list, train_loss_G_sh_cos_list],
                ['Generator adv loss', 'Generator content loss', 'Coefficient sim loss'],
                ['green', 'purple', 'red'], 
                path=path, filename='loss_curves_G', title="Generator loss curves")

    with open(f'{path}/train_losses.pickle', "wb") as file:
        pickle.dump((train_loss_D_list, train_loss_D_hr_list, train_loss_D_sr_list,
                     train_loss_G_list, train_loss_G_content_list, train_loss_G_adversarial_list, train_loss_G_sh_cos_list,
                     train_loss_G_sh_mse_list), file)
        # pickle.dump((train_loss_Dis_list, train_loss_Dis_hr_list, train_loss_Dis_recon_list,
        #              train_loss_Dec_list, train_loss_Dec_sim_list, train_loss_Dec_content_list, train_loss_Dec_gan_list, train_loss_Dec_sh_list,
        #              train_loss_Enc_list, train_loss_Enc_sim_list, train_loss_Enc_prior_list), file)

    print("TRAINING FINISHED")

def train2(config, train_prefetcher):
    """ Train the generator and discriminator models

    :param config: Config object containing model hyperparameters
    :param train_prefetcher: prefetcher for training data
    """
    # load the dataset to get the row, column angles info
    domain = config.domain
    data_dir = config.raw_hrtf_dir / config.dataset
    imp = importlib.import_module('hrtfdata.full')
    load_function = getattr(imp, config.dataset)
    ds = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate,
                                                         'side': 'left', 'domain': domain}}, subject_ids='first')
    num_row_angles = len(ds.row_angles)
    num_col_angles = len(ds.column_angles)
    num_radii = len(ds.radii)
    
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
    bs, optmizer, lr, alpha, lambda_feature, latent_dim, critic_iters = config.get_train_params()
    decay_lr = config.decay_lr

    # # get list of positive frequencies of HRTF for plotting magnitude spectrum
    # all_freqs = scipy.fft.fftfreq(256, 1 / config.hrir_samplerate)
    # pos_freqs = all_freqs[all_freqs >= 0]

    # Define VAE and transfer to CUDA
    degree = int(np.sqrt(num_row_angles*num_col_angles*num_radii/config.upscale_factor) - 1)
    vae = VAE(nbins=nbins, max_degree=degree, latent_dim=config.latent_dim).to(device)
    netD = Discriminator(nbins=nbins).to(device)
    if ('cuda' in str(device)) and (ngpu > 1):
        netD = (nn.DataParallel(netD, list(range(ngpu)))).to(device)
        vae = nn.DataParallel(vae, list(range(ngpu))).to(device)

    # Define optimizers
    # optD = optim.Adam(netD.parameters(), lr=lr*alpha)
    # optEncoder = optim.Adam(vae.encoder.parameters(), lr=lr)
    # optDecoder = optim.Adam(vae.decoder.parameters(), lr=lr)
    optEncoder = optim.RMSprop(vae.encoder.parameters(), lr=lr, alpha=0.9, eps=1e-8, weight_decay=0, momentum=0, centered=False)
    lr_encoder = ExponentialLR(optEncoder, gamma=decay_lr)
    optDecoder = optim.RMSprop(vae.decoder.parameters(), lr=lr, alpha=0.9, eps=1e-8, weight_decay=0, momentum=0, centered=False)
    lr_decoder = ExponentialLR(optDecoder, gamma=decay_lr)
    optD = optim.RMSprop(netD.parameters(), lr=0.0000015, alpha=0.9, eps=1e-8, weight_decay=0, momentum=0, centered=False)
    lr_discriminator = ExponentialLR(optD, gamma=decay_lr)

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

    margin = 1.8670232e-08

    if config.transform_flag:
        mean_std_dir = config.mean_std_coef_dir
        mean_std_full = mean_std_dir + "/mean_std_full.pickle"
        with open(mean_std_full, "rb") as f:
            mean, std = pickle.load(f)
        mean = mean.float().to(device)
        std = std.float().to(device)

    if config.start_with_existing_model:
        print(f'Initialized weights using an existing model - {config.existing_model_path}')
        vae.load_state_dict(torch.load(f'{config.existing_model_path}/Vae.pt'))
        netD.load_state_dict(torch.load(f'{config.existing_model_path}/Disc.pt'))

    train_loss_Dis_list = []
    train_loss_Dis_hr_list = []
    train_loss_Dis_recon_list = []
    train_loss_Dec_list = []
    train_loss_Dec_gan_list = []
    train_loss_Dec_sim_list = []
    train_loss_Dec_sh_list = []
    train_loss_Dec_content_list = []
    train_loss_Enc_list = []
    train_loss_Enc_prior_list = []
    train_loss_Enc_sim_list = []
    train_loss_feature_list = []

    num_epochs = config.num_epochs
    for epoch in range(num_epochs):
        with open("log.txt", "a") as f:
            f.write(f"Epoch: {epoch}\n")
        times = []

        train_loss_Dis = 0.
        train_loss_Dis_hr = 0.
        train_loss_Dis_recon = 0.
        train_loss_Dec = 0.
        train_loss_Dec_gan = 0.
        train_loss_Dec_sim = 0.
        train_loss_Dec_sh = 0.
        train_loss_Dec_content = 0.
        train_loss_Enc = 0.
        train_loss_Enc_prior = 0.
        train_loss_Enc_sim = 0.
        train_loss_feature = 0.

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
            hrtf = batch_data["hrtf"].to(device=device, memory_format=torch.contiguous_format,
                                         non_blocking=True, dtype=torch.float)
            masks = batch_data["mask"]
            
            bs = lr_coefficient.size(0)
            ones_label = Variable(torch.ones(bs,1)).to(device) # labels for real data
            zeros_label = Variable(torch.zeros(bs,1)).to(device) # labels for generated data

            # Generate fake samples using VAE
            mu, log_var, recon = vae(lr_coefficient)

            # Discriminator Training
            pred, inter_feature = netD(recon, hr_coefficient)
            # split to real and fake output
            pred_recon = pred[:bs]
            pred_real = pred[bs:]

            # train on real coefficient
            loss_D_hr = adversarial_criterion(pred_real, ones_label)
            train_loss_Dis_hr += loss_D_hr.item()
            # train on reconstructed coefficient 
            loss_D_recon = adversarial_criterion(pred_recon, zeros_label)
            train_loss_Dis_recon += loss_D_recon.item()
            gan_loss = loss_D_hr + loss_D_recon # Compute the discriminator loss
            train_loss_Dis += gan_loss.item()
            
            # training VAE
            if batch_index % int(critic_iters) == 0:
                # train decoder
                feature_recon = inter_feature[:bs]
                feature_real = inter_feature[bs:]
                feature_loss = ((feature_recon - feature_real) ** 2).mean() # feature-wise loss
                train_loss_feature += feature_loss.item()
                sh_loss = 0.001 * ((recon - hr_coefficient) ** 2).mean()  # sh coefficient loss
                train_loss_Dec_sh += sh_loss.item()
                # convert reconstructed coefficient back to hrtf
                harmonics_list = []
                for i in range(masks.size(0)):
                    SHT = SphericalHarmonicsTransform(28, ds.row_angles, ds.column_angles, ds.radii, masks[i].numpy().astype(bool))
                    harmonics = torch.from_numpy(SHT.get_harmonics()).float()
                    harmonics_list.append(harmonics)
                harmonics_tensor = torch.stack(harmonics_list).to(device)
                if config.transform_flag:  # unormalize the coefficient
                    recon = recon * std + mean
                recons = (harmonics_tensor @ recon.permute(0, 2, 1)).reshape(bs, num_row_angles, num_col_angles, num_radii, nbins)
                recons = recons.permute(0, 4, 3, 1, 2)  # bs x nbins x r x w x h
                if domain == "magnitude":
                    recons = F.relu(recons) + margin # filter out negative values and make it non-zero
                # during every 25th epoch and last epoch, save filename for mag spectrum plot
                if epoch % 25 == 0 or epoch == (num_epochs - 1):
                    generated = recons[0].permute(2, 3, 1, 0)  # w x h x r x nbins
                    target = hrtf[0].permute(2, 3, 1, 0)
                    filename = f"magnitude_{epoch}"
                    plot_hrtf(generated.detach().cpu(), target.detach().cpu(), path, filename)
                unweighted_content_loss = content_criterion(config, recons, hrtf, sd_mean, sd_std, ild_mean, ild_std)
                content_loss = config.content_weight * unweighted_content_loss
                train_loss_Dec_content += content_loss.item()
                err_dec = feature_loss + content_loss - gan_loss
                train_loss_Dec += err_dec.item()

                # train encoder
                prior_loss = 1 + log_var - mu.pow(2) - log_var.exp()
                prior_loss = (-0.5 * torch.sum(prior_loss)).mean() # prior loss
                train_loss_Enc_prior += prior_loss.item()
                # with open(f"log.txt", "a") as f:
                #     f.write(f"lr coef nan? {torch.isnan(lr_coefficient.any())}\n")
                #     f.write(f"recon nan? {torch.isnan(recon).any()}\n")
                #     f.write(f"feature recon: {torch.isnan(feature_recon).any()}\n")
                #     f.write(f"feature real: {torch.isnan(feature_real).any()}\n")
                # feature_sim_loss_E = config.beta * ((feature_recon - feature_real) ** 2).mean() # feature loss
                # train_loss_Enc_sim += feature_sim_loss_E.item()
                err_enc = prior_loss + feature_loss
                train_loss_Enc += err_enc.item()

                # Update encoder
                optEncoder.zero_grad()
                err_enc.backward(retain_graph=True)
                optEncoder.step()

                # Update decoder
                optDecoder.zero_grad()
                err_dec.backward(retain_graph=True)
                optDecoder.step()

                with open("log.txt", "a") as f:
                    f.write(f"{batch_index}/{len(train_prefetcher)}\n")
                    f.write(f"dis: {gan_loss.item()}\t dec: {err_dec.item()}\t enc: {err_enc.item()}\n")
                    f.write(f"D_real: {loss_D_hr.item()}, D_fake: {loss_D_recon.item()}\n")
                    f.write(f"feature loss: {feature_loss.item()}\n")
                    f.write(f"content loss: {content_loss.item()}, sh loss: {sh_loss.item()}\n")
                    f.write(f"prior: {prior_loss.item()}\n\n")
                    # f.write(f"content loss: {content_loss.item()}, sim_D: {feature_sim_loss_D.item()}, gan loss: {gan_loss_dec.item()}\n")
                    # f.write(f"prior: {prior_loss.item()}, sim_E: {feature_sim_loss_E.item()}\n\n")

            # Update D
            netD.zero_grad()
            gan_loss.backward()
            optD.step()

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
                    torch.save(vae.state_dict(), f'{path}/vae.pt')
                    torch.save(netD.state_dict(), f'{path}/Disc.pt')

                    progress(batch_index, batches, epoch, num_epochs, timed=np.mean(times))
                    times = []

            # Preload the next batch of data
            batch_data = train_prefetcher.next()

            # After training a batch of data, add 1 to the number of data batches to ensure that the
            # terminal print data normally
            batch_index += 1
        lr_encoder.step()
        lr_decoder.step()
        lr_discriminator.step()

        train_loss_Dis_list.append(train_loss_Dis / len(train_prefetcher))
        train_loss_Dis_hr_list.append(train_loss_Dis_hr / len(train_prefetcher))
        train_loss_Dis_recon_list.append(train_loss_Dis_recon / len(train_prefetcher))
        train_loss_Dec_list.append(train_loss_Dec / len(train_prefetcher))
        # train_loss_Dec_gan_list.append(train_loss_Dec_gan / len(train_prefetcher))
        train_loss_Dec_content_list.append(train_loss_Dec_content / len(train_prefetcher))
        # train_loss_Dec_sim_list.append(train_loss_Dec_sim / len(train_prefetcher))
        train_loss_Dec_sh_list.append(train_loss_Dec_sh / len(train_prefetcher))
        train_loss_Enc_list.append(train_loss_Enc / len(train_prefetcher))
        train_loss_Enc_prior_list.append(train_loss_Enc_prior / len(train_prefetcher))
        # train_loss_Enc_sim_list.append(train_loss_Enc_sim / len(train_prefetcher))
        train_loss_feature_list.append(train_loss_feature / len(train_prefetcher))
        print(f"Avearge epoch loss, discriminator: {train_loss_Dis_list[-1]}, decoder: {train_loss_Dec_list[-1]}, encoder: {train_loss_Enc_list[-1]}")
        print(f"Avearge epoch loss, D_real: {train_loss_Dis_hr_list[-1]}, D_fake: {train_loss_Dis_recon_list[-1]}")
        print(f"Average feature loss: {train_loss_feature_list[-1]}")
        print(f"Avearge content loss: {train_loss_Dec_content_list[-1]},sh loss:{train_loss_Dec_sh_list[-1]}")
        print(f"Average prior loss: {train_loss_Enc_prior_list[-1]}\n")
        # print(f"Avearge content loss: {train_loss_Dec_content_list[-1]},  decoder similarity loss: {train_loss_Dec_sim_list[-1]}, sh loss:{train_loss_Dec_sh_list[-1]}, gan loss: {train_loss_Dec_gan_list[-1]}")
        # print(f"Average prior loss: {train_loss_Enc_prior_list[-1]}, encoder similarity loss: {train_loss_Enc_sim_list[-1]}\n")

        # # create magnitude spectrum plot every 25 epochs and last epoch
        # if epoch % 25 == 0 or epoch == (num_epochs - 1):
        #     i_plot = 0
        #     magnitudes_real = torch.permute(hr.detach().cpu()[i_plot], (1, 2, 3, 0))
        #     magnitudes_interpolated = torch.permute(sr.detach().cpu()[i_plot], (1, 2, 3, 0))

        #     plot_label = filename[i_plot].split('/')[-1] + '_epoch' + str(epoch)
        #     plot_magnitude_spectrums(pos_freqs, magnitudes_real[:, :, :, :config.nbins_hrtf], magnitudes_interpolated[:, :, :, :config.nbins_hrtf],
        #                              "left", "training", plot_label, path, log_scale_magnitudes=True)

    plot_losses([train_loss_Dis_list, train_loss_Dec_list, train_loss_Enc_list], 
                ['Discriminator loss', 'Decoder loss', 'Encoder loss'],
                ['red', 'green', 'blue'], 
                path=path, filename='loss_curves', title="Loss curves")
    plot_losses([train_loss_Dis_list],['Discriminator loss'],['red'], path=path, filename='Discriminator_loss', title="Dis loss")
    plot_losses([train_loss_Dec_list],['Decoder loss'],['green'], path=path, filename='Decoder_loss', title="Dec loss")
    plot_losses([train_loss_Enc_list],['Encoder loss'],['blue'], path=path, filename='Encoder_loss', title="Enc loss")
    plot_losses([train_loss_Dis_hr_list, train_loss_Dis_recon_list],
                ['Discriminator loss real', 'Discriminator loss fake'],
                ["#5ec962", "#440154"], 
                path=path, filename='loss_curves_Dis', title="Discriminator loss curves")
    plot_losses([train_loss_feature_list],['Feature loss'],['blue'], path=path, filename='Feature_loss', title="Feature loss")
    plot_losses([train_loss_Dec_content_list, train_loss_Dec_sh_list],
                ['Decoder content loss', 'sh loss'],
                ['green', 'purple'], 
                path=path, filename='loss_curves_Dec', title="Decoder loss curves")
    plot_losses([train_loss_Enc_prior_list],['Prior loss'],['blue'], path=path, filename='Prior_loss', title="Prior loss")
    # plot_losses([train_loss_Dec_sim_list, train_loss_Dec_content_list, train_loss_Dec_gan_list, train_loss_Dec_sh_list],
    #             ['Decoder sim loss', 'Decoder content loss', 'Decoder gan loss', 'sh loss'],
    #             ['red', 'green', 'blue', 'purple'], 
    #             path=path, filename='loss_curves_Dec', title="Decoder loss curves")
    # plot_losses([train_loss_Enc_prior_list, train_loss_Enc_sim_list], 
    #             ['Encoder prior loss', 'Encoder sim loss'],
    #             ['#b5de2b', '#1f9e89'],
    #             path=path, filename='loss_curves_Enc', title="Encoder loss curves")

    with open(f'{path}/train_losses.pickle', "wb") as file:
        pickle.dump((train_loss_Dis_list, train_loss_Dis_hr_list, train_loss_Dis_recon_list, train_loss_feature_list,
                     train_loss_Dec_list, train_loss_Dec_content_list, train_loss_Dec_sh_list,
                     train_loss_Enc_list, train_loss_Enc_prior_list), file)
        # pickle.dump((train_loss_Dis_list, train_loss_Dis_hr_list, train_loss_Dis_recon_list,
        #              train_loss_Dec_list, train_loss_Dec_sim_list, train_loss_Dec_content_list, train_loss_Dec_gan_list, train_loss_Dec_sh_list,
        #              train_loss_Enc_list, train_loss_Enc_sim_list, train_loss_Enc_prior_list), file)

    print("TRAINING FINISHED")
