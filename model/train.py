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
    # load the dataset to get the row, column angles info
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

    # train_losses_G = []
    # train_losses_G_adversarial = []
    # train_losses_G_content = []
    # train_losses_D = []
    # train_losses_D_hr = []
    # train_losses_D_sr = []

    # train_SD_metric = []

    train_loss_Dis_list = []
    train_loss_Dis_hr_list = []
    train_loss_Dis_recon_list = []
    train_loss_Dec_list = []
    train_loss_Dec_gan_list = []
    train_loss_Dec_sim_list = []
    train_loss_Dec_content_list = []
    train_loss_Enc_list = []
    train_loss_Enc_prior_list = []
    train_loss_Enc_sim_list = []

    for epoch in range(num_epochs):
        times = []
        # train_loss_G = 0.
        # train_loss_G_adversarial = 0.
        # train_loss_G_content = 0.
        # train_loss_D = 0.
        # train_loss_D_hr = 0.
        # train_loss_D_recon = 0.

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
            # train on real coefficient
            pred_real = netD(hr_coefficient)[0]
            loss_D_hr = adversarial_criterion(pred_real, ones_label)
            train_loss_Dis_hr += loss_D_hr.item()
            # train on reconstructed coefficient 
            pred_recon = netD(recon.detach().clone())[0]
            loss_D_recon = adversarial_criterion(pred_recon, zeros_label)
            train_loss_D_recon += loss_D_recon.item()
            # Compute the discriminator loss
            gan_loss = loss_D_hr + loss_D_recon
            train_loss_Dis += gan_loss.item()
            # Update D
            netD.zero_grad()
            gan_loss.backward(retain_graph=True)
            optD.step()

            # train decoder
            pred_real, feature_real = netD(hr_coefficient)
            err_dec_real = adversarial_criterion(pred_real, ones_label)
            pred_recon, feature_recon = netD(recon)
            err_dec_recon = adversarial_criterion(pred_recon, zeros_label)
            gan_loss_dec = err_dec_real + err_dec_recon
            train_loss_Dec_gan += gan_loss_dec.item()
            feature_sim_loss_D = config.gamma * ((feature_recon - feature_real) ** 2).mean() # feature loss
            train_loss_Dec_sim += feature_sim_loss_D.item()
            # convert reconstructed coefficient back to hrir
            recon_coef_list = []
            num_row_angles = len(ds.row_angles)
            num_col_angles = len(ds.column_angles)
            num_radii = len(ds.radii)
            for i in range(masks.size(0)):
                SHT = SphericalHarmonicsTransform(28, ds.row_angles, ds.column_angles, ds.radii, masks[i].detach().cpu().numpy().astype(bool))
                recon_hrir = SHT.inverse(recon[i].T.detach().cpu().numpy())  # Compute the inverse
                recon_hrir_tensor = torch.from_numpy(recon_hrir.T).reshape(nbins, num_radii, num_row_angles, num_col_angles)
                recon_coef_list.append(recon_hrir_tensor)
            recons = torch.stack(recon_coef_list).to(device)
            unweighted_content_loss = content_criterion(config, recons, hrir, sd_mean, sd_std, ild_mean, ild_std)
            content_loss = config.content_weight * unweighted_content_loss
            train_loss_Dec_content += content_loss
            err_dec = feature_sim_loss_D + content_loss - gan_loss_dec
            train_loss_Dec += err_dec
            # Update decoder
            optDecoder.zero_grad()
            err_dec.backward(retain_graph=True)
            optDecoder.step()

            # train encoder
            mu, log_var, recon = vae(lr_coefficient)
            prior_loss = 1 + log_var - mu.pow(2) - log_var.exp()
            prior_loss = (-0.5 * torch.sum(prior_loss))/torch.numel(mu.data) # prior loss
            train_loss_Enc_prior += prior_loss.item()
            feature_recon = netD(recon)[1]
            feature_sim_loss_E = config.beta * ((feature_recon - feature_real) ** 2).mean() # feature loss
            train_loss_Enc_sim += feature_sim_loss_E.item()
            err_enc = prior_loss + feature_sim_loss_E
            train_loss_Enc += err_enc.item()
            # Update encoder
            optEncoder.zero_grad()
            err_enc.backward(retain_graph=True)
            optEncoder.step()

            # # Generator training
            # if batch_index % int(critic_iters) == 0:
            #     # Initialize generator model gradients
            #     netG.zero_grad()
            #     sr = netG(lr)
            #     label.fill_(1.)
            #     # Calculate adversarial loss
            #     output = netD(sr).view(-1)

            #     unweighted_content_loss_G = content_criterion(config, sr, hr, sd_mean, sd_std, ild_mean, ild_std)
            #     content_loss_G = config.content_weight * unweighted_content_loss_G
            #     adversarial_loss_G = config.adversarial_weight * adversarial_criterion(output, label)

            #     # Calculate the generator total loss value and backprop
            #     loss_G = content_loss_G + adversarial_loss_G
            #     loss_G.backward()

            #     train_loss_G += loss_G.item()
            #     train_loss_G_adversarial += adversarial_loss_G.item()
            #     train_loss_G_content += content_loss_G.item()
            #     train_SD_metric.append(unweighted_content_loss_G.item())

            #     optG.step()

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

        train_loss_Dis_list.append(train_loss_Dis / len(train_prefetcher))
        train_loss_Dis_hr_list.append(train_loss_Dis_hr / len(train_prefetcher))
        train_loss_Dis_recon_list.append(train_loss_Dis_recon / len(train_prefetcher))
        train_loss_Dec_list.append(train_loss_Dec / len(train_prefetcher))
        train_loss_Dec_gan_list.append(train_loss_Dec_gan / len(train_prefetcher))
        train_loss_Dec_content_list.append(train_loss_Dec_content / len(train_prefetcher))
        train_loss_Dec_sim_list.append(train_loss_Dec_sim / len(train_prefetcher))
        train_loss_Enc_list.append(train_loss_Enc / len(train_prefetcher))
        train_loss_Enc_prior_list.append(train_loss_Enc_prior / len(train_prefetcher))
        train_loss_Enc_sim_list.append(train_loss_Enc_sim / len(train_prefetcher))
        print(f"Avearge epoch loss, discriminator: {train_loss_Dis_list[-1]}, decoder: {train_loss_Dec_list[-1]}, encoder: {train_loss_Enc_list[-1]}")
        print(f"Avearge epoch loss, D_real: {train_loss_Dis_hr_list[-1]}, D_fake: {train_loss_Dis_recon_list[-1]}")
        print(f"Avearge content loss: {train_loss_Dec_content_list[-1]},  decoder similarity loss: {train_loss_Dec_sim_list[-1]}, gan loss: {train_loss_Dec_gan_list[-1]}")
        print(f"Average prior loss: {train_loss_Enc_prior_list[-1]}, encoder similarity loss: {train_loss_Enc_sim_list[-1]}")


        train_losses_D.append(train_loss_D / len(train_prefetcher))
        train_losses_D_hr.append(train_loss_D_hr / len(train_prefetcher))
        train_losses_D_sr.append(train_loss_D_sr / len(train_prefetcher))
        train_losses_G.append(train_loss_G / len(train_prefetcher))
        train_losses_G_adversarial.append(train_loss_G_adversarial / len(train_prefetcher))
        train_losses_G_content.append(train_loss_G_content / len(train_prefetcher))
        print(f"Average epoch loss, discriminator: {train_losses_D[-1]}, generator: {train_losses_G[-1]}")
        print(f"Average epoch loss, D_real: {train_losses_D_hr[-1]}, D_fake: {train_losses_D_sr[-1]}")
        print(f"Average epoch loss, G_adv: {train_losses_G_adversarial[-1]}, train_losses_G_content: {train_losses_G_content[-1]}")

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
    plot_losses([train_loss_Dis_hr_list, train_loss_Dis_recon_list],
                ['Discriminator loss real', 'Discriminator loss fake']
                ["#5ec962", "#440154"], 
                path=path, filename='loss_curves_Dis', title="Discriminator loss curves")
    plot_losses([train_loss_Dec_sim_list, train_loss_Dec_content_list, train_loss_Dec_gan_list],
                ['Decoder sim loss', 'Decoder content loss', 'Decoder gan loss'],
                ['red', 'green', 'blue'], 
                path=path, filename='loss_curves_Dec', title="Decoder loss curves")
    plot_losses([train_loss_Enc_prior_list, train_loss_Enc_sim_list], 
                ['Encoder prior loss', 'Encoder sim loss'],
                ['#b5de2b', '#1f9e89'],
                path=path, filename='loss_curves_Enc', title="Encoder loss curves")

    # plot_losses(train_losses_D, train_losses_G,
    #             label_1='Discriminator loss', label_2='Generator loss',
    #             color_1="#5ec962", color_2="#440154",
    #             path=path, filename='loss_curves', title="Loss curves")
    # plot_losses(train_losses_D_hr, train_losses_D_sr,
    #             label_1='Discriminator loss, real', label_2='Discriminator loss, fake',
    #             color_1="#b5de2b", color_2="#1f9e89",
    #             path=path, filename='loss_curves_D', title="Discriminator loss curves")
    # plot_losses(train_losses_G_adversarial, train_losses_G_content,
    #             label_1='Generator loss, adversarial', label_2='Generator loss, content',
    #             color_1="#31688e", color_2="#440154",
    #             path=path, filename='loss_curves_G', title="Generator loss curves")

    with open(f'{path}/train_losses.pickle', "wb") as file:
        pickle.dump((train_loss_Dis_list, train_loss_Dis_hr_list, train_loss_Dis_recon_list,
                     train_loss_Dec_list, train_loss_Dec_sim_list, train_loss_Dec_content_list, train_loss_Dec_gan_list,
                     train_loss_Enc_list, train_loss_Enc_sim_list, train_loss_Enc_prior_list), file)
        # pickle.dump((train_losses_G, train_losses_G_adversarial, train_losses_G_content,
        #              train_losses_D, train_losses_D_hr, train_losses_D_sr, train_SD_metric), file)

    print("TRAINING FINISHED")
