from functools import partial
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import random_split
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from config import Config

from model.util import *
from model.test import test
from evaluation.evaluation import run_lsd_evaluation, run_localisation_evaluation
from hrtfdata.transforms.hrirs import SphericalHarmonicsTransform

import importlib
from model.model import VAE, Discriminator

import time

from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler


def get_optimizer(config, vae, netD):
    if config.optimizer == 'adam':
        optEncoder = optim.Adam(vae.encoder.parameters(), lr=config.lr)
        optDecoder = optim.Adam(vae.decoder.parameters(), lr=config.lr)
        optD = optim.Adam(netD.parameters(), lr=config.lr * config.alpha)
    elif config.optimizer == 'rmsprop':
        optEncoder = optim.RMSprop(vae.encoder.parameters(), lr=config.lr)
        optDecoder = optim.RMSprop(vae.decoder.parameters(), lr=config.lr)
        optD = optim.RMSprop(netD.parameters(), lr=config.lr * config.alpha)
    return optEncoder, optDecoder, optD

def train_vae_gan(config, config_index, train_prefetcher):
    data_dir = config.raw_hrtf_dir / config.dataset
    imp = importlib.import_module('hrtfdata.full')
    load_function = getattr(imp, config.dataset)
    ds = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate,
                                                         'side': 'left', 'domain': 'time'}}, subject_ids='first')
    num_row_angles = len(ds.row_angles)
    num_col_angles = len(ds.column_angles)
    num_radii = len(ds.radii)

    batches = len(train_prefetcher)

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

    critic_iters = config.critic_iters
    lambda_feature = config.lambda_feature
    latent_dim = config.latent_dim

    degree = compute_sh_degree(config)
    vae = VAE(nbins=nbins, max_degree=degree, latent_dim=latent_dim).to(device)
    netD = Discriminator(nbins=nbins).to(device)
    if ('cuda' in str(device)) and (ngpu > 1):
        netD = (nn.DataParallel(netD, list(range(ngpu)))).to(device)
        vae = nn.DataParallel(vae, list(range(ngpu))).to(device)
    
    optEncoder, optDecoder, optD = get_optimizer(config, vae, netD)

    # Define loss functions
    adversarial_criterion = nn.BCEWithLogitsLoss()
    content_criterion = sd_ild_loss

    sd_mean = 7.387559253346883
    sd_std = 0.577364154400081
    ild_mean = 3.6508303231127868
    ild_std = 0.5261339271318863

    if config.start_with_existing_model:
        print(f'Initialized weights using an existing model - {config.existing_model_path}')
        vae.load_state_dict(torch.load(f'{config.existing_model_path}/Vae.pt'))
        netD.load_state_dict(torch.load(f'{config.existing_model_path}/Disc.pt'))

    # checkpoint = session.get_checkpoint()
    # if checkpoint:
    #     checkpoint_state = checkpoint.to_dict()
    #     start_epoch = checkpoint_state["epoch"]
    #     vae.load_state_dict(checkpoint_data["VAE_state_dict"])
    #     netD.load_state_dict(checkpoint_data["discriminator_state_dict"])
    #     optEncoder.load_state_dict(checkpoint_state["optEncoder_state_dict"])
    #     optDecoder.load_state_dict(checkpoint_state["optDecoder_state_dict"])
    #     optD.load_state_dict(checkpoint_data["optD_state_dict"])
    # else:
    #     start_epoch = 0
    num_epochs = config.num_epochs
    for epoch in range(num_epochs):
        if epoch % 20 == 0:
            with open("optimize.txt", "a") as f:
                f.write(f"Epoch: {epoch}\n")
        times = []
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
            if ('cuda' in str(device)) and (ngpu > 1):
                start_overall = torch.cuda.Event(enable_timing=True)
                end_overall = torch.cuda.Event(enable_timing=True)
                start_overall.record()
            else:
                start_overall = time.time()

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
                feature_sim_loss_D = lambda_feature * ((feature_recon - feature_real) ** 2).mean() # feature loss
                train_loss_Dec_sim += feature_sim_loss_D.item()
                # convert reconstructed coefficient back to hrtf
                harmonics_list = []
                for i in range(masks.size(0)):
                    SHT = SphericalHarmonicsTransform(28, ds.row_angles, ds.column_angles, ds.radii, masks[i].numpy().astype(bool))
                    harmonics = torch.from_numpy(SHT.get_harmonics()).float()
                    harmonics_list.append(harmonics)
                harmonics_tensor = torch.stack(harmonics_list).to(device)
                recons = harmonics_tensor @ recon.permute(0, 2, 1)
                recons =  F.softplus(recons.reshape(bs, nbins, num_radii, num_row_angles, num_col_angles))
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
                feature_sim_loss_E = ((feature_recon - feature_real) ** 2).mean() # feature loss
                train_loss_Enc_sim += feature_sim_loss_E.item()
                err_enc = prior_loss + feature_sim_loss_E
                train_loss_Enc += err_enc.item()
                # Update encoder
                optEncoder.zero_grad()
                err_enc.backward()
                optEncoder.step()

                with open("optimize.txt", "a") as f:
                    f.write(f"{batch_index}/{len(train_prefetcher)}\n")
                    f.write(f"dis: {gan_loss.item()}\t dec: {err_dec.item()}\t enc: {err_enc.item()}\n")
                    f.write(f"D_real: {loss_D_hr.item()}, D_fake: {loss_D_recon.item()}\n")
                    f.write(f"content loss: {content_loss.item()}, sim_D: {feature_sim_loss_D.item()}, gan loss: {gan_loss_dec.item()}\n")
                    f.write(f"prior: {prior_loss.item()}, sim_E: {feature_sim_loss_E.item()}\n\n")

            if ('cuda' in str(device)) and (ngpu > 1):
                end_overall.record()
                torch.cuda.synchronize()
                times.append(start_overall.elapsed_time(end_overall))
            else:
                end_overall = time.time()
                times.append(end_overall - start_overall)

            # Every 0th batch log useful metrics
            if batch_index == 0:
                progress(batch_index, batches, epoch, num_epochs, timed=np.mean(times))
                times = []

            batch_data = train_prefetcher.next()
            batch_index += 1

        avg_train_loss_Dis = train_loss_Dis / len(train_prefetcher)
        avg_train_loss_Dis_hr = train_loss_Dis_hr / len(train_prefetcher)
        avg_train_loss_Dis_recon = train_loss_Dis_recon / len(train_prefetcher)
        avg_train_loss_Dec = train_loss_Dec / len(train_prefetcher)
        avg_train_loss_Dec_gan = train_loss_Dec_gan / len(train_prefetcher)
        avg_train_loss_Dec_content = train_loss_Dec_content / len(train_prefetcher)
        avg_train_loss_Dec_sim = train_loss_Dec_sim / len(train_prefetcher)
        avg_train_loss_Enc = train_loss_Enc / len(train_prefetcher)
        avg_train_loss_Enc_prior = train_loss_Enc_prior / len(train_prefetcher)
        avg_train_loss_Enc_sim = train_loss_Enc_sim / len(train_prefetcher)

        if epoch % 20 == 0:
            print(f"Avearge epoch loss, discriminator: {avg_train_loss_Dis}, decoder: {avg_train_loss_Dec}, encoder: {avg_train_loss_Enc}")
            print(f"Avearge epoch loss, D_real: {avg_train_loss_Dis_hr}, D_fake: {avg_train_loss_Dis_recon}")
            print(f"Avearge content loss: {avg_train_loss_Dec_content},  decoder similarity loss: {avg_train_loss_Dec_sim}, gan loss: {avg_train_loss_Dec_gan}")
            print(f"Average prior loss: {avg_train_loss_Enc_prior}, encoder similarity loss: {avg_train_loss_Enc_sim}\n")

        # checkpoint_data = {
        #     "epoch": epoch,
        #     "VAE_state_dict": vae.state_dict(),
        #     "discriminator_state_dict": netD.state_dict(),
        #     "optD_state_dict": optD.state_dict(),
        #     "optEncoder_state_dict": optEncoder.state_dict(),
        #     "optDecoder_state_dict": optDecoder.state_dict(),
        # }
        # checkpoint = Checkpoint.from_dict(checkpoint_data)

        # session.report(
        #     {"loss_Dis": avg_train_loss_Dis, "loss_Dec": avg_train_loss_Dec, "loss_Enc": avg_train_loss_Enc},
        #     checkpoint=checkpoint,
        # )
    with torch.no_grad():
        torch.save(vae.state_dict(), f'{path}/weight_{config.upscale_factor}/vae_{config_index}.pt')
        torch.save(netD.state_dict(), f'{path}/weight_{config.upscale_factor}/Disc_{config_index}.pt')
    print("Finished Training")

def eval_vae(config, val_prefetcher):
    data_dir = config.raw_hrtf_dir / config.dataset
    imp = importlib.import_module('hrtfdata.full')
    load_function = getattr(imp, config.dataset)
    ds = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate,
                                                         'side': 'left', 'domain': 'time'}}, subject_ids='first')
    num_row_angles = len(ds.row_angles)
    num_col_angles = len(ds.column_angles)
    num_radii = len(ds.radii)
    degree = int(np.sqrt(num_row_angles*num_col_angles*num_radii/config.upscale_factor) - 1)

    ngpu = config.ngpu
    valid_dir = config.valid_path
    valid_gt_dir = config.valid_gt_path

    nbins = config.nbins_hrtf
    if config.merge_flag:
        nbins = config.nbins_hrtf * 2

    device = torch.device(config.device_name if (
            torch.cuda.is_available() and ngpu > 0) else "cpu")
    model = VAE(nbins=nbins, max_degree=degree, latent_dim=10).to(device)
    print("Build VAE model successfully.")

    model.load_state_dict(torch.load(f"{config.model_path}/vae.pt", map_location=torch.device('cpu')))
    print(f"Load VAE model weights `{os.path.abspath(config.model_path)}` successfully.")

    model.eval()

    # Initialize the data loader and load the first batch of data
    val_prefetcher.reset()
    batch_data = val_prefetcher.next()

    # Clear/Create directories
    shutil.rmtree(Path(valid_dir), ignore_errors=True)
    Path(valid_dir).mkdir(parents=True, exist_ok=True)

    while batch_data is not None:
        # Transfer in-memory data to CUDA devices to speed up validation 
        lr_coefficient = batch_data["lr_coefficient"].to(device=device, memory_format=torch.contiguous_format,
                                                         non_blocking=True, dtype=torch.float)
        hrtf = batch_data["hrtf"]
        masks = batch_data["mask"]
        sample_id = batch_data["id"].item()

        # Use the vae to generate fake samples
        with torch.no_grad():
            _, _, recon = model(lr_coefficient)

        SHT = SphericalHarmonicsTransform(28, ds.row_angles, ds.column_angles, ds.radii, masks[0].numpy().astype(bool))
        harmonics = torch.from_numpy(SHT.get_harmonics()).float().to(device)
        sr = harmonics @ recon[0].T
        sr = F.softplus(sr.reshape(-1, nbins, num_radii, num_row_angles, num_col_angles))
        file_name = '/' + f"{config.dataset}_{sample_id}.pickle"
        sr = torch.permute(sr[0], (2, 3, 1, 0)).detach().cpu() # w x h x r x nbins
        hr = torch.permute(hrtf[0], (1, 2, 3, 0)).detach().cpu() # r x w x h x nbins

        with open(valid_dir + file_name, "wb") as file:
            pickle.dump(sr, file)

        with open(valid_gt_dir + file_name, "wb") as file:
            pickle.dump(hr, file)
        
        # Preload the next batch of data
        batch_data = val_prefetcher.next()

def main(config_index):
    tag = 'ari-upscale-4'
    config = Config(tag, using_hpc=True)
    config_file_path = f"{config.path}/config_files/config_{config_index}.json"
    with open("optimize.txt", "a") as f:
        f.write(f"config loaded: {config_file_path}\n")
    config.load(config_index)

    train_prefetcher, val_prefetcher = get_train_val_loader(config)
    train_vae_gan(config, config_index, train_prefetcher)
    eval_vae(config, val_prefetcher)

    run_lsd_evaluation(config, config.valid_path)
    run_localisation_evaluation(config, config.valid_path)

    



# def ray_main(config, num_samples=20, gpus_per_trial=1):
    # hyperparameters = {
    #     "batch_size": tune.choice([2, 4, 8, 16]),
    #     "optimizer": tune.choice(["adam", "rmsprop"]),
    #     "lr": tune.loguniform(1e-4, 1e-1),
    #     "alpha": tune.loguniform(1e-2, 1),
    #     "gamma": tune.choice([0.15, 1.5, 15]),
    #     "beta": tune.choice([0.05, 0.5, 5]),
    #     "latent_dim": tune.choice([10, 50, 100]),
    #     "critic_iter": tune.choice([3, 4, 5, 6]),
    # }

    # scheduler = ASHAScheduler(
    #     metric=["loss_Dis", "loss_Dec", "loss_Enc"],
    #     mode="min",
    #     max_t=config.num_epochs,
    #     grace_period=30,
    #     reduction_factor=2,
    # )

    # result = tune.run(
    #     partial(train_vae_gan),
    #     resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
    #     config=hyperparameters,
    #     num_samples=num_samples,
    #     scheduler=scheduler,
    # )

    # best_trial = result.get_best_trial(["loss_Dis", "loss_Dec", "loss_Enc"], "min", "last")
    # print(f"Best trail combination: {best_trial.config}")
    # print(f"Best trail final discriminator loss: {best_trial.last_result['loss_Dis']}")
    # print(f"Best trail final decoder loss: {best_trial.last_result['loss_Dec']}")
    # print(f"Best trail final encoder loss: {best_trial.last_result['loss_Enc']}")

    # nbins = config.nbins_hrtf
    # if config.merge_flag:
    #     nbins = config.nbins_hrtf * 2

    # degree = compute_sh_degree(config)
    # best_trained_model = VAE(nbins=nbins, max_degree=degree, latent_dim=best_trial.config["latent_dim"])
    # device = "cpu"
    # if torch.cuda.is_available():
    #     device = "cuda:0"
    #     if gpus_per_trial > 1:
    #         best_trained_model = nn.DataParallel(best_trained_model)
    # best_trained_model.to(device)

    # best_checkpoint = best_trial.checkpoint.to_air_checkpoint()
    # best_checkpoint_data = best_checkpoint.to_dict()

    # best_trained_model.load_state_dict(best_checkpoint_data["VAE_state_dict"])
    # path = config.path
    # with torch.no_grad():
    #     torch.save(best_checkpoint_data["VAE_state_dict"], f'{path}/vae.pt')
    
    # _, test_prefetcher = load_hrtf(config)
    # print("Loaded all datasets successfully.")
    # test(config, test_prefetcher)
    # run_lsd_evaluation(config, config.valid_path)
    # run_localisation_evaluation(config, config.valid_path)

if __name__ == "__main__":
    print("using cuda? ", torch.cuda.is_available())
    tag = "ari-upscale-4"
    config = Config(tag, using_hpc=True)
    main(config)

    
