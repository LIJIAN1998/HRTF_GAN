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

from config import Config

from model.util import *
from model.test import test
from evaluation.evaluation import run_lsd_evaluation, run_localisation_evaluation
from hrtfdata.transforms.hrirs import SphericalHarmonicsTransform

import importlib
from model.model import VAE, Discriminator

from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler


def get_optimizer(hyperparameters, vae, netD):
    if hyperparameters['optimizer'] == 'adam':
        optEncoder = optim.Adam(vae.encoder.parameters(), lr=hyperparameters["lr"])
        optDecoder = optim.Adam(vae.decoder.parameters(), lr=hyperparameters["lr"])
        optD = optim.Adam(netD.parameters(), lr=hyperparameters["lr"] * hyperparameters["alpha"])
    elif hyperparameters['optimizer'] == 'rmsprop':
        optEncoder = optim.RMSprop(vae.encoder.parameters(), lr=hyperparameters["lr"])
        optDecoder = optim.RMSprop(vae.decoder.parameters(), lr=hyperparameters["lr"])
        optD = optim.RMSprop(netD.parameters(), lr=hyperparameters["lr"] * hyperparameters["alpha"])
    return optEncoder, optDecoder, optD

def train_vae_gan(config, hyperparameters):
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

    critic_iters = hyperparameters["critic_iters"]
    gamma = hyperparameters["gamma"]
    beta = hyperparameters["beta"]
    latent_dim = hyperparameters["latent_dim"]
    config.batch_size = hyperparameters["batch_size"]

    degree = compute_sh_degree(config)
    vae = VAE(nbins=nbins, max_degree=degree, latent_dim=latent_dim).to(device)
    netD = Discriminator(nbins=nbins).to(device)
    if ('cuda' in str(device)) and (ngpu > 1):
        netD = (nn.DataParallel(netD, list(range(ngpu)))).to(device)
        vae = nn.DataParallel(vae, list(range(ngpu))).to(device)
    
    optEncoder, optDecoder, optD = get_optimizer(hyperparameters, vae, netD)

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
        vae.load_state_dict(checkpoint_data["VAE_state_dict"])
        netD.load_state_dict(checkpoint_data["discriminator_state_dict"])
        optEncoder.load_state_dict(checkpoint_state["optEncoder_state_dict"])
        optDecoder.load_state_dict(checkpoint_state["optDecoder_state_dict"])
        optD.load_state_dict(checkpoint_data["optD_state_dict"])
    else:
        start_epoch = 0

    train_prefetcher, _ = load_hrtf(config)

    for epoch in range(start_epoch, 200):
        with open("log.txt", "a") as f:
            f.write(f"Epoch: {epoch}\n")
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
                feature_sim_loss_D = gamma * ((feature_recon - feature_real) ** 2).mean() # feature loss
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
                feature_sim_loss_E = beta * ((feature_recon - feature_real) ** 2).mean() # feature loss
                train_loss_Enc_sim += feature_sim_loss_E.item()
                err_enc = prior_loss + feature_sim_loss_E
                train_loss_Enc += err_enc.item()
                # Update encoder
                optEncoder.zero_grad()
                err_enc.backward()
                optEncoder.step()

            if batch_index % 10 == 0:
                with open("log.txt", "a") as f:
                    f.write(f"{batch_index}/{len(train_prefetcher)}\n")
                    f.write(f"dis: {gan_loss.item()}\t dec: {err_dec.item()}\t enc: {err_enc.item()}\n")
                    f.write(f"D_real: {loss_D_hr.item()}, D_fake: {loss_D_recon.item()}\n")
                    f.write(f"content loss: {content_loss.item()}, sim_D: {feature_sim_loss_D.item()}, gan loss: {gan_loss_dec.item()}\n")
                    f.write(f"prior: {prior_loss.item()}, sim_E: {feature_sim_loss_E.item()}\n\n")

            batch_data = train_prefetcher.next()

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

        print(f"Avearge epoch loss, discriminator: {avg_train_loss_Dis}, decoder: {avg_train_loss_Dec}, encoder: {avg_train_loss_Enc}")
        print(f"Avearge epoch loss, D_real: {avg_train_loss_Dis_hr}, D_fake: {avg_train_loss_Dis_recon}")
        print(f"Avearge content loss: {avg_train_loss_Dec_content},  decoder similarity loss: {avg_train_loss_Dec_sim}, gan loss: {avg_train_loss_Dec_gan}")
        print(f"Average prior loss: {avg_train_loss_Enc_prior}, encoder similarity loss: {avg_train_loss_Enc_sim}\n")

        checkpoint_data = {
            "epoch": epoch,
            "VAE_state_dict": vae.state_dict(),
            "discriminator_state_dict": netD.state_dict(),
            "optD_state_dict": optD.state_dict(),
            "optEncoder_state_dict": optEncoder.state_dict(),
            "optDecoder_state_dict": optDecoder.state_dict(),
        }
        checkpoint = Checkpoint.from_dict(checkpoint_data)

        session.report(
            {"loss_Dis": avg_train_loss_Dis, "loss_Dec": avg_train_loss_Dec, "loss_Enc": avg_train_loss_Enc},
            checkpoint=checkpoint,
        )
    print("Finished Training")


def main(config, num_samples=20, gpus_per_trial=1):
    hyperparameters = {
        "batch_size": tune.choice([2, 4, 8, 16]),
        "optimizer": tune.choice(["adam", "rmsprop"]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "alpha": tune.loguniform(1e-2, 1),
        "gamma": tune.choice([0.15, 1.5, 15]),
        "beta": tune.choice([0.05, 0.5, 5]),
        "latent_dim": tune.choice([10, 50, 100]),
        "critic_iter": tune.choice([3, 4, 5, 6]),
    }

    scheduler = ASHAScheduler(
        metric=["loss_Dis", "loss_Dec", "loss_Enc"],
        mode="min",
        max_t=config.num_epochs,
        grace_period=30,
        reduction_factor=2,
    )

    result = tune.run(
        partial(train_vae_gan),
        resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
        config=hyperparameters,
        num_samples=num_samples,
        scheduler=scheduler,
    )

    best_trial = result.get_best_trial(["loss_Dis", "loss_Dec", "loss_Enc"], "min", "last")
    print(f"Best trail combination: {best_trial.config}")
    print(f"Best trail final discriminator loss: {best_trial.last_result['loss_Dis']}")
    print(f"Best trail final decoder loss: {best_trial.last_result['loss_Dec']}")
    print(f"Best trail final encoder loss: {best_trial.last_result['loss_Enc']}")

    nbins = config.nbins_hrtf
    if config.merge_flag:
        nbins = config.nbins_hrtf * 2

    degree = compute_sh_degree(config)
    best_trained_model = VAE(nbins=nbins, max_degree=degree, latent_dim=best_trial.config["latent_dim"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint = best_trial.checkpoint.to_air_checkpoint()
    best_checkpoint_data = best_checkpoint.to_dict()

    best_trained_model.load_state_dict(best_checkpoint_data["VAE_state_dict"])
    path = config.path
    with torch.no_grad():
        torch.save(best_checkpoint_data["VAE_state_dict"], f'{path}/vae.pt')
    
    _, test_prefetcher = load_hrtf(config)
    print("Loaded all datasets successfully.")
    test(config, test_prefetcher)
    run_lsd_evaluation(config, config.valid_path)
    run_localisation_evaluation(config, config.valid_path)

if __name__ == "__main__":
    print("using cuda? ", torch.cuda.is_available())
    tag = "ari-upscale-4"
    config = Config(tag, using_hpc=True)
    main(config)

    
