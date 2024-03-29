from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from model.base_blocks import *

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)
    
class Trim(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x[:,:,:self.shape]

class IterativeBlock(nn.Module):
    def __init__(self, channels, kernel, stride, padding):
        super(IterativeBlock, self).__init__()
        self.up1 = UpBlock(channels, kernel, stride, padding)
        self.down1 = DownBlock(channels, kernel, stride, padding)
        self.up2 = UpBlock(channels, kernel, stride, padding)
        self.down2 = D_DownBlock(channels, kernel, stride, padding, 2)
        self.up3 = D_UpBlock(channels, kernel, stride, padding, 2)
        self.down3 = D_DownBlock(channels, kernel, stride, padding, 3)
        self.up4 = D_UpBlock(channels, kernel, stride, padding, 3)
        self.down4 = D_DownBlock(channels, kernel, stride, padding, 4)
        self.up5 = D_UpBlock(channels, kernel, stride, padding, 4)
        self.out_conv = ConvBlock(5*channels, channels, 3, 1, 1, activation=None)
        
    def forward(self, x):
        h1 = self.up1(x)
        l1 = self.down1(h1)
        h2 = self.up2(x)
        
        concat_h = torch.cat((h2, h1), 1)
        l = self.down2(concat_h)
        
        concat_l = torch.cat((l, l1), 1)
        h = self.up3(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down3(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up4(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down4(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up5(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        out = self.out_conv(concat_h)

        return out

class D_DBPN(nn.Module):
    def __init__(self, channels, base_channels, num_features, scale_factor, max_order):
        super(D_DBPN, self).__init__()

        max_num_coefficient = (max_order + 1) ** 2
        if scale_factor in [2, 4, 8]:
            # order = 7, coefficient of size 64
            kernel = 4
            stride = 2
            padding = 1
            num_blocks = int(np.log2(max_num_coefficient/64)) + 1  # base 2, 2 IterativeBlock
        elif scale_factor in [16 ,32, 48]:
            # order = 3, coefficient of size 16
            kernel = 8
            stride = 4
            padding = 2
            num_blocks = int(np.log2(max_num_coefficient/16) / np.log2(4)) # base 4, 2 IterativeBlock
        elif scale_factor in [72, 108, 216]:
            # order = 1, coefficient of size 4
            kernel = 8
            stride = 4
            padding = 2
            num_blocks = int(np.log2(max_num_coefficient/4) / np.log2(4)) # base 4, 3 IterativeBlock

        self.conv0 = ConvBlock(channels, num_features, 3, 1, 1)
        self.conv1 = ConvBlock(num_features, base_channels, 1, 1, 0)

        # Back-projection stages
        blocks = []
        for _ in range(num_blocks):
            blocks.append(IterativeBlock(base_channels, kernel, stride, padding))
        self.up_downsample = nn.Sequential(*blocks)
        
        # Reconstruction
        self.out_conv = ConvBlock(base_channels, channels, 3, 1, 1, activation=None)
        self.trim = Trim(max_num_coefficient)

        self.init_parameters()

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)

        x = self.up_downsample(x)
        x = self.out_conv(x)
        out = self.trim(x)
        return out

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                if hasattr(m, 'weight') and m.weight is not None and m.weight.requires_grad:
                    nn.init.kaiming_normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None and m.bias.requires_grad:
                    nn.init.constant_(m.bias, 0.0)
        

class ResBlock(nn.Module):
    def __init__(self, in_channnels, out_channels, stride=1, expansion=1, identity_downsample=None):
        super(ResBlock, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channnels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
        ) 
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels * self.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels * self.expansion)
        )
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x

class ResEncoder(nn.Module):
    def __init__(self, block, nbins: int, max_degree: int, latent_dim: int) -> None:
        super(ResEncoder, self).__init__()
        self.coefficient = (max_degree + 1) ** 2
        num_blocks = 2
        self.expansion = 1
        self.in_channels = 256
        self.conv1 = nn.Sequential(
            nn.Conv1d(nbins, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(self.in_channels),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        res_layers = []
        self.num_encode_layers = int(np.log2(self.coefficient // 2)) + 1
#         self.num_encode_layers = 4
        if self.coefficient == 4:
            self.num_encode_layers -= 1
        res_layers.append(self._make_layer(block, 256, num_blocks))
        for i in range(self.num_encode_layers):
            res_layers.append(self._make_layer(block, 512, num_blocks, stride=2))
        self.res_layers = nn.Sequential(*res_layers)
        self.fc = nn.Sequential(nn.Linear(512*2, 512, bias=False),
                                nn.BatchNorm1d(512, momentum=0.9),
                                nn.ReLU(True))
        self.compute_mean = nn.Linear(512, latent_dim)
        self.compute_log_var = nn.Linear(512, latent_dim)
    
    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * self.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, self.expansion, downsample))
        self.in_channels = out_channels * self.expansion

        for i in range(num_blocks-1):
            layers.append(block(self.in_channels, out_channels, expansion=self.expansion))
        return nn.Sequential(*layers)
    
    def encode(self, x):
        x = self.conv1(x)
#         x = self.maxpool(x)
        x = self.res_layers(x)
        out = x.view(x.size(0), -1)
        return out
    
    def reparametrize(self, mu, logvar):
        epsilon = Variable(torch.randn(mu.size(0), mu.size(1)).to(mu.device), requires_grad=True)
        z = mu + epsilon * torch.exp(logvar/2.)
        return z
    
    def forward(self, x):
        x = self.encode(x)
        x = self.fc(x)
        mu, log_var = self.compute_mean(x), self.compute_log_var(x)
        z = self.reparametrize(mu, log_var)
        return mu, log_var, z


class EncodingBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(EncodingBlock, self).__init__()
        self.encode_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode_block(x)

class Encoder(nn.Module):
    def __init__(self, nbins: int, max_degree: int, latent_dim: int) -> None:
        super(Encoder, self).__init__()

        self.nbins = nbins
        self.coefficient = (max_degree + 1) ** 2
        self.latent_dim = latent_dim

        in_channels = self.nbins
        out_channels = 256

        if self.coefficient == 4:
            self.encode_blocks = nn.Sequential(
                EncodingBlock(in_channels, out_channels),
                nn.Conv1d(out_channels, out_channels*2, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm1d(out_channels*2),
                nn.ReLU(),
                nn.Conv1d(out_channels*2, out_channels*2, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm1d(out_channels*2),
                nn.ReLU(),
            )
        else:
            self.num_encode_blocks = int(np.log2(self.coefficient // 2)) + 1
            encode_layers = []
            for _ in range(self.num_encode_blocks):
                encode_layers.append(EncodingBlock(in_channels, out_channels))
                in_channels = out_channels
                out_channels = min(out_channels*2, 512)
            self.encode_blocks = nn.Sequential(*encode_layers)

        self.compute_mean = nn.Linear(512*2, latent_dim)
        self.compute_log_var = nn.Linear(512*2, latent_dim)

    def encode(self, x):
        x = self.encode_blocks(x)
        out = x.view(x.size(0), -1)
        return out
    
    def reparametrize(self, mu, logvar):
        epsilon = torch.randn(mu.size(0), mu.size(1)).to(mu.device)
        z = mu + epsilon * torch.exp(logvar/2.)
        return z

    def forward(self, x):
        # print("input: ", x.shape)
        # print("num encode block: ", self.num_encode_blocks)
        x = self.encode(x)
        # print("encoded: ", x.shape)
        mu, log_var = self.compute_mean(x), self.compute_log_var(x)
        # print('mu: ', mu.shape)
        # print('log_var: ', log_var.shape)
        z = self.reparametrize(mu, log_var)
        return mu, log_var, z

class Decoder(nn.Module):
    def __init__(self, nbins: int, latent_dim: int, out_degree: int=28) -> None:
        super(Decoder, self).__init__()
        self.nbins = nbins
        self.latent_dim = latent_dim
        self.num_coefficient = (out_degree + 1) ** 2

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512*2),
            nn.BatchNorm1d(512*2, momentum=0.9),
            nn.ReLU(True),
            Reshape(-1, 512, 2),
            nn.ConvTranspose1d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.ConvTranspose1d(512, 512, kernel_size=3, stride=2, output_padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            # 512 x 6
            nn.ConvTranspose1d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.ConvTranspose1d(256, 256, kernel_size=3, stride=2, output_padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            # 256 x 14
            nn.ConvTranspose1d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.ConvTranspose1d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            # 256 x 28
            nn.ConvTranspose1d(256, nbins, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(nbins),
            nn.PReLU(),
            nn.ConvTranspose1d(nbins, nbins, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(nbins),
            nn.PReLU(),
            # nbins x 55
            nn.ConvTranspose1d(nbins, nbins, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(nbins),
            nn.PReLU(),
            nn.ConvTranspose1d(nbins, nbins, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(nbins),
            nn.PReLU(),
            # nbins x 109
            nn.ConvTranspose1d(nbins, nbins, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(nbins),
            nn.PReLU(),
            nn.ConvTranspose1d(nbins, nbins, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(nbins),
            nn.PReLU(),
            # nbins x 217
            nn.ConvTranspose1d(nbins, nbins, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(nbins),
            nn.PReLU(),
            nn.ConvTranspose1d(nbins, nbins, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(nbins),
            nn.PReLU(),
            # nbins x 433
            nn.ConvTranspose1d(nbins, nbins, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(nbins),
            nn.PReLU(),
            nn.ConvTranspose1d(nbins, nbins, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(nbins),
            # nbins x 865
            Trim(self.num_coefficient)
        )

        # self.decoder = nn.Sequential(
        #     nn.Linear(self.latent_dim, 512*32),
        #     Reshape(-1, 512, 32),
        #     nn.ConvTranspose1d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm1d(512),
        #     nn.PReLU(),
        #     nn.ConvTranspose1d(512, 512, kernel_size=3, stride=2, output_padding=1, bias=False),
        #     nn.BatchNorm1d(512),
        #     nn.PReLU(),
        #     # 512x66
        #     nn.ConvTranspose1d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm1d(256),
        #     nn.PReLU(),
        #     nn.ConvTranspose1d(256, 256, kernel_size=3, stride=2, output_padding=1, bias=False),
        #     nn.BatchNorm1d(256),
        #     nn.PReLU(),
        #     # 512x134
        #     nn.ConvTranspose1d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm1d(256),
        #     nn.PReLU(),
        #     nn.ConvTranspose1d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm1d(256),
        #     nn.PReLU(),
        #     # 256x267
        #     nn.ConvTranspose1d(256, nbins, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm1d(nbins),
        #     nn.PReLU(),
        #     nn.ConvTranspose1d(nbins, nbins, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm1d(nbins),
        #     nn.PReLU(),
        #     # nbins x 533
        #     nn.ConvTranspose1d(nbins, nbins, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm1d(nbins),
        #     nn.PReLU(),
        #     nn.ConvTranspose1d(nbins, nbins, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm1d(nbins),
        #     nn.PReLU(),
        #     # nbins x 1065
        #     nn.ConvTranspose1d(nbins, nbins, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm1d(nbins),
        #     nn.PReLU(),
        #     nn.ConvTranspose1d(nbins, nbins, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm1d(nbins),
        #     nn.PReLU(),
        #     Trim(self.num_coefficient) # nbins x 2116
        # )

        # self.classifier = nn.Softplus()

    def forward(self,  x: torch.Tensor) -> torch.Tensor:
        x = self.decoder(x)
        # print(x.shape)
        # out = self.classifier(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, nbins: int) -> None:
        super(Discriminator, self).__init__()
        self.nbins = nbins

        self.features = nn.Sequential(
            # input size: nbins x 484
            nn.Conv1d(self.nbins, 64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, True),
            # nbins x 242
            nn.Conv1d(64, 128, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, True),
            # nbins x 121
            nn.Conv1d(128, 256, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, True),
            # nbins x 61
            nn.Conv1d(256, 512, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, True),
            # nbins x 31
            nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, True),
            # nbins x 16
            nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, True),
            # nbins x 8
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 8, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2, True),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self,  x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out

class VAE(nn.Module):
    def __init__(self, nbins: int, max_degree: int, latent_dim: int, out_degree: int=28) -> None:
        super(VAE, self).__init__()
        self.nbins = nbins
        self.max_degree = max_degree
        self.latent_dim = latent_dim
        self.out_degree = out_degree

        # self.encoder = Encoder(self.nbins, self.max_degree, self.latent_dim)
        self.encoder = ResEncoder(ResBlock, self.nbins, self.max_degree, self.latent_dim)
        self.decoder = Decoder(self.nbins, self.latent_dim, self.out_degree)
        # self.discriminator = Discriminator(self.nbins)

        self.init_parameters()

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
                if hasattr(m, 'weight') and m.weight is not None and m.weight.requires_grad:
                    scale = 1.0 /np.sqrt(np.prod(m.weight.shape[1:]))
                    scale /= np.sqrt(3)
                    nn.init.uniform_(m.weight, -scale, scale)
                if hasattr(m, 'bias') and m.bias is not None and m.bias.requires_grad:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self,  x: torch.Tensor) -> torch.Tensor:
        # save original input
        x_original = x
        # encode
        mu, log_var, z = self.encoder(x)
        recon = self.decoder(z)
        return mu, log_var, recon


if __name__ == '__main__':
    x = torch.randn(1, 256, 64)
    generator = D_DBPN(256, 256, 512, 2, 15)
    x = generator(x)
    print(x.shape)
    D = Discriminator(256)
    out = D(x)
    # print("feature shape: ", feature.shape)
    print("classify result: ", out.shape)

    # x1 = torch.randn(1, 128, 49)
    # x2 = torch.randn(1, 128, 100)
    # x3 = torch.randn(1, 128, 196)
    # x4 = torch.randn(1, 128, 400)
    # inputs = [x1, x2, x3, x4]
    # degrees = [6, 9, 13, 19]

    # for i, d in enumerate(degrees):
    #     model = VAE(128, d, 10)
    #     x = inputs[i]
    #     out = model(x)
    #     print("out: ", out.shape)

    # x = torch.randn(1, 128, 841)
    # D = Discriminator(28, 128)
    # out = D(x)
    # print(out.shape)


# class Encoder(nn.Module):
#     def __init__(self, nbins: int, upscale_factor: int, latent_dim: int):
#         super(Encoder, self).__init__()

#         self.nbins = nbins
#         16, 1
#         8, 2
#         4, 4
#         2, 8
#         if upscale_factor == 16:
#             self.encode = nn.Sequential(
#                 CubeSpherePadding2D(1),
#                 CubeSphereConv2D(self.nbins, 256, (3, 3), (1, 1), bias=False),
#                 nn.BatchNorm3d(256),
#                 nn.ReLU(),
#                 CubeSpherePadding2D(1),
#                 CubeSphereConv2D(256, 512, (3, 3), (1, 1), bias=False),
#                 nn.BatchNorm3d(512),
#                 nn.ReLU(),
#             )

#         elif upscale_factor == 8:
#             self.encode = nn.Sequential(
#                 CubeSpherePadding2D(1),
#                 CubeSphereConv2D(self.nbins, 256, (3, 3), (1, 1), bias=False),
#                 nn.BatchNorm3d(256),
#                 nn.ReLU(),
#                 CubeSpherePadding2D(1),
#                 CubeSphereConv2D(256, 512, (3, 3), (2, 2), bias=False),
#                 nn.BatchNorm3d(512),
#                 nn.ReLU(),
#             )
        
#         elif upscale_factor == 4:
#             self.encode = nn.Sequential(
#                 CubeSpherePadding2D(1),
#                 CubeSphereConv2D(self.nbins, 256, (3, 3), (1, 1), bias=False),
#                 nn.BatchNorm3d(256),
#                 nn.ReLU(),
#                 CubeSpherePadding2D(1),
#                 CubeSphereConv2D(256, 512, (3, 3), (2, 2), bias=False),
#                 nn.BatchNorm3d(512),
#                 nn.ReLU(),
#                 CubeSpherePadding2D(1),
#                 CubeSphereConv2D(self.nbins, 256, (3, 3), (1, 1), bias=False),
#                 nn.BatchNorm3d(256),
#                 nn.ReLU(),
#                 CubeSpherePadding2D(1),
#                 CubeSphereConv2D(256, 512, (3, 3), (2, 2), bias=False),
#                 nn.BatchNorm3d(512),
#                 nn.ReLU(),
#             )


#         self.encode = nn.Sequential(
#             CubeSpherePadding2D(1),
#             CubeSphereConv2D(self.nbins, 256, (3, 3), (1, 1), bias=False),
#             nn.BatchNorm3d(256),
#             nn.ReLU(),
#             CubeSpherePadding2D(1),
#             CubeSphereConv2D(256, 256, (3, 3), (2, 2), bias=False),
#             nn.BatchNorm3d(256),
#             nn.ReLU(),
#             CubeSpherePadding2D(1),
#             CubeSphereConv2D(256, 512, (3, 3), (1, 1), bias=False),
#             nn.BatchNorm3d(256),
#             nn.ReLU(),
#             CubeSpherePadding2D(1),
#             CubeSphereConv2D(512, 512, (3, 3), (2, 2), bias=False),
#             nn.BatchNorm3d(256),
#             nn.ReLU(),
#         )

#         self.compute_mean = nn.Linear(512*2*2*5, latent_dim)
#         self.compute_log_var = nn.Linear(512*2*2*5, latent_dim)
    
#     def reparametrize(self, mu, logvar):
#         epsilon = torch.randn(mu.size(0), mu.size(1)).to(mu.get_device())
#         z = mu + epsilon * torch.exp(logvar/2.)
#         return z

#     def forward(self, x):
#         x = self.encode(x)
#         x = torch.flatten(x, 1)
#         mu, log_var = self.compute_mean(x), self.compute_log_var(x)
#         return self.reparametrize(mu, log_var)
    
# class Decoder(nn.Module):
#     def __init__(self, upscale_factor: int, nbins: int, latent_dim: int):
#         super(Decoder, self).__init__()

#         self.nbins = nbins
#         self.ngf = 512
#         self.num_upsampling_blocks = int(np.log(upscale_factor)/np.log(2))
#         self.latent_dim = latent_dim

#         self.fc = nn.Sequential(
#             nn.Linear(latent_dim, 512*2*2*5),
#             nn.BatchNorm1d(num_features=512*2*2*5),
#             nn.ReLU(),
#             Reshape(-1, 512, 5, 2, 2),
#         )

#         upsampling = []
#         for _ in range(self.num_upsampling_blocks):
#             upsampling.append(UpsampleBlock(self.ngf))
#         self.upsampling = nn.Sequential(*upsampling)

#         self.conv = nn.Sequential(
#             CubeSpherePadding2D(1),
#             CubeSphereConv2D(self.ngf, self.nbins, (3, 3), (1, 1))
#         )

#         self.classifier = nn.Softplus()

#     def forward(self, x):
#         x = self.fc(x)
#         x = self.upsampling(x)
#         x = self.classifier(x)

#         return x
    
# class Discriminator(nn.Module):
#     def __init__(self, nbins: int, alpha = 0.2, num_channels = 128):
#         super().__init__()
        
#         self.discriminator = nn.Sequential(
#             nn.Conv2d(3, num_channels, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.LeakyReLU(alpha, inplace=True),
#             nn.Conv2d(num_channels, num_channels*2, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(num_channels*2),
#             nn.LeakyReLU(alpha, inplace=True),
#             nn.Conv2d(num_channels*2, num_channels*4, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(num_channels*4),
#             nn.LeakyReLU(alpha, inplace=True),
#             nn.Conv2d(num_channels*4, 1, kernel_size=4, stride=1, padding=0, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         return self.discriminator(x)

        