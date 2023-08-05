import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.custom_conv import CubeSpherePadding2D, CubeSphereConv2D

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

class UpsampleBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super(UpsampleBlock, self).__init__()
        self.upsample_block_1 = nn.Sequential(
            CubeSpherePadding2D(1),
            CubeSphereConv2D(channels, channels * 4, (3, 3), (1, 1))
        )
        self.upsample_block_2 = nn.Sequential(
            nn.PixelShuffle(2),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.upsample_block_1(x)
        print("upsample 1: ", out1.shape)
        out = self.upsample_block_2(torch.permute(out1, dims=(0, 2, 1, 3, 4)))
        print("upsample 2: ", out.shape)
        x = torch.permute(out, dims=(0, 2, 1, 3, 4))
        print("permute: ", x.shape)

        return torch.permute(out, dims=(0, 2, 1, 3, 4))


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
        res_layers = []
        self.num_encode_layers = int(np.log2(self.coefficient // 2)) + 1
        if self.coefficient == 4:
            self.num_encode_layers -= 1
        res_layers.append(self._make_layer(block, 256, num_blocks))
        for i in range(self.num_encode_layers):
            res_layers.append(self._make_layer(block, 512, num_blocks, stride=2))
        self.res_layers = nn.Sequential(*res_layers)

        self.compute_mean = nn.Linear(512*2, latent_dim)
        self.compute_log_var = nn.Linear(512*2, latent_dim)
    
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
        x = self.res_layers(x)
        out = x.view(x.size(0), -1)
        return out
    
    def reparametrize(self, mu, logvar):
        epsilon = torch.randn(mu.size(0), mu.size(1)).to(mu.device)
        z = mu + epsilon * torch.exp(logvar/2.)
        return z
    
    def forward(self, x):
        x = self.encode(x)
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
            nn.Linear(self.latent_dim, 512*2),
            Reshape(-1, 512, 2),
            nn.ConvTranspose1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.ConvTranspose1d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            # 512x7
            nn.ConvTranspose1d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.ConvTranspose1d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            # 512x13
            nn.ConvTranspose1d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.ConvTranspose1d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            # 256x25
            nn.ConvTranspose1d(256, self.nbins, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(self.nbins),
            nn.PReLU(),
            nn.ConvTranspose1d(self.nbins, self.nbins, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(self.nbins),
            nn.PReLU(),
            # nbins x 49
            nn.ConvTranspose1d(self.nbins, self.nbins, kernel_size=3, stride=1, padding=0, bias=False), # nbins x 51 
            nn.BatchNorm1d(self.nbins),
            nn.PReLU(),
            nn.ConvTranspose1d(self.nbins, self.nbins, kernel_size=3, stride=2, output_padding=1, bias=False),
            nn.BatchNorm1d(self.nbins),
            nn.PReLU(),
            # nbins x 104
            nn.ConvTranspose1d(self.nbins, self.nbins, kernel_size=3, stride=1, padding=0, bias=False), # nbins x 106
            nn.BatchNorm1d(self.nbins),
            nn.PReLU(),
            nn.ConvTranspose1d(self.nbins, self.nbins, kernel_size=3, stride=2, output_padding=1, bias=False),
            nn.BatchNorm1d(self.nbins),
            nn.PReLU(),
            # nbins x 214
            nn.ConvTranspose1d(self.nbins, self.nbins, kernel_size=3, stride=1, padding=0, bias=False), # nbins x 216
            nn.BatchNorm1d(self.nbins),
            nn.PReLU(),
            nn.ConvTranspose1d(self.nbins, self.nbins, kernel_size=3, stride=2, output_padding=1, bias=False),
            nn.BatchNorm1d(self.nbins),
            nn.PReLU(),
            # nbins x 434
            nn.ConvTranspose1d(self.nbins, self.nbins, kernel_size=3, stride=1, padding=0, bias=False), # nbins x 436
            nn.BatchNorm1d(self.nbins),
            nn.PReLU(),
            nn.ConvTranspose1d(self.nbins, self.nbins, kernel_size=3, stride=2, output_padding=1, bias=False),
            nn.BatchNorm1d(self.nbins),
            nn.PReLU(),
            # nbins x 874
            Trim(self.num_coefficient) # nbins x 841
        )

        self.classifier = nn.Softplus()

    def forward(self,  x: torch.Tensor) -> torch.Tensor:
        x = self.decoder(x)
        # print(x.shape)
        out = self.classifier(x)
        return out

class Discriminator(nn.Module):
    def __init__(self, nbins: int) -> None:
        super(Discriminator, self).__init__()
        self.nbins = nbins

        self.features = nn.Sequential(
            # input size: nbins x 841
            nn.Conv1d(self.nbins, 64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, True),
            # nbins x 421
            nn.Conv1d(64, 128, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, True),
            # nbins x 211
            nn.Conv1d(128, 256, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, True),
            # nbins x 106
            nn.Conv1d(256, 512, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, True),
            # nbins x 53
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 53, 512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self,  x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x1 = x
        out = self.classifier(x)
        return out, x1

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
        self.discriminator = Discriminator(self.nbins)

    def forward(self,  x: torch.Tensor) -> torch.Tensor:
        mu, log_var, z = self.encoder(x)
        recon = self.decoder(z)
        return mu, log_var, recon
       

if __name__ == '__main__':
    x1 = torch.randn(1, 128, 49)
    x2 = torch.randn(1, 128, 100)
    x3 = torch.randn(1, 128, 196)
    x4 = torch.randn(1, 128, 400)
    inputs = [x1, x2, x3, x4]
    degrees = [6, 9, 13, 19]
    # for i, d in enumerate(degrees):
    #     model = Encoder(128, d, 10)
    #     x = inputs[i]
    #     out = model(x)
    #     print("out: ", out.shape)

    # model = Decoder(128, 10)
    # z = torch.randn(1, 10)
    # out = model(z)
    # print(out.shape)
    for i, d in enumerate(degrees):
        model = VAE(128, d, 10)
        x = inputs[i]
        out = model(x)
        print("out: ", out.shape)

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

        