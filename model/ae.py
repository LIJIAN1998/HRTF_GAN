import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from base_blocks import *

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
    def __init__(self, block, nbins: int, order: int, latent_dim: int) -> None:
        super(ResEncoder, self).__init__()
        self.coefficient = (order + 1) ** 2
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
        if self.coefficient in [16 ,4]:
            self.num_encode_layers -= 1
        res_layers.append(self._make_layer(block, 256, num_blocks))
        for i in range(self.num_encode_layers):
            res_layers.append(self._make_layer(block, 512, num_blocks, stride=2))
        self.res_layers = nn.Sequential(*res_layers)
        self.fc = nn.Sequential(nn.Linear(512*2, 512, bias=False),
                                nn.BatchNorm1d(512, momentum=0.9),
                                nn.ReLU(True),
                                nn.Linear(512, latent_dim))
    
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
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.res_layers(x)
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        return z
    
class D_DBPN(nn.Module):
    def __init__(self, nbins, base_channels, num_features, latent_dim, max_order):
        super(D_DBPN, self).__init__()

        max_num_coefficient = (max_order + 1) ** 2
        kernel = 8
        stride = 4
        padding = 2
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512*2),
            nn.BatchNorm1d(512*2),
            nn.ReLU(True),
        )
        self.conv0 = ConvBlock(512, num_features, 3, 1, 1)
        self.conv1 = ConvBlock(num_features, base_channels, 1, 1, 0)

        # Back-projection stages
        self.up1 = IterativeBlock(base_channels, kernel, stride, padding)
        self.up2 = IterativeBlock(base_channels, kernel, stride, padding)
        self.up3 = IterativeBlock(base_channels, kernel, stride, padding)
        self.up4 = IterativeBlock(base_channels, kernel, stride, padding)
        
        # Reconstruction
        self.out_conv = ConvBlock(base_channels, nbins, 3, 1, 1, activation=None)
        self.trim = Trim(max_num_coefficient)

        self.init_parameters()

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 512, 2)
        x = self.conv0(x)
        x = self.conv1(x)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
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

class Decoder(nn.Module):
    def __init__(self, nbins: int, latent_dim: int, out_degree: int=28) -> None:
        super(Decoder, self).__init__()
        self.nbins = nbins
        self.latent_dim = latent_dim
        self.num_coefficient = (out_degree + 1) ** 2

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512*2),
            nn.BatchNorm1d(512*2),
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
        # self.classifier = nn.Softplus()

    def forward(self,  x: torch.Tensor) -> torch.Tensor:
        x = self.decoder(x)
        # out = self.classifier(x)
        return x
    
class AutoEncoder(nn.Module):
    def __init__(self, nbins: int, in_order: int, latent_dim: int, base_channels: int, num_features: int, out_oder: int=28):
        super(AutoEncoder, self).__init__()

        self.encoder = ResEncoder(ResBlock, nbins, in_order, latent_dim)
        self.decoder = D_DBPN(nbins, base_channels=base_channels, num_features=num_features, latent_dim=latent_dim, max_order=out_oder)
        # self.decoder = Decoder(nbins, latent_dim, out_oder)


    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
                if hasattr(m, 'weight') and m.weight is not None and m.weight.requires_grad:
                    scale = 1.0 /np.sqrt(np.prod(m.weight.shape[1:]))
                    scale /= np.sqrt(3)
                    nn.init.uniform_(m.weight, -scale, scale)
                if hasattr(m, 'bias') and m.bias is not None and m.bias.requires_grad:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out
    
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
            # input size: nbins x 242
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
        )

        # self.features = nn.Sequential(
        #     # input size: nbins x 841
        #     nn.Conv1d(self.nbins, 64, kernel_size=3, padding=1, stride=1, bias=False),
        #     nn.BatchNorm1d(64),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Conv1d(64, 64, kernel_size=3, padding=1, stride=1, bias=False),
        #     nn.BatchNorm1d(64),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Conv1d(64, 64, kernel_size=3, padding=1, stride=2, bias=False),
        #     nn.BatchNorm1d(64),
        #     nn.LeakyReLU(0.2, True),
        #     # nbins x 421
        #     nn.Conv1d(64, 128, kernel_size=3, padding=1, stride=1, bias=False),
        #     nn.BatchNorm1d(128),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Conv1d(128, 128, kernel_size=3, padding=1, stride=2, bias=False),
        #     nn.BatchNorm1d(128),
        #     nn.LeakyReLU(0.2, True),
        #     # nbins x 211
        #     nn.Conv1d(128, 256, kernel_size=3, padding=1, stride=1, bias=False),
        #     nn.BatchNorm1d(256),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Conv1d(256, 256, kernel_size=3, padding=1, stride=2, bias=False),
        #     nn.BatchNorm1d(256),
        #     nn.LeakyReLU(0.2, True),
        #     # nbins x 106
        #     nn.Conv1d(256, 512, kernel_size=3, padding=1, stride=1, bias=False),
        #     nn.BatchNorm1d(512),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=2, bias=False),
        #     nn.BatchNorm1d(512),
        #     nn.LeakyReLU(0.2, True),
        #     # nbins x 53
        #     nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=1, bias=False),
        #     nn.BatchNorm1d(512),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=2, bias=False),
        #     nn.BatchNorm1d(512),
        #     nn.LeakyReLU(0.2, True),
        #     # nbins x 27
        #     nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=1, bias=False),
        #     nn.BatchNorm1d(512),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=2, bias=False),
        #     nn.BatchNorm1d(512),
        #     nn.LeakyReLU(0.2, True),
        #     # nbins x 34
        # )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 31, 512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # x1 = x
        out = self.classifier(x)
        return out
    

if __name__ == '__main__':
    x = torch.randn(2, 256, 400)
    G = AutoEncoder(nbins=256, in_order=19, latent_dim=128, base_channels=256, num_features=512, out_oder=21)
    x = G(x)
    print(x.shape)
    D = Discriminator(256)
    x = D(x)
    print(x.shape)