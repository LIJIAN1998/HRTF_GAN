import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
    def __init__(self, channels, out_channels, kernel, stride, padding, activation='prelu'):
        super(IterativeBlock, self).__init__()
        self.up1 = UpBlock(channels, kernel, stride, padding, activation=activation)
        self.down1 = DownBlock(channels, kernel, stride, padding, activation=activation)
        self.up2 = UpBlock(channels, kernel, stride, padding, activation=activation)
        self.down2 = D_DownBlock(channels, kernel, stride, padding, 2, activation=activation)
        self.up3 = D_UpBlock(channels, kernel, stride, padding, 2, activation=activation)
        self.down3 = D_DownBlock(channels, kernel, stride, padding, 3, activation=activation)
        self.up4 = D_UpBlock(channels, kernel, stride, padding, 3, activation=activation)
        # self.down4 = D_DownBlock(channels, kernel, stride, padding, 4, activation=activation)
        # self.up5 = D_UpBlock(channels, kernel, stride, padding, 4, activation=activation)
        # self.down5 = D_DownBlock(channels, kernel, stride, padding, 5, activation=activation)
        # self.up6 = D_UpBlock(channels, kernel, stride, padding, 5, activation=activation)
        self.out_conv = ConvBlock(4*channels, out_channels, 3, 1, 1, activation=None)
        
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

        # concat_h = torch.cat((h, concat_h), 1)
        # l = self.down4(concat_h)

        # concat_l = torch.cat((l, concat_l), 1)
        # h = self.up5(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        out = self.out_conv(concat_h)

        return out

class SimpleE(nn.Module):
    def __init__(self, nbins, latent_dim):
        super(SimpleE, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(nbins, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, True),
        )
        self.fc = nn.Sequential(
            nn.Linear(512*25, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, latent_dim)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        return z

class ResBlock(nn.Module):
    def __init__(self, in_channnels, out_channels, stride=1, expansion=1, identity_downsample=None):
        super(ResBlock, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channnels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels),
        ) 
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels * self.expansion, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels * self.expansion)
        )
        self.relu = nn.ReLU()
        self.prelu = nn.PReLU()
        self.leakyRelu = nn.LeakyReLU(0.2)
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()
        x = self.conv1(x)
        # x = self.relu(x)
        # x = self.leakyRelu(x)
        x = self.prelu(x)
        x = self.conv2(x)
        
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.prelu(x)
        # x = self.prelu(x)
        # x = self.leakyRelu(x)
        return x

class ResEncoder(nn.Module):
    def __init__(self, block, nbins: int, order: int, latent_dim: int):
        super(ResEncoder, self).__init__()
        self.coefficient = (order + 1) ** 2
        num_blocks = 2
        self.expansion = 1
        self.in_channels = 256
        self.conv1 = nn.Sequential(
            nn.Conv1d(nbins, self.in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(self.in_channels),
            # nn.ReLU(),
            nn.PReLU(),
            # nn.LeakyReLU(0.2, True)
        )
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        res_layers = []
        # self.num_encode_layers = int(np.log2(self.coefficient // 2)) + 1
        self.num_encode_layers = 4
        if self.coefficient in [16 ,4]:
            self.num_encode_layers -= 1
        res_layers.append(self._make_layer(block, 256, num_blocks))
        for i in range(self.num_encode_layers):
            res_layers.append(self._make_layer(block, 512, num_blocks, stride=2))
        self.res_layers = nn.Sequential(*res_layers)
        self.fc = nn.Sequential(nn.Linear(512*25, 512),
                                nn.BatchNorm1d(512),
                                # nn.ReLU(True),
                                nn.PReLU(),
                                # nn.LeakyReLU(0.2, True),
                                nn.Linear(512, latent_dim))
    
    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * self.expansion, kernel_size=1, stride=stride),
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
        kernel = 4
        stride = 2
        padding = 1
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512*16),
            nn.BatchNorm1d(512*16),
            nn.ReLU(True),
            # nn.PReLU(),
            Reshape(-1, 512, 16),
        )
        activation = 'tanh'

        self.conv0 = ConvBlock(512, base_channels, 3, 1, 1)
        # self.conv0 = ConvBlock(512, num_features, 3, 1, 1)
        # self.conv1 = ConvBlock(num_features, base_channels, 1, 1, 0)

        # Back-projection stages
        self.up1 = IterativeBlock(base_channels, base_channels*2, kernel, stride, padding)
        self.up2 = IterativeBlock(base_channels*2, base_channels*4, kernel, stride, padding)
        self.up3 = IterativeBlock(base_channels*4, base_channels*8, kernel, stride, padding)
        self.up4 = IterativeBlock(base_channels*8, base_channels*8, kernel, stride, padding)
        self.up5 = IterativeBlock(base_channels*8, base_channels*8, kernel, stride, padding)
        
        # Reconstruction
        self.out_conv = ConvBlock(base_channels*8, nbins, 3, 1, 1, activation=None)
        self.trim = Trim(max_num_coefficient)

        self.init_parameters()

    def forward(self, x):
        x = self.fc(x)
        x = self.conv0(x)
        # x = self.conv1(x)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
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
            nn.Linear(latent_dim, 512*13),
            nn.BatchNorm1d(512*13),
            nn.Tanh(),
            Reshape(-1, 512, 13),
            nn.ConvTranspose1d(512, 512, kernel_size=3, stride=1, bias=False), # 15
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.ConvTranspose1d(512, 512, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            # 512 x 31
            nn.ConvTranspose1d(512, 256, kernel_size=3, stride=1, bias=False),  # 33
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.ConvTranspose1d(256, 256, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            # 256 x 67
            nn.ConvTranspose1d(256, 256, kernel_size=3, stride=1, bias=False),  # 69
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.ConvTranspose1d(256, 256, kernel_size=3, stride=2, bias=False),  
            nn.BatchNorm1d(256),
            nn.Tanh(),
            # 256 x 139
            nn.ConvTranspose1d(256, nbins, kernel_size=3, stride=1, bias=False),  # 141
            nn.BatchNorm1d(nbins),
            nn.Tanh(),
            nn.ConvTranspose1d(nbins, nbins, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm1d(nbins),
            nn.Tanh(),
            # nbins x 283
            nn.ConvTranspose1d(nbins, nbins, kernel_size=3, stride=1, bias=False), # 285
            nn.BatchNorm1d(nbins),
            nn.Tanh(),
            nn.ConvTranspose1d(nbins, nbins, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(nbins),
            # nn.PReLU(),
            # nbins x 571
            Trim(self.num_coefficient)
        )

    def forward(self,  x: torch.Tensor) -> torch.Tensor:
        x = self.decoder(x)
        return x

class AutoEncoder(nn.Module):
    def __init__(self, nbins: int, in_order: int, latent_dim: int, base_channels: int, num_features: int, out_oder: int=22):
        super(AutoEncoder, self).__init__()

        self.encoder = ResEncoder(ResBlock, nbins, in_order, latent_dim)
        # self.encoder = SimpleE(nbins, latent_dim)
        self.decoder = D_DBPN(nbins, base_channels=base_channels, num_features=num_features,
                              latent_dim=latent_dim, max_order=out_oder)
        # self.decoder = Decoder(nbins, latent_dim, out_oder)

        # self.init_parameters()

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

        padding = 0
        self.features = nn.Sequential(
            # input size: nbins x 529     484
            nn.Conv1d(self.nbins, 64, kernel_size=3, padding=1, stride=1, bias=False),
            # nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, True),
            # nbins x 265         242
            nn.Conv1d(64, 128, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, True),
            # nbins x 133      121
            nn.Conv1d(128, 256, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, True),
            # nbins x  67     61
            nn.Conv1d(256, 512, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, True),
            # nbins x 34   31
            # nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=1, bias=False),
            # nn.BatchNorm1d(512),
            # nn.LeakyReLU(0.2, True),
            # nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=2, bias=False),
            # nn.BatchNorm1d(512),
            # nn.LeakyReLU(0.2, True),
            # nbins x 16
        )

        # self.features = nn.Sequential(
        #     # input size: nbins x 812       841
        #     nn.Conv1d(self.nbins, 64, kernel_size=3, padding=1, stride=1, bias=False),
        #     nn.BatchNorm1d(64),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Conv1d(64, 64, kernel_size=3, padding=1, stride=1, bias=False),
        #     nn.BatchNorm1d(64),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Conv1d(64, 64, kernel_size=3, padding=1, stride=2, bias=False),
        #     nn.BatchNorm1d(64),
        #     nn.LeakyReLU(0.2, True),
        #     # nbins x 406   421
        #     nn.Conv1d(64, 128, kernel_size=3, padding=1, stride=1, bias=False),
        #     nn.BatchNorm1d(128),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Conv1d(128, 128, kernel_size=3, padding=1, stride=2, bias=False),
        #     nn.BatchNorm1d(128),
        #     nn.LeakyReLU(0.2, True),
        #     # nbins x 203   211
        #     nn.Conv1d(128, 256, kernel_size=3, padding=1, stride=1, bias=False),
        #     nn.BatchNorm1d(256),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Conv1d(256, 256, kernel_size=3, padding=1, stride=2, bias=False),
        #     nn.BatchNorm1d(256),
        #     nn.LeakyReLU(0.2, True),
        #     # nbins x 102   106
        #     nn.Conv1d(256, 512, kernel_size=3, padding=1, stride=1, bias=False),
        #     nn.BatchNorm1d(512),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=2, bias=False),
        #     nn.BatchNorm1d(512),
        #     nn.LeakyReLU(0.2, True),
        #     # nbins x 51    53
        #     nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=1, bias=False),
        #     nn.BatchNorm1d(512),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=2, bias=False),
        #     nn.BatchNorm1d(512),
        #     nn.LeakyReLU(0.2, True),
        #     # nbins x 26    27
        #     # nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=1, bias=False),
        #     # nn.BatchNorm1d(512),
        #     # nn.LeakyReLU(0.2, True),
        #     # nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=2, bias=False),
        #     # nn.BatchNorm1d(512),
        #     # nn.LeakyReLU(0.2, True),
        #     # nbins x 34
        # )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 31, 512),
            # nn.BatchNorm1d(512),
            # nn.LeakyReLU(0.2, True),
            # nn.Linear(512, 512),
            # nn.BatchNorm1d(512),
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
    G = AutoEncoder(nbins=256, in_order=19, latent_dim=128, base_channels=64, num_features=512, out_oder=21)
    x = G(x)
    print(x.shape)
    D = Discriminator(256)
    x = D(x)
    print(x.shape)