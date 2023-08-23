import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from base_blocks import *
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

class ResBlock(nn.Module):
    def __init__(self, channnels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(channnels, channnels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channnels),
        ) 
        self.conv2 = nn.Sequential(
            nn.Conv1d(channnels, channnels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channnels)
        )
        self.prelu = nn.PReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.conv1(x)
        # x = self.leakyRelu(x)
        x = self.prelu(x)
        x = self.conv2(x)
        out = torch.add(x, identity)
        # x = self.prelu(x)
        # x = self.leakyRelu(x)
        return out

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
        h2 = self.up2(l1)
        
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
    
class D_DBPN(nn.Module):
    def __init__(self, nbins, max_order):
        super(D_DBPN, self).__init__()

        max_num_coefficient = (max_order + 1) ** 2
        kernel = 4
        stride = 2
        padding = 1
        base_channels = 256
        
        activation = 'tanh'

        self.conv0 = ConvBlock(nbins, 256, 3, 1, 1)
        self.conv1 = ConvBlock(256, base_channels, 1, 1, 0)

        # Back-projection stages
        self.up1 = IterativeBlock(base_channels, base_channels, kernel, stride, padding)
        self.up2 = IterativeBlock(base_channels, base_channels, kernel, stride, padding)
        self.up3 = IterativeBlock(base_channels, base_channels, kernel, stride, padding)
        # self.up4 = IterativeBlock(base_channels*8, base_channels*8, kernel, stride, padding)
        # self.up5 = IterativeBlock(base_channels*8, base_channels*8, kernel, stride, padding, activation=activation)
        
        # Reconstruction
        self.out_conv = ConvBlock(base_channels, nbins, 3, 1, 1, activation=None)
        self.trim = Trim(max_num_coefficient)

        self.init_parameters()

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        # x = self.up4(x)
        # x = self.up5(x)
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

        self.classifier = nn.Sequential(
            nn.Linear(512 * 31, 512),
            # nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out

if __name__ == '__main__':
    x = torch.randn(2, 256, 64)
    G = D_DBPN(256, 21)
    x = G(x)
    print(x.shape)
    D = Discriminator(256)
    x = D(x)
    print(x.shape)