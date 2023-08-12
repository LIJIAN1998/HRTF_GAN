import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm1d(out_channels)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)
        if self.activation is not None:
            return self.act(out)
        else:
            return out

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm1d(out_channels)
        
        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)
        if self.activation is not None:
            return self.act(out)
        else:
            return out

class UpBlock(nn.Module):
    def __init__(self, channels, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(UpBlock, self).__init__()
        self.conv1 = DeconvBlock(channels, channels, kernel_size, stride, padding, bias, activation, norm=None)
        self.conv2 = ConvBlock(channels, channels, kernel_size, stride, padding, bias, activation, norm=None)
        self.conv3 = DeconvBlock(channels, channels, kernel_size, stride, padding, bias, activation, norm=None)

    def forward(self, x):
        h0 = self.conv1(x)
        l0 = self.conv2(h0)
        h1 = self.conv3(l0 - x)
        return h1 + h0
    
class DownBlock(nn.Module):
    def __init__(self, channels, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(DownBlock, self).__init__()
        self.conv1 = ConvBlock(channels, channels, kernel_size, stride, padding, bias, activation, norm)
        self.conv2 = DeconvBlock(channels, channels, kernel_size, stride, padding, bias, activation, norm)
        self.conv3 = ConvBlock(channels, channels, kernel_size, stride, padding, bias, activation, norm)

    def forward(self, x):
        l0 = self.conv1(x)
        h0 = self.conv2(l0)
        l1 = self.conv3(h0 - x)
        return l1 + l0
    
class D_DownBlock(nn.Module):
    def __init__(self, channels, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu', norm=None):
        super(D_DownBlock, self).__init__()
        self.conv = ConvBlock(channels*num_stages, channels, 1, 1, 0, bias, activation, norm)
        self.down1 = ConvBlock(channels, channels, kernel_size, stride, padding, bias, activation, norm)
        self.down2 = DeconvBlock(channels, channels, kernel_size, stride, padding, bias, activation, norm)
        self.down3 = ConvBlock(channels, channels, kernel_size, stride, padding, bias, activation, norm)

    def forward(self, x):
        x = self.conv(x)
        l0 = self.down1(x)
        h0 = self.down2(l0)
        l1 = self.down3(h0 - x)
        return l1 + l0

class D_UpBlock(nn.Module):
    def __init__(self, channels, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu', norm=None):
        super(D_UpBlock, self).__init__()
        self.conv = ConvBlock(channels*num_stages, channels, 1, 1, 0, bias, activation, norm)
        self.up1 = DeconvBlock(channels, channels, kernel_size, stride, padding, bias, norm)
        self.up2 = ConvBlock(channels, channels, kernel_size, stride, padding, bias, activation, norm)
        self.up3 = DeconvBlock(channels, channels, kernel_size, stride, padding, bias, activation, norm)

    def forward(self, x):
        x = self.conv(x)
        h0 = self.up1(x)
        l0 = self.up2(h0)
        h1 = self.up3(l0 - x)
        return h1 + h0
    
