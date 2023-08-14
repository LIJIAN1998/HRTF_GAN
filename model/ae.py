import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        # self.classifier = nn.Softplus()

    def forward(self,  x: torch.Tensor) -> torch.Tensor:
        x = self.decoder(x)
        # out = self.classifier(x)
        return x
    
class AutoEncoder(nn.Module):
    def __init__(self, nbins: int, in_order: int, latent_dim: int, out_oder: int=28):
        super(AutoEncoder, self).__init__()

        self.encoder = ResEncoder(ResBlock, nbins, in_order, latent_dim)
        self.decoder = Decoder(nbins, latent_dim, out_oder)


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
            nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, True),
            # nbins x 27
            # nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=1, bias=False),
            # nn.BatchNorm1d(512),
            # nn.LeakyReLU(0.2, True),
            # nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=2, bias=False),
            # nn.BatchNorm1d(512),
            # nn.LeakyReLU(0.2, True),
            # nbins x 34
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 27, 512),
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
    x = torch.randn(2, 256, 25)
    G = AutoEncoder(256, 4, 128, 28)
    x = G(x)
    print(x.shape)
    D = Discriminator(256)
    x = D(x)
    print(x.shape)