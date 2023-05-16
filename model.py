import torch
import torch.nn as nn
import torch.nn.functional as F

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)
    
class Trim(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        return x[:,:,:28,:28]
    
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        self.encode = nn.Sequential(
            nn.Conv2d(3, 32, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ReLU(),
            nn.Conv2d(64, 64, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, stride=1, kernel_size=3, padding=1),
            nn.Flatten(),
        )

        self.compute_mean = nn.Linear(64*7*7, latent_dim)
        self.compute_log_var = nn.Linear(64*7*7, latent_dim)
    
    def reparametrize(self, mu, logvar):
        epsilon = torch.randn(mu.size(0), mu.size(1)).to(mu.get_device())
        z = mu + epsilon * torch.exp(logvar/2.)
        return z

    def forward(self, x):
        x = self.encode(x)
        mu, log_var = self.compute_mean(x), self.compute_log_var(x)
        return self.reparametrize(mu, log_var)
    
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()

        self.decode = nn.Sequential(
            nn.Linear(latent_dim, 64*7*7),
            Reshape(-1, 64, 7, 7),
            nn.ConvTranspose2d(64, 64, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, stride=2, kernel_size=3, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, stride=1, kernel_size=3, padding=0),
            Trim(),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decode(x)
    
class Discriminator(nn.Module):
    def __init__(self, alpha = 0.2, num_channels = 128):
        super().__init__()
        
        self.discriminator = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(alpha, inplace=True),
            nn.Conv2d(num_channels, num_channels*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_channels*2),
            nn.LeakyReLU(alpha, inplace=True),
            nn.Conv2d(num_channels*2, num_channels*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_channels*4),
            nn.LeakyReLU(alpha, inplace=True),
            nn.Conv2d(num_channels*4, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.discriminator(x)

        