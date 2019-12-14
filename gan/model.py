import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.nn.utils import spectral_norm

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Debug(nn.Module):
    def forward(self, x):
        print(x.shape)


    
class Generator(nn.Module):
    def __init__(self, ngpu, nc, nz, ngf):
        """
        Generator

        Args:
          ngpu: number of gpus
          nc: number of channels in training image
          nz: length of latent vector (size of generator input)
          ngf: size of feature maps in generator
        """
        self.ngpu = ngpu
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

class Discriminator(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        """
        Discriminator

        Args:
          ngpu: number of gpus
          nc: number of channels in training images
          ndf: size of feature maps in discriminator
        """
        self.ngpu = ngpu
        self.nc = nc
        self.ndf = ndf
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        self.debug = Debug()
        self.conv1 = spectral_norm(nn.Conv2d(self.nc, self.ndf, 3, stride=1, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(self.ndf, self.ndf, 4, stride=2, padding=1))
        self.conv3 = spectral_norm(nn.Conv2d(self.ndf, self.ndf*2, 3, stride=1, padding=1))
        self.conv4 = spectral_norm(nn.Conv2d(self.ndf*2, self.ndf*2, 4, stride=2, padding=1))
        self.conv5 = spectral_norm(nn.Conv2d(self.ndf*2, self.ndf*4, 3, stride=1, padding=1))
        self.conv6 = spectral_norm(nn.Conv2d(self.ndf*4, self.ndf*4, 4, stride=2, padding=1))
        self.conv7 = spectral_norm(nn.Conv2d(self.ndf*4, self.ndf*8, 3, stride=1, padding=1))

        self.fc = spectral_norm(nn.Linear(8*8*512, 1))

    def forward(self, x):
        l = 0.1
        m = x
        # self.debug(m)
        m = nn.LeakyReLU(l)(self.conv1(m))
        # self.debug(m)
        m = nn.LeakyReLU(l)(self.conv2(m))
        # self.debug(m)
        m = nn.LeakyReLU(l)(self.conv3(m))
        # self.debug(m)
        m = nn.LeakyReLU(l)(self.conv4(m))
        # self.debug(m)
        m = nn.LeakyReLU(l)(self.conv5(m))
        # self.debug(m)
        m = nn.LeakyReLU(l)(self.conv6(m))
        # self.debug(m)
        m = nn.LeakyReLU(l)(self.conv7(m))
        # self.debug(m)

        return self.fc(m.view(-1,8*8*512))
        
