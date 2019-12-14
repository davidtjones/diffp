from .model import *


def generate(config, samples, output_directory):

    # Load GAN
    netG = Generator(1, config.nc, config.nz, config.ngf).to(config.device)
    netG.load_state_dict(torch.load("gen_state_dict"))

    netD = Discriminator(1, config.nc, config.ndf).to(config.device)
    netD.load_state_dict(torch.load("disc_state_dict"))

    for i in range(samples):
        noise = torch.randn(1, config.nz, 1, 1, device=config.device())
        fake = netG(noise).detach().cpu()

        # need to break fake image into R and Ls
