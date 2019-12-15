import time
from pathlib import Path
from .model import *
from matplotlib.pyplot import imshow, imsave
import matplotlib.pyplot as plt


def generate(config, samples, output_directory):
    out = Path(output_directory)
    start_time = time.time()
    # Load GAN
    netG = Generator(1, config.nc, config.nz, config.ngf).to(config.device)
    netG.load_state_dict(torch.load("gen_state_dict"))

    # netD = Discriminator(1, config.nc, config.ndf).to(config.device)
    # netD.load_state_dict(torch.load("disc_state_dict"))

    for i in range(samples):
        noise = torch.randn(1, config.nz, 1, 1, device=config.device)
        fake = netG(noise).detach().cpu().numpy().squeeze()
        fake = ((fake - fake.min())/(fake.max() - fake.min()))
        imsave(out / f"sample_{i}.jpeg", fake.transpose(1, 2, 0))

    print(f"Generated {samples} samples in {time.time() - start_time} seconds!")

        

        

        
