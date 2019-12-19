import time
from pathlib import Path
from .model import *
from matplotlib.pyplot import imshow, imsave
import matplotlib.pyplot as plt


def generate(config, samples, output_directory, idx_start=0, state_dict="gen_state_dict"):
    out = Path(output_directory)
    start_time = time.time()
    # Load GAN
    netG = Generator(1, config.nc, config.nz, config.ngf).to(config.device)
    netG.load_state_dict(torch.load(state_dict))
    
    for i in range(samples):
        noise = torch.randn(1, config.nz, 1, 1, device=config.device)
        fake = netG(noise).detach().cpu().numpy().squeeze()
        fake = ((fake - fake.min())/(fake.max() - fake.min()))
        imsave(out / f"sample_{idx_start+i}.jpeg", fake.transpose(1, 2, 0))

    print(f"Generated {samples} samples in {time.time() - start_time} seconds!")

        

        

        
