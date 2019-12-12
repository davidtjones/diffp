import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.utils as vutils
from .model import *
import torch.optim as optim
from .util.tboard import TBoard
from datetime import datetime
import os

def train(dataset, config, test_dataset=None,
          phases=['train'], results_dir=None):

    results_dir = Path(results_dir)
    results_dir = results_dir / str(datetime.fromtimestamp(time.time()))
    os.mkdir(results_dir)
    tb = TBoard(results_dir=results_dir)
    # Split training dataset into train/validation via dataset_split
    sample_count = len(dataset)
    
    indices = list(range(sample_count))
    dataset_size = len(indices)

    # Create dataloaders
    train_sampler = SubsetRandomSampler(indices)

    dataloader = DataLoader(
                    dataset,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    sampler=train_sampler)


    netG = Generator(1, config.nc, config.nz, config.ngf).to(config.device)
    netG.apply(weights_init)
    # if opt.netG != '':
    #     netG.load_state_dict(torch.load(opt.netG))
    # print(netG)

    netD = Discriminator(1, config.nc, config.ndf).to(config.device)
    netD.apply(weights_init)
    # if opt.netD != '':
    #    netD.load_state_dict(torch.load(opt.netD))
    # print(netD)
    
    optimizerD = optim.Adam(netD.parameters(), lr=config.lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=config.lr, betas=(0.5, 0.999))

    fixed_noise = torch.randn(config.batch_size, config.nz, 1, 1, device=config.device)

    real_label = 1
    fake_label = 0
    criterion = nn.BCELoss()

    iterations = 0
    start = time.time()
    best_loss = 1000.0

    losses = {}
    
    
    for epoch in range(config.epochs):
        epoch_time = time.time()
        print('\nEpoch {}/{}'.format(epoch+1, config.epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase

        running_loss_D = 0.0
        running_loss_G = 0.0

        # Iterate over data
        for batch_idx, batch in enumerate(dataloader):
            # print(batch_idx)
            train_time = time.time()
            netD.zero_grad()
            
            # Train with all-real batch
            images = batch['image'].to(config.device)
            
            b_size = images.size(0)
            # labels = batch['label'].to(config.device)
            # labels for GAN are 1 for a real image and 0 for a fake image
            # label = torch.full((b_size,), real_label, device=config.device)
            # Label smoothing: 
            label = np.random.random_sample((b_size,))
            label_real = label*(.5) + .7
            label_real = torch.tensor(label_real, device=config.device).float()

            label_fake = label * .3
            label_fake = torch.tensor(label_fake, device=config.device).float()
            
            ## Update D Network
            # Train with all real batch         
            output = netD(images).view(-1)
            # noisy labels: (flip occassionally)
            errD_real = criterion(output, label_real)
            errD_real.backward()
            D_x = output.mean().item()
            
            ## Train with all fake batch
            noise = torch.randn(b_size, config.nz, 1, 1, device=config.device)
            fake = netG(noise)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label_fake)
                
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()
            
            ## Update G Network
            netG.zero_grad()
            output = netD(fake).view(-1)
            errG = criterion(output, label_fake)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            
            
            # Output training stats
            if batch_idx % 50 == 0:
                print(f'[{epoch}/{config.epochs}][{batch_idx}/{len(dataloader)}]\tLoss_D: {errD.item()}\tLoss_G: {errG.item()}, D(x): {D_x}, D(G(z)): {D_G_z1} / {D_G_z2}')
                
            
            running_loss_G += errG.item() * label_real.shape[0]
            running_loss_D += errD.item() * label_real.shape[0]
                                
            iterations += 1
            
        # Epoch stats:
        running_loss_D /= dataset_size
        running_loss_G /= dataset_size

        losses["loss_D"] = running_loss_D
        losses["loss_G"] = running_loss_G
        tb.add_to_metrics_plot(epoch, losses)

        # Generate sample generated images after each epoch
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        tb.add_image(vutils.make_grid(fake, padding=2, normalize=True), epoch)



