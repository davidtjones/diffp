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

def train(dataset, config, use_tb=False, results_dir=None):

    if use_tb:
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

    # number of updates to discriminator for every update to generator
    disc_iters = 5 
    
    optimizerD = optim.Adam(netD.parameters(), lr=config.lrD, betas=(0.0, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=config.lrG, betas=(0.0, 0.9))

    # use an exponentially decaying learning rate
    scheduler_d = optim.lr_scheduler.ExponentialLR(optimizerD, gamma=0.99)
    scheduler_g = optim.lr_scheduler.ExponentialLR(optimizerG, gamma=0.99)
    
    fixed_noise = torch.randn(config.batch_size, config.nz, 1, 1, device=config.device)

    real_label = .9
    fake_label = .1
    criterion = nn.BCEWithLogitsLoss()

    iterations = 0
    start = time.time()

    losses = {}
    
    for epoch in range(config.epochs):
        epoch_time = time.time()
        print('\nEpoch {}/{}'.format(epoch+1, config.epochs))
        print('-' * 10)

        running_loss_D = 0.0
        running_loss_G = 0.0

        # Iterate over data
        for batch_idx, batch in enumerate(dataloader):
            train_time = time.time()
            netD.zero_grad()
            
            # Train with all-real batch
            images = batch['image'].to(config.device)

            # Show a sample of real images
            if batch_idx == 0 and epoch == 0 and use_tb:
                tb.add_image('sample images L',
                             vutils.make_grid(images[:,:3,:,:],
                                              padding=2,
                                              normalize=True),
                             epoch)
                tb.add_image('sample images R',
                             vutils.make_grid(images[:,3:,:,:],
                                              padding=2,
                                              normalize=True),
                             epoch)

            b_size = images.size(0)
            # labels for GAN are 1 for a real image and 0 for a fake image
            label = torch.full((b_size,), real_label, device=config.device)

            ## Update D Network
            # Train with all real batch
            for _ in range(disc_iters):
                optimizerD.zero_grad()
                optimizerG.zero_grad()
                output = netD(images).view(-1)

                errD_real = criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()
            
                ## Train with all fake batch
                noise = torch.randn(b_size, config.nz, 1, 1, device=config.device)
                fake = netG(noise)
                label = label.fill_(fake_label)
                output = netD(fake.detach()).view(-1)
                errD_fake = criterion(output, label)
                
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                optimizerD.step()
            
            ## Update G Network
            # netG.zero_grad()
            optimizerD.zero_grad()
            optimizerG.zero_grad()
            
            label.fill_(real_label)
            output = netD(fake).view(-1)

            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            
            
            # Output training stats
            if batch_idx % 50 == 0:
                print(f'[{batch_idx}/{len(dataloader)}]\tLoss_D: {errD.item()}\tLoss_G: {errG.item()}, D(x): {D_x}, D(G(z)): {D_G_z1} / {D_G_z2}')

                if use_tb:
                    with torch.no_grad():
                        fake = netG(fixed_noise).detach().cpu()
                        tb.add_image('generated images L',
                             vutils.make_grid(fake[:,:3,:,:],
                                              padding=2,
                                              normalize=True),
                             epoch)
                        tb.add_image('generated images R',
                             vutils.make_grid(fake[:,3:,:,:],
                                              padding=2,
                                              normalize=True),
                             epoch)

                        losses["loss_D"] = errD.item()
                        losses["loss_G"] = errG.item()
                        tb.add_to_metrics_plot(epoch, losses)
                
            
            running_loss_G += errG.item() * label.shape[0]
            running_loss_D += errD.item() * label.shape[0]
                                
            iterations += 1

            scheduler_d.step()
            scheduler_g.step()

        # Epoch stats:
        running_loss_D /= dataset_size
        running_loss_G /= dataset_size

        print(f"Epoch loss: D: {running_loss_D}\t G: {running_loss_G}")

    # Save GAN
    torch.save(netD.state_dict(), "disc_state_dict")
    torch.save(netG.state_dict(), "gen_state_dict")

