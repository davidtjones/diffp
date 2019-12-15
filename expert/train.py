import os
import time
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.optim as optim
import torch.nn as nn
from .util.tboard import TBoard


def train(dataset, config, use_tb=False):

    if use_tb:
        results_dir = Path(results_dir)
        results_dir = results_dir / str(datetime.fromtimestamp(time.time()))
        os.mkdir(results_dir)
        tb = TBoard(results_dir=results_dir)

    sample_count = len(dataset)

    split = sample_count // 5
    indices = list(range(sample_count))

    valid_idx = np.random.choice(
        indices,
        size=split,
        replace=False)

    train_idx = list(set(indices) - set(valid_idx))
    dataset_sizes = {'train': len(train_idx), 'val': len(valid_idx)}

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    dataloaders = {
        loader: DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            sampler=sampler
            )
        for (loader, sampler) in [('train', train_sampler), ('val', valid_sampler)]}

    print("Training samples: %s" % len(train_sampler))
    print("Validation samples: %s" % len(valid_sampler))
    print("Samples: %s, %s" % (sample_count, sample_count == len(train_sampler) + len(valid_sampler)))
    print("Sample batch shape: ", end='')

    # Instantiate Model
    model = None

    # Fit data to model

    criterion = nn.CrossEntropyLoss()
    optimizer = nn.Adam(model.parameters())

    iterations = 0
    start = time.time()
    best_loss = 1000.0

    for epoch in range(config.num_epochs):
        epoch_time = time.time()
        print('\nEpoch {}/{}'.format(epoch+1, config.num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                if config.scheduler:
                    config.scheduler.step()
                model.train()
            else:
                model.eval()

            running_f1 = running_fb = running_precision = running_recall = running_loss = 0.0

            # Iterate over data
            for batch_idx, batch in enumerate(dataloaders[phase]):
                train_time = time.time()
                images = batch['image'].to(config.device)
                labels = batch['labels'].to(config.device)

                optimizer.zero_grad()

                iterations += 1

                # forward pass
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
    
