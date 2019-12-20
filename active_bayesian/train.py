import os
import time
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.optim as optim
import torch.nn as nn
from .util.tboard import TBoard
from .model import ActiveBayesianModel
import torch.optim as optim
from .acquisition import AcquisitionFn
from scipy.stats import mode

def train_on_gan(healthy_ds, diabetic_ds, dataset, config):
    training_accuracy = []
    training_loss = []
    accuracy_values = []
    loss_values = []
    sizes = []

    samples_to_train_on = 50
    model = ActiveBayesianModel(3, 5).to(config.device)
    # model.load_state_dict(torch.load("ab_state_dict"))
    aqfn = AcquisitionFn(model)

    sample_count = len(dataset) // 10

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
            sampler=sampler,
            drop_last=True
            )
        for (loader, sampler) in [('train', train_sampler), ('val', valid_sampler)]}

    # Fit data to model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    iterations = 0
    start = time.time()
    best_loss = 1000.0
    # scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001)

    for epoch in range(config.epochs * 10):
        epoch_time = time.time()
        print('\nEpoch {}/{}'.format(epoch+1, config.epochs * 10))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            print(f"Starting {phase} phase")
            if phase == 'train':
                model.train()
            else:
                model.eval()

            training_images = []
            training_labels = []
            if phase == 'train':
                for x in range(samples_to_train_on):
                    idx = x + samples_to_train_on * epoch
                    healthy_image = torch.Tensor(healthy_ds[x]).transpose(2,0).unsqueeze(0).to(config.device)
                    diabetic_image = torch.Tensor(diabetic_ds[x]).transpose(2,0).unsqueeze(0).to(config.device)
                    if not aqfn.is_image_certain(healthy_image):
                        training_images.append(healthy_image)
                        training_labels.append(0)
                    if not aqfn.is_image_certain(diabetic_image):
                        import random
                        training_images.append(diabetic_image)
                        training_labels.append(random.randint(1,4))

            # print(f"First iter training images: {training_images}")
            # print(f"First iter training labels: {training_labels}")
            # print(f"Lens: {len(training_images)}, {len(training_labels)}")
            # input("Enter to continue")

            running_loss = running_acc = total = 0.0

            if phase == 'train':
                dl = training_images
            else:
                dl = dataloaders[phase]

            if len(training_images) == 0 and phase == 'train':
                print("Skipping round...")
                continue

            # Iterate over data
            for batch_idx, batch in enumerate(dl):
                train_time = time.time()

                if phase == 'val':
                    images = batch['image'].to(config.device)
                    labels = batch['label'].to(config.device)
                else:
                    images = torch.stack(training_images)
                    if len(training_images) > 1:
                        images = images.squeeze(1)
                    images = images.to(config.device)
                    labels = torch.from_numpy(np.array(training_labels)).to(config.device)

                print(f"Size of images: {len(images)}")

                optimizer.zero_grad()

                iterations += 1

                # forward pass
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(images)

                    _, predicted = torch.max(output.data, -1)
                    loss = criterion(output, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                predicted[predicted!=0] = 1
                labels[labels!=0] = 1

                # print(f"Predicted: {predicted}")
                # print(f"Labels: {labels}")
                # input("Enter to continue")

                running_loss += loss.item()*output.shape[0]
                running_acc += (predicted == labels).sum().item()
                total += output.size(0)

                if phase == 'train':
                    print(f"[{batch_idx}/{len(dataloaders[phase])}] loss: {running_loss/total:.3f}\t acc: {running_acc/total:.3f}")
                    break

                # scheduler.step()  # step learning rate scheduler

            if phase == 'val':
                size = dataset_sizes[phase]
            else:
                size = len(training_images)

            running_loss = running_loss/size
            running_acc = running_acc/size

            print(f"{phase.capitalize()}: Loss: {running_loss:.3f}\t Acc: {running_acc:.3f}")
            if (phase == 'val'):
                loss_values.append(running_loss)
                accuracy_values.append(running_acc)
                np.savetxt("loss_values.txt", loss_values)
                np.savetxt("accuracy_values.txt", accuracy_values)
                if running_loss < best_loss:
                    print("New best loss! Saving model...")
                    torch.save(model.state_dict(), "ab_gan_state_dict")
                    best_loss = running_loss
            else:
                sizes.append(size)
                training_accuracy.append(running_acc)
                training_loss.append(running_loss)
                np.savetxt("training_loss.txt", training_loss)
                np.savetxt("training_accuracy.txt", training_accuracy)
                np.savetxt("sizes.txt", sizes)


    print(f"finished in {time.time() - start_time}")

def train(dataset, config, use_tb=False):
    start_time = time.time()
    if use_tb:
        results_dir = Path(results_dir)
        results_dir = results_dir / str(datetime.fromtimestamp(time.time()))
        os.mkdir(results_dir)
        tb = TBoard(results_dir=results_dir)

    sample_count = len(dataset) // 10

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
            sampler=sampler,
            drop_last=True
            )
        for (loader, sampler) in [('train', train_sampler), ('val', valid_sampler)]}

    print("Training samples: %s" % len(train_sampler))
    print("Validation samples: %s" % len(valid_sampler))
    print("Samples: %s, %s" % (sample_count, sample_count == len(train_sampler) + len(valid_sampler)))


    # Instantiate Model
    model = ActiveBayesianModel(3, 5).to(config.device)
    # print(model)

    # Fit data to model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    iterations = 0
    start = time.time()
    best_loss = 1000.0
    # scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001)
    
    for epoch in range(config.epochs):
        epoch_time = time.time()
        print('\nEpoch {}/{}'.format(epoch+1, config.epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            print(f"Starting {phase} phase")
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = running_acc = total = 0.0

            # Iterate over data
            for batch_idx, batch in enumerate(dataloaders[phase]):
                train_time = time.time()
                images = batch['image'].to(config.device)
                labels = batch['label'].to(config.device)

                optimizer.zero_grad()

                iterations += 1

                # forward pass
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(images)

                    _, predicted = torch.max(output.data, 1)
                    loss = criterion(output, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()*output.shape[0]
                running_acc += (predicted == labels).sum().item()
                total += output.size(0)

                if batch_idx % 150 == 0 and phase == 'train':
                    print(f"[{batch_idx}/{len(dataloaders[phase])}] loss: {running_loss/total:.3f}\t acc: {running_acc/total:.3f}")

                # scheduler.step()  # step learning rate scheduler
                
            running_loss = running_loss/dataset_sizes[phase]
            running_acc = running_acc/dataset_sizes[phase]
                        
            print(f"{phase.capitalize()}: Loss: {running_loss:.3f}\t Acc: {running_acc:.3f}")
            if (phase == 'val'):
                if running_loss < best_loss:
                    print("New best loss! Saving model...")
                    torch.save(model.state_dict(), "ab_state_dict")
                    best_loss = running_loss

    print(f"finished in {time.time() - start_time}")
