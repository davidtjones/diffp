import argparse
import torch
from pathlib import Path

from dataset import DiabeticRetinopathyDataset
from transforms import LoadImage
from torchvision.transforms import Resize, CenterCrop, Compose, RandomHorizontalFlip, Resize, Normalize, ToTensor

import gan
from gan.train import train as gan_train
from gan.config import Config as gan_Config
from gan.generate import generate


parser = argparse.ArgumentParser(description="run script for the diabetic retinopathy challenge")
parser.add_argument("network", help="either 'gan' for generation or 'expert' for classification")

group = parser.add_mutually_exclusive_group()
group.add_argument("-t", "--train", help="train network", action="store_true")
group.add_argument("-c", "--classify", help="classify samples", action="store_true")
group.add_argument("-g", "--generate", help="generate samples", action="store_true")

parser.add_argument("-s", "--sample_count", type=int, help="number of samples to generate")
parser.add_argument("-d", "--directory", help="directory containing samples to be trained, classified, or generated")
parser.add_argument("--tboard", help="turn on tensorboard output", action="store_true")

args = parser.parse_args()


image_size = 64  # resize image to this length/width - changing this requires changes to the model!


if args.train:
    # only load dataset if we plan to do training
    input_transform = Compose([
        LoadImage(Path(r"data/train")),
        Resize(image_size),
        CenterCrop(image_size),
        ToTensor(),
        Normalize([.5,.5,.5], [.5,.5,.5])
    ])
    

    dataset = DiabeticRetinopathyDataset(r"data/trainLabels.csv",
                                         Path(r"data/train"),
                                         use_rl=True,
                                         transform_input=input_transform)


if args.network == "gan":
    config = gan_Config(20, 128, .00004, 0.0001)
    print("%s: Starting up" % config.name)
    print("%s: Device: %s : %s" % (config.name, config.device, config.device_name))

    if args.train:
        gan_train(dataset, config, use_tb=args.tboard, results_dir=args.directory)
    elif args.generate:
        generate(args.sample_count, output_directory=args.directory)
    else:
        print("nothing to do")

if args.network == "expert":
    if args.train:
        pass
    elif args.classify:
        pass
    else:
        print("nothing to do")


# config needs to take arguments from parser















