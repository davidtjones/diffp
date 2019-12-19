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

import expert
from expert.train import train as expert_train
from expert.config import Config as expert_Config
from expert.evaluate import evaluate as classify

from util.make_dataset import make_dataset
from util.split_dataset_dg import split_dataset_dg

parser = argparse.ArgumentParser(description="run script for the diabetic retinopathy challenge")
parser.add_argument("network", help="either 'gan' for generation or 'expert' for classification")

group = parser.add_mutually_exclusive_group()
group.add_argument("-t", "--train", help="train network", action="store_true")
group.add_argument("-c", "--classify", help="classify samples", action="store_true")
group.add_argument("-g", "--generate", help="generate samples", action="store_true")

parser.add_argument("-s", "--sample_count", type=int, help="number of samples to generate")
parser.add_argument("-d", "--directory", help="directory containing samples to be trained, classified, or generated")
parser.add_argument("--tboard", help="turn on tensorboard output", action="store_true")
parser.add_argument("-dg", "--double_gan", help="use two gans to model retinopathy", action="store_true")

args = parser.parse_args()


image_size = 64  # resize image to this length/width - changing this requires changes to the model!

input_transform = Compose([
    LoadImage(Path(r"data/train")),
    Resize(image_size),
    CenterCrop(image_size),
    ToTensor(),
    Normalize([.5,.5,.5], [.5,.5,.5])
])

if args.train:
    # only load dataset if we plan to do training
    dataset = DiabeticRetinopathyDataset(r"data/trainLabels.csv",
                                         Path(r"data/train"),
                                         # use_rl=True,
                                         transform_input=input_transform)


if args.network == "gan":
    config = gan_Config(20, 128, .00004, 0.0001)
    print("%s: Starting up" % config.name)
    print("%s: Device: %s : %s" % (config.name, config.device, config.device_name))
    if args.double_gan:
        print("using the double gan method")
        # kind of hacky, fix later
        
        healthy_dataset = DiabeticRetinopathyDataset(r"data/trainLabels.csv",
                                         Path(r"data/train"),
                                         # use_rl=True,
                                         transform_input=input_transform)

        
        diabetic_dataset = DiabeticRetinopathyDataset(r"data/trainLabels.csv",
                                         Path(r"data/train"),
                                         # use_rl=True,
                                         transform_input=input_transform)

        healthy_samples, diabetic_samples = split_dataset_dg(r"data/trainLabels.csv")
        healthy_dataset.dr_frame = healthy_samples
        diabetic_dataset.dr_frame = diabetic_samples

        if args.train:
            gan_train(healthy_dataset, config, use_tb=args.tboard, results_dir=args.directory, out_prefix="gan_healthy_")
            gan_train(diabetic_dataset, config, use_tb=args.tboard, results_dir=args.directory, out_prefix="gan_diabetic_")

        elif args.generate:
            generate(config,
                     args.sample_count//2,
                     output_directory=args.directory,
                     state_dict="gan_healthy_gen_state_dict")

            generate(config,
                     args.sample_count - args.sample_count//2,
                     output_directory=args.directory,
                     idx_start=args.sample_count//2,
                     state_dict="gan_diabetic_gen_state_dict")


    else:
        print("using the single gan method")
        if args.train:
            gan_train(dataset, config, use_tb=args.tboard, results_dir=args.directory)
        elif args.generate:
            generate(config, args.sample_count, output_directory=args.directory)
        else:
            print("nothing to do")

if args.network == "expert":
    config = expert_Config(10, 25)
    print("%s: Starting up" % config.name)
    print("%s: Device: %s : %s" % (config.name, config.device, config.device_name))
    
    if args.train:
        expert_train(dataset, config)
    elif args.classify:
        # Create dataset from given directory to pass to the classifier
        df = make_dataset(args.directory)
        classify(config, "eval_dataset.csv")
    else:
        print("nothing to do")


# config needs to take arguments from parser















