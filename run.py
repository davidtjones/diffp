import argparse
from pathlib import Path
import learner
from learner.dataset import DiabeticRetinopathyDataset
from learner.transforms import LoadImage, Norm
from learner.train import train
from learner.config import Config
from torchvision.transforms import Resize, CenterCrop, Compose, RandomHorizontalFlip, Resize, Normalize, ToTensor
import torch
import torch.optim as optim


parser = argparse.ArgumentParser(description="run script for the diabetic retinopathy challenge")
parser.add_argument("results_dir", type=str, help="where to store results")
args = parser.parse_args()

# config needs to take arguments from parser
config = Config(5, 128, .0002, )
print("%s: Starting up" % config.name)
print("%s: Device: %s : %s" % (config.name, config.device, config.device_name))

image_size = 64  # resize image to this length/width - changing this requires changes to the model!

input_transform = Compose([
    LoadImage(Path(r"data/train")),
    Resize(image_size),
    CenterCrop(image_size),
    ToTensor(),
    Norm()
    ])


dataset = DiabeticRetinopathyDataset(r"data/trainLabels.csv",
                                     Path(r"data/train"),
                                     transform_input=input_transform)

# evalset = DiabeticRetinopathyDataset(...)


train(dataset, config, results_dir=args.results_dir)









