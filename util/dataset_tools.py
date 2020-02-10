from pathlib import Path

import json


from torchvision import datasets

def activate_dataset(args):
    # delete this
    dataset = args.dataset
    print(f"activating {dataset}")

    with open('options/base_options.json', 'r') as f:
        opts = json.load(f)

    opts["base_options"]["dataset"] = str(dataset)
    with open('options/base_options.json', 'w') as f:
        json.dump(opts, f)

def get_dataset(dataset, train=True, transform=None, target_transform=None):
    if dataset == "mnist":
         d = datasets.MNIST("./datasets/", train=train, download=True)
    if dataset == "cifar10":
        d = datasets.CIFAR10("./datasets/", train=train, download=True)
    if dataset == "diabeticretinopathy":
        pass

    return d
    


