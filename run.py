import code
import json
import argparse
import torch
from pathlib import Path
import numpy as np

from util.dataset_tools import activate_dataset

from active_bayesian.train import train as al_train
from active_bayesian.evaluate import evaluate as al_evaluate
from active_bayesian.classify import classify as al_classify

from gan.train import train as gan_train
from gan.evaluate import evaluate as gan_evaluate
from gan.generate import generate

from expert.train import train as exp_train
from expert.evaluate import evaluate as exp_evaluate
from expert.classify import classify as exp_classify


parser = argparse.ArgumentParser(description="entry point for the differential privacy project")
subparsers = parser.add_subparsers(help="sub-command help")

# Gan commands
parser_gan = subparsers.add_parser('gan', help='sub-commands for the GAN')
parser_gan_subparsers = parser_gan.add_subparsers(help="GAN sub-command help")

parser_gan_train = parser_gan_subparsers.add_parser("train", help="train the GAN")
parser_gan_train.add_argument("dataset", help="dataset to use, either cifar10, mnist, ...", action="store")
parser_gan_train.set_defaults(func=gan_train)

parser_gan_eval = parser_gan_subparsers.add_parser("evaluate", help="evaluate the GAN")
parser_gan_eval.add_argument("dataset", help="dataset to use, either cifar10, mnist, ...", action="store")
parser_gan_eval.set_defaults(func=gan_evaluate)

parser_gan_gen = parser_gan_subparsers.add_parser("generate", help="generate new images")
parser_gan_gen.set_defaults(func=generate)


# Expert commands
parser_exp = subparsers.add_parser('expert', help='sub-commands for the expert model')
parser_exp_subparsers = parser_exp.add_subparsers(help="Expert Model sub-command help")

parser_exp_train = parser_exp_subparsers.add_parser("train", help="train the Expert Model")
parser_exp_train.add_argument("dataset", help="dataset to use, either cifar10, mnist, ...", action="store")
parser_exp_train.set_defaults(func=exp_train)

parser_exp_eval = parser_exp_subparsers.add_parser("evaluate", help="evaluate the Expert Model")
parser_exp_eval.add_argument("dataset", help="dataset to use, either cifar10, mnist, ...", action="store")
parser_exp_eval.set_defaults(func=exp_evaluate)

parser_exp_classify = parser_exp_subparsers.add_parser("classify", help="classify using the Expert Model")
parser_exp_classify.set_defaults(func=exp_classify)


# Active Learning Model commands
parser_al = subparsers.add_parser("active", help="sub-commands for the active learning model")
parser_al_subparsers = parser_al.add_subparsers(help="Active Learner sub-command help")

parser_al_train = parser_al_subparsers.add_parser("train", help="train the Active Learning Model")
parser_al_train.add_argument("dataset", help="dataset to use, either cifar10, mnist, ...", action="store")
parser_al.set_defaults(func=al_train)

parser_al_eval = parser_al_subparsers.add_parser("evaluate", help="evaluate the Active Learning Model")
parser_al_eval.add_argument("dataset", help="dataset to use, either cifar10, mnist, ...", action="store")
parser_al_eval.set_defaults(func=al_evaluate)

parser_al_classify = parser_al_subparsers.add_parser("classify", help="classify using the Active Learning Model")
parser_al_classify.set_defaults(func=al_classify)


# Parse arguments
args = parser.parse_args()

with open("options/base_options.json", 'r') as f:
    opts = json.load(f)

kwargs = vars(args)
func = kwargs['func']
del kwargs['func']

# pass parameters to func
func(**kwargs)

