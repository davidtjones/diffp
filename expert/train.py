import os
import time
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.optim as optim
from .util.tboard import TBoard


def train(dataset, config, use_tb=False):

    if use_tb:
        results_dir = Path(results_dir)
        results_dir = results_dir / str(datetime.fromtimestamp(time.time()))
        os.mkdir(results_dir)
        tb = TBoard(results_dir=results_dir)
