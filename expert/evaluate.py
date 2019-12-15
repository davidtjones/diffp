import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from .model import ExpertModel


def evaluate(config, eval_dataset):
    start_time = time.time()

    # load model:
    model = ExpertModel(3, 5).to(config.device)
    model.load_state_dict(torch.load("expert_state_dict"))
    model.eval()  # get model in eval mode

    # setup output dataframe
    df = pd.DataFrame(columns=["sample_image", "prediction"])

    # iterate over dataset

    # classify

    # store

    # export as csv
    df.to_csv("predictions.csv")

    
    
    
