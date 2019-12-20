import os
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.pyplot import imread
from .model import ExpertModel
from .eval_dataset import DiabeticRetinopathyEvalDataset
from .transforms import LoadImage
from torchvision.transforms import Resize, CenterCrop, Compose, Normalize, ToTensor
from torch.utils.data import DataLoader


def evaluate(config, eval_dataset_csv, verify_input=False):
    start_time = time.time()

    # load model:
    model = ExpertModel(3, 5).to(config.device)
    model.load_state_dict(torch.load("expert_state_dict"))
    model.eval()  # get model in eval mode

    input_transform = Compose([
        LoadImage(Path(r"")),
        Resize(64),
        CenterCrop(64),
        ToTensor(),
        Normalize([.5,.5,.5], [.5,.5,.5])
        ])

    
    eval_dataset = DiabeticRetinopathyEvalDataset(eval_dataset_csv, input_transform)

    
    dataloader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        )
    
    predictions = []
    if verify_input:
        print("Verifying input integrity...")
        error = 0
        try:
            for idx, data in enumerate(eval_dataset):
                print(idx, end='|', flush=True)
                error = idx
        except:
            print(f"issue at sample {error+1}, verify or remove sample {eval_dataset.dr_frame.iloc[error+1]}")
            exit()
    # classify
    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].to(config.device)
        output = model(images)
        _, prediction = torch.max(output.data, 1)
        predictions.append(prediction)
    
    
    df = eval_dataset.dr_frame
    predictions_column = torch.cat([prediction for prediction in predictions])
    print(predictions_column.shape)
    df['predictions'] = predictions_column.detach().cpu().numpy()

    # export as csv
    df.to_csv("predictions.csv")

    
    
    
