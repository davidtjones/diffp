import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset


class DiabeticRetinopathyDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform_input=None, transform_label=None):
        self.dr_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform_input = transform_input
        self.transform_label = transform_label

    def __len__(self):
        return len(self.dr_frame)

    def __getitem__(self, idx):
        sample = self.dr_frame.iloc[idx]
        image = sample['image']
        label = sample['label']

        if transform_input:
            transform_input(image)

        if transform_label:
            transform_label(label)
        
        return {"image": sample['image'], "label": sample['level']}


if __name__ == '__main__':
    drd = DiabeticRetinopathyDataset("data/trainLabels.csv", Path("../data/train/"))
    print(drd[0])
