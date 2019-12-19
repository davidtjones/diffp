import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset

class DiabeticRetinopathyEvalDataset(Dataset):
    def __init__(self, csv_file, transform_input=None):
        self.dr_frame = pd.read_csv(csv_file)
        self.transform_input = transform_input

    def __len__(self):
        return len(self.dr_frame)

    def __getitem__(self, idx):
        image = self.dr_frame.iloc[idx]['image_path']

        if self.transform_input:
            image = self.transform_input(image)

        return {'image': image}



if __name__ == '__main__':
    drd = DiabeticRetinopathyEvalDataset("eval_dataset.csv")
    print(drd[0])
