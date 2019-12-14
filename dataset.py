import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset


class DiabeticRetinopathyDataset(Dataset):
    def __init__(self, csv_file, root_dir, use_rl=False, transform_input=None, transform_label=None):
        self.dr_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.use_rl = use_rl
        self.transform_input = transform_input
        self.transform_label = transform_label

        
    def __len__(self):
        if self.use_rl:
            return len(self.dr_frame)//2
        else:
            return len(self.dr_frame)

        
    def __getitem__(self, idx):
        if self.use_rl:
            idx = idx*2
            sample_l = self.dr_frame.iloc[idx]
            sample_r = self.dr_frame.iloc[idx+1]
            image_l = sample_l['image']+".jpeg"
            image_r = sample_r['image']+".jpeg"
            label = torch.tensor(sample_l['level'])

        else:
            sample = self.dr_frame.iloc[idx]
            image = sample['image']+".jpeg"
            label = torch.tensor(sample['level'])

        if self.transform_input:
            if self.use_rl:
                image_r = self.transform_input(image_r)
                image_l = self.transform_input(image_l)
                # concatenate and align images
                image = torch.cat((image_l, torch.flip(image_r, [1])))
            else:
                image = self.transform_input(image)

        if self.transform_label:
            label = self.transform_label(label)
        
        return {"image": image, "label": label}


if __name__ == '__main__':
    drd = DiabeticRetinopathyDataset("data/trainLabels.csv", Path("../data/train/"))
    print(drd[0])
