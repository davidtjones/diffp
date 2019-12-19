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
    from util.class_balance import class_balance
    from sklearn.decomposition import PCA
    import numpy as np
    from sklearn.manifold import TSNE
    from transforms import LoadImage
    from torchvision.transforms import Resize, CenterCrop, Compose, RandomHorizontalFlip, Resize, Normalize, ToTensor
    from multiprocessing import Pool

    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    
    image_size = 64
    input_transform = Compose([
        LoadImage(Path(r"data/train")),
        Resize(image_size),
        CenterCrop(image_size),
        ToTensor(),
        Normalize([.5,.5,.5], [.5,.5,.5])
    ])

    
    drd = DiabeticRetinopathyDataset("data/trainLabels.csv", Path("../data/train/"), transform_input=input_transform)
    print(drd[0]['image'].shape)

    counts = class_balance(drd)
    print(counts)
    def func(idx):
        return drd[idx]['image'].reshape(-1).numpy()

    def func2(idx):
        return drd[idx]['label'].numpy()

    p = Pool(8)
    indices = np.arange(len(drd))
    samples = np.stack(p.map(func, indices))
    labels = np.stack(p.map(func2, indices))

    print(samples.shape)
    print(labels.shape)

    tsne = TSNE()
    X_embedded = tsne.fit_transform(samples)
    plt.scatter(X_embedded[0], X_embedded[1], c=labels)
    plt.savefig('preembed.png')

    
    

    
        
        
    # Plot TSNE of data before balancing (benchmark)
    
    

    
