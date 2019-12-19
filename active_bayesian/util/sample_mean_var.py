import numpy as np
from pathlib import Path
from matplotlib.pyplot import imread
from tqdm import tqdm
import time
from multiprocessing import Pool
import json


def get_sample(img_path):
    sample_count = 10
    channels = 3
    img = imread(img_path)
    img_sample = np.stack(
        [np.mean(
            np.random.choice(
                img[:,:,i].flatten(), sample_count, replace=False)
        ) for i in range(channels)])
    return img_sample

def main():
    start_time = time.time()
    p = Pool(10)
    data_path = Path(r"../data/train/")
    images = data_path.glob("*.jpeg")

    img_samples = p.map(get_sample, images)
    p.close()
    p.join()
    img_samples = np.array(img_samples)
    
    
    means = np.mean(img_samples, axis=0)
    var = np.var(img_samples, axis=0)
    meanvars = np.stack((means, var))
    print(f'Means: {means}')
    print(f'Variance: {var}')
    print(f"Finished in {time.time() - start_time} seconds")

    with open('../data/meanvar.npy', 'w') as outfile:
        numpy.save(outfile, meanvars)
        
if __name__ == '__main__':
    main()
