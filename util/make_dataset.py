# Create a dataset (csv) from a directory of loose images

from pathlib import Path
import pandas as pd

def make_dataset(image_dir):
    image_path = Path(image_dir)

    files = [path for path in image_path.glob('*.jpeg')]
    df = pd.DataFrame(files, columns=['image_path'])
        
    df.to_csv("eval_dataset.csv")
    return df


if __name__ == '__main__':
    df = make_directory("gan/generated_images")

    print(df)    
    
    



