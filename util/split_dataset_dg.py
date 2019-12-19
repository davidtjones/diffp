import pandas as pd

def split_dataset_dg(csv_file):
    df = pd.read_csv(csv_file)
    hl = df.loc[df['level'] == 0]
    dr = df.loc[df['level'] > 0]
    return dr, hl
