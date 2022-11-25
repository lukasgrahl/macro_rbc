import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def time_format(arr: pd.Series,
                t_format: str = "%Y-%m-%d"):
    arr = arr.astype(str)
    return arr.apply(lambda x: dt.datetime.strptime(str(x), t_format))

def plot_df(df: pd.DataFrame,
            row_ratio: float =1.5,
            figsize_col: int=10):
    
    row = int(np.ceil(df.shape[1]/2))
    fig, ax = plt.subplots(row, 2, figsize=(10, row_ratio * row))
    for i,col in enumerate(df.columns):
        if i > row-1:
            c=1
            r=i-row
        else:
            c=0
            r=i

        ax[r, c].plot(df[col])
        ax[r, c].set_title(col)

    fig.tight_layout()
    pass
