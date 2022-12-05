import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import wraps
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.filters.hp_filter import hpfilter

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

def plot_sm_results(res, filter_output='predicted'):
    fig = plt.figure(figsize=(14,8))
    
    endog_vars = res.data.ynames
    states = mle_res.states.predicted.columns
    
    gs, plot_locs = gp.prepare_gridspec_figure(n_cols=3, n_plots=len(states))
    
    
    for i, (name, loc) in enumerate(zip(states, plot_locs)):
        axis = fig.add_subplot(gs[loc])

        mu = getattr(res.states, filter_output)[name]
        sigma = getattr(res.states, filter_output + '_cov').loc[name, name]

        upper = mu + 1.98 * np.sqrt(sigma + 1e-8)
        lower = mu - 1.98 * np.sqrt(sigma + 1e-8)

        start_idx = 1 if filter_output == 'predicted' else 0
        axis.plot(mle_res.data.dates, mu.values[start_idx:], label='Predicted')
        axis.fill_between(mle_res.data.dates, lower.values[start_idx:], upper.values[start_idx:], color='tab:blue', alpha=0.25)

        if name in endog_vars:
            mle_res.data.orig_endog[name].plot(label='Data', ax=axis)

        axis.set(title=name)
    fig.tight_layout()
    title_text = 'One-Step Ahead' if filter_output =='predicted' else filter_output.title()
    fig.suptitle(f'Kalman {title_text} Predictions', y=1.05)
    fig.axes[1].legend(bbox_to_anchor=(0.5, 0.98), loc='lower center', bbox_transform=fig.transFigure, ncols=2)

    plt.show()


def skipna(func): 

    @wraps(func)
    def wrapper(arr, *args, skipna: bool=False, **kwargs):
        arr_ = arr.copy()
        if skipna:
            arr_na = arr_[~arr_.isna()].copy()
            arr_na = func(arr_na, **kwargs)
            arr_[~arr_.isna()] = arr_na
            return arr_
        else:
            return func(arr_, *args, **kwargs)
    return wrapper



@skipna
def apply_func(arr, func, **kwargs):
    """
    *args:
    arr: array
    
    **kwargs:
    skipna: wether to skipna or note
    func: transformation function
    """
    return func(arr)


@skipna
def get_seasonal_decompose(arr: pd.Series, plot: bool=False, **kwargs):
    """
    returns remainder after trend and seasonal
    """
    res = sm.tsa.seasonal_decompose(arr, **kwargs)
    res.plot() if plot is True else 0
    return res.trend, res.seasonal


@skipna
def get_seasonal_hp(arr: pd.Series, plot: bool=False, lamb:float = 6.25, return_cycle: bool = False, **kwargs):
    """
    returns: cycle, trend
    """
    cycle, trend = hpfilter(arr, lamb=lamb)
    plt.plot(np.array([cycle, trend]).transpose()) if plot is True else 0
    if return_cycle:
        return cycle
    else:
        return trend


@skipna
def arr_adf(arr, maxlag: int=10, p_level: float=.05):
    """
    The null hypothesis of the Augmented Dickey-Fuller states there is a unit root, hence data is non-stationary.
    """
    test = adfuller(arr, maxlag=maxlag)
    print(arr.name, f" p-val: {test[1]},  reject: {test[1] <= p_level}")
    pass
    

@skipna
def poly_detrend(arr, poly_order: int=2, return_pred: bool = False, **kwargs):
    x = range(0, len(arr))
    y = arr.values
    model = np.polyfit(x, y, poly_order, **kwargs)
    predicted = np.polyval(model, x)
    detrend = y - predicted
    if return_pred:
        return predicted
    else:
        return detrend