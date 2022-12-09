import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import wraps
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.filters.hp_filter import hpfilter
from scipy.signal import periodogram

def time_format(arr: pd.Series,
                t_format: str = "%Y-%m-%d"):
    arr = arr.astype(str)
    return arr.apply(lambda x: dt.datetime.strptime(str(x), t_format))

def make_var_names(var, n_lags, reg):
    names = [f'L1.{var}']
    for lag in range(1, n_lags + 1):
        names.append(f'D{lag}L1.{var}')
    if reg != 'n':
        names.append('Constant')
    if 't' in reg:
        names.append('Trend')

    return names

def ADF_test_summary(df, maxlag=None, autolag='BIC', missing='error'):
    if missing == 'error':
        if df.isna().any().any():
            raise ValueError("df has missing data; handle it or pass missing='drop' to automatically drop it.")
            
    if isinstance(df, pd.Series):
        df = df.to_frame()
        
    for series in df.columns:
        data = df[series].copy()
        if missing == 'drop':
            data.dropna(inplace=True)
            
        print(series.center(110))
        print(('=' * 110))
        line = 'Specification' + ' ' * 15 + 'Coeff' + ' ' * 10 + 'Statistic' + ' ' * 5 + 'P-value' + ' ' * 6 + 'Lags' + ' ' * 6 + '1%'
        line += ' ' * 10 + '5%' + ' ' * 8 + '10%'
        print(line)
        print(('-' * 110))
        spec_fixed = False
        for i, (name, reg) in enumerate(zip(['Constant and Trend', 'Constant Only', 'No Constant'], ['ct', 'c', 'n'])):
            stat, p, crit, regresult = sm.tsa.adfuller(data, regression=reg, regresults=True, maxlag=maxlag,
                                                       autolag=autolag)
            n_lag = regresult.usedlag
            gamma = regresult.resols.params[0]
            names = make_var_names(series, n_lag, reg)
            reg_coefs = pd.Series(regresult.resols.params, index=names)
            reg_tstat = pd.Series(regresult.resols.tvalues, index=names)
            reg_pvals = pd.Series(regresult.resols.pvalues, index=names)

            line = f'{name:<21}{gamma:13.3f}{stat:15.3f}{p:13.3f}{n_lag:11}{crit["1%"]:10.3f}{crit["5%"]:12.3f}{crit["10%"]:11.3f}'
            print(line)

            for coef in reg_coefs.index:
                if coef in name:
                    line = f"\t{coef:<13}{reg_coefs[coef]:13.3f}{reg_tstat[coef]:15.3f}{reg_pvals[coef]:13.3f}"
                    print(line)
    pass


def plot_df(df: pd.DataFrame,
            fill_arr: np.array=None,
            row_ratio: float =1.5,
            figsize_col: int=10,
            if_skpina: bool = True):
    start = df.index.min()
    end = df.index.max()
    
    
    row = int(np.ceil(df.shape[1]/2))
    fig, ax = plt.subplots(row, 2, figsize=(10, row_ratio * row))
    for i,col in enumerate(df.columns):
        if i > row-1:
            c=1
            r=i-row
        else:
            c=0
            r=i

        if fill_arr is not None:
            # only inlcude relevant recessions
            for t in fill_arr:
                if t[1] < start:
                    continue
                if t[0] < start:
                    t[0]=start
                if t[0] > end:
                    continue
                if t[1] > end:
                    t[1]=end
                # plot recessions
                ax[r, c].axvspan(t[0], t[1], alpha=.2, color='red')

        # plot col                
        ax[r, c].plot(df[col].dropna())
        ax[r, c].set_title(col)

    fig.legend(['recession'])
    fig.tight_layout()
    pass



def plot_sm_results(res, extra_data=None, filter_output='predicted', var_names=None):
    fig = plt.figure(figsize=(14,8))
    
    endog_vars = res.data.ynames
    states = res.states.predicted.columns
    if var_names:
        states = [x for x in states if x in var_names]
    
    gs, plot_locs = gp.prepare_gridspec_figure(n_cols=3, n_plots=len(states))
    
    for i, (name, loc) in enumerate(zip(states, plot_locs)):
        axis = fig.add_subplot(gs[loc])

        mu = getattr(res.states, filter_output)[name]
        sigma = getattr(res.states, filter_output + '_cov').loc[name, name]

        upper = mu + 1.98 * np.sqrt(sigma + 1e-8)
        lower = mu - 1.98 * np.sqrt(sigma + 1e-8)

        start_idx = 1 if filter_output == 'predicted' else 0
        axis.plot(res.data.dates, mu.values[start_idx:], label='Predicted')
        axis.fill_between(res.data.dates, lower.values[start_idx:], upper.values[start_idx:], color='tab:blue', alpha=0.25)

        if name in endog_vars:
            res.data.orig_endog[name].plot(label='Data', ax=axis)
        
        elif extra_data is not None:
            if name in extra_data.columns:
                extra_data[name].plot(label='Data', ax=axis)

        axis.set(title=name)
    fig.tight_layout()
    title_text = 'One-Step Ahead' if filter_output =='predicted' else filter_output.title()
    fig.suptitle(f'Kalman {title_text} Predictions', y=1.05)
    fig.axes[1].legend(bbox_to_anchor=(0.5, 0.98), loc='lower center', bbox_transform=fig.transFigure, ncols=2)

    plt.show()
    pass


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
def poly_detrend(arr, poly_order: int=2, return_pred: bool = False, return_func_form: bool=False, **kwargs):
    x = range(0, len(arr))
    y = arr.values
    model = np.polyfit(x, y, poly_order, **kwargs)
    predicted = np.polyval(model, x)
    detrend = y - predicted
    if return_pred:
        return predicted
    elif return_func_form:
        return model
    else:
        return detrend
    
    
@skipna
def apply_func(arr, func, **kwargs):
    """
    *args:
    arr: array
    
    **kwargs:
    skipna: wether to skipna or note
    func: transformation function
    """
    return func(arr, **kwargs)
    

@skipna
def get_seasonal_decompose(arr: pd.Series, plot: bool=False, **kwargs):
    """
    returns remainder after trend and seasonal
    """
    res = sm.tsa.seasonal_decompose(arr, **kwargs)
    res.plot() if plot is True else 0
    return res.trend, res.seasonal


@skipna
def get_seasonal_hp(arr: pd.Series, plot: bool=False, lamb:float = 6.25, return_trend: bool = False, **kwargs):
    """
    returns: cylce or trend based on "return_trend"
    """
    cycle, trend = hpfilter(arr, lamb=lamb)
    plt.plot(np.array([cycle, trend]).transpose()) if plot is True else 0
    if return_trend:
        return trend
    else:
        return cycle


@skipna
def arr_adf(arr, maxlag: int=10, p_level: float=.05):
    """
    The null hypothesis of the Augmented Dickey-Fuller states there is a unit root, hence data is non-stationary.
    """
    test = adfuller(arr, maxlag=maxlag)
    print(arr.name, f" p-val: {test[1]},  reject: {test[1] <= p_level}")
    pass

def get_max_seasonal(df: pd.DataFrame,
                    lags: int = 15):
    out = pd.DataFrame()
    for col in df.columns:
        freq, Pxx = periodogram(df[col].dropna(), window='triang', nfft=lags*2)
        out[col] = Pxx
        print(f'{col}: strongest effect at {np.where((Pxx == Pxx.max())==True)[0][0] + 1} lags')
    return out

class OLS_dummie_deseasonal:
    
    def __init__(self):
        
        self.y = None
        self.model = None
        self.m_res = None
        pass
    
    def summary(self):
        print(self.m_res.summary())
        pass
    
    def _get_quarter_dummiesX(self, arr):
        X = pd.Series(arr.index)
        X = X.apply(lambda x: np.ceil(x.month/3)).values
        X = pd.get_dummies(X).iloc[:, :-1]
        X.set_index(arr.index, inplace=True)
        X.columns = [f'quarter_{item}' for item in X.columns]
    
        X['constant'] = list([1] * len(X))
        X['trend'] = np.linspace(0, len(X)-1, len(X))
        X['trend^2'] = X.trend ** 2
                
        return X
    
    def get_trend(self, 
                  y):
        self.y = y.copy()
        
        # for skipping na
        self.detrend = y.copy()
        self._y_na = y.dropna().copy()
        
        self.model = sm.OLS(self._y_na, exog=self._get_quarter_dummiesX(self._y_na), hasconst=True)
        self.m_res = self.model.fit()
    
        # for skipping na, get original shape back
        self.detrend.loc[self._y_na.index] = self.m_res.resid
        return self.detrend
    
    def plot_fit(self):
        fig, ax = plt.subplots()
        ax.plot(self._y_na, label='data')
        ax.plot(self.detrend.dropna(), label='fit')
        ax.set_title(self.y.name)
        fig.legend()
        pass
    
    def predict(self,
                y):
        pred = self.m_res.predict(self._get_quarter_dummiesX(y))
        return y - pred
    
