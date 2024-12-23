import pandas as pd
import numpy as np
import re
import math
import datetime
from typing import Union, List
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import warnings
from arch import arch_model
from collections import defaultdict
from scipy.stats import norm
from portfolio_management.utils import _filter_columns_and_indexes

pd.options.display.float_format = "{:,.4f}".format
warnings.filterwarnings("ignore")



def calc_tangency_weights(
    returns: pd.DataFrame,
    cov_mat: str = 1,
    return_graphic: bool = False,
    return_port_ret: bool = False,
    target_ret_rescale_weights: Union[None, float] = None,
    annual_factor: int = 12,
    name: str = 'Tangency',
    expected_returns: Union[pd.Series, pd.DataFrame] = None,
    expected_returns_already_annualized: bool = False
):
    """
    Calculates tangency portfolio weights based on the covariance matrix of returns.

    Parameters:
    returns (pd.DataFrame): Time series of returns.
    cov_mat (str, default=1): Covariance matrix for calculating tangency weights.
    return_graphic (bool, default=False): If True, plots the tangency weights.
    return_port_ret (bool, default=False): If True, returns the portfolio returns.
    target_ret_rescale_weights (float or None, default=None): Target return for rescaling weights.
    annual_factor (int, default=12): Factor for annualizing returns.
    name (str, default='Tangency'): Name for labeling the weights and portfolio.

    Returns:
    pd.DataFrame or pd.Series: Tangency portfolio weights or portfolio returns if `return_port_ret` is True.
    """
    returns = returns.copy()
    
    if 'date' in returns.columns.str.lower():
        returns = returns.rename({'Date': 'date'}, axis=1)
        returns = returns.set_index('date')
    returns.index.name = 'date'

    if cov_mat == 1:
        cov_inv = np.linalg.inv((returns.cov() * annual_factor))
    else:
        cov = returns.cov()
        covmat_diag = np.diag(np.diag((cov)))
        covmat = cov_mat * cov + (1 - cov_mat) * covmat_diag
        cov_inv = np.linalg.pinv((covmat * annual_factor))  
        
    ones = np.ones(returns.columns.shape) 
    if expected_returns is not None:
        mu = expected_returns
        if not expected_returns_already_annualized:
            mu *= annual_factor
    else:
        mu = returns.mean() * annual_factor
    scaling = 1 / (np.transpose(ones) @ cov_inv @ mu)
    tangent_return = scaling * (cov_inv @ mu)
    tangency_wts = pd.DataFrame(
        index=returns.columns,
        data=tangent_return,
        columns=[f'{name} Weights']
    )
    port_returns = returns @ tangency_wts.rename({f'{name} Weights': f'{name} Portfolio'}, axis=1)

    if return_graphic:
        tangency_wts.plot(kind='bar', title=f'{name} Weights')

    if isinstance(target_ret_rescale_weights, (float, int)):
        scaler = target_ret_rescale_weights / port_returns[f'{name} Portfolio'].mean()
        tangency_wts[[f'{name} Weights']] *= scaler
        port_returns *= scaler
        tangency_wts = tangency_wts.rename(
            {f'{name} Weights': f'{name} Weights Rescaled Target {target_ret_rescale_weights:.2%}'},
            axis=1
        )
        port_returns = port_returns.rename(
            {f'{name} Portfolio': f'{name} Portfolio Rescaled Target {target_ret_rescale_weights:.2%}'},
            axis=1
        )

    if cov_mat != 1:
        port_returns = port_returns.rename(columns=lambda c: c.replace('Tangency', f'Tangency Regularized {cov_mat:.2f}'))
        tangency_wts = tangency_wts.rename(columns=lambda c: c.replace('Tangency', f'Tangency Regularized {cov_mat:.2f}'))
        
    if return_port_ret:
        return port_returns
    return tangency_wts



def calc_equal_weights(
    returns: pd.DataFrame,
    return_graphic: bool = False,
    return_port_ret: bool = False,
    target_ret_rescale_weights: Union[float, None] = None,
    name: str = 'Equal Weights'
):
    """
    Calculates equal weights for the portfolio based on the provided returns.

    Parameters:
    returns (pd.DataFrame): Time series of returns.
    return_graphic (bool, default=False): If True, plots the equal weights.
    return_port_ret (bool, default=False): If True, returns the portfolio returns.
    target_ret_rescale_weights (float or None, default=None): Target return for rescaling weights.
    name (str, default='Equal Weights'): Name for labeling the portfolio.

    Returns:
    pd.DataFrame or pd.Series: Equal portfolio weights or portfolio returns if `return_port_ret` is True.
    """
    returns = returns.copy()

    if 'date' in returns.columns.str.lower():
        returns = returns.rename({'Date': 'date'}, axis=1)
        returns = returns.set_index('date')
    returns.index.name = 'date'

    equal_wts = pd.DataFrame(
        index=returns.columns,
        data=[1 / len(returns.columns)] * len(returns.columns),
        columns=[f'{name}']
    )
    port_returns = returns @ equal_wts.rename({f'{name}': f'{name} Portfolio'}, axis=1)

    if return_graphic:
        equal_wts.plot(kind='bar', title=f'{name}')

    if isinstance(target_ret_rescale_weights, (float, int)):
        scaler = target_ret_rescale_weights / port_returns[f'{name} Portfolio'].mean()
        equal_wts[[f'{name}']] *= scaler
        port_returns *= scaler
        equal_wts = equal_wts.rename(
            {f'{name}': f'{name} Rescaled Target {target_ret_rescale_weights:.2%}'},
            axis=1
        )
        port_returns = port_returns.rename(
            {f'{name} Portfolio': f'{name} Portfolio Rescaled Target {target_ret_rescale_weights:.2%}'},
            axis=1
        )
        
    if return_port_ret:
        return port_returns
    return equal_wts



def calc_risk_parity_weights(
    returns: pd.DataFrame,
    return_graphic: bool = False,
    return_port_ret: bool = False,
    target_ret_rescale_weights: Union[None, float] = None,
    name: str = 'Risk Parity'
):
    """
    Calculates risk parity portfolio weights based on the variance of each asset.

    Parameters:
    returns (pd.DataFrame): Time series of returns.
    return_graphic (bool, default=False): If True, plots the risk parity weights.
    return_port_ret (bool, default=False): If True, returns the portfolio returns.
    target_ret_rescale_weights (float or None, default=None): Target return for rescaling weights.
    name (str, default='Risk Parity'): Name for labeling the portfolio.

    Returns:
    pd.DataFrame or pd.Series: Risk parity portfolio weights or portfolio returns if `return_port_ret` is True.
    """
    returns = returns.copy()

    if 'date' in returns.columns.str.lower():
        returns = returns.rename({'Date': 'date'}, axis=1)
        returns = returns.set_index('date')
    returns.index.name = 'date'

    risk_parity_wts = pd.DataFrame(
        index=returns.columns,
        data=[1 / returns[asset].var() for asset in returns.columns],
        columns=[f'{name} Weights']
    )
    port_returns = returns @ risk_parity_wts.rename({f'{name} Weights': f'{name} Portfolio'}, axis=1)

    if return_graphic:
        risk_parity_wts.plot(kind='bar', title=f'{name} Weights')

    if isinstance(target_ret_rescale_weights, (float, int)):
        scaler = target_ret_rescale_weights / port_returns[f'{name} Portfolio'].mean()
        risk_parity_wts[[f'{name} Weights']] *= scaler
        port_returns *= scaler
        risk_parity_wts = risk_parity_wts.rename(
            {f'{name} Weights': f'{name} Weights Rescaled Target {target_ret_rescale_weights:.2%}'},
            axis=1
        )
        port_returns = port_returns.rename(
            {f'{name} Portfolio': f'{name} Portfolio Rescaled Target {target_ret_rescale_weights:.2%}'},
            axis=1
        )
        
    if return_port_ret:
        return port_returns
    return risk_parity_wts


def calc_gmv_weights(
    returns: pd.DataFrame,
    return_graphic: bool = False,
    return_port_ret: bool = False,
    target_ret_rescale_weights: Union[float, None] = None,
    name: str = 'GMV'
):
    """
    Calculates Global Minimum Variance (GMV) portfolio weights.

    Parameters:
    returns (pd.DataFrame): Time series of returns.
    return_graphic (bool, default=False): If True, plots the GMV weights.
    return_port_ret (bool, default=False): If True, returns the portfolio returns.
    target_ret_rescale_weights (float or None, default=None): Target return for rescaling weights.
    name (str, default='GMV'): Name for labeling the portfolio.

    Returns:
    pd.DataFrame or pd.Series: GMV portfolio weights or portfolio returns if `return_port_ret` is True.
    """
    returns = returns.copy()

    if 'date' in returns.columns.str.lower():
        returns = returns.rename({'Date': 'date'}, axis=1)
        returns = returns.set_index('date')
    returns.index.name = 'date'

    ones = np.ones(returns.columns.shape)
    cov = returns.cov()
    cov_inv = np.linalg.inv(cov)
    scaling = 1 / (np.transpose(ones) @ cov_inv @ ones)
    gmv_tot = scaling * cov_inv @ ones
    gmv_wts = pd.DataFrame(
        index=returns.columns,
        data=gmv_tot,
        columns=[f'{name} Weights']
    )
    port_returns = returns @ gmv_wts.rename({f'{name} Weights': f'{name} Portfolio'}, axis=1)

    if isinstance(target_ret_rescale_weights, (float, int)):
        scaler = target_ret_rescale_weights / port_returns[f'{name} Portfolio'].mean()
        gmv_wts[[f'{name} Weights']] *= scaler
        port_returns *= scaler
        gmv_wts = gmv_wts.rename(
            {f'{name} Weights': f'{name} Weights Rescaled Target {target_ret_rescale_weights:.2%}'},
            axis=1
        )
        port_returns = port_returns.rename(
            {f'{name} Portfolio': f'{name} Portfolio Rescaled Target {target_ret_rescale_weights:.2%}'},
            axis=1
        )

    if return_graphic:
        gmv_wts.plot(kind='bar', title=f'{name} Weights')

    if return_port_ret:
        return port_returns

    return gmv_wts



def calc_target_ret_weights(
    target_ret: float,
    returns: pd.DataFrame,
    return_graphic: bool = False,
    return_port_ret: bool = False
):
    """
    Calculates the portfolio weights to achieve a target return by combining Tangency and GMV portfolios.

    Parameters:
    target_ret (float): Target return for the portfolio.
    returns (pd.DataFrame): Time series of asset returns.
    return_graphic (bool, default=False): If True, plots the portfolio weights.
    return_port_ret (bool, default=False): If True, returns the portfolio returns.

    Returns:
    pd.DataFrame: Weights of the Tangency and GMV portfolios, along with the combined target return portfolio.
    """
    returns = returns.copy()
    
    if 'date' in returns.columns.str.lower():
        returns = returns.rename({'Date': 'date'}, axis=1)
        returns = returns.set_index('date')
    returns.index.name = 'date'
    
    mu_tan = returns.mean() @ calc_tangency_weights(returns, cov_mat = 1)
    mu_gmv = returns.mean() @ calc_gmv_weights(returns)
    
    delta = (target_ret - mu_gmv[0]) / (mu_tan[0] - mu_gmv[0])
    mv_weights = (delta * calc_tangency_weights(returns, cov_mat=1)).values + ((1 - delta) * calc_gmv_weights(returns)).values
    
    mv_weights = pd.DataFrame(
        index=returns.columns,
        data=mv_weights,
        columns=[f'Target {target_ret:.2%} Weights']
    )
    port_returns = returns @ mv_weights.rename({f'Target {target_ret:.2%} Weights': f'Target {target_ret:.2%} Portfolio'}, axis=1)

    if return_graphic:
        mv_weights.plot(kind='bar', title=f'Target Return of {target_ret:.2%} Weights')

    if return_port_ret:
        return port_returns

    mv_weights['Tangency Weights'] = calc_tangency_weights(returns, cov_mat=1).values
    mv_weights['GMV Weights'] = calc_gmv_weights(returns).values

    return mv_weights




def create_portfolio(
    returns: pd.DataFrame,
    weights: Union[dict, list],
    port_name: Union[None, str] = None
):
    """
    Creates a portfolio by applying the specified weights to the asset returns.

    Parameters:
    returns (pd.DataFrame): Time series of asset returns.
    weights (dict or list): Weights to apply to the returns. If a list is provided, it will be converted into a dictionary.
    port_name (str or None, default=None): Name for the portfolio. If None, a name will be generated based on asset weights.

    Returns:
    pd.DataFrame: The portfolio returns based on the provided weights.
    """
    if 'date' in returns.columns.str.lower():
        returns = returns.rename({'Date': 'date'}, axis=1)
        returns = returns.set_index('date')
    returns.index.name = 'date'

    if isinstance(weights, list):
        returns = returns.iloc[:, :len(weights)]
        weights = dict(zip(returns.columns, weights))

    returns = returns[list(weights.keys())]
    port_returns = pd.DataFrame(returns @ list(weights.values()))

    if port_name is None:
        port_name = " + ".join([f"{n} ({w:.2%})" for n, w in weights.items()])
    port_returns.columns = [port_name]
    return port_returns

