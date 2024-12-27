
import pandas as pd
from sklearn.datasets import make_sparse_spd_matrix
import numpy as np
from typing import Union, Literal


def create_returns_df(
        n_samples: int = 1000,
        n_assets: int = 5,
        avg_return: float = .004,
        alpha_sparsity: float = .3,
        seed: int = 42,
        end_date: str = "2024-01-01",
        date_frequecy: Union[Literal["ME", "BM", "BQ", "BA", "W", "D"]] = "ME", # For month
        variance_multiplier: float = .03,
        truncate: bool = True
    ) -> pd.DataFrame:
    if variance_multiplier > 0.5 or variance_multiplier <= 0:
        raise ValueError("variance_multiplier must be between 0 and 0.5")
    rng = np.random.RandomState(seed)
    asset_names = ["".join(rng.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), 3)) for i in range(n_assets)]
    cov_matrix = make_sparse_spd_matrix(n_dim=n_assets, alpha=alpha_sparsity)
    cov_matrix /= (np.max(cov_matrix) / variance_multiplier)
    returns = np.random.multivariate_normal(np.ones(n_assets) * avg_return, cov_matrix, n_samples)
    if truncate:
        returns[returns < -1] = -.95
    returns_df = pd.DataFrame(returns, columns=asset_names)
    returns_df.index = pd.date_range(end=end_date, periods=n_samples, freq=date_frequecy)
    return returns_df


def create_rf_returns_df(
    n_samples: int = 1000,
    avg_rf_rate: float = .002,
    ts_auto_correlation: float = .8,
    seed: int = 42,
    std_rf_rate: float = .01,
    end_date: str = "2024-01-01",
    date_frequecy: Union[Literal["ME", "BM", "BQ", "BA", "W", "D"]] = "ME"
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    rf_returns = rng.normal(avg_rf_rate, std_rf_rate, n_samples)
    rf_returns = pd.Series(rf_returns)
    for i in range(1, n_samples):
        rf_returns[i] = rf_returns[i] * (1 - ts_auto_correlation) + rf_returns[i - 1] * ts_auto_correlation
    rf_returns.index = pd.date_range(end=end_date, periods=n_samples, freq=date_frequecy)
    return rf_returns.to_frame("RF")


if __name__ == "__main__":
    returns_df = create_returns_df()
    returns_rf_df = create_rf_returns_df()
    print(returns_df.head())
