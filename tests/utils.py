
import pandas as pd
from sklearn.datasets import make_sparse_spd_matrix
import numpy as np
from typing import Union, Literal


def create_returns_df(
        n_samples: int = 1000,
        n_assets: int = 5,
        alpha_sparsity: float = .3,
        seed: int = 42,
        end_date: str = "2024-01-01",
        date_frequecy: Union[Literal["ME", "BM", "BQ", "BA", "W", "D"]] = "ME", # For month
        variance_multiplier: float = .03,
        truncate: bool = True
    ) -> pd.DataFrame:
    if variance_multiplier > 0.5 or variance_multiplier <= 0:
        raise ValueError("variance_multiplier must be between 0 and 0.5")
    np.random.seed(seed)
    asset_names = ["".join(np.random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), 3)) for i in range(n_assets)]
    cov_matrix = make_sparse_spd_matrix(n_dim=n_assets, alpha=alpha_sparsity)
    cov_matrix /= (np.max(cov_matrix) / variance_multiplier)
    returns = np.random.multivariate_normal(np.zeros(n_assets), cov_matrix, n_samples)
    if truncate:
        returns[returns < -1] = -.95
    returns_df = pd.DataFrame(returns, columns=asset_names)
    returns_df.index = pd.date_range(end=end_date, periods=n_samples, freq=date_frequecy)
    return returns_df


if __name__ == "__main__":
    returns_df = create_returns_df()
    print(returns_df.head())
