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

pd.options.display.float_format = "{:,.4f}".format
warnings.filterwarnings("ignore")


def read_excel_default(
    excel_name: str,
    index_col: int = 0,
    parse_dates: bool = True,
    print_sheets: bool = False,
    sheet_name: str = None,
    **kwargs,
):
    """
    Reads an Excel file and returns a DataFrame with specified options.

    Parameters:
    excel_name (str): The path to the Excel file.
    index_col (int, default=0): Column to use as the row labels of the DataFrame.
    parse_dates (bool, default=True): Boolean to parse dates.
    print_sheets (bool, default=False): If True, prints the names and first few rows of all sheets.
    sheet_name (str or int, default=None): Name or index of the sheet to read. If None, reads the first sheet.
    **kwargs: Additional arguments passed to `pd.read_excel`.

    Returns:
    pd.DataFrame: DataFrame containing the data from the specified Excel sheet.

    Notes:
    - If `print_sheets` is True, the function will print the names and first few rows of all sheets and return None.
    - The function ensures that the index name is set to 'date' if the index column name is 'date' or 'dates', or if the index contains date-like values.
    """
    if print_sheets:
        n = 0
        while True:
            try:
                sheet = pd.read_excel(excel_name, sheet_name=n)
                print(f"Sheet {n}:")
                print(", ".join(list(sheet.columns)))
                print(sheet.head(3))
                n += 1
                print("\n" * 2)
            except:
                return
    sheet_name = 0 if sheet_name is None else sheet_name
    returns = pd.read_excel(
        excel_name,
        index_col=index_col,
        parse_dates=parse_dates,
        sheet_name=sheet_name,
        **kwargs,
    )
    if returns.index.name is not None:
        if returns.index.name.lower() in ["date", "dates"]:
            returns.index.name = "date"
    elif isinstance(returns.index[0], (datetime.date, datetime.datetime)):
        returns.index.name = "date"
    return returns


def _filter_columns_and_indexes(
    df: pd.DataFrame,
    keep_columns: Union[list, str],
    drop_columns: Union[list, str],
    keep_indexes: Union[list, str],
    drop_indexes: Union[list, str],
    drop_before_keep: bool = False,
):
    """
    Filters a DataFrame based on specified columns and indexes.

    Parameters:
    df (pd.DataFrame): DataFrame to be filtered.
    keep_columns (list or str): Columns to keep in the DataFrame.
    drop_columns (list or str): Columns to drop from the DataFrame.
    keep_indexes (list or str): Indexes to keep in the DataFrame.
    drop_indexes (list or str): Indexes to drop from the DataFrame.
    drop_before_keep (bool, default=False): Whether to drop specified columns/indexes before keeping.

    Returns:
    pd.DataFrame: The filtered DataFrame.
    """
    if not isinstance(df, (pd.DataFrame, pd.Series)):
        return df
    df = df.copy()
    # Columns
    if keep_columns is not None:
        keep_columns = (
            "(?i)" + "|".join(keep_columns)
            if isinstance(keep_columns, list)
            else "(?i)" + keep_columns
        )
    else:
        keep_columns = None
    if drop_columns is not None:
        drop_columns = (
            "(?i)" + "|".join(drop_columns)
            if isinstance(drop_columns, list)
            else "(?i)" + drop_columns
        )
    else:
        drop_columns = None
    if not drop_before_keep:
        if keep_columns is not None:
            df = df.filter(regex=keep_columns)
    if drop_columns is not None:
        df = df.drop(columns=df.filter(regex=drop_columns).columns)
    if drop_before_keep:
        if keep_columns is not None:
            df = df.filter(regex=keep_columns)
    # Indexes
    if keep_indexes is not None:
        keep_indexes = (
            "(?i)" + "|".join(keep_indexes)
            if isinstance(keep_indexes, list)
            else "(?i)" + keep_indexes
        )
    else:
        keep_indexes = None
    if drop_indexes is not None:
        drop_indexes = (
            "(?i)" + "|".join(drop_indexes)
            if isinstance(drop_indexes, list)
            else "(?i)" + drop_indexes
        )
    else:
        drop_indexes = None
    if not drop_before_keep:
        if keep_indexes is not None:
            df = df.filter(regex=keep_indexes, axis=0)
    if drop_indexes is not None:
        df = df.drop(index=df.filter(regex=drop_indexes, axis=0).index)
    if drop_before_keep:
        if keep_indexes is not None:
            df = df.filter(regex=keep_indexes, axis=0)
    return df
