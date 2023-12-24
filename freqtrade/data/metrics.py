import logging
import math
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def calculate_market_change(data: Dict[str, pd.DataFrame], column: str = "close") -> float:
    """
    Calculate market change based on "column".
    Calculation is done by taking the first non-null and the last non-null element of each DataFrame
    and calculating the pctchange as "(last - first) / first".
    Then the results per pair are combined as mean.

    :param data: Dict of DataFrames, dict key should be pair.
    :param column: Column in the original dataframes to use
    :return: Market change as a percentage
    """
    try:
        if not data:
            raise ValueError("Empty data dictionary provided.")

        pair_changes = []

        for pair, df in data.items():
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame for pair '{pair}'.")

            non_null_values = df[column].dropna()

            if len(non_null_values) < 2:
                raise ValueError(f"Not enough data points for pair '{pair}'.")

            first_value = non_null_values.iloc[0]
            last_value = non_null_values.iloc[-1]

            pair_change = (last_value - first_value) / first_value
            pair_changes.append(pair_change)

        if not pair_changes:
            raise ValueError("No valid data found for market change calculation.")

        market_change = np.mean(pair_changes) * 100.0  # Return the mean market change as a percentage
        return market_change

    except Exception as e:
        # Handle exceptions and log errors
        logging.error(f"Error in calculate_market_change: {str(e)}")
        return 0.0  # Return a default value or handle the error as needed


def combine_dataframes_with_mean(data: Dict[str, pd.DataFrame],
                                 column: str = "close") -> pd.DataFrame:
    """
    Combine multiple dataframes "column"
    :param data: Dict of Dataframes, dict key should be pair.
    :param column: Column in the original dataframes to use
    :return: DataFrame with the column renamed to the dict key, and a column
        named mean, containing the mean of all pairs.
    :raise: ValueError if no data is provided.
    """
    df_comb = pd.concat([data[pair].set_index('date').rename(
        {column: pair}, axis=1)[pair] for pair in data], axis=1)

    df_comb['mean'] = df_comb.mean(axis=1)

    return df_comb


def create_cum_profit(df: pd.DataFrame, trades: pd.DataFrame, col_name: str,
                      timeframe: str) -> pd.DataFrame:
    """
    Adds a column `col_name` with the cumulative profit for the given trades array.
    :param df: DataFrame with date index
    :param trades: DataFrame containing trades (requires columns close_date and profit_abs)
    :param col_name: Column name that will be assigned the results
    :param timeframe: Timeframe used during the operations
    :return: Returns df with one additional column, col_name, containing the cumulative profit.
    :raise: ValueError if trade-dataframe was found empty.
    """
    if len(trades) == 0:
        raise ValueError("Trade dataframe empty.")
    from freqtrade.exchange import timeframe_to_minutes
    timeframe_minutes = timeframe_to_minutes(timeframe)
    # Resample to timeframe to make sure trades match candles
    _trades_sum = trades.resample(f'{timeframe_minutes}min', on='close_date'
                                  )[['profit_abs']].sum()
    df.loc[:, col_name] = _trades_sum['profit_abs'].cumsum()
    # Set first value to 0
    df.loc[df.iloc[0].name, col_name] = 0
    # FFill to get continuous
    df[col_name] = df[col_name].ffill()
    return df


def _calc_drawdown_series(profit_results: pd.DataFrame, *, date_col: str, value_col: str,
                          starting_balance: float) -> pd.DataFrame:
    max_drawdown_df = pd.DataFrame()
    max_drawdown_df['cumulative'] = profit_results[value_col].cumsum()
    max_drawdown_df['high_value'] = max_drawdown_df['cumulative'].cummax()
    max_drawdown_df['drawdown'] = max_drawdown_df['cumulative'] - max_drawdown_df['high_value']
    max_drawdown_df['date'] = profit_results.loc[:, date_col]
    if starting_balance:
        cumulative_balance = starting_balance + max_drawdown_df['cumulative']
        max_balance = starting_balance + max_drawdown_df['high_value']
        max_drawdown_df['drawdown_relative'] = ((max_balance - cumulative_balance) / max_balance)
    else:
        # NOTE: This is not completely accurate,
        # but might good enough if starting_balance is not available
        max_drawdown_df['drawdown_relative'] = (
            (max_drawdown_df['high_value'] - max_drawdown_df['cumulative'])
            / max_drawdown_df['high_value'])
    return max_drawdown_df


def calculate_underwater(trades: pd.DataFrame, *, date_col: str = 'close_date',
                         value_col: str = 'profit_ratio', starting_balance: float = 0.0
                         ):
    """
    Calculate max drawdown and the corresponding close dates
    :param trades: DataFrame containing trades (requires columns close_date and profit_ratio)
    :param date_col: Column in DataFrame to use for dates (defaults to 'close_date')
    :param value_col: Column in DataFrame to use for values (defaults to 'profit_ratio')
    :return: Tuple (float, highdate, lowdate, highvalue, lowvalue) with absolute max drawdown,
             high and low time and high and low value.
    :raise: ValueError if trade-dataframe was found empty.
    """
    if len(trades) == 0:
        raise ValueError("Trade dataframe empty.")
    profit_results = trades.sort_values(date_col).reset_index(drop=True)
    max_drawdown_df = _calc_drawdown_series(
        profit_results,
        date_col=date_col,
        value_col=value_col,
        starting_balance=starting_balance)

    return max_drawdown_df


def calculate_max_drawdown(trades: pd.DataFrame, *, date_col: str = 'close_date',
                           value_col: str = 'profit_abs', starting_balance: float = 0,
                           relative: bool = False
                           ) -> Tuple[float, pd.Timestamp, pd.Timestamp, float, float, float]:
    """
    Calculate max drawdown and the corresponding close dates
    :param trades: DataFrame containing trades (requires columns close_date and profit_ratio)
    :param date_col: Column in DataFrame to use for dates (defaults to 'close_date')
    :param value_col: Column in DataFrame to use for values (defaults to 'profit_abs')
    :param starting_balance: Portfolio starting balance - properly calculate relative drawdown.
    :return: Tuple (float, highdate, lowdate, highvalue, lowvalue, relative_drawdown)
             with absolute max drawdown, high and low time and high and low value,
             and the relative account drawdown
    :raise: ValueError if trade-dataframe was found empty.
    """
    if len(trades) == 0:
        raise ValueError("Trade dataframe empty.")
    profit_results = trades.sort_values(date_col).reset_index(drop=True)
    max_drawdown_df = _calc_drawdown_series(
        profit_results,
        date_col=date_col,
        value_col=value_col,
        starting_balance=starting_balance
    )

    idxmin = max_drawdown_df['drawdown_relative'].idxmax() if relative \
        else max_drawdown_df['drawdown'].idxmin()
    if idxmin == 0:
        raise ValueError("No losing trade, therefore no drawdown.")
    high_date = profit_results.loc[max_drawdown_df.iloc[:idxmin]['high_value'].idxmax(), date_col]
    low_date = profit_results.loc[idxmin, date_col]
    high_val = max_drawdown_df.loc[max_drawdown_df.iloc[:idxmin]
                                   ['high_value'].idxmax(), 'cumulative']
    low_val = max_drawdown_df.loc[idxmin, 'cumulative']
    max_drawdown_rel = max_drawdown_df.loc[idxmin, 'drawdown_relative']

    return (
        abs(max_drawdown_df.loc[idxmin, 'drawdown']),
        high_date,
        low_date,
        high_val,
        low_val,
        max_drawdown_rel
    )


def calculate_csum(trades: pd.DataFrame, starting_balance: float = 0) -> Tuple[float, float]:
    """
    Calculate min/max cumsum of trades, to show if the wallet/stake amount ratio is sane
    :param trades: DataFrame containing trades (requires columns close_date and profit_percent)
    :param starting_balance: Add starting balance to results, to show the wallets high / low points
    :return: Tuple (float, float) with cumsum of profit_abs
    :raise: ValueError if trade-dataframe was found empty.
    """
    if len(trades) == 0:
        raise ValueError("Trade dataframe empty.")

    csum_df = pd.DataFrame()
    csum_df['sum'] = trades['profit_abs'].cumsum()
    csum_min = csum_df['sum'].min() + starting_balance
    csum_max = csum_df['sum'].max() + starting_balance

    return csum_min, csum_max


def calculate_cagr(days_passed: int, starting_balance: float, final_balance: float) -> float:
    """
    Calculate CAGR
    :param days_passed: Days passed between start and ending balance
    :param starting_balance: Starting balance
    :param final_balance: Final balance to calculate CAGR against
    :return: CAGR
    """
    return (final_balance / starting_balance) ** (1 / (days_passed / 365)) - 1


def calculate_expectancy(trades: pd.DataFrame, risk_free_rate: float = 0.0) -> Dict[str, float]:
    """
    Calculate trading expectancy metrics.
    
    :param trades: DataFrame containing trades (requires columns close_date and profit_abs)
    :param risk_free_rate: Annual risk-free rate for calculating risk-adjusted metrics (default is 0.0)
    
    :return: A dictionary containing trading performance metrics.
    """
    if len(trades) == 0:
        return {"Error": "Trade dataframe is empty"}

    metrics = {
        "Total Trades": len(trades),
        "Total Wins": len(trades[trades["profit_abs"] > 0]),
        "Total Losses": len(trades[trades["profit_abs"] < 0])
    }

    metrics["Win Rate"] = (metrics["Total Wins"] / metrics["Total Trades"]) * 100
    metrics["Loss Rate"] = 100 - metrics["Win Rate"]

    if metrics["Total Wins"] > 0:
        metrics["Average Win"] = trades.loc[trades["profit_abs"] > 0, "profit_abs"].mean()
    else:
        metrics["Average Win"] = 0

    if metrics["Total Losses"] > 0:
        metrics["Average Loss"] = abs(trades.loc[trades["profit_abs"] < 0, "profit_abs"].mean())
    else:
        metrics["Average Loss"] = 0

    metrics["Expectancy"] = (metrics["Average Win"] * metrics["Win Rate"] - metrics["Average Loss"] * metrics["Loss Rate"]) / 100

    max_drawdown = calculate_max_drawdown(trades)
    metrics["Max Drawdown"] = max_drawdown["max_drawdown"]
    metrics["Max Drawdown Start Date"] = max_drawdown["max_drawdown_start_date"]
    metrics["Max Drawdown End Date"] = max_drawdown["max_drawdown_end_date"]

    if metrics["Total Losses"] > 0:
        metrics["Profit Factor"] = metrics["Total Wins"] / metrics["Total Losses"]
    else:
        metrics["Profit Factor"] = None

    metrics["Risk-Adjusted Return"] = calculate_risk_adjusted_return(metrics["Expectancy"], metrics["Max Drawdown"], risk_free_rate)

    return metrics

def calculate_max_drawdown(trades: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate the maximum drawdown.
    
    :param trades: DataFrame containing trades (requires columns close_date and profit_abs)
    
    :return: A dictionary containing the maximum drawdown metrics.
    """
    if len(trades) == 0:
        return {"max_drawdown": 0.0, "max_drawdown_start_date": None, "max_drawdown_end_date": None}

    cumulative_profit = (1 + (trades["profit_abs"] / 100)).cumprod()
    max_drawdown = ((cumulative_profit.cummax() - cumulative_profit) / cumulative_profit.cummax()).max()
    max_drawdown_start_date = cumulative_profit[cumulative_profit.idxmax():].idxmin()
    max_drawdown_end_date = cumulative_profit[cumulative_profit.idxmax():].idxmax()

    return {"max_drawdown": max_drawdown, "max_drawdown_start_date": max_drawdown_start_date, "max_drawdown_end_date": max_drawdown_end_date}

def calculate_risk_adjusted_return(expectancy: float, max_drawdown: float, risk_free_rate: float) -> float:
    """
    Calculate the risk-adjusted return using the Sharpe ratio.
    
    :param expectancy: Trading expectancy
    :param max_drawdown: Maximum drawdown
    :param risk_free_rate: Annual risk-free rate
    
    :return: Risk-adjusted return (Sharpe ratio)
    """
    if max_drawdown == 0:
        return 0.0

    sharpe_ratio = (expectancy - (risk_free_rate / 252)) / (max_drawdown / 100)
    return sharpe_ratio


def calculate_sortino(trades: pd.DataFrame, min_date: datetime, max_date: datetime,
                      starting_balance: float) -> float:
    """
    Calculate sortino
    :param trades: DataFrame containing trades (requires columns profit_abs)
    :return: sortino
    """
    if (len(trades) == 0) or (min_date is None) or (max_date is None) or (min_date == max_date):
        return 0

    total_profit = trades['profit_abs'] / starting_balance
    days_period = max(1, (max_date - min_date).days)

    expected_returns_mean = total_profit.sum() / days_period

    down_stdev = np.std(trades.loc[trades['profit_abs'] < 0, 'profit_abs'] / starting_balance)

    if down_stdev != 0 and not np.isnan(down_stdev):
        sortino_ratio = expected_returns_mean / down_stdev * np.sqrt(365)
    else:
        # Define high (negative) sortino ratio to be clear that this is NOT optimal.
        sortino_ratio = -100

    # print(expected_returns_mean, down_stdev, sortino_ratio)
    return sortino_ratio


def calculate_sharpe(trades: pd.DataFrame, min_date: datetime, max_date: datetime,
                     starting_balance: float) -> float:
    """
    Calculate sharpe
    :param trades: DataFrame containing trades (requires column profit_abs)
    :return: sharpe
    """
    if (len(trades) == 0) or (min_date is None) or (max_date is None) or (min_date == max_date):
        return 0

    total_profit = trades['profit_abs'] / starting_balance
    days_period = max(1, (max_date - min_date).days)

    expected_returns_mean = total_profit.sum() / days_period
    up_stdev = np.std(total_profit)

    if up_stdev != 0:
        sharp_ratio = expected_returns_mean / up_stdev * np.sqrt(365)
    else:
        # Define high (negative) sharpe ratio to be clear that this is NOT optimal.
        sharp_ratio = -100

    # print(expected_returns_mean, up_stdev, sharp_ratio)
    return sharp_ratio


def calculate_calmar(trades: pd.DataFrame, min_date: datetime, max_date: datetime,
                     starting_balance: float) -> float:
    """
    Calculate calmar
    :param trades: DataFrame containing trades (requires columns close_date and profit_abs)
    :return: calmar
    """
    if (len(trades) == 0) or (min_date is None) or (max_date is None) or (min_date == max_date):
        return 0

    total_profit = trades['profit_abs'].sum() / starting_balance
    days_period = max(1, (max_date - min_date).days)

    # adding slippage of 0.1% per trade
    # total_profit = total_profit - 0.0005
    expected_returns_mean = total_profit / days_period * 100

    # calculate max drawdown
    try:
        _, _, _, _, _, max_drawdown = calculate_max_drawdown(
            trades, value_col="profit_abs", starting_balance=starting_balance
        )
    except ValueError:
        max_drawdown = 0

    if max_drawdown != 0:
        calmar_ratio = expected_returns_mean / max_drawdown * math.sqrt(365)
    else:
        # Define high (negative) calmar ratio to be clear that this is NOT optimal.
        calmar_ratio = -100

    # print(expected_returns_mean, max_drawdown, calmar_ratio)
    return calmar_ratio
