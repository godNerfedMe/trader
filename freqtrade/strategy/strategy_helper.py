import pandas as pd
import logging
from typing import Optional

from freqtrade.exchange import timeframe_to_minutes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def merge_informative_pair(dataframe: pd.DataFrame, informative: pd.DataFrame,
                           timeframe: str, timeframe_inf: str, ffill: bool = True,
                           append_timeframe: bool = True,
                           date_column: str = 'date',
                           suffix: Optional[str] = None) -> pd.DataFrame:
    """
    Merge informative samples into the original dataframe, avoiding lookahead bias.

    This function correctly aligns the timestamps to ensure that there's no lookahead bias when merging
    informative data with the original dataframe.

    :param dataframe: Original dataframe.
    :param informative: Informative pair dataframe.
    :param timeframe: Timeframe of the original pair sample.
    :param timeframe_inf: Timeframe of the informative pair sample.
    :param ffill: Forward-fill missing values (optional but recommended).
    :param append_timeframe: Rename columns by appending timeframe.
    :param date_column: Name of the date column.
    :param suffix: Suffix to add to informative columns (if specified, append_timeframe must be False).
    :return: Merged dataframe.
    :raise: ValueError if the secondary timeframe is shorter than the dataframe timeframe.
    """
    try:
        # Input validation
        assert isinstance(dataframe, pd.DataFrame), "Input 'dataframe' must be a DataFrame."
        assert isinstance(informative, pd.DataFrame), "Input 'informative' must be a DataFrame."
        assert date_column in dataframe.columns, f"'{date_column}' not found in 'dataframe' columns."
        assert date_column in informative.columns, f"'{date_column}' not found in 'informative' columns."

        informative = informative.copy()
        minutes_inf = timeframe_to_minutes(timeframe_inf)
        minutes = timeframe_to_minutes(timeframe)

        if minutes == minutes_inf:
            # No need to forwardshift if the timeframes are identical
            informative['date_merge'] = informative[date_column]
        elif minutes < minutes_inf:
            if not informative.empty:
                if timeframe_inf == '1M':
                    informative['date_merge'] = (
                        (informative[date_column] + pd.offsets.MonthBegin(1))
                        - pd.to_timedelta(minutes, 'm')
                    )
                else:
                    informative['date_merge'] = (
                        informative[date_column] + pd.to_timedelta(minutes_inf, 'm') -
                        pd.to_timedelta(minutes, 'm')
                    )
            else:
                informative['date_merge'] = informative[date_column]
        else:
            raise ValueError("Tried to merge a faster timeframe to a slower timeframe. "
                            "This can create new rows and affect your indicators.")

        date_merge = 'date_merge'
        if suffix and append_timeframe:
            raise ValueError("You cannot specify `append_timeframe` as True and a `suffix`.")
        elif append_timeframe:
            date_merge = f'date_merge_{timeframe_inf}'
            informative.columns = [f"{col}_{timeframe_inf}" for col in informative.columns]
        elif suffix:
            date_merge = f'date_merge_{suffix}'
            informative.columns = [f"{col}_{suffix}" for col in informative.columns]

        if ffill:
            dataframe = pd.merge_ordered(dataframe, informative, fill_method="ffill", left_on='date',
                                         right_on=date_merge, how='left')
        else:
            dataframe = pd.merge(dataframe, informative, left_on='date',
                                 right_on=date_merge, how='left')
        dataframe = dataframe.drop(date_merge, axis=1)

        return dataframe
    except Exception as e:
        logger.error(f"An error occurred in 'merge_informative_pair': {str(e)}")
        raise

def stoploss_from_open(open_relative_stop: float, current_profit: float,
                       is_short: bool = False, leverage: float = 1.0) -> float:
    """
    Calculate stop loss relative to the current price based on open-relative stop.

    Given the current profit, an open-relative stop, and leverage, this function calculates
    a stop loss value relative to the current price.

    :param open_relative_stop: Desired stop loss percentage, relative to the open price (adjusted for leverage).
    :param current_profit: The current profit percentage.
    :param is_short: True for short trades, False for long trades.
    :param leverage: Leverage used for the calculation.
    :return: Stop loss value relative to the current price.
    """
    try:
        assert 0 <= open_relative_stop <= 1, "Invalid open_relative_stop value. Must be between 0 and 1."
        assert -1 <= current_profit <= 1, "Invalid current_profit value. Must be between -1 and 1."

        _current_profit = current_profit / leverage

        if (_current_profit == -1 and not is_short) or (is_short and _current_profit == 1):
            return 1

        if is_short:
            stoploss = -1 + ((1 - open_relative_stop / leverage) / (1 - _current_profit))
        else:
            stoploss = 1 - ((1 + open_relative_stop / leverage) / (1 + _current_profit))

        return max(stoploss * leverage, 0.0)
    except Exception as e:
        logger.error(f"An error occurred in 'stoploss_from_open': {str(e)}")
        raise

def stoploss_from_absolute(stop_rate: float, current_rate: float,
                           is_short: bool = False, leverage: float = 1.0) -> float:
    """
    Calculate stop loss relative to the current price based on an absolute stop rate.

    Given the current rate, a stop rate, and leverage, this function calculates a stop loss value
    relative to the current price.

    :param stop_rate: Stop loss price.
    :param current_rate: Current asset price.
    :param is_short: True for short trades, False for long trades.
    :param leverage: Leverage used for the calculation.
    :return: Positive stop loss value relative to the current price.
    """
    try:
        assert current_rate > 0, "Invalid current_rate. Must be greater than 0."

        stoploss = 1 - (stop_rate / current_rate)

        if is_short:
            stoploss = -stoploss

        return max(min(stoploss, 1.0), 0.0) * leverage
    except Exception as e:
        logger.error(f"An error occurred in 'stoploss_from_absolute': {str(e)}")
        raise
