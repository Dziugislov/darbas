import pandas as pd
import numpy as np
from input import INITIAL_CAPITAL


class SMAStrategy:
    """
    SMA (Simple Moving Average) trading strategy that is always in the market

    Features:
    - SMA relationship determines long or short position
      - If short SMA > long SMA → LONG
      - If short SMA < long SMA → SHORT
    - Always in the market (either long or short)
    - No stop losses or trailing stops
    """

    def __init__(self, short_sma, long_sma, contract_size, fixed_stop_loss=0.95, slippage=0):
        """
        Initialize the SMA strategy with specific parameters
        """
        self.short_sma = short_sma
        self.long_sma = long_sma
        self.contract_size = contract_size
        self.slippage = slippage

    def apply_strategy(self, data, strategy_name="Strategy", initial_capital=INITIAL_CAPITAL):
        """
        Apply the SMA strategy that is always in the market based on SMA relationship
        """
        # Create a copy of the DataFrame to avoid modifying the original
        sim_data = data.copy()

        # Calculate SMAs
        sim_data[f'SMA_Short_{strategy_name}'] = sim_data['Close'].rolling(window=self.short_sma).mean()
        sim_data[f'SMA_Long_{strategy_name}'] = sim_data['Close'].rolling(window=self.long_sma).mean()

        # Initialize signal column (0 for days where we don't have enough data for both SMAs)
        sim_data[f'Signal_{strategy_name}'] = 0

        # Set up columns to track position and entry
        sim_data[f'Entry_Price_{strategy_name}'] = np.nan
        sim_data[f'Position_{strategy_name}'] = 0

        # Find where both SMAs are available (not NaN)
        valid_sma_mask = sim_data[f'SMA_Short_{strategy_name}'].notna() & sim_data[f'SMA_Long_{strategy_name}'].notna()

        # For all days where we have valid SMAs, determine position based on SMA relationship
        sim_data.loc[valid_sma_mask & (sim_data[f'SMA_Short_{strategy_name}'] > sim_data[f'SMA_Long_{strategy_name}']),
        f'Signal_{strategy_name}'] = 1  # Long position when short SMA > long SMA

        sim_data.loc[valid_sma_mask & (sim_data[f'SMA_Short_{strategy_name}'] < sim_data[f'SMA_Long_{strategy_name}']),
        f'Signal_{strategy_name}'] = -1  # Short position when short SMA < long SMA

        # Also update the Position column to match the Signal
        sim_data[f'Position_{strategy_name}'] = sim_data[f'Signal_{strategy_name}']

        # Calculate entry prices (when position changes)
        position_changes = (sim_data[f'Signal_{strategy_name}'] != sim_data[f'Signal_{strategy_name}'].shift(1)) & \
                           (sim_data[f'Signal_{strategy_name}'] != 0)

        # For entries into long positions, add slippage
        long_entries = position_changes & (sim_data[f'Signal_{strategy_name}'] == 1)
        sim_data.loc[long_entries, f'Entry_Price_{strategy_name}'] = sim_data.loc[long_entries, 'Close'] + self.slippage

        # For entries into short positions, subtract slippage
        short_entries = position_changes & (sim_data[f'Signal_{strategy_name}'] == -1)
        sim_data.loc[short_entries, f'Entry_Price_{strategy_name}'] = sim_data.loc[
                                                                          short_entries, 'Close'] - self.slippage

        # Calculate P&L based on fixed contract size
        sim_data[f'Daily_PnL_{strategy_name}'] = sim_data['Close'].diff() * sim_data[f'Signal_{strategy_name}'].shift(
            1) * self.contract_size
        sim_data[f'Daily_PnL_{strategy_name}'] = sim_data[f'Daily_PnL_{strategy_name}'].fillna(
            0)  # Replace NaN values in first row

        # Calculate cumulative P&L
        sim_data[f'Cumulative_PnL_{strategy_name}'] = sim_data[f'Daily_PnL_{strategy_name}'].cumsum()

        # Calculate capital over time
        sim_data[f'Capital_{strategy_name}'] = initial_capital + sim_data[f'Cumulative_PnL_{strategy_name}']

        # Calculate returns as multiple of initial capital
        sim_data[f'Cumulative_Returns_{strategy_name}'] = sim_data[f'Capital_{strategy_name}'] / initial_capital

        # Logarithmic Returns
        epsilon = 1e-10
        sim_data[f'Returns_{strategy_name}'] = np.log(
            (sim_data[f'Capital_{strategy_name}'] + epsilon) /
            (sim_data[f'Capital_{strategy_name}'].shift(1) + epsilon)
        ).fillna(0)

        return sim_data