import pandas as pd
import numpy as np
import time
import os
import pickle

class SMAStrategy:
    #Vectorized SMA (Simple Moving Average) trading strategy with ATR-based position sizing

    def __init__(self, short_sma, long_sma, big_point_value, slippage=0, capital=6000, atr_period=50):
        """
        Initialize the SMA strategy with specific parameters

        Parameters:
        short_sma (int): Short SMA period in days
        long_sma (int): Long SMA period in days
        big_point_value (int): Contract big point value for calculating dollar P&L
        slippage (float): Slippage in price units added/subtracted from execution price
        capital (float): The capital allocation for position sizing
        atr_period (int): Period for ATR calculation for position sizing
        """
        self.short_sma = short_sma
        self.long_sma = long_sma
        self.big_point_value = big_point_value
        self.slippage = slippage
        self.capital = capital
        self.atr_period = atr_period

    def calculate_atr(self, data, period=None):
        """
        Calculate Average True Range (ATR)
        
        Parameters:
        data: DataFrame with OHLC data
        period: ATR calculation period, defaults to self.atr_period if None
        
        Returns:
        Series: ATR values
        """
        if period is None:
            period = self.atr_period
            
        # Calculate True Range
        # True High is the maximum of today's high and yesterday's close
        high = data['High'].copy()
        prev_close = data['Close'].shift(1)
        true_high = pd.DataFrame({'high': high, 'prev_close': prev_close}).max(axis=1)
        
        # True Low is the minimum of today's low and yesterday's close
        low = data['Low'].copy()
        true_low = pd.DataFrame({'low': low, 'prev_close': prev_close}).min(axis=1)
        
        # True Range = True High - True Low
        true_range = true_high - true_low
        
        # ATR is the moving average of True Range
        atr = true_range.rolling(window=period).mean()
        
        return atr

    def apply_strategy(self, data, strategy_name="Strategy"):
        """
        Apply the simple SMA crossover strategy to the price data using vectorized operations
        with ATR-based position sizing
        """
        # Create a copy of the DataFrame to avoid modifying the original
        sim_data = data.copy()
        
        # Calculate SMAs
        sim_data[f'SMA_Short_{strategy_name}'] = sim_data['Close'].rolling(window=self.short_sma).mean()
        sim_data[f'SMA_Long_{strategy_name}'] = sim_data['Close'].rolling(window=self.long_sma).mean()
        
        # Calculate ATR for position sizing
        sim_data[f'ATR_{strategy_name}'] = self.calculate_atr(sim_data, self.atr_period)
        
        # Calculate position size based on ATR
        sim_data[f'Position_Size_{strategy_name}'] = np.round(
            self.capital / (sim_data[f'ATR_{strategy_name}'] * self.big_point_value) + 0.5
        )
        
        # Determine position direction
        sim_data[f'Position_Dir_{strategy_name}'] = np.where(
            sim_data[f'SMA_Short_{strategy_name}'] > sim_data[f'SMA_Long_{strategy_name}'], 1, -1
        )
        
        # Fill NaN values at the beginning
        sim_data[f'Position_Dir_{strategy_name}'] = sim_data[f'Position_Dir_{strategy_name}'].fillna(0)
        
        # Identify position changes
        sim_data[f'Position_Change_{strategy_name}'] = sim_data[f'Position_Dir_{strategy_name}'].diff() != 0
        
        # Calculate P&L
        market_pnl = sim_data['Close'].diff() * self.big_point_value
        sim_data[f'Market_PnL_{strategy_name}'] = market_pnl
        
        # Strategy P&L
        sim_data[f'Daily_PnL_{strategy_name}'] = (
            market_pnl * 
            sim_data[f'Position_Dir_{strategy_name}'].shift(1) * 
            sim_data[f'Position_Size_{strategy_name}'].shift(1)
        )
        
        # Apply slippage at position changes
        position_changed = sim_data[f'Position_Change_{strategy_name}']
        sim_data.loc[position_changed, f'Daily_PnL_{strategy_name}'] -= (
            self.slippage * sim_data[f'Position_Size_{strategy_name}'][position_changed]
        )
        
        # Replace NaN values in first row
        sim_data[f'Daily_PnL_{strategy_name}'] = sim_data[f'Daily_PnL_{strategy_name}'].fillna(0)
        
        # Calculate cumulative P&L
        sim_data[f'Cumulative_PnL_{strategy_name}'] = sim_data[f'Daily_PnL_{strategy_name}'].cumsum()
        
        return sim_data

    def optimize(self, data, sma_range, train_test_split=0.7, results_file=None, warm_up_idx=None, ticker=None):
        best_sharpe = -np.inf
        best_sma = (0, 0)
        best_trades = 0
        best_sim_data = None
        all_results = []
        sim_data_cache = {}

        if warm_up_idx is None:
            raise ValueError("warm_up_idx must be provided to ensure data is trimmed properly.")

        if results_file:
            with open(results_file, 'w') as f:
                f.write("short_SMA,long_SMA,trades,sharpe_ratio\n")

        total_combinations = sum(1 for a, b in [(s, l) for s in sma_range for l in sma_range] if a < b)
        completed = 0
        total_sim_time = 0

        print(f"Running {total_combinations} simulations...")

        for short_sma in sma_range:
            for long_sma in sma_range:
                if short_sma >= long_sma:
                    continue

                sim_start_time = time.time()

                orig_short_sma = self.short_sma
                orig_long_sma = self.long_sma

                self.short_sma = short_sma
                self.long_sma = long_sma

                sim_data = self.apply_strategy(data.copy(), strategy_name=f"Sim_{short_sma}_{long_sma}")
                sim_data_eval = sim_data.iloc[warm_up_idx:].copy()
                split_index = int(len(sim_data_eval) * train_test_split)

                trade_col = sim_data_eval.filter(like="Position_Change").columns[0]
                trade_entries = sim_data_eval[trade_col]
                trade_count = trade_entries.sum()

                pnl_col = sim_data_eval.filter(like="Daily_PnL").columns[0]
                sim_data_eval['Daily_Returns'] = sim_data_eval[pnl_col]
                in_sample_returns = sim_data_eval['Daily_Returns'].iloc[:split_index]

                if len(in_sample_returns.dropna()) == 0 or in_sample_returns.std() == 0:
                    sharpe_ratio = 0
                else:
                    sharpe_ratio = in_sample_returns.mean() / in_sample_returns.std() * np.sqrt(252)

                if results_file:
                    with open(results_file, 'a') as f:
                        f.write(f"{short_sma},{long_sma},{trade_count},{sharpe_ratio:.6f}\n")

                sim_data_cache[(short_sma, long_sma)] = sim_data
                all_results.append((short_sma, long_sma, trade_count, sharpe_ratio))

                if sharpe_ratio > best_sharpe:
                    best_sharpe = sharpe_ratio
                    best_sma = (short_sma, long_sma)
                    best_trades = trade_count
                    best_sim_data = sim_data_eval.copy()

                self.short_sma = orig_short_sma
                self.long_sma = orig_long_sma

                sim_end_time = time.time()
                sim_time = sim_end_time - sim_start_time
                total_sim_time += sim_time

                completed += 1
                if completed % 1500 == 0 or completed == total_combinations:
                    avg_sim_time = total_sim_time / completed
                    est_remaining = avg_sim_time * (total_combinations - completed)
                    print(
                        f"Progress: {completed}/{total_combinations} simulations completed ({(completed / total_combinations * 100):.1f}%)"
                        f" - Avg sim time: {avg_sim_time:.4f}s - Est. remaining: {est_remaining:.1f}s")

        if best_sim_data is not None:
            print("\n--- OPTIMIZATION SHARPE VERIFICATION ---")
            verify_split_idx = int(len(best_sim_data) * train_test_split)
            verify_returns = best_sim_data['Daily_Returns'].iloc[:verify_split_idx]

            if verify_returns.std() > 0:
                verify_sharpe = verify_returns.mean() / verify_returns.std() * np.sqrt(252)
                print(f"Optimization best Sharpe = {best_sharpe:.6f}")
                print(f"Verification Sharpe = {verify_sharpe:.6f}")
                print(f"Data points used: {len(best_sim_data)}")
                print(f"In-sample data points: {len(verify_returns)}")
            else:
                print("Cannot verify Sharpe (std = 0)")

        return best_sma, best_sharpe, best_trades, all_results

    def calculate_performance_metrics(self, data, strategy_name="Strategy", train_test_split=0.7):
        """
        Calculate detailed performance metrics for the strategy

        Parameters:
        data: DataFrame with strategy results
        strategy_name: Name suffix for the strategy columns
        train_test_split: Portion of data used for in-sample testing

        Returns:
        dict: Dictionary with performance metrics
        """
        # Calculate split index for in-sample/out-of-sample
        split_index = int(len(data) * train_test_split)
        split_date = data.index[split_index]

        # Extract daily P&L data
        daily_pnl = data[f'Daily_PnL_{strategy_name}']
        
        # Split returns for performance comparison
        returns_in_sample = daily_pnl.iloc[:split_index]
        returns_out_sample = daily_pnl.iloc[split_index:]

        # Calculate separate Sharpe ratios on dollar P&L (annualized)
        sharpe_in_sample = returns_in_sample.mean() / returns_in_sample.std() * np.sqrt(
            252) if returns_in_sample.std() > 0 else 0
        sharpe_out_sample = returns_out_sample.mean() / returns_out_sample.std() * np.sqrt(
            252) if returns_out_sample.std() > 0 else 0
        sharpe_full = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252) if daily_pnl.std() > 0 else 0

        # Calculate trade counts using position changes
        position_changes = data[f'Position_Change_{strategy_name}']
        
        total_trades = position_changes.sum()
        in_sample_trades = position_changes.iloc[:split_index].sum()
        out_sample_trades = position_changes.iloc[split_index:].sum()

        # Calculate max drawdown in dollar terms
        pnl_series = data[f'Cumulative_PnL_{strategy_name}']
        # Use loc to avoid SettingWithCopyWarning
        data.loc[:, 'Peak'] = pnl_series.cummax()
        data.loc[:, 'Drawdown_Dollars'] = pnl_series - data['Peak']
        max_drawdown_dollars = data['Drawdown_Dollars'].min()

        # Calculate profit/loss metrics
        total_pnl = data[f'Cumulative_PnL_{strategy_name}'].iloc[-1]

        # Calculate in-sample and out-of-sample P&L
        in_sample_pnl = data[f'Daily_PnL_{strategy_name}'].iloc[:split_index].sum()
        out_sample_pnl = data[f'Daily_PnL_{strategy_name}'].iloc[split_index:].sum()

        # Calculate average position size
        avg_position_size = data[f'Position_Size_{strategy_name}'].mean()
        max_position_size = data[f'Position_Size_{strategy_name}'].max()

        # Assemble the results in a dictionary
        metrics = {
            'split_date': split_date,
            'total_pnl': total_pnl,
            'sharpe_full': sharpe_full,
            'sharpe_in_sample': sharpe_in_sample,
            'sharpe_out_sample': sharpe_out_sample,
            'max_drawdown_dollars': max_drawdown_dollars,
            'total_trades': total_trades,
            'in_sample_trades': in_sample_trades,
            'out_sample_trades': out_sample_trades,
            'in_sample_pnl': in_sample_pnl,
            'out_sample_pnl': out_sample_pnl,
            'avg_position_size': avg_position_size,
            'max_position_size': max_position_size
        }

        return metrics