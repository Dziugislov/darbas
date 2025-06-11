import pandas as pd
import numpy as np
import time
from read_ts import mavg  # using built-in SMA function from read_ts

class AlligatorSMAStrategy:
    """
    Vectorized Alligator SMA trading strategy with ATR-based position sizing

    Features:
    - Three SMA periods: short, medium, and long, calculated using read_ts.mavg
    - Entry/exit rules designed to be less strict (more frequent trades):
        * Go long when either:
            - Short SMA > Medium SMA
            OR
            - Medium SMA > Long SMA
        * Go short when either:
            - Short SMA < Medium SMA
            OR
            - Medium SMA < Long SMA
      This “OR” logic ensures more signals and more time spent in a position.
    - ATR-based position sizing for volatility-adjusted sizing
    - Slippage modeling for realistic execution
    - P&L calculated in absolute dollar terms using big point value
    - Fully vectorized implementation for efficiency
    """

    def __init__(self, short_sma, med_sma, long_sma, big_point_value, slippage=0, capital=6000, atr_period=50):
        """
        Initialize the Alligator SMA strategy with specific parameters

        Parameters:
        short_sma (int): Short SMA period in days
        med_sma (int): Medium SMA period in days
        long_sma (int): Long SMA period in days
        big_point_value (int): Contract big point value for calculating dollar P&L
        slippage (float): Slippage in price units added/subtracted from execution price
        capital (float): The capital allocation for position sizing
        atr_period (int): Period for ATR calculation for position sizing
        """
        self.short_sma = short_sma
        self.med_sma = med_sma
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

        high = data['High'].copy()
        prev_close = data['Close'].shift(1)
        true_high = pd.DataFrame({'high': high, 'prev_close': prev_close}).max(axis=1)

        low = data['Low'].copy()
        true_low = pd.DataFrame({'low': low, 'prev_close': prev_close}).min(axis=1)

        true_range = true_high - true_low
        atr = true_range.rolling(window=period).mean()

        return atr

    def apply_strategy(self, data, strategy_name="Alligator"):
        sim_data = data.copy()

        # 1) Calculate the three SMAs (same as before)
        close_array = sim_data['Close'].values
        sim_data[f'SMA_Short_{strategy_name}'] = mavg(close_array, self.short_sma)
        sim_data[f'SMA_Med_{strategy_name}']   = mavg(close_array, self.med_sma)
        sim_data[f'SMA_Long_{strategy_name}']  = mavg(close_array, self.long_sma)

        # 2) Calculate ATR and position size (unchanged)
        sim_data[f'ATR_{strategy_name}'] = self.calculate_atr(sim_data, self.atr_period)
        sim_data[f'Position_Size_{strategy_name}'] = np.round(
            self.capital / (sim_data[f'ATR_{strategy_name}'] * self.big_point_value) + 0.5
        )

        # 3) Strict SMA‐stack logic:
        #    Long  only when  Short > Med > Long
        #    Short only when  Short < Med < Long
        sma_short = sim_data[f'SMA_Short_{strategy_name}']
        sma_med   = sim_data[f'SMA_Med_{strategy_name}']
        sma_long  = sim_data[f'SMA_Long_{strategy_name}']

        cond_strict_long  = (sma_short > sma_med) & (sma_med > sma_long)
        cond_strict_short = (sma_short < sma_med) & (sma_med < sma_long)

        # 4) Build Position_Dir step-by-step:
        #    - On the very first row, default to 0 (flat)
        #    - On each subsequent row:
        #         if strict_long  →  +1
        #         elif strict_short →  −1
        #         else → carry forward yesterday’s position
        sim_data[f'Position_Dir_{strategy_name}'] = 0
        for i in range(1, len(sim_data)):
            if cond_strict_long.iloc[i]:
                sim_data[f'Position_Dir_{strategy_name}'].iat[i] = 1
            elif cond_strict_short.iloc[i]:
                sim_data[f'Position_Dir_{strategy_name}'].iat[i] = -1
            else:
                # carry forward previous bar’s position
                sim_data[f'Position_Dir_{strategy_name}'].iat[i] = sim_data[f'Position_Dir_{strategy_name}'].iat[i-1]

        # 5) Position changes (entry/exit) occur whenever Position_Dir flips
        sim_data[f'Position_Change_{strategy_name}'] = sim_data[f'Position_Dir_{strategy_name}'].diff().fillna(0) != 0

        # 6) P&L calculation (unchanged)
        market_pnl = sim_data['Close'].diff() * self.big_point_value
        sim_data[f'Market_PnL_{strategy_name}'] = market_pnl

        sim_data[f'Daily_PnL_{strategy_name}'] = (
            market_pnl
            * sim_data[f'Position_Dir_{strategy_name}'].shift(1)
            * sim_data[f'Position_Size_{strategy_name}'].shift(1)
        )

        # 7) Slippage on each trade
        position_changed = sim_data[f'Position_Change_{strategy_name}']
        sim_data.loc[position_changed, f'Daily_PnL_{strategy_name}'] -= (
            self.slippage * sim_data[f'Position_Size_{strategy_name}'][position_changed]
        )

        sim_data[f'Daily_PnL_{strategy_name}'] = sim_data[f'Daily_PnL_{strategy_name}'].fillna(0)
        sim_data[f'Cumulative_PnL_{strategy_name}'] = sim_data[f'Daily_PnL_{strategy_name}'].cumsum()

        return sim_data

    def optimize(self, data, sma_range, train_test_split=0.7, results_file=None, warm_up_idx=None):
        """
        Find the optimal SMA parameters (short, medium, long) and record all simulations using vectorized operations

        Parameters:
        data: DataFrame with market data
        sma_range: Range of SMA periods to test (list or iterable of ints)
        train_test_split: Portion of data to use for in-sample testing
        results_file: Path to save simulation results (CSV)
        warm_up_idx: Index to trim warm-up period (if provided)

        Returns:
        best_sma_params: Tuple with (short_sma, med_sma, long_sma)
        best_sharpe: Best Sharpe ratio found
        best_trades: Number of trades with best parameters
        all_results: List of tuples with all simulation results
        """
        best_sharpe = -np.inf
        best_sma = (0, 0, 0)
        best_trades = 0
        best_sim_data = None

        all_results = []

        # Prepare results file header if needed
        if results_file:
            with open(results_file, 'w') as f:
                f.write("short_SMA,med_SMA,long_SMA,trades,sharpe_ratio\n")

        # Count total valid combinations short < med < long
        total_combinations = sum(
            1 for s in sma_range for m in sma_range for l in sma_range if s < m < l
        )
        completed = 0
        total_sim_time = 0

        print(f"Running {total_combinations} simulations...")

        for short_sma in sma_range:
            for med_sma in sma_range:
                if med_sma <= short_sma:
                    continue
                for long_sma in sma_range:
                    if long_sma <= med_sma:
                        continue

                    sim_start_time = time.time()

                    # Backup original params
                    orig_short = self.short_sma
                    orig_med = self.med_sma
                    orig_long = self.long_sma

                    # Set new SMA parameters
                    self.short_sma = short_sma
                    self.med_sma = med_sma
                    self.long_sma = long_sma

                    # Apply strategy
                    sim_data = self.apply_strategy(data.copy(), strategy_name="Sim")

                    # Trim warm-up if provided
                    if warm_up_idx is not None:
                        sim_data_eval = sim_data.iloc[warm_up_idx:].copy()
                        split_index = int(len(sim_data_eval) * train_test_split)
                    else:
                        sim_data_eval = sim_data
                        split_index = int(len(sim_data_eval) * train_test_split)

                    # Count trades
                    trade_entries = sim_data_eval['Position_Change_Sim']
                    trade_count = trade_entries.sum()

                    # Calculate daily returns
                    sim_data_eval.loc[:, 'Daily_Returns'] = sim_data_eval['Daily_PnL_Sim']

                    # Calculate in-sample Sharpe ratio
                    in_sample_returns = sim_data_eval['Daily_Returns'].iloc[:split_index]
                    if len(in_sample_returns.dropna()) == 0 or in_sample_returns.std() == 0:
                        sharpe_ratio = 0
                    else:
                        sharpe_ratio = in_sample_returns.mean() / in_sample_returns.std() * np.sqrt(252)

                    # Save result to file if requested
                    if results_file:
                        with open(results_file, 'a') as f:
                            f.write(f"{short_sma},{med_sma},{long_sma},{trade_count},{sharpe_ratio:.6f}\n")

                    all_results.append((short_sma, med_sma, long_sma, trade_count, sharpe_ratio))

                    # Update best if improved
                    if sharpe_ratio > best_sharpe:
                        best_sharpe = sharpe_ratio
                        best_sma = (short_sma, med_sma, long_sma)
                        best_trades = trade_count
                        best_sim_data = sim_data_eval.copy()

                    # Restore original params
                    self.short_sma = orig_short
                    self.med_sma = orig_med
                    self.long_sma = orig_long

                    sim_end_time = time.time()
                    sim_time = sim_end_time - sim_start_time
                    total_sim_time += sim_time
                    completed += 1

                    if completed % 100 == 0 or completed == total_combinations:
                        avg_sim_time = total_sim_time / completed
                        est_remaining = avg_sim_time * (total_combinations - completed)
                        print(
                            f"Progress: {completed}/{total_combinations} "
                            f"({(completed / total_combinations * 100):.1f}%) - "
                            f"Avg sim time: {avg_sim_time:.4f}s - "
                            f"Est. remaining: {est_remaining:.1f}s"
                        )

        # Verification for best parameters
        if best_sim_data is not None:
            print("\n--- OPTIMIZATION VERIFICATION ---")
            verify_split_idx = int(len(best_sim_data) * train_test_split)
            verify_returns = best_sim_data['Daily_Returns'].iloc[:verify_split_idx]
            if verify_returns.std() > 0:
                verify_sharpe = verify_returns.mean() / verify_returns.std() * np.sqrt(252)
                print(f"Best Sharpe (optimization): {best_sharpe:.6f}")
                print(f"Best Sharpe (verification): {verify_sharpe:.6f}")
                print(f"In-sample points: {len(verify_returns)}, Total points: {len(best_sim_data)}")
            else:
                print("Cannot verify Sharpe (std = 0)")

        return best_sma, best_sharpe, best_trades, all_results

    def calculate_performance_metrics(self, data, strategy_name="Alligator", train_test_split=0.7):
        """
        Calculate detailed performance metrics for the strategy

        Parameters:
        data: DataFrame with strategy results
        strategy_name: Name suffix for the strategy columns
        train_test_split: Portion of data used for in-sample testing

        Returns:
        dict: Dictionary with performance metrics
        """
        split_index = int(len(data) * train_test_split)
        split_date = data.index[split_index]

        daily_pnl = data[f'Daily_PnL_{strategy_name}']
        returns_in_sample = daily_pnl.iloc[:split_index]
        returns_out_sample = daily_pnl.iloc[split_index:]

        sharpe_in_sample = (returns_in_sample.mean() / returns_in_sample.std() * np.sqrt(252)
                            if returns_in_sample.std() > 0 else 0)
        sharpe_out_sample = (returns_out_sample.mean() / returns_out_sample.std() * np.sqrt(252)
                             if returns_out_sample.std() > 0 else 0)
        sharpe_full = (daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
                       if daily_pnl.std() > 0 else 0)

        print("\n--- PERFORMANCE METRICS VERIFICATION ---")
        print(f"In-sample Sharpe = {sharpe_in_sample:.6f}")
        print(f"Mean (in-sample): {returns_in_sample.mean():.6f}, Std (in-sample): {returns_in_sample.std():.6f}")

        position_changes = data[f'Position_Change_{strategy_name}']
        total_trades = position_changes.sum()
        in_sample_trades = position_changes.iloc[:split_index].sum()
        out_sample_trades = position_changes.iloc[split_index:].sum()

        pnl_series = data[f'Cumulative_PnL_{strategy_name}']
        data.loc[:, 'Peak'] = pnl_series.cummax()
        data.loc[:, 'Drawdown_Dollars'] = pnl_series - data['Peak']
        max_drawdown_dollars = data['Drawdown_Dollars'].min()

        total_pnl = data[f'Cumulative_PnL_{strategy_name}'].iloc[-1]
        in_sample_pnl = data[f'Daily_PnL_{strategy_name}'].iloc[:split_index].sum()
        out_sample_pnl = data[f'Daily_PnL_{strategy_name}'].iloc[split_index:].sum()

        avg_position_size = data[f'Position_Size_{strategy_name}'].mean()
        max_position_size = data[f'Position_Size_{strategy_name}'].max()

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
