import pandas as pd
import numpy as np
from input import INITIAL_CAPITAL


class SMAStrategy:
    """
    Simple SMA (Simple Moving Average) trading strategy
    
    Features:
    - Basic SMA crossover entry signals (long when short SMA crosses above long SMA, short when it crosses below)
    - Pure long/short positions based on crossover direction
    - Slippage modeling for realistic execution
    - Log returns for Sharpe ratio calculation
    """

    def __init__(self, short_sma, long_sma, contract_size, slippage=0):
        """
        Initialize the SMA strategy with specific parameters

        Parameters:
        short_sma (int): Short SMA period in days
        long_sma (int): Long SMA period in days
        contract_size (int): Fixed number of contracts to trade
        slippage (float): Slippage in price units added/subtracted from execution price
        """
        self.short_sma = short_sma
        self.long_sma = long_sma
        self.contract_size = contract_size
        self.slippage = slippage

    def apply_strategy(self, data, strategy_name="Strategy", initial_capital=INITIAL_CAPITAL):
        """
        Apply the simple SMA crossover strategy to the price data
        """
        # Create a copy of the DataFrame to avoid modifying the original
        sim_data = data.copy()

        # Calculate SMAs
        sim_data[f'SMA_Short_{strategy_name}'] = sim_data['Close'].rolling(window=self.short_sma).mean()
        sim_data[f'SMA_Long_{strategy_name}'] = sim_data['Close'].rolling(window=self.long_sma).mean()

        # Initialize position column
        sim_data[f'Position_{strategy_name}'] = 0
        sim_data[f'Signal_{strategy_name}'] = 0
        sim_data[f'Entry_Price_{strategy_name}'] = np.nan

        # Loop through data to apply signals
        for i in range(1, len(sim_data)):
            curr_row = sim_data.iloc[i]
            prev_row = sim_data.iloc[i - 1]

            # Check for SMA crossover signals
            if (prev_row[f'SMA_Short_{strategy_name}'] <= prev_row[f'SMA_Long_{strategy_name}'] and
                    curr_row[f'SMA_Short_{strategy_name}'] > curr_row[f'SMA_Long_{strategy_name}']):
                # Bullish crossover - Go LONG
                executed_price = curr_row['Close'] + self.slippage
                sim_data.iloc[i, sim_data.columns.get_loc(f'Signal_{strategy_name}')] = 1
                sim_data.iloc[i, sim_data.columns.get_loc(f'Position_{strategy_name}')] = 1
                sim_data.iloc[i, sim_data.columns.get_loc(f'Entry_Price_{strategy_name}')] = executed_price

            elif (prev_row[f'SMA_Short_{strategy_name}'] >= prev_row[f'SMA_Long_{strategy_name}'] and
                  curr_row[f'SMA_Short_{strategy_name}'] < curr_row[f'SMA_Long_{strategy_name}']):
                # Bearish crossover - Go SHORT
                executed_price = curr_row['Close'] - self.slippage
                sim_data.iloc[i, sim_data.columns.get_loc(f'Signal_{strategy_name}')] = -1
                sim_data.iloc[i, sim_data.columns.get_loc(f'Position_{strategy_name}')] = -1
                sim_data.iloc[i, sim_data.columns.get_loc(f'Entry_Price_{strategy_name}')] = executed_price

            else:
                # No crossover - maintain previous position
                sim_data.iloc[i, sim_data.columns.get_loc(f'Signal_{strategy_name}')] = sim_data.iloc[i-1][f'Signal_{strategy_name}']
                sim_data.iloc[i, sim_data.columns.get_loc(f'Position_{strategy_name}')] = sim_data.iloc[i-1][f'Position_{strategy_name}']
                if sim_data.iloc[i-1][f'Entry_Price_{strategy_name}'] > 0:
                    sim_data.iloc[i, sim_data.columns.get_loc(f'Entry_Price_{strategy_name}')] = sim_data.iloc[i-1][f'Entry_Price_{strategy_name}']

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

    def optimize(self, data, sma_range, train_test_split=0.7, initial_capital=INITIAL_CAPITAL, results_file=None):
        """
        Find the optimal SMA parameters and record all simulations

        Parameters:
        data: DataFrame with market data
        sma_range: Range of SMA periods to test
        train_test_split: Portion of data to use for in-sample testing
        initial_capital: Initial capital for calculating returns
        results_file: Path to save simulation results

        Returns:
        best_sma_params: Tuple with (short_sma, long_sma)
        best_sharpe: Best Sharpe ratio found
        best_trades: Number of trades with best parameters
        all_results: List of tuples with all simulation results
        """
        # Initialize variables to track the best performance
        best_sharpe = -np.inf  # Start with negative infinity to ensure any valid Sharpe ratio will be better
        best_sma = (0, 0)  # Tuple to store the best (short_sma, long_sma) combination
        best_trades = 0  # Number of trades with the best parameters

        # Create a list to store all simulation results
        all_results = []

        # Create the output file for all simulations and write the header
        if results_file:
            with open(results_file, 'w') as f:
                f.write("short_SMA,long_SMA,trades,sharpe_ratio\n")

        # Calculate split index for in-sample/out-of-sample
        split_index = int(len(data) * train_test_split)

        # Count total combinations to test
        total_combinations = sum(1 for a, b in [(s, l) for s in sma_range for l in sma_range] if a < b)
        completed = 0

        print(f"Running {total_combinations} simulations...")

        # Iterate through all possible combinations of short and long SMA periods
        for short_sma in sma_range:
            for long_sma in sma_range:
                # Skip invalid combinations where short SMA is not actually shorter than long SMA
                if short_sma >= long_sma:
                    continue

                # Save original parameters
                orig_short_sma = self.short_sma
                orig_long_sma = self.long_sma

                # Set new parameters for this simulation
                self.short_sma = short_sma
                self.long_sma = long_sma

                # Apply strategy to the data
                sim_data = self.apply_strategy(data.copy(), strategy_name="Sim", initial_capital=initial_capital)

                # Count trades
                trade_entries = (sim_data['Signal_Sim'] != sim_data['Signal_Sim'].shift(1))
                if not pd.isna(sim_data['Signal_Sim'].iloc[0]):
                    trade_entries.iloc[0] = False
                trade_count = trade_entries.sum()

                # Calculate Sharpe ratio using only in-sample data
                in_sample_returns = sim_data['Returns_Sim'].iloc[:split_index]

                # Skip if there are no returns or all returns are 0
                if len(in_sample_returns.dropna()) == 0 or in_sample_returns.std() == 0:
                    sharpe_ratio = 0
                else:
                    sharpe_ratio = in_sample_returns.mean() / in_sample_returns.std() * np.sqrt(252)  # Annualized

                # Append the results to our file
                if results_file:
                    with open(results_file, 'a') as f:
                        f.write(f"{short_sma},{long_sma},{trade_count},{sharpe_ratio:.6f}\n")

                # Store the results
                result = (short_sma, long_sma, trade_count, sharpe_ratio)
                all_results.append(result)

                # Update best parameters if current combination performs better
                if sharpe_ratio > best_sharpe:
                    best_sharpe = sharpe_ratio
                    best_sma = (short_sma, long_sma)
                    best_trades = trade_count

                # Restore original parameters
                self.short_sma = orig_short_sma
                self.long_sma = orig_long_sma

                # Update progress
                completed += 1
                if completed % 10 == 0 or completed == total_combinations:
                    print(
                        f"Progress: {completed}/{total_combinations} simulations completed ({(completed / total_combinations * 100):.1f}%)")

        # Return the optimal SMA parameters, corresponding Sharpe ratio, and all results
        return best_sma, best_sharpe, best_trades, all_results

    def calculate_performance_metrics(self, data, strategy_name="Strategy", train_test_split=0.7,
                                      initial_capital=INITIAL_CAPITAL):
        """
        Calculate detailed performance metrics for the strategy

        Parameters:
        data: DataFrame with strategy results
        strategy_name: Name suffix for the strategy columns
        train_test_split: Portion of data used for in-sample testing
        initial_capital: Initial capital for calculating returns

        Returns:
        dict: Dictionary with performance metrics
        """
        # Calculate split index for in-sample/out-of-sample
        split_index = int(len(data) * train_test_split)
        split_date = data.index[split_index]

        # Extract returns data
        daily_returns = data[f'Returns_{strategy_name}']

        # Split returns for performance comparison
        returns_in_sample = daily_returns.iloc[:split_index]
        returns_out_sample = daily_returns.iloc[split_index:]

        # Calculate separate Sharpe ratios (annualized)
        sharpe_in_sample = returns_in_sample.mean() / returns_in_sample.std() * np.sqrt(
            252) if returns_in_sample.std() > 0 else 0
        sharpe_out_sample = returns_out_sample.mean() / returns_out_sample.std() * np.sqrt(
            252) if returns_out_sample.std() > 0 else 0
        sharpe_full = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0

        # Calculate trade counts
        trade_entries = (data[f'Signal_{strategy_name}'] != data[f'Signal_{strategy_name}'].shift(1))
        if not pd.isna(data[f'Signal_{strategy_name}'].iloc[0]):
            trade_entries.iloc[0] = False

        total_trades = trade_entries.sum()
        in_sample_trades = trade_entries.iloc[:split_index].sum()
        out_sample_trades = trade_entries.iloc[split_index:].sum()

        # Calculate max drawdown
        data['Peak'] = data[f'Capital_{strategy_name}'].cummax()
        data['Drawdown'] = (data[f'Capital_{strategy_name}'] - data['Peak']) / data['Peak'] * 100
        max_drawdown = data['Drawdown'].min()

        # Calculate profit/loss metrics
        final_capital = data[f'Capital_{strategy_name}'].iloc[-1]
        total_return_dollars = final_capital - initial_capital
        total_return_percent = (final_capital / initial_capital - 1) * 100

        # Calculate in-sample and out-of-sample returns
        in_sample_pnl = data[f'Daily_PnL_{strategy_name}'].iloc[:split_index].sum()
        out_sample_pnl = data[f'Daily_PnL_{strategy_name}'].iloc[split_index:].sum()
        in_sample_return_pct = (in_sample_pnl / initial_capital) * 100
        out_sample_return_pct = (out_sample_pnl / initial_capital) * 100

        # Assemble the results in a dictionary
        metrics = {
            'split_date': split_date,
            'final_capital': final_capital,
            'total_return_dollars': total_return_dollars,
            'total_return_percent': total_return_percent,
            'sharpe_full': sharpe_full,
            'sharpe_in_sample': sharpe_in_sample,
            'sharpe_out_sample': sharpe_out_sample,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'in_sample_trades': in_sample_trades,
            'out_sample_trades': out_sample_trades,
            'in_sample_pnl': in_sample_pnl,
            'out_sample_pnl': out_sample_pnl,
            'in_sample_return_pct': in_sample_return_pct,
            'out_sample_return_pct': out_sample_return_pct
        }

        return metrics