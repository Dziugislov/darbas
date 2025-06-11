import pandas as pd
import numpy as np
from input import INITIAL_CAPITAL


class PortfolioSMAStrategy:
    """
    Portfolio implementation of the SMA (Simple Moving Average) trading strategy.
    Manages multiple instruments with individual SMA parameters, but combines PnL
    into a single portfolio performance.
    
    Features:
    - Multiple instruments with individual SMA parameters
    - Basic SMA crossover entry signals for each instrument
    - Pure long/short positions based on crossover direction
    - Slippage modeling for realistic execution
    - Log returns for Sharpe ratio calculation
    - Combined portfolio performance metrics
    """

    def __init__(self, instruments, contract_sizes, slippages):
        """
        Initialize the Portfolio SMA strategy with multiple instruments
        
        Parameters:
        instruments (dict): Dictionary of instruments with their SMA parameters
                           Format: {'symbol': {'short_sma': X, 'long_sma': Y}}
        contract_sizes (dict): Dictionary of contract sizes for each instrument
                              Format: {'symbol': contract_size}
        slippages (dict): Dictionary of slippage values for each instrument
                         Format: {'symbol': slippage}
        """
        self.instruments = instruments
        self.contract_sizes = contract_sizes
        self.slippages = slippages
        
    def apply_strategy(self, data_dict, strategy_name="Strategy", initial_capital=INITIAL_CAPITAL):
        """
        Apply the SMA crossover strategy to multiple instruments and calculate combined portfolio performance
        
        Parameters:
        data_dict (dict): Dictionary of DataFrames, one for each instrument
                         Format: {'symbol': dataframe}
        strategy_name (str): Name suffix for the strategy columns
        initial_capital (float): Initial capital for calculating returns
        
        Returns:
        tuple: (combined_data, instrument_data_dict)
               - combined_data: DataFrame with combined portfolio performance
               - instrument_data_dict: Dictionary of DataFrames with individual instrument performance
        """
        # Dictionary to store processed data for each instrument
        instrument_data_dict = {}
        
        # Apply strategy to each instrument separately
        for symbol, params in self.instruments.items():
            if symbol not in data_dict:
                print(f"Warning: No data for {symbol}, skipping...")
                continue
                
            # Get instrument-specific parameters
            short_sma = params['short_sma']
            long_sma = params['long_sma']
            contract_size = self.contract_sizes.get(symbol, 1)
            slippage = self.slippages.get(symbol, 0)
            
            # Create a copy of the DataFrame to avoid modifying the original
            sim_data = data_dict[symbol].copy()
            
            # Calculate SMAs
            sim_data[f'SMA_Short_{strategy_name}'] = sim_data['Close'].rolling(window=short_sma).mean()
            sim_data[f'SMA_Long_{strategy_name}'] = sim_data['Close'].rolling(window=long_sma).mean()
            
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
                    executed_price = curr_row['Close'] + slippage
                    sim_data.iloc[i, sim_data.columns.get_loc(f'Signal_{strategy_name}')] = 1
                    sim_data.iloc[i, sim_data.columns.get_loc(f'Position_{strategy_name}')] = 1
                    sim_data.iloc[i, sim_data.columns.get_loc(f'Entry_Price_{strategy_name}')] = executed_price
                    
                elif (prev_row[f'SMA_Short_{strategy_name}'] >= prev_row[f'SMA_Long_{strategy_name}'] and
                      curr_row[f'SMA_Short_{strategy_name}'] < curr_row[f'SMA_Long_{strategy_name}']):
                    # Bearish crossover - Go SHORT
                    executed_price = curr_row['Close'] - slippage
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
            sim_data[f'Daily_PnL_{strategy_name}'] = sim_data['Close'].diff() * sim_data[f'Signal_{strategy_name}'].shift(1) * contract_size
            sim_data[f'Daily_PnL_{strategy_name}'] = sim_data[f'Daily_PnL_{strategy_name}'].fillna(0)  # Replace NaN values in first row
            
            # Calculate cumulative P&L for this instrument
            sim_data[f'Cumulative_PnL_{strategy_name}'] = sim_data[f'Daily_PnL_{strategy_name}'].cumsum()
            
            # Add symbol-specific column for reference
            sim_data['Symbol'] = symbol
            
            # Store processed data
            instrument_data_dict[symbol] = sim_data
        
        # Create combined DataFrame for portfolio performance
        # Start by finding the common date range
        all_dates = set()
        for symbol, data in instrument_data_dict.items():
            all_dates.update(data.index)
        
        common_dates = sorted(all_dates)
        
        # Create a DataFrame with the combined dates
        combined_data = pd.DataFrame(index=common_dates)
        combined_data.sort_index(inplace=True)
        
        # Initialize PnL columns
        combined_data[f'Daily_PnL_{strategy_name}'] = 0
        
        # Add each instrument's PnL to the combined DataFrame
        for symbol, data in instrument_data_dict.items():
            # Align dates and add this instrument's PnL to the combined PnL
            aligned_data = data.reindex(combined_data.index)
            combined_data[f'Daily_PnL_{strategy_name}'].fillna(0, inplace=True)
            aligned_pnl = aligned_data[f'Daily_PnL_{strategy_name}'].fillna(0)
            combined_data[f'Daily_PnL_{strategy_name}'] += aligned_pnl
            
            # Add individual instrument columns for reference
            combined_data[f'{symbol}_Daily_PnL_{strategy_name}'] = aligned_pnl
        
        # Calculate portfolio-level metrics
        combined_data[f'Cumulative_PnL_{strategy_name}'] = combined_data[f'Daily_PnL_{strategy_name}'].cumsum()
        combined_data[f'Capital_{strategy_name}'] = initial_capital + combined_data[f'Cumulative_PnL_{strategy_name}']
        combined_data[f'Cumulative_Returns_{strategy_name}'] = combined_data[f'Capital_{strategy_name}'] / initial_capital
        
        # Logarithmic Returns
        epsilon = 1e-10
        combined_data[f'Returns_{strategy_name}'] = np.log(
            (combined_data[f'Capital_{strategy_name}'] + epsilon) /
            (combined_data[f'Capital_{strategy_name}'].shift(1) + epsilon)
        ).fillna(0)
        
        return combined_data, instrument_data_dict
    
    def optimize(self, data_dict, sma_range, train_test_split=0.7, initial_capital=INITIAL_CAPITAL, results_file=None):
        """
        Find the optimal SMA parameters for each instrument in the portfolio
        
        Parameters:
        data_dict (dict): Dictionary of DataFrames, one for each instrument
                         Format: {'symbol': dataframe}
        sma_range (range): Range of SMA periods to test
        train_test_split (float): Portion of data to use for in-sample testing
        initial_capital (float): Initial capital for calculating returns
        results_file (str): Path to save simulation results
        
        Returns:
        tuple: (best_params, best_sharpe, best_trades, all_results)
               - best_params: Dictionary of best parameters for each instrument
               - best_sharpe: Best Sharpe ratio found
               - best_trades: Dictionary of number of trades with best parameters for each instrument
               - all_results: List of simulation results
        """
        # Initialize results storage
        all_results = []
        best_sharpe = -np.inf
        best_params = {}
        best_trades = {}
        
        # Create the output file if specified
        if results_file:
            with open(results_file, 'w') as f:
                header = "symbol,short_SMA,long_SMA,trades,sharpe_ratio\n"
                f.write(header)
        
        # For each instrument, individually optimize SMA parameters
        for symbol, data in data_dict.items():
            print(f"\nOptimizing SMA parameters for {symbol}...")
            
            # Calculate split index for in-sample/out-of-sample
            split_index = int(len(data) * train_test_split)
            
            # Count total combinations to test
            total_combinations = sum(1 for a, b in [(s, l) for s in sma_range for l in sma_range] if a < b)
            completed = 0
            
            print(f"Running {total_combinations} simulations for {symbol}...")
            
            # Track best parameters for this instrument
            best_instrument_sharpe = -np.inf
            best_instrument_params = (0, 0)
            best_instrument_trades = 0
            
            # Iterate through all possible combinations of short and long SMA periods
            for short_sma in sma_range:
                for long_sma in sma_range:
                    # Skip invalid combinations where short SMA is not actually shorter than long SMA
                    if short_sma >= long_sma:
                        continue
                    
                    # Set up parameters for this simulation
                    instrument_params = {symbol: {'short_sma': short_sma, 'long_sma': long_sma}}
                    
                    # Run a single-instrument simulation
                    temp_strategy = PortfolioSMAStrategy(
                        instruments=instrument_params,
                        contract_sizes={symbol: self.contract_sizes.get(symbol, 1)},
                        slippages={symbol: self.slippages.get(symbol, 0)}
                    )
                    
                    # Apply strategy to get results
                    sim_data, _ = temp_strategy.apply_strategy(
                        {symbol: data.copy()},
                        strategy_name="Sim",
                        initial_capital=initial_capital
                    )
                    
                    # Count trades
                    instrument_data = sim_data.copy()
                    trade_entries = np.zeros(len(instrument_data))
                    
                    # Add column for trade entries/exits
                    trade_col = f'{symbol}_Daily_PnL_Sim'
                    if trade_col in instrument_data.columns:
                        # Use changes in PnL to detect trades
                        trade_entries = (instrument_data[trade_col] != 0) & (instrument_data[trade_col].shift(1) == 0)
                    
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
                            f.write(f"{symbol},{short_sma},{long_sma},{trade_count},{sharpe_ratio:.6f}\n")
                    
                    # Store the results
                    result = (symbol, short_sma, long_sma, trade_count, sharpe_ratio)
                    all_results.append(result)
                    
                    # Update best parameters for this instrument if current combination performs better
                    if sharpe_ratio > best_instrument_sharpe:
                        best_instrument_sharpe = sharpe_ratio
                        best_instrument_params = (short_sma, long_sma)
                        best_instrument_trades = trade_count
                        
                        # If this is also the best overall, update the portfolio best
                        if sharpe_ratio > best_sharpe:
                            best_sharpe = sharpe_ratio
                    
                    # Update progress
                    completed += 1
                    if completed % 10 == 0 or completed == total_combinations:
                        print(f"Progress: {completed}/{total_combinations} simulations completed ({(completed / total_combinations * 100):.1f}%)")
            
            # Store best parameters for this instrument
            best_params[symbol] = {'short_sma': best_instrument_params[0], 'long_sma': best_instrument_params[1]}
            best_trades[symbol] = best_instrument_trades
            
            print(f"Best parameters for {symbol}: Short SMA = {best_instrument_params[0]}, Long SMA = {best_instrument_params[1]}")
            print(f"Sharpe ratio: {best_instrument_sharpe:.4f}, Trades: {best_instrument_trades}")
        
        # Print overall best results
        print("\nOptimization complete!")
        print(f"Best portfolio Sharpe ratio: {best_sharpe:.4f}")
        print("Best parameters for each instrument:")
        for symbol, params in best_params.items():
            print(f"{symbol}: Short SMA = {params['short_sma']}, Long SMA = {params['long_sma']}, Trades = {best_trades[symbol]}")
        
        return best_params, best_sharpe, best_trades, all_results
    
    def calculate_performance_metrics(self, combined_data, instrument_data_dict, strategy_name="Strategy", 
                                    train_test_split=0.7, initial_capital=INITIAL_CAPITAL):
        """
        Calculate detailed performance metrics for the portfolio strategy
        
        Parameters:
        combined_data: DataFrame with combined portfolio performance
        instrument_data_dict: Dictionary of DataFrames with individual instrument performance
        strategy_name: Name suffix for the strategy columns
        train_test_split: Portion of data used for in-sample testing
        initial_capital: Initial capital for calculating returns
        
        Returns:
        dict: Dictionary with performance metrics
        """
        # Calculate split index for in-sample/out-of-sample
        split_index = int(len(combined_data) * train_test_split)
        split_date = combined_data.index[split_index]
        
        # Extract returns data
        daily_returns = combined_data[f'Returns_{strategy_name}']
        
        # Split returns for performance comparison
        returns_in_sample = daily_returns.iloc[:split_index]
        returns_out_sample = daily_returns.iloc[split_index:]
        
        # Calculate separate Sharpe ratios (annualized)
        sharpe_in_sample = returns_in_sample.mean() / returns_in_sample.std() * np.sqrt(
            252) if returns_in_sample.std() > 0 else 0
        sharpe_out_sample = returns_out_sample.mean() / returns_out_sample.std() * np.sqrt(
            252) if returns_out_sample.std() > 0 else 0
        sharpe_full = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
        
        # Calculate trade counts for each instrument
        total_trades = 0
        in_sample_trades = 0
        out_sample_trades = 0
        
        # Instrument-specific metrics
        instrument_metrics = {}
        
        for symbol, data in instrument_data_dict.items():
            # Create a mask for trade entries
            trade_entries = (data[f'Signal_{strategy_name}'] != data[f'Signal_{strategy_name}'].shift(1))
            if not pd.isna(data[f'Signal_{strategy_name}'].iloc[0]):
                trade_entries.iloc[0] = False
            
            # Count trades
            symbol_total_trades = trade_entries.sum()
            symbol_in_sample = trade_entries.iloc[:split_index].sum()
            symbol_out_sample = trade_entries.iloc[split_index:].sum()
            
            # Update overall counts
            total_trades += symbol_total_trades
            in_sample_trades += symbol_in_sample
            out_sample_trades += symbol_out_sample
            
            # Store instrument-specific metrics
            instrument_metrics[symbol] = {
                'total_trades': symbol_total_trades,
                'in_sample_trades': symbol_in_sample,
                'out_sample_trades': symbol_out_sample,
                'pnl': data[f'Cumulative_PnL_{strategy_name}'].iloc[-1] if not data.empty else 0
            }
        
        # Calculate max drawdown
        combined_data['Peak'] = combined_data[f'Capital_{strategy_name}'].cummax()
        combined_data['Drawdown'] = (combined_data[f'Capital_{strategy_name}'] - combined_data['Peak']) / combined_data['Peak'] * 100
        max_drawdown = combined_data['Drawdown'].min()
        
        # Calculate profit/loss metrics
        final_capital = combined_data[f'Capital_{strategy_name}'].iloc[-1]
        total_return_dollars = final_capital - initial_capital
        total_return_percent = (final_capital / initial_capital - 1) * 100
        
        # Calculate in-sample and out-of-sample returns
        in_sample_pnl = combined_data[f'Daily_PnL_{strategy_name}'].iloc[:split_index].sum()
        out_sample_pnl = combined_data[f'Daily_PnL_{strategy_name}'].iloc[split_index:].sum()
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
            'out_sample_return_pct': out_sample_return_pct,
            'instruments': instrument_metrics
        }
        
        return metrics