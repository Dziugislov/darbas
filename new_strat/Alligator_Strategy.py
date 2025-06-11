import pandas as pd
import numpy as np
from input import INITIAL_CAPITAL


class AlligatorStrategy:
    """
    Alligator trading strategy using three SMAs with flexible entry and exit rules
    
    Features:
    - Jaw (Blue): Longest SMA (slow)
    - Teeth (Red): Medium SMA (medium)
    - Lips (Green): Shortest SMA (fast)
    
    Entry Rules (More Flexible):
    - Long: Lips crosses above Teeth (no need to cross Jaw), Lips and Teeth expanding upward, price above Lips and Teeth
    - Short: Lips crosses below Teeth (no need to cross Jaw), Lips and Teeth expanding downward, price below Lips and Teeth
    
    Exit Rules (Less Strict):
    - When Lips crosses back against Teeth (trend weakening)
    - When both Lips and Teeth are contracting AND price closes against them
    """

    def __init__(self, jaw_period, teeth_period, lips_period, contract_size, slippage=0):
        """
        Initialize the Alligator strategy with specific parameters

        Parameters:
        jaw_period (int): Jaw (longest/blue) SMA period in days
        teeth_period (int): Teeth (medium/red) SMA period in days
        lips_period (int): Lips (shortest/green) SMA period in days
        contract_size (int): Fixed number of contracts to trade
        slippage (float): Slippage in price units added/subtracted from execution price
        """
        self.jaw_period = jaw_period
        self.teeth_period = teeth_period
        self.lips_period = lips_period
        self.contract_size = contract_size
        self.slippage = slippage

    def apply_strategy(self, data, strategy_name="Strategy", initial_capital=INITIAL_CAPITAL):
        """
        Apply the Alligator strategy with entry and exit rules
        """
        # Create a copy of the DataFrame to avoid modifying the original
        sim_data = data.copy()

        # Calculate SMAs for the Alligator indicator
        sim_data[f'Jaw_{strategy_name}'] = sim_data['Close'].rolling(window=self.jaw_period).mean()
        sim_data[f'Teeth_{strategy_name}'] = sim_data['Close'].rolling(window=self.teeth_period).mean()
        sim_data[f'Lips_{strategy_name}'] = sim_data['Close'].rolling(window=self.lips_period).mean()

        # Calculate SMA directions (expanding/contracting)
        sim_data[f'Jaw_Dir_{strategy_name}'] = sim_data[f'Jaw_{strategy_name}'].diff().rolling(window=3).mean()
        sim_data[f'Teeth_Dir_{strategy_name}'] = sim_data[f'Teeth_{strategy_name}'].diff().rolling(window=3).mean()
        sim_data[f'Lips_Dir_{strategy_name}'] = sim_data[f'Lips_{strategy_name}'].diff().rolling(window=3).mean()

        # Initialize columns
        sim_data[f'Signal_{strategy_name}'] = 0
        sim_data[f'Position_{strategy_name}'] = 0
        sim_data[f'Entry_Price_{strategy_name}'] = np.nan

        # Loop through data to apply signals and manage positions
        for i in range(3, len(sim_data)):
            # Skip if we don't have all SMAs yet
            if (pd.isna(sim_data.iloc[i][f'Jaw_{strategy_name}']) or 
                pd.isna(sim_data.iloc[i][f'Teeth_{strategy_name}']) or 
                pd.isna(sim_data.iloc[i][f'Lips_{strategy_name}'])):
                continue
                
            # Get previous position
            prev_position = sim_data.iloc[i-1][f'Position_{strategy_name}']
            
            # Get current data
            curr_price = sim_data.iloc[i]['Close']
            curr_jaw = sim_data.iloc[i][f'Jaw_{strategy_name}']
            curr_teeth = sim_data.iloc[i][f'Teeth_{strategy_name}']
            curr_lips = sim_data.iloc[i][f'Lips_{strategy_name}']
            
            # Get previous data for crossover detection
            prev_lips = sim_data.iloc[i-1][f'Lips_{strategy_name}']
            prev_teeth = sim_data.iloc[i-1][f'Teeth_{strategy_name}']
            prev_jaw = sim_data.iloc[i-1][f'Jaw_{strategy_name}']
            
            # SMA directions
            jaw_dir = sim_data.iloc[i][f'Jaw_Dir_{strategy_name}']
            teeth_dir = sim_data.iloc[i][f'Teeth_Dir_{strategy_name}']
            lips_dir = sim_data.iloc[i][f'Lips_Dir_{strategy_name}']
            
            # Carry previous position forward by default
            sim_data.iloc[i, sim_data.columns.get_loc(f'Position_{strategy_name}')] = prev_position
            
            if prev_position != 0:
                sim_data.iloc[i, sim_data.columns.get_loc(f'Entry_Price_{strategy_name}')] = sim_data.iloc[i-1][f'Entry_Price_{strategy_name}']
                sim_data.iloc[i, sim_data.columns.get_loc(f'Signal_{strategy_name}')] = prev_position
            
            # Check for long exit conditions - LESS STRICT
            if prev_position == 1:
                # Exit long if:
                # 1. Lips crosses back below Teeth (more lenient, only care about Teeth crossing)
                lips_crosses_inside = (prev_lips > prev_teeth and curr_lips <= curr_teeth)
                
                # 2. Only exit if both Lips and Teeth are contracting (more lenient)
                lines_contracting = (lips_dir < 0 and teeth_dir < 0)
                
                # 3. Only exit if price closes below both Lips and Teeth (more lenient)
                price_closes_against_trend = curr_price < curr_teeth and curr_price < curr_lips
                
                if lips_crosses_inside or (lines_contracting and price_closes_against_trend):
                    sim_data.iloc[i, sim_data.columns.get_loc(f'Position_{strategy_name}')] = 0
                    sim_data.iloc[i, sim_data.columns.get_loc(f'Signal_{strategy_name}')] = 0
                    sim_data.iloc[i, sim_data.columns.get_loc(f'Entry_Price_{strategy_name}')] = np.nan
            
            # Check for short exit conditions - LESS STRICT
            elif prev_position == -1:
                # Exit short if:
                # 1. Lips crosses back above Teeth (more lenient, only care about Teeth crossing)
                lips_crosses_inside = (prev_lips < prev_teeth and curr_lips >= curr_teeth)
                
                # 2. Only exit if both Lips and Teeth are contracting (more lenient)
                lines_contracting = (lips_dir > 0 and teeth_dir > 0)
                
                # 3. Only exit if price closes above both Lips and Teeth (more lenient)
                price_closes_against_trend = curr_price > curr_teeth and curr_price > curr_lips
                
                if lips_crosses_inside or (lines_contracting and price_closes_against_trend):
                    sim_data.iloc[i, sim_data.columns.get_loc(f'Position_{strategy_name}')] = 0
                    sim_data.iloc[i, sim_data.columns.get_loc(f'Signal_{strategy_name}')] = 0
                    sim_data.iloc[i, sim_data.columns.get_loc(f'Entry_Price_{strategy_name}')] = np.nan
            
            # Entry conditions - only check if we're not already in a position
            if sim_data.iloc[i][f'Position_{strategy_name}'] == 0:
                # Long entry conditions - MORE FLEXIBLE:
                # 1. Lips crosses above Teeth only (no need to cross above Jaw)
                lips_crosses_above_teeth = (prev_lips <= prev_teeth and curr_lips > curr_teeth)
                
                # 2. At least Lips and Teeth are expanding upward (Jaw can lag)
                minimal_expansion_up = lips_dir > 0 and teeth_dir > 0
                
                # 3. Price stays above at least Lips and Teeth (more flexible condition)
                price_above_key_lines = curr_price > curr_teeth and curr_price > curr_lips
                
                if lips_crosses_above_teeth and minimal_expansion_up and price_above_key_lines:
                    entry_price = curr_price + self.slippage  # Add slippage to buy price
                    
                    sim_data.iloc[i, sim_data.columns.get_loc(f'Position_{strategy_name}')] = 1
                    sim_data.iloc[i, sim_data.columns.get_loc(f'Signal_{strategy_name}')] = 1
                    sim_data.iloc[i, sim_data.columns.get_loc(f'Entry_Price_{strategy_name}')] = entry_price
                
                # Short entry conditions - MORE FLEXIBLE:
                # 1. Lips crosses below Teeth only (no need to cross below Jaw)
                lips_crosses_below_teeth = (prev_lips >= prev_teeth and curr_lips < curr_teeth)
                
                # 2. At least Lips and Teeth are expanding downward (Jaw can lag)
                minimal_expansion_down = lips_dir < 0 and teeth_dir < 0
                
                # 3. Price stays below at least Lips and Teeth (more flexible condition)
                price_below_key_lines = curr_price < curr_teeth and curr_price < curr_lips
                
                if lips_crosses_below_teeth and minimal_expansion_down and price_below_key_lines:
                    entry_price = curr_price - self.slippage  # Subtract slippage from sell price
                    
                    sim_data.iloc[i, sim_data.columns.get_loc(f'Position_{strategy_name}')] = -1
                    sim_data.iloc[i, sim_data.columns.get_loc(f'Signal_{strategy_name}')] = -1
                    sim_data.iloc[i, sim_data.columns.get_loc(f'Entry_Price_{strategy_name}')] = entry_price

        # Calculate P&L based on fixed contract size
        sim_data[f'Daily_PnL_{strategy_name}'] = sim_data['Close'].diff() * sim_data[f'Position_{strategy_name}'].shift(1) * self.contract_size
        sim_data[f'Daily_PnL_{strategy_name}'].fillna(0, inplace=True)

        # Calculate cumulative P&L
        sim_data[f'Cumulative_PnL_{strategy_name}'] = sim_data[f'Daily_PnL_{strategy_name}'].cumsum()

        # Calculate capital over time
        sim_data[f'Capital_{strategy_name}'] = initial_capital + sim_data[f'Cumulative_PnL_{strategy_name}']

        # Calculate returns as multiple of initial capital
        sim_data[f'Cumulative_Returns_{strategy_name}'] = sim_data[f'Capital_{strategy_name}'] / initial_capital

        # Calculate logarithmic returns
        epsilon = 1e-10
        sim_data[f'Returns_{strategy_name}'] = np.log(
            (sim_data[f'Capital_{strategy_name}'] + epsilon) /
            (sim_data[f'Capital_{strategy_name}'].shift(1) + epsilon)
        ).fillna(0)

        return sim_data

    def optimize_jaw_lips(self, data, jaw_range, lips_ratio_range, teeth_period, 
                          train_test_split=0.7, initial_capital=INITIAL_CAPITAL, results_file=None):
        """
        Optimize the Jaw and Lips periods (keeping Teeth fixed)
        """
        # Initialize variables to track the best performance
        best_sharpe = -np.inf
        best_params = (0, teeth_period, 0)  # (jaw, teeth, lips)
        best_trades = 0

        # Create a list to store all simulation results
        all_results = []

        # Create the output file for all simulations and write the header
        if results_file:
            with open(results_file, 'w') as f:
                f.write("jaw_period,teeth_period,lips_period,trades,sharpe_ratio\n")

        # Calculate split index for in-sample/out-of-sample
        split_index = int(len(data) * train_test_split)

        # Count total combinations to test
        total_combinations = len(jaw_range) * len(lips_ratio_range)
        completed = 0

        print(f"Running {total_combinations} simulations for Jaw-Lips optimization...")

        # Iterate through all possible combinations of Jaw and Lips periods
        for jaw_period in jaw_range:
            for lips_ratio in lips_ratio_range:
                # Calculate lips period based on ratio to jaw
                lips_period = max(2, int(jaw_period * lips_ratio))
                
                # Skip invalid combinations where lips_period >= teeth_period
                if lips_period >= teeth_period:
                    continue

                # Save original parameters
                orig_jaw = self.jaw_period
                orig_teeth = self.teeth_period
                orig_lips = self.lips_period

                # Set new parameters for this simulation
                self.jaw_period = jaw_period
                self.teeth_period = teeth_period
                self.lips_period = lips_period

                # Apply strategy to the data
                sim_data = self.apply_strategy(data.copy(), strategy_name="Sim", initial_capital=initial_capital)

                # Count trades (position changes)
                position_changes = (sim_data['Position_Sim'] != sim_data['Position_Sim'].shift(1)) & (sim_data['Position_Sim'] != 0)
                if not pd.isna(sim_data['Position_Sim'].iloc[0]):
                    position_changes.iloc[0] = False  # Ignore first row
                trade_count = position_changes.sum()

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
                        f.write(f"{jaw_period},{teeth_period},{lips_period},{trade_count},{sharpe_ratio:.6f}\n")

                # Store the results
                result = (jaw_period, teeth_period, lips_period, trade_count, sharpe_ratio)
                all_results.append(result)

                # Update best parameters if current combination performs better
                if sharpe_ratio > best_sharpe and trade_count > 0:
                    best_sharpe = sharpe_ratio
                    best_params = (jaw_period, teeth_period, lips_period)
                    best_trades = trade_count

                # Restore original parameters
                self.jaw_period = orig_jaw
                self.teeth_period = orig_teeth
                self.lips_period = orig_lips

                # Update progress
                completed += 1
                if completed % 10 == 0 or completed == total_combinations:
                    print(f"Progress: {completed}/{total_combinations} simulations completed ({(completed / total_combinations * 100):.1f}%)")

        # Return the optimal parameters, corresponding Sharpe ratio, and all results
        return best_params, best_sharpe, best_trades, all_results

    def optimize_teeth(self, data, jaw_period, teeth_range, lips_period,
                       train_test_split=0.7, initial_capital=INITIAL_CAPITAL, results_file=None):
        """
        Optimize the Teeth period (keeping Jaw and Lips fixed)
        """
        # Initialize variables to track the best performance
        best_sharpe = -np.inf
        best_params = (jaw_period, 0, lips_period)  # (jaw, teeth, lips)
        best_trades = 0

        # Create a list to store all simulation results
        all_results = []

        # Create the output file for all simulations and write the header
        if results_file:
            with open(results_file, 'w') as f:
                f.write("jaw_period,teeth_period,lips_period,trades,sharpe_ratio\n")

        # Calculate split index for in-sample/out-of-sample
        split_index = int(len(data) * train_test_split)

        # Count total combinations to test
        total_combinations = len(teeth_range)
        completed = 0

        print(f"Running {total_combinations} simulations for Teeth optimization...")

        # Iterate through all possible Teeth periods
        for teeth_period in teeth_range:
            # Skip invalid combinations
            if teeth_period >= jaw_period or teeth_period <= lips_period:
                continue

            # Save original parameters
            orig_jaw = self.jaw_period
            orig_teeth = self.teeth_period
            orig_lips = self.lips_period

            # Set new parameters for this simulation
            self.jaw_period = jaw_period
            self.teeth_period = teeth_period
            self.lips_period = lips_period

            # Apply strategy to the data
            sim_data = self.apply_strategy(data.copy(), strategy_name="Sim", initial_capital=initial_capital)

            # Count trades (position changes)
            position_changes = (sim_data['Position_Sim'] != sim_data['Position_Sim'].shift(1)) & (sim_data['Position_Sim'] != 0)
            if not pd.isna(sim_data['Position_Sim'].iloc[0]):
                position_changes.iloc[0] = False  # Ignore first row
            trade_count = position_changes.sum()

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
                    f.write(f"{jaw_period},{teeth_period},{lips_period},{trade_count},{sharpe_ratio:.6f}\n")

            # Store the results
            result = (jaw_period, teeth_period, lips_period, trade_count, sharpe_ratio)
            all_results.append(result)

            # Update best parameters if current combination performs better
            if sharpe_ratio > best_sharpe and trade_count > 0:
                best_sharpe = sharpe_ratio
                best_params = (jaw_period, teeth_period, lips_period)
                best_trades = trade_count

            # Restore original parameters
            self.jaw_period = orig_jaw
            self.teeth_period = orig_teeth
            self.lips_period = orig_lips

            # Update progress
            completed += 1
            if completed % 5 == 0 or completed == total_combinations:
                print(f"Progress: {completed}/{total_combinations} simulations completed ({(completed / total_combinations * 100):.1f}%)")

        # Return the optimal parameters, corresponding Sharpe ratio, and all results
        return best_params, best_sharpe, best_trades, all_results

    def calculate_performance_metrics(self, data, strategy_name="Strategy", train_test_split=0.7, initial_capital=INITIAL_CAPITAL):
        """
        Calculate detailed performance metrics for the strategy
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
        sharpe_in_sample = returns_in_sample.mean() / returns_in_sample.std() * np.sqrt(252) if returns_in_sample.std() > 0 else 0
        sharpe_out_sample = returns_out_sample.mean() / returns_out_sample.std() * np.sqrt(252) if returns_out_sample.std() > 0 else 0
        sharpe_full = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0

        # Calculate trade counts (position changes)
        position_changes = (data[f'Position_{strategy_name}'] != data[f'Position_{strategy_name}'].shift(1)) & (data[f'Position_{strategy_name}'] != 0)
        if not pd.isna(data[f'Position_{strategy_name}'].iloc[0]):
            position_changes.iloc[0] = False  # Ignore first row
        total_trades = position_changes.sum()
        in_sample_trades = position_changes.iloc[:split_index].sum()
        out_sample_trades = position_changes.iloc[split_index:].sum()

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