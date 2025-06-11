# Configuration file for Alligator optimization strategy
import os

# Specify the correct directory for output files
OUTPUT_DIR = r"D:\dziug\Documents\darbas\new_strat"

# Ticker symbol for the asset
TICKER = 'INTC'

# Date range for analysis
START_DATE = '2017-01-01'  # Original start date, kept for compatibility
END_DATE = '2025-01-01'
SIMULATION_START_DATE = '2018-01-01'  # Actual start date for trading simulation

# Alligator parameters ranges for optimization
JAW_MIN = 35      # Minimum period for Jaw (longest SMA)
JAW_MAX = 150      # Maximum period for Jaw (longest SMA)
JAW_STEP = 5    # Step size for Jaw periods

# Lips ratio ranges for optimization (as percentage of Jaw period)
LIPS_RATIO_MIN = 0.1    # Minimum ratio of Lips to Jaw
LIPS_RATIO_MAX = 0.4    # Maximum ratio of Lips to Jaw
LIPS_RATIO_STEP = 0.05  # Step size for Lips ratio

# Teeth range for optimization (explicit periods)
TEETH_MIN = 8     # Minimum period for Teeth
TEETH_MAX = 34    # Maximum period for Teeth
TEETH_STEP = 2    # Step size for Teeth periods

# Initial Teeth period for Jaw-Lips optimization
INITIAL_TEETH = 13

# Data splitting ratio (percentage of data used for in-sample testing)
TRAIN_TEST_SPLIT = 0.7

# Results output files - updated with absolute paths
JAW_LIPS_RESULTS_FILE = os.path.join(OUTPUT_DIR, 'alligator_jaw_lips_results.txt')
TEETH_RESULTS_FILE = os.path.join(OUTPUT_DIR, 'alligator_teeth_results.txt')

# Capital management
INITIAL_CAPITAL = 1000000  # Starting capital in USD
CONTRACT_SIZE = 100       # Number of contracts/shares to trade

# Trade execution parameters
SLIPPAGE = 10      # Constant slippage in price units

# Optional: Pre-determined optimal parameters
OPTIMAL_JAW = 34
OPTIMAL_TEETH = 21
OPTIMAL_LIPS = 8