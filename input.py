# Configuration file for SMA optimization strategy

# Ticker symbol for the asset
TICKER = 'GC=F'

# Date range for analysis
START_DATE = '2014-01-01'
END_DATE = '2025-01-01'

# SMA range parameters
SMA_MIN = 10
SMA_MAX = 300
SMA_STEP = 5

# Data splitting ratio (percentage of data used for in-sample testing)
TRAIN_TEST_SPLIT = 0.7

# Results output file
RESULTS_FILE = 'sma_all_results.txt'

# Capital management
INITIAL_CAPITAL = 1000000  # Starting capital in USD
CONTRACT_SIZE = 100       # Number of contracts/shares to trade

# Clustering parameters
MIN_TRADES = 10          # Minimum number of trades to consider in clustering
MAX_TRADES = 2000         # Maximum number of trades to consider in clustering
MIN_ELEMENTS_PER_CLUSTER = 5  # Minimum elements required for a valid cluster
DEFAULT_NUM_CLUSTERS = 5     # Default number of clusters if not specified by user

# Trade execution parameters
STOP_LOSS = 0.95       # Fixed stop loss multiplier (5% loss from entry)
SLIPPAGE = 27      # Constant slippage in price units

# Dynamic trailing stop parameters
TRAILING_STOP_LEVELS = [
    {'profit_threshold': 5, 'trailing_stop': 2},   # 2% trailing stop when up 5%
    {'profit_threshold': 10, 'trailing_stop': 5},  # 5% trailing stop when up 10%
    {'profit_threshold': 20, 'trailing_stop': 8}  # 10% trailing stop when up 20%
]