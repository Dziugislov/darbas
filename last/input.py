# Configuration file for SMA optimization strategy
# Ticker symbol for the asset
TICKER = 'ES=F'

# Date range for analysis
START_DATE = '2014-01-01'
END_DATE = '2025-01-01'

# SMA range parameters
SMA_MIN = 10
SMA_MAX = 300
SMA_STEP = 20

# Data splitting ratio (percentage of data used for in-sample testing)
TRAIN_TEST_SPLIT = 0.7

# Results output file
RESULTS_FILE = 'sma_all_results.txt'

# Capital management
INITIAL_CAPITAL = 1000000  # Starting capital in USD

# Clustering parameters
MIN_TRADES = 10         # Minimum number of trades to consider in clustering
MAX_TRADES = 2000         # Maximum number of trades to consider in clustering
MIN_ELEMENTS_PER_CLUSTER = 10  # Minimum elements required for a valid cluster
DEFAULT_NUM_CLUSTERS = 10     # Default number of clusters if not specified by user
