# Configuration file for SMA optimization strategy using genetic algorithm
# Ticker symbol for the asset
TICKER = 'WBS'
SHORT_SMA = 34
LONG_SMA = 40

# Date range for analysis
START_DATE = '2014-01-01'
END_DATE = '2025-01-01'

# SMA range parameters (for genetic algorithm bounds)
SMA_MIN = 10
SMA_MAX = 300
SMA_STEP = 2  # Kept for compatibility with original code, not used in GA

# Data splitting ratio (percentage of data used for in-sample testing)
TRAIN_TEST_SPLIT = 0.7

# Results output file
RESULTS_FILE = 'optimized_strategies.pkl'

# ATR-based position sizing parameters
ATR_PERIOD = 30          # Period for ATR calculation (days)
TRADING_CAPITAL = 6000   # Capital allocation for position sizing

# Genetic Algorithm parameters
POPULATION_SIZE = 1000    # Number of individuals in population
NUM_GENERATIONS = 10    # Number of generations to evolve
HALL_OF_FAME_SIZE = 100000   #didelis skaicius kad visus paimtu
CROSSOVER_PROB = 0.8     # Probability of crossover
MUTATION_PROB = 0.3      # Probability of mutation
RANDOM_SEED = 22         # Random seed for reproducibility

# Clustering parameters (kept for compatibility)
MIN_TRADES = 10              # Minimum number of trades to consider in clustering
MAX_TRADES = 2000            # Maximum number of trades to consider in clustering
MIN_ELEMENTS_PER_CLUSTER = 20  # Minimum elements required for a valid cluster
DEFAULT_NUM_CLUSTERS = 50     # Default number of clusters if not specified by user
ELBOW_THRESHOLD = 5  # Threshold for elbow method to determine number of clusters
MIN_SHARPE = 0.2  # Minimum Sharpe ratio to consider in clustering