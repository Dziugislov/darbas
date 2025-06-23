from data_analysis_gen import bimonthly_out_of_sample_comparison
import pandas as pd
import numpy as np

def test_smoke_run():
    dates = pd.date_range("2024-01-01", periods=20, freq="B")
    data = pd.DataFrame(index=dates)
    top_clusters = [(5, 20, 1.5, [])]

    df = bimonthly_out_of_sample_comparison(
        data=data,
        best_short_sma=5,
        best_long_sma=20,
        top_clusters=top_clusters,
        big_point_value=1.0,
        slippage=0.0,
        capital=10000,
        atr_period=14,
        min_sharpe=1.0,
        ANALYSIS_METHOD='Test'
    )

    assert hasattr(df, 'Best_sharpe')
    assert hasattr(df, 'Avg_cluster_sharpe')
    assert len(df) >= 1
