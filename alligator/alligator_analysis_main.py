"""
Classic Alligator Strategy Analysis Main Script

This script performs a comprehensive analysis of the Classic Alligator trading strategy:
1. Checks and visualizes optimization results
2. Performs hierarchical clustering to find robust parameter regions
3. Plots performance comparison of the best parameters and top parameter clusters
4. Performs bimonthly out-of-sample analysis to evaluate strategy robustness
"""

import os
import pandas as pd
import numpy as np
from input import OUTPUT_DIR, JAW_LIPS_RESULTS_FILE, INITIAL_TEETH

# Import the analysis functions
from alligator_hierarchical_analysis import (
    analyze_alligator_results,
    hierarchical_cluster_analysis,
    plot_alligator_performance,
    bimonthly_out_of_sample_comparison
)


def check_results_file():
    """Check if the optimization results file exists and has the correct structure"""
    if not os.path.exists(JAW_LIPS_RESULTS_FILE):
        print(f"Error: Results file not found at {JAW_LIPS_RESULTS_FILE}")
        print("Please run the Alligator optimization first or check file paths.")
        return False
    
    try:
        data = pd.read_csv(JAW_LIPS_RESULTS_FILE)
        expected_columns = ['jaw_period', 'teeth_period', 'lips_period', 'trades', 'sharpe_ratio']
        
        # If using a different column structure, map it
        if not all(col in data.columns for col in expected_columns):
            print(f"Warning: Some expected columns {expected_columns} not found in {data.columns.tolist()}")
            
            # Check if we can map columns
            if 'short_SMA' in data.columns and 'long_SMA' in data.columns:
                print("Detected SMA column names - renaming to Alligator parameters")
                data = data.rename(columns={
                    'short_SMA': 'lips_period',
                    'long_SMA': 'jaw_period'
                })
                
                if 'teeth_period' not in data.columns:
                    print(f"Adding teeth_period column with value {INITIAL_TEETH}")
                    data['teeth_period'] = INITIAL_TEETH
                
                # Save the modified data
                data.to_csv(JAW_LIPS_RESULTS_FILE, index=False)
                print(f"Updated file saved to {JAW_LIPS_RESULTS_FILE}")
                
            else:
                return False
        
        # Quick validation
        if len(data) == 0:
            print("Error: Results file is empty.")
            return False
            
        print(f"Results file validated. Contains {len(data)} rows of optimization data.")
        return True
    
    except Exception as e:
        print(f"Error checking results file: {e}")
        return False


def main():
    """Main execution function"""
    print("\n======================================================")
    print("         CLASSIC ALLIGATOR STRATEGY CLUSTER ANALYSIS          ")
    print("======================================================\n")
    
    # Make sure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check if the results file exists and has the correct structure
    if not check_results_file():
        return
    
    # Run the basic analysis and create the heatmap
    print("\n----- STEP 1: Basic Analysis and Heatmap -----")
    data, best_jaw, best_lips, best_sharpe, best_trades = analyze_alligator_results()
    
    if data is None:
        print("Error in basic analysis. Stopping.")
        return
    
    # Run the hierarchical cluster analysis
    print("\n----- STEP 2: Hierarchical Clustering -----")
    X_filtered, medoids, top_medoids, max_sharpe_point, labels = hierarchical_cluster_analysis()
    
    if X_filtered is None:
        print("Error in hierarchical clustering. Stopping.")
        return
    
    # Get the teeth period from the results file
    try:
        with open(JAW_LIPS_RESULTS_FILE, 'r') as f:
            header = f.readline().strip().split(',')
            if 'teeth_period' in header:
                teeth_idx = header.index('teeth_period')
                # Read first data line to get teeth_period
                data_line = f.readline().strip().split(',')
                teeth_period = int(data_line[teeth_idx])
                print(f"Using teeth period from file: {teeth_period}")
            else:
                teeth_period = INITIAL_TEETH
                print(f"Using default teeth period: {teeth_period}")
    except Exception as e:
        teeth_period = INITIAL_TEETH
        print(f"Error reading teeth period, using default: {teeth_period}")
    
    # Plot performance comparison
    print("\n----- STEP 3: Performance Comparison -----")
    market_data = plot_alligator_performance(
        jaw_period=int(best_jaw),
        lips_period=int(best_lips),
        teeth_period=teeth_period,
        top_medoids=top_medoids
    )
    
    # Run bimonthly out-of-sample comparison
    print("\n----- STEP 4: Bimonthly Out-of-Sample Analysis -----")
    if top_medoids and len(top_medoids) > 0:
        bimonthly_data = bimonthly_out_of_sample_comparison(
            market_data,
            best_jaw=int(best_jaw),
            best_teeth=teeth_period,
            best_lips=int(best_lips),
            top_medoid=top_medoids[0]  # Use the top-performing medoid
        )
    else:
        print("No top medoids found. Cannot run bimonthly comparison.")
    
    print("\n======================================================")
    print("         ANALYSIS COMPLETE - Results in:              ")
    print(f"         {OUTPUT_DIR}")
    print("======================================================\n")


if __name__ == "__main__":
    main()