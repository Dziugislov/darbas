#!/usr/bin/env python
"""
Main script to run portfolio analysis steps in sequence:
1. portfolio_data_gather.py - Load data and run SMA optimization for portfolio
2. portfolio_data_analysis.py - Perform K-means clustering analysis on portfolio
3. portfolio_data_analysis_hierarchy.py - Perform hierarchical clustering analysis on portfolio
"""

import os
import sys
import subprocess
import time

def run_script(script_name):
    """Run a Python script and wait for it to complete"""
    print("\n" + "="*80)
    print(f"RUNNING: {script_name}")
    print("="*80 + "\n")
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, script_name)
    
    # Check if the script exists
    if not os.path.exists(script_path):
        print(f"Error: Script {script_path} not found!")
        return False
    
    # Run the script using the same Python interpreter
    python_executable = sys.executable
    try:
        # Using subprocess.run to capture output
        result = subprocess.run(
            [python_executable, script_path],
            check=True,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        return False

def main():
    """Main function to run all portfolio analysis scripts in sequence"""
    start_time = time.time()
    
    print("\n📊 PORTFOLIO SMA STRATEGY ANALYSIS PIPELINE 📊")
    print("Running all portfolio analysis steps in sequence...\n")
    
    # Step 1: Portfolio Data Gathering and Optimization
    if not run_script("portfolio_data_gather.py"):
        print(f"❌ Portfolio data gathering failed. Stopping pipeline.")
        return
    
    print(f"\n✅ Portfolio data gathering and optimization completed successfully.")
    
    # Step 2: K-means Clustering Analysis for Portfolio
    if not run_script("portfolio_data_analysis.py"):
        print(f"❌ K-means clustering analysis for portfolio failed. Continuing to next step...\n")
    else:
        print(f"\n✅ K-means clustering analysis for portfolio completed successfully.")
    
    # Step 3: Hierarchical Clustering Analysis for Portfolio
    if not run_script("portfolio_data_analysis_hierarchy.py"):
        print(f"❌ Hierarchical clustering analysis for portfolio failed.")
    else:
        print(f"\n✅ Hierarchical clustering analysis for portfolio completed successfully.")
    
    # Calculate total runtime
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "="*80)
    print(f"PORTFOLIO ANALYSIS PIPELINE COMPLETE")
    print(f"Total runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()