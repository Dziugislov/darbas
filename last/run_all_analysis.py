#!/usr/bin/env python
"""
Main script to run all analysis steps in sequence:
1. data_gather.py - Load data and run SMA optimization
2. data_analysis.py - Perform K-means clustering analysis
3. data_analysis_hierarchy.py - Perform hierarchical clustering analysis
"""

import os
import sys
import subprocess
import time

def update_ticker(symbol):
    """Update the TICKER in input.py"""
    input_file = "input.py"
    with open(input_file, "r") as file:
        lines = file.readlines()

    # Find the line with TICKER and update it
    for i, line in enumerate(lines):
        if line.startswith("TICKER"):
            lines[i] = f"TICKER = '{symbol}'\n"
            break
    
    # Write the updated lines back to the file
    with open(input_file, "w") as file:
        file.writelines(lines)

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
    """Main function to run all analysis scripts in sequence"""
    start_time = time.time()
    
    print("\n📊 SMA STRATEGY ANALYSIS PIPELINE 📊")
    print("Running all analysis steps in sequence...\n")
    
    # List of underlying symbols (Futures)
    underlyings = ['ES=F', 'SI=F', 'GC=F', 'CC']


    # Loop through all underlyings
    for symbol in underlyings:
        print(f"\nRunning analysis for {symbol}...")

        # Update the TICKER in input.py for the current symbol
        update_ticker(symbol)

        # Step 1: Data Gathering and Optimization
        if not run_script("data_gather.py"):
            print(f"❌ Data gathering for {symbol} failed. Stopping pipeline for this symbol.")
            continue
        
        print(f"\n✅ Data gathering and optimization for {symbol} completed successfully.")
        
        # Step 2: K-means Clustering Analysis
        if not run_script("data_analysis.py"):
            print(f"❌ K-means clustering analysis for {symbol} failed. Continuing to next step...\n")
        else:
            print(f"\n✅ K-means clustering analysis for {symbol} completed successfully.")
        
        # Step 3: Hierarchical Clustering Analysis
        if not run_script("data_analysis_hierarchy.py"):
            print(f"❌ Hierarchical clustering analysis for {symbol} failed.")
        else:
            print(f"\n✅ Hierarchical clustering analysis for {symbol} completed successfully.")
    
    # Calculate total runtime
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "="*80)
    print(f"ANALYSIS PIPELINE COMPLETE")
    print(f"Total runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
