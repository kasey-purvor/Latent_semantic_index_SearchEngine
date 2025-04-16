#!/usr/bin/env python3
"""
Run the complete search engine evaluation pipeline.

This script:
1. Runs all test queries through each search engine variant
2. Collects search results
3. Evaluates results using the judge blender system
4. Calculates nDCG@10 metrics for search engine comparison
5. Generates a summary of the evaluations
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_pipeline():
    """Run the complete evaluation pipeline."""
    logging.info("Starting search engine evaluation pipeline")
    
    # Step 1: Run all test queries and collect results
    logging.info("Step 1: Running test queries")
    run_queries_script = Path(__file__).parent / "run_test_queries.py"
    if not run_queries_script.exists():
        logging.error(f"Script not found: {run_queries_script}")
        return False
    
    logging.info(f"Running: {run_queries_script}")
    result = os.system(f"python {run_queries_script}")
    if result != 0:
        logging.error("Failed to run test queries")
        return False
    
    # Step 2: Prepare pooled results
    logging.info("Step 2: Preparing pooled results")
    pooling_script = Path(__file__).parent / "prepare_pooled_results.py"
    if not pooling_script.exists():
        logging.error(f"Script not found: {pooling_script}")
        return False
    
    logging.info(f"Running: {pooling_script}")
    result = os.system(f"python {pooling_script}")
    if result != 0:
        logging.error("Failed to prepare pooled results")
        return False
    
    # Step 3: Run judge blender evaluation
    logging.info("Step 3: Running judge blender evaluation")
    judge_blender_script = Path(__file__).parent / "judge_blender.py"
    if not judge_blender_script.exists():
        logging.error(f"Script not found: {judge_blender_script}")
        return False
    
    # Use the pooled results file
    pooled_results_file = Path(__file__).parent / "results" / "pooled_results.json"
    logging.info(f"Running: {judge_blender_script} --results-file {pooled_results_file}")
    result = os.system(f"python {judge_blender_script} --results-file {pooled_results_file}")
    if result != 0:
        logging.error("Failed to run judge blender evaluation")
        return False
    
    # Step 4: Calculate nDCG@10 metrics
    logging.info("Step 4: Calculating nDCG@10 metrics")
    ndcg_script = Path(__file__).parent / "ndcg_calculator.py"
    if not ndcg_script.exists():
        logging.error(f"Script not found: {ndcg_script}")
        return False
    
    logging.info(f"Running: {ndcg_script}")
    result = os.system(f"python {ndcg_script}")
    if result != 0:
        logging.error("Failed to calculate nDCG@10 metrics")
        return False
    
    # Look for most recent evaluation summary
    eval_dir = Path(__file__).parent / "evaluations"
    summary_files = list(eval_dir.glob("*_summary.json"))
    if not summary_files:
        logging.error(f"No evaluation summary found in {eval_dir}")
        return False
    
    # Sort by modification time (newest first)
    latest_summary = max(summary_files, key=lambda f: f.stat().st_mtime)
    logging.info(f"Evaluation summary generated at: {latest_summary}")
    logging.info("Evaluation pipeline completed successfully")
    return True

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run search engine evaluation pipeline")
    args = parser.parse_args()
    
    success = run_pipeline()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 