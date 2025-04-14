#!/usr/bin/env python3
"""
Visualize the results of the search engine evaluation.

This script generates visualizations of the evaluation results,
including comparison charts of the different search engine variants.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
EVALUATION_DIR = Path(__file__).parent / "evaluations"
CHARTS_DIR = Path(__file__).parent / "charts"

def ensure_charts_directory():
    """Create charts directory if it doesn't exist."""
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

def load_evaluation_data(summary_file_path):
    """
    Load evaluation data from the summary file.
    
    Args:
        summary_file_path: Path to the evaluation summary JSON file
        
    Returns:
        Dictionary containing the evaluation summary data
    """
    try:
        with open(summary_file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading evaluation data: {e}")
        return None

def visualize_overall_scores(summary_data, output_file=None):
    """
    Create a bar chart of overall scores for each search engine variant.
    
    Args:
        summary_data: Evaluation summary data
        output_file: Optional path to save the chart
    """
    variant_scores = summary_data["variant_scores"]
    variants = list(variant_scores.keys())
    scores = list(variant_scores.values())
    
    # Sort by score
    sorted_items = sorted(zip(variants, scores), key=lambda x: x[1], reverse=True)
    variants = [item[0] for item in sorted_items]
    scores = [item[1] for item in sorted_items]
    
    plt.figure(figsize=(10, 6))
    
    # Create bar chart
    bars = plt.bar(variants, scores, color=sns.color_palette("viridis", len(variants)))
    
    # Add score labels on top of bars
    for bar, score in zip(bars, scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{score:.2f}",
            ha='center',
            fontweight='bold'
        )
    
    plt.title("Overall Relevance Scores by Search Engine Variant", fontsize=14, fontweight='bold')
    plt.ylabel("Average Relevance Score (1-4)", fontsize=12)
    plt.ylim(0, 4.5)  # Set y-axis limits for better visualization
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=15, ha='right')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logging.info(f"Overall scores chart saved to: {output_file}")
    else:
        plt.show()

def visualize_category_comparison(summary_data, output_file=None):
    """
    Create a grouped bar chart comparing performance across query categories.
    
    Args:
        summary_data: Evaluation summary data
        output_file: Optional path to save the chart
    """
    category_scores = summary_data["query_category_scores"]
    categories = ["high", "moderate", "low"]
    
    # Create DataFrame for easier plotting
    data = {}
    for category in categories:
        cat_data = category_scores.get(category, {})
        data[category] = cat_data
    
    df = pd.DataFrame(data)
    
    # Sort variants by overall score
    variant_order = [item["variant"] for item in summary_data["overall_ranking"]]
    df = df.reindex(variant_order)
    
    # Create the plot
    plt.figure(figsize=(12, 7))
    
    bar_width = 0.25
    x = np.arange(len(df.index))
    
    # Plot bars for each category
    colors = sns.color_palette("viridis", 3)
    for i, category in enumerate(categories):
        plt.bar(
            x + (i - 1) * bar_width,
            df[category],
            width=bar_width,
            label=f"{category.capitalize()} Representation",
            color=colors[i]
        )
    
    # Add labels and legend
    plt.xlabel("Search Engine Variant", fontsize=12)
    plt.ylabel("Average Relevance Score (1-4)", fontsize=12)
    plt.title("Performance by Topic Representation Category", fontsize=14, fontweight='bold')
    plt.xticks(x, df.index, rotation=15, ha='right')
    plt.ylim(0, 4.5)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logging.info(f"Category comparison chart saved to: {output_file}")
    else:
        plt.show()

def load_detailed_evaluations(eval_file_path):
    """
    Load detailed evaluation data.
    
    Args:
        eval_file_path: Path to the detailed evaluations JSON file
        
    Returns:
        Dictionary containing the detailed evaluation data
    """
    try:
        with open(eval_file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading detailed evaluations: {e}")
        return None

def visualize_query_distribution(detailed_evaluations, output_file=None):
    """
    Create a box plot showing the distribution of scores for each variant.
    
    Args:
        detailed_evaluations: Detailed evaluation data
        output_file: Optional path to save the chart
    """
    # Collect all scores for each variant
    variant_scores = {}
    
    for query_id, query_data in detailed_evaluations.items():
        evaluations = query_data.get("evaluations", {})
        
        for variant_name, variant_eval in evaluations.items():
            if variant_name not in variant_scores:
                variant_scores[variant_name] = []
            
            result_evaluations = variant_eval.get("result_evaluations", [])
            for result_eval in result_evaluations:
                variant_scores[variant_name].append(result_eval.get("average_score", 0))
    
    # Create DataFrame
    data = []
    for variant, scores in variant_scores.items():
        for score in scores:
            data.append({"Variant": variant, "Score": score})
    
    df = pd.DataFrame(data)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    sns.boxplot(x="Variant", y="Score", data=df, palette="viridis")
    
    plt.title("Distribution of Relevance Scores by Search Engine Variant", fontsize=14, fontweight='bold')
    plt.xlabel("Search Engine Variant", fontsize=12)
    plt.ylabel("Relevance Score (1-4)", fontsize=12)
    plt.ylim(0, 4.5)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=15, ha='right')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logging.info(f"Score distribution chart saved to: {output_file}")
    else:
        plt.show()

def create_all_visualizations(summary_file_path, detailed_file_path):
    """
    Generate all visualizations for the evaluation results.
    
    Args:
        summary_file_path: Path to the evaluation summary file
        detailed_file_path: Path to the detailed evaluations file
    """
    ensure_charts_directory()
    
    # Load data
    summary_data = load_evaluation_data(summary_file_path)
    if not summary_data:
        return False
    
    detailed_data = load_detailed_evaluations(detailed_file_path)
    if not detailed_data:
        return False
    
    # Create visualizations
    visualize_overall_scores(
        summary_data,
        CHARTS_DIR / "overall_scores.png"
    )
    
    visualize_category_comparison(
        summary_data,
        CHARTS_DIR / "category_comparison.png"
    )
    
    visualize_query_distribution(
        detailed_data,
        CHARTS_DIR / "score_distribution.png"
    )
    
    logging.info(f"All visualizations saved to: {CHARTS_DIR}")
    return True

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Visualize search engine evaluation results")
    parser.add_argument(
        "--summary-file",
        default=str(EVALUATION_DIR / "evaluation_summary.json"),
        help="Path to evaluation summary JSON file"
    )
    parser.add_argument(
        "--detailed-file",
        default=str(EVALUATION_DIR / "evaluations.json"),
        help="Path to detailed evaluations JSON file"
    )
    args = parser.parse_args()
    
    summary_file = Path(args.summary_file)
    detailed_file = Path(args.detailed_file)
    
    if not summary_file.exists():
        logging.error(f"Summary file not found: {summary_file}")
        return 1
    
    if not detailed_file.exists():
        logging.error(f"Detailed evaluations file not found: {detailed_file}")
        return 1
    
    success = create_all_visualizations(summary_file, detailed_file)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 