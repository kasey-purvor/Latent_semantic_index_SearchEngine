#!/usr/bin/env python3
"""
NDCG Metrics Visualization

This script generates visualizations of NDCG metrics to compare
the performance of different search engine variants.
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
EVALUATION_DIR = Path(__file__).parent / "evaluations"
OUTPUT_DIR = Path(__file__).parent / "visualizations"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_ndcg_metrics(filepath: Optional[str] = None) -> Dict[str, Any]:
    """
    Load NDCG metrics from file. If filepath is None, load the most recent
    NDCG metrics file from the evaluations directory.
    
    Args:
        filepath: Path to the NDCG metrics file (optional)
        
    Returns:
        Dictionary containing NDCG metrics
    """
    if filepath:
        metrics_file = Path(filepath)
    else:
        # Find the most recent NDCG metrics file
        metrics_files = list(EVALUATION_DIR.glob("ndcg_metrics_*.json"))
        if not metrics_files:
            raise FileNotFoundError("No NDCG metrics files found.")
        
        # Sort by modification time (newest first)
        metrics_file = max(metrics_files, key=lambda f: f.stat().st_mtime)
    
    logging.info(f"Loading NDCG metrics from {metrics_file}")
    
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    # Verify that the file has the expected structure
    required_keys = ["query_ndcg", "system_averages", "overall_comparison"]
    missing_keys = [key for key in required_keys if key not in data]
    
    if missing_keys:
        logging.warning(f"NDCG metrics file is missing expected keys: {missing_keys}")
    
    return data

def plot_overall_comparison(ndcg_metrics: Dict[str, Any], output_dir: Path) -> str:
    """
    Create a bar chart showing the average NDCG@10 scores for each system.
    
    Args:
        ndcg_metrics: NDCG metrics data
        output_dir: Directory to save the plot
        
    Returns:
        Path to the saved plot file
    """
    plt.figure(figsize=(14, 10))
    
    # Extract systems and scores from the metrics
    systems = [item["system"] for item in ndcg_metrics["overall_comparison"]]
    scores = [item["average_ndcg@10"] for item in ndcg_metrics["overall_comparison"]]
    
    # Create a dictionary of system names to scores for easier manipulation
    system_scores = dict(zip(systems, scores))
    
    # Split systems into regular and BERT-reranked variants
    bert_systems = [s for s in systems if "BERT reranking" in s]
    regular_systems = [s for s in systems if s not in bert_systems]
    
    # Move BM25 baseline to be first in lists (which will be rightmost in the plot)
    bm25_system = next((s for s in regular_systems if "BM25" in s or "baseline" in s.lower()), None)
    if bm25_system:
        regular_systems.remove(bm25_system)
        regular_systems = [bm25_system] + regular_systems
    
    # Combine the lists: regular systems first, then BERT-reranked variants
    # This will place regular systems on the left and BERT-reranked on the right in the plot
    ordered_systems = regular_systems + bert_systems
    ordered_scores = [system_scores[s] for s in ordered_systems]
    
    # Reverse for plotting (so BM25 is rightmost)
    ordered_systems.reverse()
    ordered_scores.reverse()
    
    # Use different colors for regular vs BERT-reranked systems
    colors = []
    for system in ordered_systems:
        if "BERT reranking" in system:
            # Use a different color palette for BERT-reranked systems
            colors.append(plt.cm.plasma(0.2))
        elif "BM25" in system or "baseline" in system.lower():
            # Highlight BM25 baseline in a distinct color
            colors.append(plt.cm.viridis(0.9))
        else:
            # Use the regular color palette for other systems
            colors.append(plt.cm.viridis(0.5))
    
    # Calculate the min/max values for better scaling
    min_score = min(ordered_scores)
    max_score = max(ordered_scores)
    
    # Set the y-axis lower limit to better visualize small differences
    # If values are very close, set the lower limit to a percentage of the minimum
    if max_score - min_score < 0.1 and min_score > 0:
        # Start y-axis at e.g., 80% of the min score to magnify differences
        lower_limit = max(0, min_score * 0.8)
    else:
        lower_limit = 0
    
    # Create the bar chart with hatching pattern for BERT-reranked variants
    bars = []
    for i, (system, score) in enumerate(zip(ordered_systems, ordered_scores)):
        if "BERT reranking" in system:
            bar = plt.bar(i, score, color=colors[i], hatch='/', alpha=0.9)
        else:
            bar = plt.bar(i, score, color=colors[i])
        bars.append(bar[0])  # Store the bar object for label positioning
    
    plt.title("Average NDCG@10 Scores by Search System", fontsize=18)
    plt.xlabel("Search System", fontsize=16)
    plt.ylabel("NDCG@10", fontsize=16)
    
    # Set y-axis limits to highlight differences
    plt.ylim(lower_limit, max_score * 1.1)
    
    # Add value labels on top of each bar with more precision for small differences
    for bar, score in zip(bars, ordered_scores):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + (max_score - lower_limit) * 0.01,  # Small offset based on y-range
            f"{height:.5f}",  # More decimal places for precision
            ha='center',
            va='bottom',
            fontsize=14,
            fontweight='bold'
        )
    
    # Add a horizontal dashed line at the mean score to help visualize differences
    mean_score = sum(ordered_scores) / len(ordered_scores)
    plt.axhline(y=mean_score, color='red', linestyle='--', alpha=0.5, 
                label=f"Mean: {mean_score:.5f}")
    
    # Add gridlines to help compare values
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add a legend for the different system types
    if any("BERT reranking" in s for s in ordered_systems):
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=plt.cm.viridis(0.5), label='Regular Search'),
            Patch(facecolor=plt.cm.plasma(0.2), hatch='/', label='With BERT Reranking'),
            Patch(facecolor=plt.cm.viridis(0.9), label='BM25 Baseline')
        ]
        plt.legend(handles=legend_elements, fontsize=12, loc='upper right')
    else:
        plt.legend(fontsize=12)
    
    # Adjust x-axis tick labels and positions
    plt.xticks(range(len(ordered_systems)), [s.replace(' with BERT reranking', '\n+ BERT') for s in ordered_systems], 
               rotation=45, ha='right', fontsize=12)
    
    plt.tight_layout()
    
    # Save the plot in high resolution
    output_file = output_dir / "ndcg_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # Optional: Create a version with a broken y-axis for extreme differences
    if max_score - min_score < 0.1 and min_score > 0:
        plt.close()
        
        # Create a version with a "zoomed in" view
        plt.figure(figsize=(14, 10))
        
        # Create the zoomed bar chart with hatching pattern for BERT-reranked variants
        bars = []
        for i, (system, score) in enumerate(zip(ordered_systems, ordered_scores)):
            if "BERT reranking" in system:
                bar = plt.bar(i, score, color=colors[i], hatch='/', alpha=0.9)
            else:
                bar = plt.bar(i, score, color=colors[i])
            bars.append(bar[0])
        
        plt.title("Average NDCG@10 Scores (Zoomed View)", fontsize=18)
        plt.xlabel("Search System", fontsize=16)
        plt.ylabel("NDCG@10", fontsize=16)
        
        # Set a very narrow y-range to exaggerate differences
        plt.ylim(min_score * 0.98, max_score * 1.02)
        
        # Add value labels
        for bar, score in zip(bars, ordered_scores):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + (max_score - min_score) * 0.05,
                f"{height:.5f}",
                ha='center',
                va='bottom',
                fontsize=14,
                fontweight='bold'
            )
        
        plt.axhline(y=mean_score, color='red', linestyle='--', alpha=0.5, 
                    label=f"Mean: {mean_score:.5f}")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add the same legend as the main plot
        if any("BERT reranking" in s for s in ordered_systems):
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=plt.cm.viridis(0.5), label='Regular Search'),
                Patch(facecolor=plt.cm.plasma(0.2), hatch='/', label='With BERT Reranking'),
                Patch(facecolor=plt.cm.viridis(0.9), label='BM25 Baseline')
            ]
            plt.legend(handles=legend_elements, fontsize=12, loc='upper right')
        else:
            plt.legend(fontsize=12)
        
        plt.xticks(range(len(ordered_systems)), [s.replace(' with BERT reranking', '\n+ BERT') for s in ordered_systems], 
                   rotation=45, ha='right', fontsize=12)
        plt.tight_layout()
        
        # Save the zoomed version
        zoom_output_file = output_dir / "ndcg_comparison_zoomed.png"
        plt.savefig(zoom_output_file, dpi=300, bbox_inches='tight')
        
    plt.close()
    
    logging.info(f"Saved NDCG comparison to {output_file}")
    return str(output_file)

def create_summary_report(ndcg_metrics: Dict[str, Any], plot_path: str, output_dir: Path) -> str:
    """
    Create a simple HTML report summarizing the NDCG results.
    
    Args:
        ndcg_metrics: NDCG metrics data
        plot_path: Path to the main plot
        output_dir: Directory to save the report
        
    Returns:
        Path to the saved report file
    """
    # Create HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>NDCG Evaluation Results</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2, h3 { color: #2c3e50; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { text-align: left; padding: 8px; }
            th { background-color: #3498db; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            .plot { margin: 20px 0; text-align: center; }
            .plot img { max-width: 100%; height: auto; border: 1px solid #ddd; }
            .summary { background-color: #eef; padding: 15px; border-radius: 5px; }
            .bert { background-color: #ffe; }
            .baseline { font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>NDCG Evaluation Results</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <p>This report shows the performance comparison of different search engine variants based on NDCG@10 metrics.</p>
        </div>
        
        <h2>Overall System Performance</h2>
        <table>
            <tr>
                <th>Rank</th>
                <th>System</th>
                <th>Average NDCG@10</th>
            </tr>
    """
    
    # Get systems and sort: BM25 first, then other regular systems, then BERT-reranked variants
    systems = [item["system"] for item in ndcg_metrics["overall_comparison"]]
    scores = [item["average_ndcg@10"] for item in ndcg_metrics["overall_comparison"]]
    
    # Create a dictionary for easy lookup
    system_scores = dict(zip(systems, scores))
    
    # Split systems into regular and BERT-reranked variants
    bert_systems = [(s, system_scores[s]) for s in systems if "BERT reranking" in s]
    regular_systems = [(s, system_scores[s]) for s in systems if s not in [b[0] for b in bert_systems]]
    
    # Separate BM25 baseline
    bm25_system = next(((s, score) for s, score in regular_systems if "BM25" in s or "baseline" in s.lower()), None)
    if bm25_system:
        regular_systems = [sys for sys in regular_systems if sys[0] != bm25_system[0]]
    
    # Sort each group by NDCG score (descending)
    regular_systems.sort(key=lambda x: x[1], reverse=True)
    bert_systems.sort(key=lambda x: x[1], reverse=True)
    
    # Combine with BM25 first, then other regular systems, then BERT-reranked
    all_systems = []
    if bm25_system:
        all_systems.append(bm25_system)
    all_systems.extend(regular_systems)
    all_systems.extend(bert_systems)
    
    # Add system comparison table
    for i, (system, score) in enumerate(all_systems):
        css_class = ""
        if "BM25" in system or "baseline" in system.lower():
            css_class = "baseline"
        elif "BERT reranking" in system:
            css_class = "bert"
            
        html_content += f"""
            <tr class="{css_class}">
                <td>{i+1}</td>
                <td>{system}</td>
                <td>{score:.5f}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>Visualization</h2>
    """
    
    # Add the plot
    plot_filename = os.path.basename(plot_path)
    html_content += f"""
        <div class="plot">
            <img src="{plot_filename}" alt="NDCG Comparison">
        </div>
    """
    
    # Check if zoomed version exists
    zoomed_plot_path = os.path.join(os.path.dirname(plot_path), "ndcg_comparison_zoomed.png")
    if os.path.exists(zoomed_plot_path):
        zoomed_filename = os.path.basename(zoomed_plot_path)
        html_content += f"""
            <div class="plot">
                <h3>Zoomed View (Magnified Differences)</h3>
                <img src="{zoomed_filename}" alt="NDCG Comparison (Zoomed)">
            </div>
        """
    
    html_content += """
        <footer>
            <p>Generated at: <script>document.write(new Date().toLocaleString());</script></p>
        </footer>
    </body>
    </html>
    """
    
    # Save HTML report
    output_file = output_dir / "ndcg_report.html"
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    logging.info(f"Saved evaluation report to {output_file}")
    return str(output_file)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate NDCG metrics visualizations")
    parser.add_argument(
        "--metrics-file",
        type=str,
        help="Path to NDCG metrics JSON file (if not provided, uses the most recent file)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Directory to save visualizations"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load NDCG metrics
        ndcg_metrics = load_ndcg_metrics(args.metrics_file)
        
        # Generate the overall comparison chart
        logging.info("Generating overall comparison chart...")
        plot_path = plot_overall_comparison(ndcg_metrics, output_dir)
        
        # Create a simple HTML report
        logging.info("Generating summary report...")
        report_path = create_summary_report(ndcg_metrics, plot_path, output_dir)
        
        logging.info(f"Visualization saved to {output_dir}")
        logging.info(f"Summary report available at: {report_path}")
        
        return 0
    except Exception as e:
        logging.error(f"Error generating visualizations: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 