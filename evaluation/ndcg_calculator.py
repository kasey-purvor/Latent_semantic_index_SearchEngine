#!/usr/bin/env python3
"""
NDCG@10 Calculator for Search Engine Evaluation

This script calculates nDCG@10 scores for each search engine variant based on:
1. The original search results (ranking order) from each engine
2. The relevance scores assigned by JudgeBlender
"""

import os
import json
import math
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
RESULTS_DIR = Path(__file__).parent / "results"
EVALUATION_DIR = Path(__file__).parent / "evaluations"
OUTPUT_DIR = EVALUATION_DIR

def load_original_search_results() -> Dict[str, Any]:
    """Load original search results with original ranking order."""
    results_file = RESULTS_DIR / "search_results.json"
    logging.info(f"Loading original search results from {results_file}")
    with open(results_file, 'r') as f:
        return json.load(f)

def load_judgeblender_evaluations() -> Dict[str, Any]:
    """Load the most recent JudgeBlender evaluations."""
    # We want to exclude summary files which don't have the detailed evaluations
    eval_files = list(EVALUATION_DIR.glob("pooled_results_eval_*.json"))
    eval_files = [f for f in eval_files if not f.name.endswith("_summary.json")]
    
    if not eval_files:
        raise FileNotFoundError("No evaluation files found")
    
    # Sort by modification time (newest first)
    latest_eval = max(eval_files, key=lambda f: f.stat().st_mtime)
    logging.info(f"Loading JudgeBlender evaluations from {latest_eval}")
    with open(latest_eval, 'r') as f:
        data = json.load(f)
    
    # Debug: log the keys in the evaluation file
    logging.info(f"Evaluation file keys: {list(data.keys())}")
    if "search_results" not in data and "query_evaluations" not in data:
        logging.error("Expected 'search_results' or 'query_evaluations' key not found in evaluation file")
        # Check if search_results are stored under a different key
        logging.info(f"Available keys: {list(data.keys())}")
    
    return data

def calculate_dcg(scores: List[float]) -> float:
    """
    Calculate DCG for a list of relevance scores.
    
    Uses the continuous floating-point relevance scores without rounding.
    
    Args:
        scores: List of floating-point relevance scores
        
    Returns:
        DCG value
    """
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(scores))

def calculate_ndcg(scores: List[float]) -> float:
    """
    Calculate nDCG for a list of relevance scores.
    
    Args:
        scores: List of floating-point relevance scores
        
    Returns:
        nDCG value between 0.0 and 1.0
    """
    dcg = calculate_dcg(scores)
    # Create ideal ordering (sorted by relevance score, descending)
    ideal_scores = sorted(scores, reverse=True)
    idcg = calculate_dcg(ideal_scores)
    return dcg / idcg if idcg > 0 else 0.0

def compute_ndcg_metrics(original_results: Dict[str, Any], 
                         evaluations: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute nDCG@10 metrics for each search engine variant.
    
    Args:
        original_results: Original search results with ranking order
        evaluations: JudgeBlender evaluations
        
    Returns:
        Dictionary with nDCG@10 metrics
    """
    metrics = {
        "query_ndcg": {},
        "system_averages": {},
        "overall_comparison": {}
    }
    
    # Debug: log the top-level structure of evaluations
    logging.info(f"Evaluation file keys: {list(evaluations.keys())}")
    
    # More detailed logging of the evaluations structure
    if 'query_evaluations' in evaluations:
        query_ids = list(evaluations['query_evaluations'].keys())
        logging.info(f"Found {len(query_ids)} queries in evaluation data: {query_ids[:5]}...")
        
        sample_query_id = query_ids[0]
        sample_query = evaluations['query_evaluations'][sample_query_id]
        logging.info(f"Sample query {sample_query_id} keys: {list(sample_query.keys())}")
        
        if 'result_evaluations' in sample_query:
            result_count = len(sample_query['result_evaluations'])
            logging.info(f"Sample query {sample_query_id} has {result_count} result_evaluations")
            if result_count > 0:
                sample_result = sample_query['result_evaluations'][0]
                logging.info(f"Sample result keys: {list(sample_result.keys())}")
                if 'paper_id' in sample_result:
                    logging.info(f"Sample paper_id: {sample_result['paper_id']}")
    
    # Debug: log the structure of original_results
    original_query_ids = list(original_results.keys())
    logging.info(f"Found {len(original_query_ids)} queries in original results: {original_query_ids[:5]}...")
    sample_query_id = original_query_ids[0]
    sample_query = original_results[sample_query_id]
    logging.info(f"Original results sample query {sample_query_id} keys: {list(sample_query.keys())}")
    
    if 'results' in sample_query:
        engines = list(sample_query['results'].keys())
        logging.info(f"Found {len(engines)} search engine variants: {engines}")
        for engine in engines:
            result_count = len(sample_query['results'][engine])
            logging.info(f"Engine {engine} has {result_count} results for query {sample_query_id}")
            if result_count > 0:
                sample_result = sample_query['results'][engine][0]
                logging.info(f"Sample search result keys: {list(sample_result.keys())}")
                if 'paper_id' in sample_result:
                    logging.info(f"Sample search result paper_id: {sample_result['paper_id']}")
    
    # Extract evaluation results from query_evaluations structure
    eval_results = {}
    if 'query_evaluations' in evaluations:
        logging.info("Found 'query_evaluations' structure - extracting result evaluations")
        for query_id, query_data in evaluations['query_evaluations'].items():
            eval_results[query_id] = query_data.get('result_evaluations', [])
    else:
        # Try to find the search results in the evaluations file using old approach
        search_results_key = None
        for key in evaluations.keys():
            if key == "search_results":
                search_results_key = key
                break
            # Check if it might be under a different key
            if isinstance(evaluations[key], dict) and any(isinstance(v, list) for v in evaluations[key].values()):
                # Inspect the structure to see if it contains the expected data
                sample_values = next((v for v in evaluations[key].values() if isinstance(v, list)), [])
                if sample_values and isinstance(sample_values[0], dict) and "paper_id" in sample_values[0]:
                    search_results_key = key
                    logging.info(f"Found potential search results under key: {key}")
                    break
        
        if search_results_key:
            eval_results = evaluations[search_results_key]
            logging.info(f"Using '{search_results_key}' as the source of evaluation data")
        else:
            # If we can't determine the structure, check if 'search_results' exists
            eval_results = evaluations.get("search_results", {})
            if not eval_results:
                logging.error("Could not find search results in the evaluation file")
                # Look for alternative structures
                if "test_queries" in evaluations and isinstance(evaluations["test_queries"], dict):
                    logging.info("Found 'test_queries' - checking structure")
                    sample_query = next(iter(evaluations["test_queries"].values()))
                    logging.info(f"Sample query structure: {list(sample_query.keys())}")
    
    # Count how many queries we have in the original results
    logging.info(f"Number of queries in original results: {len(original_results)}")
    first_query_id = next(iter(original_results.keys()))
    logging.info(f"First query structure: {list(original_results[first_query_id].keys())}")
    
    # Debug: Check if each query has a 'results' field
    missing_results = []
    for query_id, query_data in original_results.items():
        if 'results' not in query_data:
            missing_results.append(query_id)
    if missing_results:
        logging.error(f"Queries missing 'results' field: {missing_results}")
    
    # Helper function to normalize titles for comparison
    def normalize_title(title):
        if not title:
            return ""
        # Convert to lowercase, remove extra spaces, and strip punctuation
        return ' '.join(title.lower().strip().split())
    
    # For each query
    for query_id, query_data in original_results.items():
        query_metrics = {}
        
        # Get evaluated results for this query
        query_evaluations = eval_results.get(query_id, [])
        logging.info(f"Query {query_id}: found {len(query_evaluations)} evaluation results")
        
        # Debug: if we have no results for a query but there should be some, log more details
        if not query_evaluations and query_id in evaluations.get('query_evaluations', {}):
            logging.error(f"Query {query_id} exists in evaluations but has no results in eval_results!")
            evaled_query = evaluations['query_evaluations'][query_id]
            if 'result_evaluations' in evaled_query:
                result_count = len(evaled_query['result_evaluations'])
                logging.error(f"Query {query_id} actually has {result_count} result_evaluations in the raw data")
                if result_count > 0:
                    sample_result = evaled_query['result_evaluations'][0]
                    logging.error(f"Sample result keys: {list(sample_result.keys())}")
        
        # Create mapping of normalized title to relevance score
        title_scores = {}
        paper_scores = {}
        missing_blended_scores = 0
        
        # First, create a title lookup dict for the evaluation results
        for result in query_evaluations:
            paper_id = result.get("paper_id")
            title = result.get("title")
            normalized_title = normalize_title(title)
            
            # Get the score (blended_score or fallback)
            score = None
            if "blended_score" in result:
                score = result["blended_score"]
            else:
                missing_blended_scores += 1
                # Try fallbacks
                if "average_score" in result:
                    score = result["average_score"]
                    logging.info(f"Using 'average_score' as fallback for result {title or paper_id}")
                elif "relevance_score" in result:
                    score = result["relevance_score"]
                    logging.info(f"Using 'relevance_score' as fallback for result {title or paper_id}")
                elif "score" in result:
                    score = result["score"]
                    logging.info(f"Using 'score' as fallback for result {title or paper_id}")
            
            # Store in both lookups if we have a score
            if score is not None:
                if paper_id:
                    paper_scores[paper_id] = score
                if title:
                    title_scores[normalized_title] = score
        
        if missing_blended_scores > 0:
            logging.warning(f"Query {query_id}: {missing_blended_scores} results missing 'blended_score'")
        
        logging.info(f"Query {query_id}: mapped {len(paper_scores)} paper IDs and {len(title_scores)} normalized titles to scores")
        
        # Debug: Show some sample title mappings
        sample_titles = list(title_scores.keys())[:3]
        if sample_titles:
            logging.info(f"Sample normalized titles: {sample_titles}")
        
        # For each search engine variant
        for variant_name, variant_results in query_data.get("results", {}).items():
            # Get top 10 results
            top10_results = variant_results[:10]
            logging.info(f"Query {query_id}, Variant {variant_name}: {len(top10_results)} top results")
            
            # Get relevance scores in the original order
            relevance_scores = []
            missing_scores = 0
            title_matches = 0
            
            # Collect original search result titles for debugging
            search_result_titles = []
            
            for result in top10_results:
                paper_id = result.get("paper_id")
                title = result.get("title")
                search_result_titles.append(title)
                normalized_title = normalize_title(title)
                
                # Try paper_id first, fall back to normalized title
                score = 0.0
                
                if paper_id and paper_id in paper_scores:
                    score = paper_scores[paper_id]
                elif normalized_title and normalized_title in title_scores:
                    score = title_scores[normalized_title]
                    title_matches += 1
                else:
                    # Try fuzzy matching if exact match fails
                    matched = False
                    if normalized_title:
                        for eval_title in title_scores.keys():
                            # Simple substring check - if eval title contains search result title or vice versa
                            if normalized_title in eval_title or eval_title in normalized_title:
                                score = title_scores[eval_title]
                                title_matches += 1
                                matched = True
                                logging.info(f"Fuzzy match: '{normalized_title}' ~ '{eval_title}'")
                                break
                    
                    if not matched:
                        missing_scores += 1
                
                relevance_scores.append(score)
            
            if title_matches > 0:
                logging.info(f"Query {query_id}, Variant {variant_name}: {title_matches} results matched by title")
            
            if missing_scores > 0:
                logging.warning(f"Query {query_id}, Variant {variant_name}: {missing_scores}/{len(top10_results)} results not found in evaluations")
                # Show the first few titles that didn't match
                logging.info(f"First few search result titles: {search_result_titles[:3]}")
            
            # Check if all scores are zero
            if all(score == 0.0 for score in relevance_scores):
                logging.error(f"Query {query_id}, Variant {variant_name}: All relevance scores are zero!")
            
            # Pad with zeros if less than 10 results
            while len(relevance_scores) < 10:
                relevance_scores.append(0.0)
            
            # Calculate nDCG@10
            ndcg10 = calculate_ndcg(relevance_scores[:10])
            query_metrics[variant_name] = ndcg10
            logging.info(f"Query {query_id}, Variant {variant_name}: nDCG@10 = {ndcg10}")
        
        metrics["query_ndcg"][query_id] = query_metrics
    
    # Calculate average nDCG@10 for each system
    variant_scores = {}
    for query_id, query_metrics in metrics["query_ndcg"].items():
        for variant_name, ndcg in query_metrics.items():
            if variant_name not in variant_scores:
                variant_scores[variant_name] = []
            variant_scores[variant_name].append(ndcg)
    
    # Calculate mean nDCG@10 for each system
    for variant_name, scores in variant_scores.items():
        metrics["system_averages"][variant_name] = sum(scores) / len(scores)
    
    # Sort systems by average nDCG@10 (descending)
    sorted_systems = sorted(
        metrics["system_averages"].items(),
        key=lambda x: x[1],
        reverse=True
    )
    metrics["overall_comparison"] = [
        {"system": name, "average_ndcg@10": score}
        for name, score in sorted_systems
    ]
    
    return metrics

def main():
    """Main entry point."""
    # Load data
    try:
        original_results = load_original_search_results()
        evaluations = load_judgeblender_evaluations()
        
        # Compute nDCG@10 metrics
        ndcg_metrics = compute_ndcg_metrics(original_results, evaluations)
        
        # Save metrics to file
        timestamp = evaluations.get("timestamp", "unknown")
        output_file = OUTPUT_DIR / f"ndcg_metrics_{timestamp.replace(' ', '_').replace(':', '')}.json"
        with open(output_file, 'w') as f:
            json.dump(ndcg_metrics, f, indent=2)
        
        logging.info(f"nDCG@10 metrics saved to {output_file}")
        
        # Print overall comparison
        logging.info("Overall nDCG@10 system comparison:")
        for system in ndcg_metrics["overall_comparison"]:
            logging.info(f"  {system['system']}: {system['average_ndcg@10']:.4f}")
        
        return True
    except Exception as e:
        logging.error(f"Error calculating nDCG@10 metrics: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    main() 