#!/usr/bin/env python3
"""
Prepare Pooled Results for Judge Blender

This script:
1. Reads search results from all variants
2. Creates a unified pool of documents for each query
3. Formats the data for judge_blender.py
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
RESULTS_DIR = Path(__file__).parent / "results"
INPUT_FILE = RESULTS_DIR / "search_results.json"
OUTPUT_FILE = RESULTS_DIR / "pooled_results.json"
TOP_N = 50  # Number of results to pool from each variant


def load_search_results() -> Dict[str, Any]:
    """Load search results from file."""
    logging.info(f"Loading search results from {INPUT_FILE}")
    with open(INPUT_FILE, 'r') as f:
        return json.load(f)


def create_document_pool(search_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a unified pool of unique documents for each query.
    
    Args:
        search_results: Original search results from all variants
        
    Returns:
        Reformatted data with pooled results
    """
    pooled_data = {
        "system_name": "Latent Semantic Index Search Engine",
        "test_queries": {},
        "search_results": {}
    }
    
    for query_id, query_data in search_results.items():
        # Extract query information
        query_info = query_data["query_data"]
        pooled_data["test_queries"][query_id] = query_info
        
        # Create a unified pool of documents
        document_pool = []
        seen_paper_ids = set()
        
        # For each variant, get top N results
        for variant_name, variant_results in query_data["results"].items():
            # Limit to top N from each variant
            variant_top_n = variant_results[:TOP_N] if variant_results else []
            
            for result in variant_top_n:
                paper_id = result.get("paper_id")
                
                # Skip if we've seen this paper before
                if paper_id in seen_paper_ids:
                    continue
                
                # Add to pool and mark as seen
                seen_paper_ids.add(paper_id)
                document_pool.append(result)
        
        # Store pooled results
        pooled_data["search_results"][query_id] = document_pool
        logging.info(f"Query {query_id}: Pooled {len(document_pool)} unique documents from all variants")
    
    return pooled_data


def main():
    """Main entry point."""
    # Load search results
    search_results = load_search_results()
    
    # Create document pool
    pooled_data = create_document_pool(search_results)
    
    # Save pooled data
    logging.info(f"Saving pooled results to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(pooled_data, f, indent=2)
    
    logging.info(f"Pooling complete. Created pool with {len(pooled_data['test_queries'])} queries")
    
    # Print stats
    total_documents = sum(len(docs) for docs in pooled_data["search_results"].values())
    logging.info(f"Total unique documents in pool: {total_documents}")
    logging.info(f"Average documents per query: {total_documents / len(pooled_data['test_queries']):.1f}")


if __name__ == "__main__":
    main() 