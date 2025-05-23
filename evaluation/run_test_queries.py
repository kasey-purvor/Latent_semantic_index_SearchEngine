#!/usr/bin/env python3
"""
Run all test queries against each search engine variant and save the results.
This script automates the process of querying each search engine variant with
the 30 test queries and storing the results for evaluation.

Each search engine variant is run both with and without BERT reranking to
enable comparison of the effect of reranking on search quality.
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional

# Add project root to path to allow imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import test queries
from evaluation.test_query_generation.test_queries import get_all_queries

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# Constants
QUERY_RESULTS = 50
SEARCH_ENGINE_VARIANTS = {
    1: "BM25 baseline",
    2: "Basic LSI with k=150 dimensions",
    3: "Field-weighted LSI",
    4: "Field-weighted LSI with BERT-enhanced indexing"
}

INDEX_TYPES = {
    1: "bm25",
    2: "lsi_basic",
    3: "lsi_field_weighted",
    4: "lsi_bert_enhanced"
}

RESULTS_DIR = Path(__file__).parent / "results"

# Caching mechanism to avoid reloading for each query
loaded_indexers = {}  # Cache for loaded indexers
papers_data = None    # Global cache for papers data
bert_reranker = None  # Global cache for BERT reranker


def ensure_results_directory():
    """Create results directory if it doesn't exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    

def load_bert_reranker():
    """Initialize and return the BERT reranker."""
    global bert_reranker
    
    if bert_reranker is None:
        try:
            # Import the reranker
            from faiss_bert_reranking import BertFaissReranker
            
            logging.info("Initializing BERT-FAISS reranker")
            bert_reranker = BertFaissReranker(model_name="C-KAI/sbert-academic-group44")
            bert_reranker.initialize()
            logging.info("BERT-FAISS reranker initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize BERT reranker: {e}")
            return None
    
    return bert_reranker


def run_query_on_search_engine(query: str, variant_id: int, apply_reranking: bool = False, top_n: int = 50):
    """
    Run a query on a specific search engine variant
    
    Args:
        query: The search query to run
        variant_id: The identifier for the search engine variant
        apply_reranking: Whether to apply BERT reranking
        top_n: Number of top results to return (default 50)
        
    Returns:
        List of search results with expanded content
    """
    global papers_data, loaded_indexers
    
    # Import main search functionality
    sys.path.append(str(project_root / "src"))
    
    from indexing import load_papers
    from src.main import DEFAULT_DATA_DIR, DEFAULT_MODEL_DIR
    
    # Load papers data only once across all queries
    if papers_data is None:
        logging.info(f"Loading original papers data from {DEFAULT_DATA_DIR} (once for all queries)")
        papers_data = load_papers(DEFAULT_DATA_DIR)
    
    # Determine which indexer to use and load if not cached
    index_type = INDEX_TYPES[variant_id]
    
    if index_type not in loaded_indexers:
        logging.info(f"Loading {index_type} index (first time)")
        
        if index_type == 'bm25':
            from indexing.bm25 import BM25Indexer
            indexer = BM25Indexer()
        elif index_type == 'lsi_basic':
            from indexing.lsi import LSIIndexer
            indexer = LSIIndexer()
        elif index_type == 'lsi_field_weighted':
            from indexing.field_weighted_lsi import FieldWeightedLSIIndexer
            indexer = FieldWeightedLSIIndexer()
        elif index_type == 'lsi_bert_enhanced':
            from indexing.bert_enhanced import BertEnhancedIndexer
            indexer = BertEnhancedIndexer()
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Load the index
        index_dir = os.path.join(DEFAULT_MODEL_DIR, index_type)
        indexer.load_index(index_dir)
        
        # Cache the loaded indexer
        loaded_indexers[index_type] = indexer
    else:
        # Use cached indexer
        indexer = loaded_indexers[index_type]
        logging.info(f"Using cached {index_type} index")
    
    # Track the applied methods for result naming
    variant_name = SEARCH_ENGINE_VARIANTS[variant_id]
    
    # Search for documents
    logging.info(f"Running query '{query}' on {variant_name}")
    results = indexer.search(query, top_k=top_n)
    
    # Ensure consistent paper_id format
    for result in results:
        if 'paper_id' in result and result['paper_id'] is not None:
            result['paper_id'] = str(result['paper_id'])
    
    # Apply BERT reranking if requested
    if apply_reranking:
        reranker = load_bert_reranker()
        if reranker:
            try:
                logging.info(f"Reranking {len(results)} results with BERT")
                
                # Apply reranking
                reranking_top_k = min(10, len(results))  # Rerank top 10 results or all if less than 10
                reranked_results = reranker.rerank(query, results, top_k=reranking_top_k)
                
                # Preserve original rank and score information
                for i, result in enumerate(reranked_results):
                    if i < len(results):
                        result['original_rank'] = i + 1
                        result['original_score'] = results[i].get('score', 0.0)
                
                logging.info(f"Reranking complete. Returning top {reranking_top_k} results ranked by BERT similarity.")
                results = reranked_results
            except Exception as e:
                logging.error(f"BERT reranking failed: {e}")
                logging.info("Continuing with non-reranked results")
    else:
        logging.info(f"Skipping BERT reranking as per configuration")
    
    # Debug: Check what fields the search engine returns directly
    if results and len(results) > 0:
        logging.info(f"Sample search result fields: {list(results[0].keys())}")
    
    # Expand results with paper data if needed
    expanded_results = []
    for result in results:
        expanded_result = result.copy()
        paper_id = result.get('paper_id')
        
        # Find the corresponding paper in papers_data
        matching_paper = None
        for paper in papers_data:
            if paper.get('coreId') == paper_id:
                matching_paper = paper
                # Add any missing fields from the paper, including full text
                for field in ['title', 'abstract', 'fullText', 'body_text', 'full_text', 'text', 'authors', 'year', 'venue']:
                    if field in paper and field not in expanded_result and paper[field] is not None:
                        expanded_result[field] = paper[field]
                break
        
        # Only log warnings when papers aren't found
        if not matching_paper:
            logging.warning(f"No matching paper found for coreId {paper_id} - ID field mismatch or missing document")
            
        expanded_results.append(expanded_result)
    
    return expanded_results


def run_all_queries():
    """Run all test queries on all search engine variants and save results."""
    ensure_results_directory()
    
    test_queries = get_all_queries()
    all_results = {}
    
    for query_data in test_queries:
        query_id = query_data['id']
        query_text = query_data['query']
        query_results = {}
        
        logging.info(f"Processing query {query_id}: '{query_text}'")
        
        for variant_id in SEARCH_ENGINE_VARIANTS.keys():
            variant_name = SEARCH_ENGINE_VARIANTS[variant_id]
            
            # Run each variant without BERT reranking first
            try:
                results = run_query_on_search_engine(query_text, variant_id, apply_reranking=False)
                query_results[variant_name] = results
                logging.info(f"  Got {len(results)} results from {variant_name}")
            except Exception as e:
                logging.error(f"Error running query {query_id} on {variant_name}: {e}")
            
            # Run with BERT reranking (except for BM25 baseline which should remain pure)
            if variant_id != 1:  # Skip reranking for BM25
                try:
                    reranked_results = run_query_on_search_engine(query_text, variant_id, apply_reranking=True)
                    reranked_name = f"{variant_name} with BERT reranking"
                    query_results[reranked_name] = reranked_results
                    logging.info(f"  Got {len(reranked_results)} results from {reranked_name}")
                except Exception as e:
                    logging.error(f"Error running query {query_id} on {reranked_name}: {e}")
        
        # Store query results
        all_results[query_id] = {
            "query_data": query_data,
            "results": query_results
        }
        
        # Save intermediate results after each query
        results_file = RESULTS_DIR / "search_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logging.info(f"Saved results for query {query_id} to {results_file}")
        
        # Small delay to avoid overwhelming the system
        time.sleep(1)
    
    logging.info(f"All queries completed. Results saved to {results_file}")
    return all_results


if __name__ == "__main__":
    run_all_queries() 