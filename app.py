"""
Flask API for Latent Semantic Search Engine

This module provides a RESTful API interface to the search engine functionality
"""

import os
import sys
import logging
from typing import Dict, List, Any, Optional
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import argparse

# Import search engine functionality
# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.indexing import load_papers
from src.indexing.bm25 import BM25Indexer
from src.indexing.lsi import LSIIndexer
from src.indexing.field_weighted_lsi import FieldWeightedLSIIndexer
from src.indexing.bert_enhanced import BertEnhancedIndexer
from src.faiss_bert_reranking import BertFaissReranker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Constants
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "irCOREdata")
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "data", "processed_data")

# Available index types
INDEX_TYPES = {
    'bm25': 'BM25 baseline',
    'lsi_basic': 'Basic LSI with k=150 dimensions',
    'lsi_field_weighted': 'Field-weighted LSI',
    'lsi_bert_enhanced': 'Field-weighted LSI with BERT-enhanced indexing'
}

# Query methods with descriptions
QUERY_METHODS = {
    'binary': 'Binary (keyword search, best for specific terms)',
    'tfidf': 'TF-IDF (better for document similarity, finds semantically similar documents)',
    'log_entropy': 'Log-Entropy (advanced weighting, good for varied term importance)'
}

# Cache for loaded indexers to avoid reloading
indexer_cache = {}
papers_data = None
bert_reranker = None

def get_indexer(index_type: str):
    """
    Get or create an indexer instance
    
    Args:
        index_type: Type of index to use
        
    Returns:
        Initialized indexer instance
    """
    global indexer_cache
    
    if index_type in indexer_cache:
        return indexer_cache[index_type]
    
    # Initialize the appropriate indexer
    if index_type == 'bm25':
        indexer = BM25Indexer()
    elif index_type == 'lsi_basic':
        indexer = LSIIndexer()
    elif index_type == 'lsi_field_weighted':
        indexer = FieldWeightedLSIIndexer()
    elif index_type == 'lsi_bert_enhanced':
        indexer = BertEnhancedIndexer()
    else:
        raise ValueError(f"Unknown index type: {index_type}")
    
    # Load the index
    index_dir = os.path.join(DEFAULT_MODEL_DIR, index_type)
    logging.info(f"Loading {index_type} index from {index_dir}")
    try:
        indexer.load_index(index_dir)
        indexer_cache[index_type] = indexer
        return indexer
    except Exception as e:
        logging.error(f"Failed to load index: {e}")
        raise ValueError(f"Failed to load {index_type} index: {e}")

def get_papers_data():
    """
    Load papers data if not already loaded
    
    Returns:
        List of paper dictionaries
    """
    global papers_data
    
    if papers_data is None:
        logging.info(f"Loading papers data from {DEFAULT_DATA_DIR}")
        papers_data = load_papers(DEFAULT_DATA_DIR)
        logging.info(f"Loaded {len(papers_data)} papers")
    
    return papers_data

def get_bert_reranker():
    """
    Initialize BERT reranker if not already initialized
    
    Returns:
        Initialized BertFaissReranker instance or None if initialization fails
    """
    global bert_reranker
    
    if bert_reranker is None:
        try:
            logging.info("Initializing BERT reranker")
            bert_reranker = BertFaissReranker(model_name="C-KAI/sbert-academic-group44")
            bert_reranker.initialize()
        except Exception as e:
            logging.error(f"Failed to initialize BERT reranker: {e}")
            return None
    
    return bert_reranker

@app.route('/api/index-types', methods=['GET'])
def get_index_types():
    """Get available index types"""
    return jsonify({
        'index_types': [
            {'id': key, 'name': value, 'methods': get_supported_methods(key)}
            for key, value in INDEX_TYPES.items()
        ]
    })

def get_supported_methods(index_type: str):
    """Get query methods supported by the index type"""
    if index_type == 'bm25':
        # BM25 only supports binary method
        return [{'id': 'binary', 'name': QUERY_METHODS['binary']}]
    else:
        # Other index types support all methods
        return [{'id': key, 'name': value} for key, value in QUERY_METHODS.items()]

@app.route('/api/search', methods=['POST'])
def search():
    """Search for documents"""
    data = request.json
    
    # Extract parameters from request
    query = data.get('query', '')
    index_type = data.get('indexType', 'lsi_field_weighted')
    method = data.get('method', 'binary')
    top_n = int(data.get('topN', 50))
    reranking_top_k = int(data.get('rerankingTopK', 10))
    use_reranking = bool(data.get('useReranking', True))
    
    # For BM25, always use binary method and disable reranking
    is_bm25 = index_type == 'bm25'
    if is_bm25:
        method = 'binary'
        use_reranking = False
    
    try:
        # Get indexer instance
        indexer = get_indexer(index_type)
        
        # Load papers data for result expansion
        papers = get_papers_data()
        
        # Initialize BERT reranker if needed
        reranker = None
        if use_reranking and not is_bm25:
            reranker = get_bert_reranker()
        
        # Create a query processor to properly handle different query methods
        from src.search.query_processor import QueryProcessor
        
        # IMPORTANT: Create QueryProcessor with necessary data from the indexer
        if hasattr(indexer, 'vectorizer') and hasattr(indexer, 'svd'):
            # For LSI-based indexers, create model_data dictionary for QueryProcessor
            model_data = {
                'vectorizer': indexer.vectorizer,
                'svd_model': indexer.svd,
                'normalized_lsi_vectors': indexer._doc_vectors,
                'paper_ids': [doc['paper_id'] for doc in indexer._doc_info],
                'field_weights': getattr(indexer, '_field_weights', {})
            }
            
            # Create query processor
            query_processor = QueryProcessor(model_data)
            
            # Process query with the specified method
            # Instead of using the indexer's search method directly, use query processor
            results = query_processor.search(query, method=method, top_n=top_n)
        else:
            # Fallback for BM25 or other non-LSI indexers that don't have vectorizer/svd
            results = indexer.search(query, top_k=top_n)
            
        # Add paper_id as string
        for result in results:
            if 'paper_id' in result and result['paper_id'] is not None:
                result['paper_id'] = str(result['paper_id'])
        
        # Apply BERT reranking if enabled and available
        if reranker is not None and use_reranking and not is_bm25:
            logging.info("Applying BERT-FAISS reranking to search results")
            results = reranker.rerank(query, results, top_k=reranking_top_k)
        elif is_bm25:
            logging.info("Skipping BERT-FAISS reranking for BM25 baseline")
            # For BM25, only return top results to match reranked count
            if len(results) > reranking_top_k:
                results = results[:reranking_top_k]
        
        # Expand results with paper metadata
        # Create a lookup dictionary for faster paper retrieval
        papers_by_id = {str(paper.get('coreId')): paper for paper in papers}
        
        expanded_results = []
        for result in results:
            paper_id = result['paper_id']
            paper_data = papers_by_id.get(paper_id, {})
            
            # Create a copy of the result with additional metadata
            expanded_result = result.copy()
            
            # Add selected metadata fields
            expanded_result['title'] = paper_data.get('title', '')
            expanded_result['abstract'] = paper_data.get('abstract', '')
            expanded_result['url'] = paper_data.get('url', '')
            expanded_result['year'] = paper_data.get('year')
            
            # Add topics if available
            topics = paper_data.get('topics', [])
            if isinstance(topics, list):
                expanded_result['topics'] = topics
            
            expanded_results.append(expanded_result)
        
        # Return search results
        return jsonify({
            'query': query,
            'indexType': index_type,
            'method': method,
            'useReranking': use_reranking and not is_bm25,
            'totalResults': len(expanded_results),
            'results': expanded_results
        })
        
    except Exception as e:
        logging.error(f"Search error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Latent Semantic Search Engine API')
    parser.add_argument('--host', default='0.0.0.0', help='Host to run the API on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the API on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Pre-load indexers and papers data
    try:
        # Load papers data
        get_papers_data()
        
        # Load indexers
        for index_type in INDEX_TYPES.keys():
            try:
                get_indexer(index_type)
                logging.info(f"Successfully loaded {index_type} index")
            except Exception as e:
                logging.warning(f"Could not pre-load {index_type} index: {e}")
        
        # Initialize BERT reranker
        get_bert_reranker()
        
    except Exception as e:
        logging.error(f"Initialization error: {e}", exc_info=True)
    
    # Run the Flask app
    app.run(host=args.host, port=args.port, debug=args.debug) 