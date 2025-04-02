"""
Query processing module for latent semantic search.
Implements multiple query representation methods:
- Binary (default)
- TF-IDF
- Log-entropy
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, cast
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# Default paths for data and models
DEFAULT_DATA_DIR = "../data/irCOREdata"
DEFAULT_MODEL_DIR = "../data/processed_data"
DEFAULT_OUTPUT_DIR = "../data/processed_data"

class QueryProcessor:
    """
    Handles the processing of search queries and retrieval of relevant documents.
    """
    
    # Query representation methods
    BINARY = 'binary'
    TFIDF = 'tfidf'
    LOG_ENTROPY = 'log_entropy'
    
    def __init__(self, model_data: Dict[str, Any]):
        """
        Initialize the query processor with model data
        
        Args:
            model_data: Dictionary containing model components:
                - vectorizer: TF-IDF vectorizer
                - svd_model: LSI/SVD model
                - normalized_lsi_vectors: Document vectors in LSI space
                - paper_ids: List of paper IDs corresponding to document vectors
                - field_weights: Dictionary of field weights used in indexing
        """
        self.vectorizer = model_data['vectorizer']
        self.svd_model = model_data['svd_model']
        self.document_vectors = model_data['normalized_lsi_vectors']
        self.paper_ids = model_data['paper_ids']
        self.field_weights = model_data.get('field_weights', {})
        
        # Validate model
        if self.document_vectors.shape[0] != len(self.paper_ids):
            raise ValueError(f"Number of document vectors ({self.document_vectors.shape[0]}) "
                            f"doesn't match number of paper IDs ({len(self.paper_ids)})")
        
        logging.info(f"Query processor initialized with {len(self.paper_ids)} documents "
                    f"and {self.document_vectors.shape[1]} LSI dimensions")
    
    def process_query(self, query_text: str, method: str = BINARY) -> np.ndarray:
        """
        Process a query text and convert it to the LSI vector space
        
        Args:
            query_text: The search query text
            method: Query representation method (binary, tfidf, log_entropy)
            
        Returns:
            Normalized query vector in LSI space
        """
        # 1. Convert query to TF-IDF space using the same vectorizer as documents
        query_tfidf = self.vectorizer.transform([query_text])
        
        # 2. Apply the selected query representation method
        if method == self.BINARY:
            # Binary: Convert to presence/absence (1/0)
            query_tfidf.data = np.ones_like(query_tfidf.data)
            logging.info("Applied binary query representation")
            
        elif method == self.LOG_ENTROPY:
            # Log-entropy: Apply log transformation to term frequencies
            query_tfidf.data = np.log1p(query_tfidf.data)
            logging.info("Applied log-entropy query representation")
            
        elif method == self.TFIDF:
            # Standard TF-IDF is already applied by the vectorizer
            logging.info("Using standard TF-IDF query representation")
            
        else:
            logging.warning(f"UNKNOWN QUERY REPRESENTATION METHOD '{method}', USING BINARY")
            query_tfidf.data = np.ones_like(query_tfidf.data)
        
        # 3. Project into LSI space using the SVD model
        query_lsi = self.svd_model.transform(query_tfidf)
        
        # 4. Normalize to unit length for cosine similarity
        query_lsi_normalized = normalize(query_lsi)
        
        # Cast to ensure correct return type
        return cast(np.ndarray, query_lsi_normalized[0])  # Return as 1D array
    
    def search(self, query_text: str, 
               method: str = BINARY, 
               top_n: int = 100,
               min_score: float = 0.1) -> List[Dict[str, Any]]:
        """
        Search for documents matching the query
        
        Args:
            query_text: The search query
            method: Query representation method
            top_n: Number of top results to return
            min_score: Minimum similarity score to include in results
            
        Returns:
            List of dictionaries with search results
        """
        # Process the query
        query_vector = self.process_query(query_text, method)
        
        # Calculate similarity with all documents
        similarities = cosine_similarity(
            query_vector.reshape(1, -1), 
            self.document_vectors
        )[0]
        
        # Get top N document indices sorted by similarity
        top_indices = similarities.argsort()[::-1][:top_n]
        
        # Create results list
        results = []
        for idx in top_indices:
            score = similarities[idx]
            if score >= min_score:
                results.append({
                    'paper_id': self.paper_ids[idx],
                    'score': float(score),
                    'document_index': int(idx)
                })
        
        logging.info(f"Found {len(results)} results for query: '{query_text}'")
        return results

    def expand_results(self, results: List[Dict[str, Any]], 
                       papers_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Expand search results with document metadata
        
        Args:
            results: List of search result dictionaries
            papers_data: List of paper dictionaries with full metadata
            
        Returns:
            Enhanced results with document metadata
        """
        # Create a lookup dictionary for faster paper retrieval
        papers_by_id = {paper.get('coreId'): paper for paper in papers_data}
        
        expanded_results = []
        for result in results:
            paper_id = result['paper_id']
            paper_data = papers_by_id.get(paper_id)
            
            if paper_data:
                # Create a copy of the result with additional metadata
                expanded_result = result.copy()
                
                # Add selected metadata fields
                expanded_result['title'] = paper_data.get('title', '')
                expanded_result['abstract'] = paper_data.get('abstract', '')
                expanded_result['url'] = paper_data.get('url', '')
                expanded_result['year'] = paper_data.get('year')
                
                # Add topics if available
                topics = paper_data.get('topics')
                if isinstance(topics, list):
                    expanded_result['topics'] = topics
                
                if 'abstract' in result and result['abstract'] is not None:
                    abstract = result['abstract']
                    if len(abstract) > 200:
                        abstract = abstract[:200] + "..."
                    print(f"   Abstract: {abstract}")
                
                expanded_results.append(expanded_result)
            else:
                # If paper not found, just add the original result
                expanded_results.append(result)
                logging.warning(f"Paper with ID {paper_id} not found in papers data")
        
        return expanded_results 