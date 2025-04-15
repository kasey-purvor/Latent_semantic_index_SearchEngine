"""
BERT Reranker implementation.
Reranks search results using BERT embeddings and FAISS for efficient filtering.
"""

import os
import numpy as np
import faiss
import logging
import time
from typing import Dict, List, Any, Optional
from sentence_transformers import SentenceTransformer

class BertFaissReranker:
    """
    Reranks search results using BERT embeddings with FAISS for efficient filtering.
    
    This implementation:
    1. Takes the search results from the base search engine
    2. Computes BERT embeddings for just these specific results
    3. Uses FAISS to efficiently find the most semantically similar results to the query
    4. Returns the top k results ranked by BERT similarity
    
    This avoids ID matching issues since we're working directly with the search results.
    """
    
    def __init__(self, model_name: str = "C-KAI/sbert-academic-group44"):
        """
        Initialize the BERT reranker.
        
        Args:
            model_name: Name or path of the SentenceTransformer model to use
        """
        self.model_name = model_name
        self.model = None
        self.model_path = None
    
    def initialize(self, cache_dir: str = "../data/processed_data/faiss_bert_reranking"):
        """
        Initialize the BERT model.
        
        Args:
            cache_dir: Directory to save/load model
        """
        os.makedirs(cache_dir, exist_ok=True)
        
        self.model_path = os.path.join(cache_dir, "bert_model")
        
        # Check if model is already saved
        if os.path.exists(self.model_path):
            logging.info(f"Loading BERT model from {self.model_path}")
            self.model = SentenceTransformer(self.model_path)
        else:
            logging.info(f"Downloading BERT model {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            # Save model for future use
            logging.info(f"Saving BERT model to {self.model_path}")
            self.model.save(self.model_path)
    
    def build_index(self, documents: List[Dict[str, Any]], force_rebuild: bool = False) -> None:
        """
        This method is maintained for backward compatibility.
        In the new implementation, we don't pre-build a FAISS index - we just need the BERT model.
        
        Args:
            documents: List of document dictionaries (not used)
            force_rebuild: Force rebuild flag (not used)
        """
        logging.info("The revised BERT reranking approach doesn't require pre-building an index.")
        logging.info("We only need to download and save the BERT model, which is done in initialize().")
        
        # Make sure model is initialized
        if self.model is None:
            self.initialize()
    
    def rerank(self, query: str, search_results: List[Dict[str, Any]], top_k: int = 10, faiss_filter_k: int = None) -> List[Dict[str, Any]]:
        """
        Rerank search results using BERT embeddings with FAISS for efficient filtering.
        
        Args:
            query: Search query string
            search_results: List of search results from a base indexer
            top_k: Number of results to return after reranking
            faiss_filter_k: Number of results to retrieve using FAISS filtering (if None, use all results)
            
        Returns:
            List of reranked search results
        """
        if self.model is None:
            raise ValueError("BERT model not initialized. Call initialize() first.")
        
        if not search_results:
            logging.warning("No search results to rerank")
            return []
            
        logging.info(f"Reranking {len(search_results)} results with BERT")
        
        # Extract text from search results, including body text with truncation
        search_texts = []
        max_tokens = 500  # Maximum tokens per document
        
        for result in search_results:
            # Get title (handle None/empty)
            title = result.get('title', '') or ''
            
            # Get abstract (handle None/empty)
            abstract = result.get('abstract', '') or ''
            
            # Get body (try different possible field names, handle None/empty)
            body = (result.get('body', '') or 
                   result.get('fullText', '') or 
                   result.get('body_text', '') or 
                   result.get('full_text', '') or 
                   result.get('text', '') or '')
            
            # Combine all fields
            combined = f"{title} {abstract} {body}".strip()
            
            # Simple token count approximation (split by spaces)
            # This is a rough approximation; BERT tokenization may differ
            tokens = combined.split()
            if len(tokens) > max_tokens:
                # Truncate to max_tokens
                combined = ' '.join(tokens[:max_tokens])
            
            if combined:
                search_texts.append(combined)
            else:
                search_texts.append("No title, abstract, or body text available")
        
        # Encode query
        logging.info("Encoding query with BERT")
        query_vector = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vector)
        
        # Encode search results
        logging.info(f"Encoding {len(search_texts)} search results with BERT")
        search_vectors = self.model.encode(search_texts, convert_to_numpy=True, show_progress_bar=True)
        faiss.normalize_L2(search_vectors)
        
        # Create a temporary FAISS index with just these search result vectors
        logging.info("Building temporary FAISS index for filtering")
        d = search_vectors.shape[1]  # Embedding dimension
        index = faiss.IndexFlatIP(d)  # Inner product for cosine similarity with normalized vectors
        index.add(search_vectors)
        
        # If faiss_filter_k is not specified, use all results
        if faiss_filter_k is None or faiss_filter_k >= len(search_results):
            faiss_filter_k = len(search_results)
        
        # Use FAISS to efficiently find the most similar results to the query
        logging.info(f"Using FAISS to find top {faiss_filter_k} most similar results")
        similarities, indices = index.search(query_vector, faiss_filter_k)
        
        # Extract the filtered results and add BERT scores
        reranked_results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(search_results):  # Safety check
                result = search_results[idx].copy()  # Copy to avoid modifying original
                result['bert_score'] = float(similarities[0][i])
                result['original_score'] = result.get('score', 0.0)  # Save original score
                result['score'] = result['bert_score']  # Use BERT score as primary score
                reranked_results.append(result)
        
        # Sort by BERT score (descending) and limit to top_k
        reranked_results = sorted(reranked_results, key=lambda x: x.get('bert_score', 0), reverse=True)[:top_k]
        
        logging.info(f"Reranking complete. Returning top {len(reranked_results)} results ranked by BERT similarity.")
        return reranked_results 